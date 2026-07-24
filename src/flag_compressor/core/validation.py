from __future__ import annotations

import json
from pathlib import Path

import torch

from flag_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint


def _validate_fused_moe_entry(
    logical_name: str,
    spec: dict,
    tensor_meta: dict,
    logical_shape_values: dict,
    errors: list,
) -> int:
    """Validate the per-expert tensors expanded from a fused-bank manifest entry.

    Returns the number of quantized per-expert projections checked. The bank
    name itself is logical and is not expected on disk; each expert projection
    is stored as ``<experts>.<e>.<proj>.weight_packed`` (+ scale, shape).
    """
    prefix = logical_name.rsplit(".", 1)[0]
    num_experts = int(spec.get("num_experts", 0))
    proj_kind = spec.get("proj_kind")
    out_features, in_features = spec.get("per_expert_logical_shape", [0, 0])
    group_size = int(spec.get("group_size", 32)) or 1
    proj_names = (
        ("gate_proj", "up_proj") if proj_kind == "gate_up_proj" else (proj_kind,)
    )
    expected_packed_shape = [out_features, in_features // 8]
    expected_scale_shape = [out_features, in_features // group_size]
    checked = 0
    for expert_id in range(num_experts):
        for proj_name in proj_names:
            base = f"{prefix}.{expert_id}.{proj_name}"
            weight_name = f"{base}.weight_packed"
            scale_name = f"{base}.weight_scale"
            shape_name = f"{base}.weight_shape"
            if weight_name not in tensor_meta:
                errors.append(f"Fused MoE weight is missing: {weight_name}")
                continue
            checked += 1
            weight_shape, weight_dtype = tensor_meta[weight_name]
            if weight_dtype != torch.int32:
                errors.append(
                    f"Fused MoE weight {weight_name} is {weight_dtype}, expected int32"
                )
            if list(weight_shape) != expected_packed_shape:
                errors.append(f"Fused MoE packed shape mismatch for {weight_name}")
            if scale_name not in tensor_meta:
                errors.append(f"Fused MoE scale is missing: {scale_name}")
            else:
                scale_shape, scale_dtype = tensor_meta[scale_name]
                if scale_dtype != torch.bfloat16:
                    errors.append(
                        f"Fused MoE scale {scale_name} is {scale_dtype}, expected bfloat16"
                    )
                if list(scale_shape) != expected_scale_shape:
                    errors.append(f"Fused MoE scale shape mismatch for {scale_name}")
            if shape_name not in tensor_meta:
                errors.append(f"Fused MoE logical shape tensor is missing: {shape_name}")
            elif logical_shape_values.get(shape_name) != [out_features, in_features]:
                errors.append(
                    f"Fused MoE logical shape tensor {shape_name} has the wrong value"
                )
    return checked


def validate_artifact(model_path: str | Path) -> dict:
    path = Path(model_path)
    checkpoint = HfSafetensorsCheckpoint(path)
    errors: list[str] = []
    observed: set[str] = set()
    tensor_meta: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    logical_shape_values: dict[str, list[int]] = {}
    total_size = 0
    for shard_name, shard_path in checkpoint.iter_shards():
        if not shard_path.exists():
            errors.append(f"Missing shard: {shard_name}")
            continue
        state = checkpoint.load_shard(shard_name)
        expected = {name for name, shard in checkpoint.weight_map.items() if shard == shard_name}
        actual = set(state)
        if expected != actual:
            errors.append(
                f"Shard {shard_name} index mismatch: missing={sorted(expected-actual)[:3]}, "
                f"extra={sorted(actual-expected)[:3]}"
            )
        observed.update(actual)
        tensor_meta.update({name: (tuple(tensor.shape), tensor.dtype) for name, tensor in state.items()})
        logical_shape_values.update(
            {
                name: [int(value) for value in tensor.tolist()]
                for name, tensor in state.items()
                if name.endswith(".weight_shape")
                and tensor.dtype == torch.int64
                and tensor.shape == (2,)
            }
        )
        total_size += sum(t.numel() * t.element_size() for t in state.values())

    if observed != set(checkpoint.weight_map):
        errors.append("Checkpoint index keys do not match stored tensors")
    indexed_size = checkpoint.index.get("metadata", {}).get("total_size")
    if indexed_size is not None and int(indexed_size) != total_size:
        errors.append(f"metadata.total_size={indexed_size} but actual size is {total_size}")

    manifest_path = path / "quantization_manifest.json"
    int4_tensors = 0
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        schema = manifest.get("schema")
        if schema != "flag-compressor.provenance.v1":
            errors.append("Unsupported quantization manifest schema")
        for name, spec in manifest.get("tensors", {}).items():
            tensor_format = spec.get("format")
            if tensor_format == "compressed-tensors-pack-quantized-int4-moe-fused":
                # ``name`` is the logical fused-bank name, which no longer
                # exists on disk: it was expanded into per-expert projections.
                int4_tensors += _validate_fused_moe_entry(
                    name, spec, tensor_meta, logical_shape_values, errors
                )
                continue
            if name not in tensor_meta:
                errors.append(f"Manifest weight is missing: {name}")
                continue
            if tensor_format != "compressed-tensors-pack-quantized-int4":
                continue
            int4_tensors += 1
            weight_shape, weight_dtype = tensor_meta[name]
            scale_name = spec.get("scale")
            if weight_dtype != torch.int32:
                errors.append(
                    f"INT4 weight {name} is {weight_dtype}, expected {torch.int32}"
                )
            if list(weight_shape) != spec.get("storage_shape"):
                errors.append(f"INT4 storage shape mismatch for {name}")
            if scale_name not in tensor_meta:
                errors.append(f"INT4 scale is missing for {name}: {scale_name}")
            else:
                scale_shape, scale_dtype = tensor_meta[scale_name]
                if scale_dtype != torch.bfloat16:
                    errors.append(
                        f"INT4 scale {scale_name} is {scale_dtype}, expected bfloat16"
                    )
                if list(scale_shape) != spec.get("scale_shape"):
                    errors.append(f"INT4 scale shape mismatch for {name}")
            shape_name = spec.get("shape")
            if shape_name not in tensor_meta:
                errors.append(f"INT4 logical shape tensor is missing: {shape_name}")
            else:
                shape_shape, shape_dtype = tensor_meta[shape_name]
                if shape_shape != (2,) or shape_dtype != torch.int64:
                    errors.append(
                        f"INT4 logical shape tensor {shape_name} must be int64[2]"
                    )
                elif logical_shape_values.get(shape_name) != spec.get(
                    "logical_shape"
                ):
                    errors.append(
                        f"INT4 logical shape tensor {shape_name} has the wrong value"
                    )

        config_path = path / "config.json"
        if not config_path.exists():
            errors.append("compressed-tensors artifact is missing config.json")
        else:
            with config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
            quant_config = config.get("quantization_config") or {}
            if quant_config.get("quant_method") != "compressed-tensors":
                errors.append("config.json does not declare compressed-tensors")
            if quant_config.get("format") != "pack-quantized":
                errors.append("config.json does not declare pack-quantized format")

    return {
        "valid": not errors,
        "errors": errors,
        "tensors": len(observed),
        "shards": len(checkpoint.shard_files()),
        "total_size": total_size,
        "int4_tensors": int4_tensors,
        "has_manifest": manifest_path.exists(),
    }
