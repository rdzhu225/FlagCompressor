from __future__ import annotations

import json
from pathlib import Path

import torch

from flagos_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint


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
    int8_tensors = 0
    observed_quant_bits: set[int] = set()
    observed_strategies: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        schema = manifest.get("schema")
        if schema != "flagos-compressor.provenance.v1":
            errors.append("Unsupported quantization manifest schema")
        for name, spec in manifest.get("tensors", {}).items():
            tensor_format = spec.get("format")
            if name not in tensor_meta:
                errors.append(f"Manifest weight is missing: {name}")
                continue
            if tensor_format not in {
                "compressed-tensors-pack-quantized-int4",
                "compressed-tensors-pack-quantized-int8",
            }:
                continue
            num_bits = 4 if tensor_format.endswith("int4") else 8
            observed_quant_bits.add(num_bits)
            observed_strategies.add(spec.get("strategy", "group"))
            if spec.get("num_bits", num_bits) != num_bits:
                errors.append(
                    f"Manifest bit width mismatch for {name}: "
                    f"{spec.get('num_bits')} vs format INT{num_bits}"
                )
            if num_bits == 4:
                int4_tensors += 1
            else:
                int8_tensors += 1
            weight_shape, weight_dtype = tensor_meta[name]
            scale_name = spec.get("scale")
            if weight_dtype != torch.int32:
                errors.append(
                    f"INT{num_bits} weight {name} is {weight_dtype}, "
                    f"expected {torch.int32}"
                )
            if list(weight_shape) != spec.get("storage_shape"):
                errors.append(f"INT{num_bits} storage shape mismatch for {name}")
            if scale_name not in tensor_meta:
                errors.append(
                    f"INT{num_bits} scale is missing for {name}: {scale_name}"
                )
            else:
                scale_shape, scale_dtype = tensor_meta[scale_name]
                if scale_dtype != torch.bfloat16:
                    errors.append(
                        f"INT{num_bits} scale {scale_name} is {scale_dtype}, "
                        "expected bfloat16"
                    )
                if list(scale_shape) != spec.get("scale_shape"):
                    errors.append(
                        f"INT{num_bits} scale shape mismatch for {name}"
                    )
            shape_name = spec.get("shape")
            if shape_name not in tensor_meta:
                errors.append(
                    f"INT{num_bits} logical shape tensor is missing: {shape_name}"
                )
            else:
                shape_shape, shape_dtype = tensor_meta[shape_name]
                if shape_shape != (2,) or shape_dtype != torch.int64:
                    errors.append(
                        f"INT{num_bits} logical shape tensor {shape_name} "
                        "must be int64[2]"
                    )
                elif logical_shape_values.get(shape_name) != spec.get(
                    "logical_shape"
                ):
                    errors.append(
                        f"INT{num_bits} logical shape tensor {shape_name} "
                        "has the wrong value"
                    )

        config_path = path / "config.json"
        artifact = manifest.get("artifact") or {}
        if len(observed_quant_bits) == 1:
            manifest_bits = next(iter(observed_quant_bits))
            expected_encoding = (
                "uint4b8" if manifest_bits == 4 else "uint8b128"
            )
            if (
                "num_bits" in artifact
                and artifact.get("num_bits") != manifest_bits
            ):
                errors.append("Manifest artifact bit width is inconsistent")
            if artifact.get("weight_encoding") != expected_encoding:
                errors.append("Manifest artifact weight encoding is inconsistent")
        if (
            len(observed_strategies) == 1
            and "strategy" in artifact
            and artifact.get("strategy") != next(iter(observed_strategies))
        ):
            errors.append("Manifest artifact weight strategy is inconsistent")
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
            config_bits = {
                group.get("weights", {}).get("num_bits")
                for group in (quant_config.get("config_groups") or {}).values()
            }
            config_bits.discard(None)
            if observed_quant_bits and config_bits != observed_quant_bits:
                errors.append(
                    "config.json quantized bit widths do not match the manifest"
                )
            config_strategies = {
                group.get("weights", {}).get("strategy")
                for group in (quant_config.get("config_groups") or {}).values()
            }
            config_strategies.discard(None)
            if (
                observed_strategies
                and config_strategies != observed_strategies
            ):
                errors.append(
                    "config.json weight strategies do not match the manifest"
                )

    return {
        "valid": not errors,
        "errors": errors,
        "tensors": len(observed),
        "shards": len(checkpoint.shard_files()),
        "total_size": total_size,
        "int4_tensors": int4_tensors,
        "int8_tensors": int8_tensors,
        "has_manifest": manifest_path.exists(),
    }
