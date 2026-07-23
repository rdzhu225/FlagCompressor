from __future__ import annotations

import json
from pathlib import Path

from flag_compressor.core.dtypes import tensor_dtype_name
from flag_compressor.core.profile import ModelProfile, TensorInfo
from flag_compressor.inspect.scale_pairing import build_scale_map
from flag_compressor.inspect.tensor_classifier import (
    classify_weight,
    infer_logical_shape,
    infer_storage_format,
)
from flag_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint


def scan_hf_safetensors(model_path: str | Path) -> ModelProfile:
    model_dir = Path(model_path)
    checkpoint = HfSafetensorsCheckpoint(model_dir)
    scale_map = build_scale_map(set(checkpoint.weight_map.keys()))
    manifest_specs: dict[str, dict] = {}
    manifest_path = model_dir / "quant_manifest.json"
    manifest_abi = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_abi = manifest.get("abi_version")
        manifest_specs = dict(manifest.get("tensors") or {})
        for weight_name, spec in manifest_specs.items():
            scale_name = spec.get("scale")
            if weight_name in checkpoint.weight_map and scale_name in checkpoint.weight_map:
                scale_map[weight_name] = scale_name
    scale_targets = set(scale_map.values())

    profile = ModelProfile(
        model_path=str(model_dir),
        format="hf_safetensors",
        metadata={
            "num_index_keys": len(checkpoint.weight_map),
            "num_shards": len(checkpoint.shard_files()),
            "quant_manifest_abi": manifest_abi,
        },
    )

    shapes: dict[str, tuple[int, ...]] = {}
    raw: dict[str, dict] = {}
    for shard_name, _ in checkpoint.iter_shards():
        state = checkpoint.load_shard(shard_name)
        for tensor_name, tensor in state.items():
            shape = tuple(int(x) for x in tensor.shape)
            shapes[tensor_name] = shape
            raw[tensor_name] = {
                "shape": shape,
                "dtype": tensor_dtype_name(tensor),
                "shard": shard_name,
                "element_size": tensor.element_size(),
            }

    for tensor_name, info in raw.items():
        scale_name = scale_map.get(tensor_name)
        role = "scale" if tensor_name in scale_targets else "weight"
        storage_format = None
        manifest_spec = manifest_specs.get(tensor_name)
        if role == "weight" and scale_name:
            if manifest_spec and manifest_spec.get("format"):
                storage_format = manifest_spec["format"]
            else:
                storage_format = infer_storage_format(
                    info["shape"],
                    info["element_size"],
                    shapes.get(scale_name),
                )
        module_kind, tags = classify_weight(tensor_name) if role == "weight" else (None, ())
        logical_shape = (
            tuple(manifest_spec["logical_shape"])
            if manifest_spec and manifest_spec.get("logical_shape")
            else infer_logical_shape(info["shape"], storage_format)
        )
        profile.tensors[tensor_name] = TensorInfo(
            name=tensor_name,
            shape=info["shape"],
            dtype=info["dtype"],
            shard=info["shard"],
            element_size=info["element_size"],
            scale_name=scale_name,
            role=role,
            storage_format=storage_format,
            logical_shape=logical_shape,
            module_kind=module_kind,
            tags=tags,
        )

    return profile
