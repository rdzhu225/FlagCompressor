from __future__ import annotations

from pathlib import Path

from flag_compressor.core.dtypes import tensor_dtype_name
from flag_compressor.core.profile import ModelProfile, TensorInfo
from flag_compressor.inspect.scale_pairing import build_scale_map
from flag_compressor.inspect.tensor_classifier import classify_weight_module, infer_source_format
from flag_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint


def scan_hf_safetensors(model_path: str | Path) -> ModelProfile:
    checkpoint = HfSafetensorsCheckpoint(model_path)
    scale_map = build_scale_map(set(checkpoint.weight_map.keys()))

    profile = ModelProfile(
        model_path=str(Path(model_path)),
        format="hf_safetensors",
        metadata={
            "num_index_keys": len(checkpoint.weight_map),
            "num_shards": len(checkpoint.shard_files()),
        },
    )

    for shard_name, _ in checkpoint.iter_shards():
        state = checkpoint.load_shard(shard_name)
        for tensor_name, tensor in state.items():
            scale_name = scale_map.get(tensor_name)
            role = "scale" if tensor_name in scale_map.values() else "weight"
            module_kind = classify_weight_module(tensor_name) if role == "weight" else None
            source_format = infer_source_format(
                tensor_name,
                tensor.element_size(),
                scale_name,
                module_kind,
            )
            profile.tensors[tensor_name] = TensorInfo(
                name=tensor_name,
                shape=tuple(int(x) for x in tensor.shape),
                dtype=tensor_dtype_name(tensor),
                shard=shard_name,
                element_size=tensor.element_size(),
                scale_name=scale_name,
                role=role,
                source_format=source_format,
                module_kind=module_kind,
            )

    return profile
