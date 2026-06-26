from __future__ import annotations

from pathlib import Path

from quant_engine.core.dtypes import tensor_dtype_name
from quant_engine.core.profile import ModelProfile, TensorInfo
from quant_engine.inspect.scale_pairing import build_scale_map
from quant_engine.io.hf_checkpoint import HfSafetensorsCheckpoint


def infer_source_format(name: str, element_size: int, scale_name: str | None) -> str | None:
    if scale_name is None:
        return None
    lower_name = name.lower()
    if element_size == 1 and ("experts" in lower_name or "fp4" in lower_name):
        return "fp4_e2m1_e8m0"
    if element_size == 1:
        return "fp8_block_e8m0"
    return None


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
            source_format = infer_source_format(tensor_name, tensor.element_size(), scale_name)
            profile.tensors[tensor_name] = TensorInfo(
                name=tensor_name,
                shape=tuple(int(x) for x in tensor.shape),
                dtype=tensor_dtype_name(tensor),
                shard=shard_name,
                element_size=tensor.element_size(),
                scale_name=scale_name,
                role=role,
                source_format=source_format,
            )

    return profile

