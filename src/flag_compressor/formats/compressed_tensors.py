from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from flag_compressor.backends.base import BackendRunContext, QuantBackend
from flag_compressor.formats.base import (
    ArtifactResult,
    WeightFormat,
    register_weight_format,
)


@dataclass(frozen=True)
class CompressedTensorNames:
    weight: str
    scale: str
    shape: str


def compressed_tensor_names(tensor_name: str) -> CompressedTensorNames:
    if not tensor_name.endswith(".weight"):
        raise ValueError(
            "compressed-tensors export requires a source tensor ending in "
            f"'.weight', got {tensor_name!r}"
        )
    prefix = tensor_name[: -len(".weight")]
    return CompressedTensorNames(
        weight=f"{prefix}.weight_packed",
        scale=f"{prefix}.weight_scale",
        shape=f"{prefix}.weight_shape",
    )


class CompressedTensorsInt4GroupwiseFormat(WeightFormat):
    """Stable ``compressed-tensors`` W4A16 ``pack-quantized`` serialization."""

    name = "compressed_tensors_int4_groupwise"

    def from_canonical(
        self,
        tensor_name: str,
        weight: torch.Tensor,
        backend: QuantBackend,
        context: BackendRunContext,
        params: dict[str, Any],
    ) -> ArtifactResult:
        quantizer = params.get("quantizer", "mse")
        if quantizer != "mse":
            raise NotImplementedError(f"Unsupported INT4 quantizer: {quantizer}")
        logical_shape = tuple(int(dim) for dim in weight.shape)
        if len(logical_shape) != 2:
            raise ValueError(
                f"compressed-tensors W4A16 requires a 2D weight, got {logical_shape}"
            )
        int4_values, scales = backend.run(
            "mse_int4_quant",
            weight,
            group_size=int(params.get("group_size", 32)),
            n_candidates=int(params.get("n_candidates", 200)),
            chunk_size=int(params.get("chunk_size", 4096)),
            context=context,
        )
        packed = backend.run(
            "int4_pack_uint4b8_int32",
            int4_values,
            context=context,
        )
        names = compressed_tensor_names(tensor_name)
        shape = torch.tensor(logical_shape, dtype=torch.int64)
        return ArtifactResult(
            tensors={
                names.weight: packed.cpu(),
                names.scale: scales.cpu(),
                names.shape: shape,
            },
            generated_tensor_names=(names.weight, names.scale, names.shape),
        )


register_weight_format(CompressedTensorsInt4GroupwiseFormat())
