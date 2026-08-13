from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from flagos_compressor.backends.base import BackendRunContext, QuantBackend
from flagos_compressor.core.dtypes import parse_w8a8_scale_dtype
from flagos_compressor.formats.base import (
    ArtifactResult,
    WeightFormat,
    register_weight_format,
)


@dataclass(frozen=True)
class CompressedTensorNames:
    weight: str
    scale: str
    shape: str


@dataclass(frozen=True)
class IntQuantizedTensorNames:
    weight: str
    scale: str


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


def int_quantized_tensor_names(tensor_name: str) -> IntQuantizedTensorNames:
    """Names used by compressed-tensors ``int-quantized`` checkpoints."""
    if not tensor_name.endswith(".weight"):
        raise ValueError(
            "compressed-tensors int-quantized export requires a source tensor "
            f"ending in '.weight', got {tensor_name!r}"
        )
    prefix = tensor_name[: -len(".weight")]
    return IntQuantizedTensorNames(
        weight=tensor_name,
        scale=f"{prefix}.weight_scale",
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


class CompressedTensorsInt8GroupwiseFormat(WeightFormat):
    """Stable ``compressed-tensors`` W8A16 ``pack-quantized`` serialization."""

    name = "compressed_tensors_int8_groupwise"
    default_strategy = "group"

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
            raise NotImplementedError(f"Unsupported INT8 quantizer: {quantizer}")
        logical_shape = tuple(int(dim) for dim in weight.shape)
        if len(logical_shape) != 2:
            raise ValueError(
                f"compressed-tensors W8A16 requires a 2D weight, got {logical_shape}"
            )
        strategy = params.get("strategy", self.default_strategy)
        if strategy == "channel":
            effective_group_size = logical_shape[1]
        elif strategy == "group":
            effective_group_size = int(params.get("group_size", 128))
        else:
            raise ValueError(f"Unsupported INT8 weight strategy: {strategy}")
        int8_values, scales = backend.run(
            "mse_int8_quant",
            weight,
            group_size=effective_group_size,
            n_candidates=int(params.get("n_candidates", 200)),
            chunk_size=int(params.get("chunk_size", 1024)),
            context=context,
        )
        packed = backend.run(
            "int8_pack_uint8b128_int32",
            int8_values,
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


register_weight_format(CompressedTensorsInt8GroupwiseFormat())


class CompressedTensorsInt8ChannelwiseFormat(
    CompressedTensorsInt8GroupwiseFormat
):
    """Per-output-channel W8A16 serialization."""

    name = "compressed_tensors_int8_channelwise"
    default_strategy = "channel"


register_weight_format(CompressedTensorsInt8ChannelwiseFormat())


class CompressedTensorsW8A8ChannelwiseFormat(WeightFormat):
    """Per-output-channel weights with dynamic per-token W8A8 activations."""

    name = "compressed_tensors_w8a8_channelwise"

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
            raise NotImplementedError(f"Unsupported INT8 quantizer: {quantizer}")
        logical_shape = tuple(int(dim) for dim in weight.shape)
        if len(logical_shape) != 2:
            raise ValueError(
                f"compressed-tensors W8A8 requires a 2D weight, got {logical_shape}"
            )
        if params.get("strategy", "channel") != "channel":
            raise ValueError("compressed-tensors W8A8 requires channel weight strategy")

        int8_values, scales = backend.run(
            "mse_int8_quant",
            weight,
            group_size=logical_shape[1],
            n_candidates=int(params.get("n_candidates", 200)),
            chunk_size=int(params.get("chunk_size", 1024)),
            scale_dtype=parse_w8a8_scale_dtype(
                params.get("scale_dtype", "float32")
            ),
            context=context,
        )
        names = int_quantized_tensor_names(tensor_name)
        return ArtifactResult(
            tensors={
                names.weight: int8_values.cpu(),
                names.scale: scales.cpu(),
            },
            # The raw INT8 weight replaces the source tensor under the same
            # name; only the scale is newly generated.
            generated_tensor_names=(names.scale,),
        )


register_weight_format(CompressedTensorsW8A8ChannelwiseFormat())
