from __future__ import annotations

from typing import Any

import torch

from flagos_compressor.backends.base import BackendRunContext, QuantBackend
from flagos_compressor.core.moe_layout import layout_by_name
from flagos_compressor.formats.base import (
    ArtifactResult,
    WeightFormat,
    register_weight_format,
)


def fused_expert_bank_prefix(tensor_name: str) -> str:
    """Return the ``...experts`` module prefix of a fused expert bank name.

    ``model...mlp.experts.gate_up_proj`` -> ``model...mlp.experts``.
    """
    proj = tensor_name.split(".")[-1]
    suffix = "." + proj
    if not tensor_name.endswith(suffix):
        raise ValueError(
            f"fused expert export expects a projection leaf, got {tensor_name!r}"
        )
    return tensor_name[: -len(suffix)]


def _per_expert_projections(
    proj_kind: str, expert: torch.Tensor
) -> list[tuple[str, torch.Tensor]]:
    """Split one expert's projection into standard 2D linear weights.

    ``expert`` is already oriented as 2D ``[out, in]`` (the caller applies any
    layout transpose first). ``gate_up_proj`` fuses gate and up along the output
    axis and is split into ``gate_proj`` and ``up_proj``; every other projection
    maps to a single standard linear weight of the same leaf name.
    """
    if proj_kind == "gate_up_proj":
        gate, up = expert.chunk(2, dim=0)
        return [("gate_proj", gate), ("up_proj", up)]
    return [(proj_kind, expert)]


class _CompressedTensorsIntMoEFusedFormat(WeightFormat):
    """Pack-quantized serialization of a fused 3D routed-expert bank.

    Each expert of the ``[num_experts, out, in]`` bank is quantized as an
    independent 2D weight and written under the standard per-expert
    compressed-tensors names (``experts.<e>.<proj>.weight_packed`` /
    ``weight_scale`` / ``weight_shape``), matching the layout emitted by
    llm-compressor and consumed by vLLM's non-fused expert loader.
    """

    num_bits: int

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
            raise NotImplementedError(
                f"Unsupported INT{self.num_bits} quantizer: {quantizer}"
            )
        if weight.dim() != 3:
            raise ValueError(
                f"fused MoE W{self.num_bits}A16 requires a 3D bank, "
                f"got shape {tuple(weight.shape)}"
            )
        proj_kind = params.get("proj_kind") or tensor_name.split(".")[-1]
        group_size = int(params.get("group_size", 32 if self.num_bits == 4 else 128))
        n_candidates = int(params.get("n_candidates", 200))
        chunk_size = int(
            params.get("chunk_size", 4096 if self.num_bits == 4 else 1024)
        )
        layout = layout_by_name(params["layout"])

        prefix = fused_expert_bank_prefix(tensor_name)
        num_experts = int(weight.shape[0])
        tensors: dict[str, torch.Tensor] = {}
        generated: list[str] = []

        for expert_id in range(num_experts):
            expert = weight[expert_id]
            # Orient the storage slice to logical [out, in] before splitting.
            if not layout.out_in_order:
                expert = expert.transpose(0, 1)
            for proj_name, proj_weight in _per_expert_projections(proj_kind, expert):
                proj_weight = proj_weight.contiguous()
                out_features, in_features = (int(d) for d in proj_weight.shape)
                quantized, scales = backend.run(
                    f"mse_int{self.num_bits}_quant",
                    proj_weight,
                    group_size=group_size,
                    n_candidates=n_candidates,
                    chunk_size=chunk_size,
                    context=context,
                )
                packed = backend.run(
                    (
                        "int4_pack_uint4b8_int32"
                        if self.num_bits == 4
                        else "int8_pack_uint8b128_int32"
                    ),
                    quantized,
                    context=context,
                )
                base = f"{prefix}.{expert_id}.{proj_name}"
                weight_name = f"{base}.weight_packed"
                scale_name = f"{base}.weight_scale"
                shape_name = f"{base}.weight_shape"
                tensors[weight_name] = packed.cpu()
                tensors[scale_name] = scales.cpu()
                tensors[shape_name] = torch.tensor(
                    (out_features, in_features), dtype=torch.int64
                )
                generated.extend((weight_name, scale_name, shape_name))

        return ArtifactResult(
            tensors=tensors,
            generated_tensor_names=tuple(generated),
        )


class CompressedTensorsInt4MoEFusedFormat(_CompressedTensorsIntMoEFusedFormat):
    """W4A16 serialization of a fused routed-expert bank."""

    name = "compressed_tensors_int4_moe_fused"
    num_bits = 4


class CompressedTensorsInt8MoEFusedFormat(_CompressedTensorsIntMoEFusedFormat):
    """W8A16 serialization of a fused routed-expert bank."""

    name = "compressed_tensors_int8_moe_fused"
    num_bits = 8


class CompressedTensorsW8A8Int8MoEFusedFormat(WeightFormat):
    """Canonical W8A8 export of a fused 3D routed-expert bank.

    The source bank is expanded to the per-expert Linear names consumed by
    vLLM's compressed-tensors MoE loader. Each 2D projection is stored as raw
    INT8 with an FP32 ``[out_features, 1]`` channel scale.
    """

    name = "compressed_tensors_w8a8_int8_moe_fused"

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
        if params.get("strategy", "channel") != "channel":
            raise ValueError("compressed-tensors MoE W8A8 requires channel strategy")
        if weight.dim() != 3:
            raise ValueError(
                "fused MoE W8A8 requires a 3D bank, "
                f"got shape {tuple(weight.shape)}"
            )

        proj_kind = params.get("proj_kind") or tensor_name.split(".")[-1]
        n_candidates = int(params.get("n_candidates", 200))
        chunk_size = int(params.get("chunk_size", 1024))
        layout = layout_by_name(params["layout"])
        prefix = fused_expert_bank_prefix(tensor_name)
        tensors: dict[str, torch.Tensor] = {}
        generated: list[str] = []

        for expert_id in range(int(weight.shape[0])):
            expert = weight[expert_id]
            if not layout.out_in_order:
                expert = expert.transpose(0, 1)
            for proj_name, proj_weight in _per_expert_projections(proj_kind, expert):
                proj_weight = proj_weight.contiguous()
                in_features = int(proj_weight.shape[1])
                quantized, scales = backend.run(
                    "mse_int8_quant",
                    proj_weight,
                    group_size=in_features,
                    n_candidates=n_candidates,
                    chunk_size=chunk_size,
                    scale_dtype=torch.float32,
                    context=context,
                )
                base = f"{prefix}.{expert_id}.{proj_name}"
                weight_name = f"{base}.weight"
                scale_name = f"{base}.weight_scale"
                tensors[weight_name] = quantized.cpu()
                tensors[scale_name] = scales.cpu()
                generated.extend((weight_name, scale_name))

        return ArtifactResult(
            tensors=tensors,
            generated_tensor_names=tuple(generated),
        )


register_weight_format(CompressedTensorsInt4MoEFusedFormat())
register_weight_format(CompressedTensorsInt8MoEFusedFormat())
register_weight_format(CompressedTensorsW8A8Int8MoEFusedFormat())
