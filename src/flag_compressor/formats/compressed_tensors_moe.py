from __future__ import annotations

from typing import Any

import torch

from flag_compressor.backends.base import BackendRunContext, QuantBackend
from flag_compressor.formats.base import (
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
    """Split one expert's fused projection into standard 2D linear weights.

    ``expert`` is 2D ``[out, in]``. ``gate_up_proj`` fuses gate and up along the
    output axis and is split into ``gate_proj`` and ``up_proj``; every other
    projection maps to a single standard linear weight of the same leaf name.
    """
    if proj_kind == "gate_up_proj":
        gate, up = expert.chunk(2, dim=0)
        return [("gate_proj", gate), ("up_proj", up)]
    return [(proj_kind, expert)]


class CompressedTensorsInt4MoEFusedFormat(WeightFormat):
    """W4A16 ``pack-quantized`` serialization of a fused 3D routed-expert bank.

    Each expert of the ``[num_experts, out, in]`` bank is quantized as an
    independent 2D weight and written under the standard per-expert
    compressed-tensors names (``experts.<e>.<proj>.weight_packed`` /
    ``weight_scale`` / ``weight_shape``), matching the layout emitted by
    llm-compressor and consumed by vLLM's non-fused expert loader.
    """

    name = "compressed_tensors_int4_moe_fused"

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
        if weight.dim() != 3:
            raise ValueError(
                f"fused MoE W4A16 requires a 3D bank, got shape {tuple(weight.shape)}"
            )
        proj_kind = params.get("proj_kind") or tensor_name.split(".")[-1]
        group_size = int(params.get("group_size", 32))
        n_candidates = int(params.get("n_candidates", 200))
        chunk_size = int(params.get("chunk_size", 4096))

        prefix = fused_expert_bank_prefix(tensor_name)
        num_experts = int(weight.shape[0])
        tensors: dict[str, torch.Tensor] = {}
        generated: list[str] = []

        for expert_id in range(num_experts):
            expert = weight[expert_id]
            for proj_name, proj_weight in _per_expert_projections(proj_kind, expert):
                proj_weight = proj_weight.contiguous()
                out_features, in_features = (int(d) for d in proj_weight.shape)
                int4_values, scales = backend.run(
                    "mse_int4_quant",
                    proj_weight,
                    group_size=group_size,
                    n_candidates=n_candidates,
                    chunk_size=chunk_size,
                    context=context,
                )
                packed = backend.run(
                    "int4_pack_uint4b8_int32",
                    int4_values,
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


register_weight_format(CompressedTensorsInt4MoEFusedFormat())
