from __future__ import annotations

from typing import Any

import torch

from flagos_compressor.backends.base import BackendRunContext, QuantBackend
from flagos_compressor.formats.base import ArtifactResult, WeightFormat, register_weight_format


class Bf16Format(WeightFormat):
    name = "bf16"

    def to_canonical(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
        backend: QuantBackend,
        context: BackendRunContext,
        params: dict[str, Any],
    ) -> torch.Tensor:
        del scale, backend, context, params
        return weight.to(torch.bfloat16)

    def from_canonical(
        self,
        tensor_name: str,
        weight: torch.Tensor,
        backend: QuantBackend,
        context: BackendRunContext,
        params: dict[str, Any],
    ) -> ArtifactResult:
        del backend, context, params
        return ArtifactResult(tensors={tensor_name: weight.to(torch.bfloat16).cpu()})


register_weight_format(Bf16Format())
