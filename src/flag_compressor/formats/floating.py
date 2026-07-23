from __future__ import annotations

from typing import Any

import torch

from flag_compressor.backends.base import BackendRunContext, QuantBackend
from flag_compressor.formats.base import WeightFormat, register_weight_format


class FloatingInputFormat(WeightFormat):
    def __init__(self, name: str) -> None:
        self.name = name

    def to_canonical(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
        backend: QuantBackend,
        context: BackendRunContext,
        params: dict[str, Any],
    ) -> torch.Tensor:
        del scale, backend, context, params
        return weight


register_weight_format(FloatingInputFormat("fp16"))
register_weight_format(FloatingInputFormat("fp32"))
