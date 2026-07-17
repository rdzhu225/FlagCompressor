from __future__ import annotations

import torch

from flag_compressor.backends.base import BackendRunContext, QuantBackend
from flag_compressor.core.plan import TensorAction
from flag_compressor.transforms._int4_common import quantize_to_int4
from flag_compressor.transforms.base import Transform, TransformResult, register_transform


class Bf16ToInt4Transform(Transform):
    name = "bf16_to_int4"

    def apply(
        self,
        action: TensorAction,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
        backend: QuantBackend,
        context: BackendRunContext,
    ) -> TransformResult:
        dequant = weight if weight.dtype == torch.bfloat16 else weight.to(torch.bfloat16)
        return quantize_to_int4(action, dequant, backend, context)


register_transform(Bf16ToInt4Transform())
