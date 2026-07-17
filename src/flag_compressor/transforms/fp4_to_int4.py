from __future__ import annotations

import torch

from flag_compressor.backends.base import BackendRunContext, QuantBackend
from flag_compressor.core.plan import TensorAction
from flag_compressor.transforms._int4_common import quantize_to_int4
from flag_compressor.transforms.base import Transform, TransformResult, register_transform


class Fp4ToInt4Transform(Transform):
    name = "fp4_to_int4"

    def apply(
        self,
        action: TensorAction,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
        backend: QuantBackend,
        context: BackendRunContext,
    ) -> TransformResult:
        if scale is None:
            raise ValueError(f"{action.tensor.name} requires a scale tensor for fp4_to_int4")
        dequant = backend.run("fp4_dequant", weight, scale, context=context)
        return quantize_to_int4(action, dequant, backend, context)


register_transform(Fp4ToInt4Transform())
