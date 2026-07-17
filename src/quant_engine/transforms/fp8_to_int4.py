from __future__ import annotations

import torch

from quant_engine.backends.base import BackendRunContext, QuantBackend
from quant_engine.core.plan import TensorAction
from quant_engine.transforms._int4_common import quantize_to_int4
from quant_engine.transforms.base import Transform, TransformResult, register_transform


class Fp8ToInt4Transform(Transform):
    name = "fp8_to_int4"

    def apply(
        self,
        action: TensorAction,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
        backend: QuantBackend,
        context: BackendRunContext,
    ) -> TransformResult:
        if scale is None:
            raise ValueError(f"{action.tensor.name} requires a scale tensor for fp8_to_int4")
        block_size = int(action.params.get("block_size", 128))
        dequant = backend.run("fp8_dequant", weight, scale, block_size=block_size, context=context)
        return quantize_to_int4(action, dequant, backend, context)


register_transform(Fp8ToInt4Transform())
