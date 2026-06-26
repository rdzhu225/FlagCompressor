from __future__ import annotations

import torch

from quant_engine.backends.base import BackendRunContext, QuantBackend
from quant_engine.core.plan import TensorAction
from quant_engine.transforms.base import Transform, TransformResult, register_transform


class Fp4ToBf16Transform(Transform):
    name = "fp4_to_bf16"

    def apply(
        self,
        action: TensorAction,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
        backend: QuantBackend,
        context: BackendRunContext,
    ) -> TransformResult:
        if scale is None:
            raise ValueError(f"{action.tensor.name} requires a scale tensor for fp4_to_bf16")
        result = backend.run("fp4_dequant", weight, scale, context=context)
        return TransformResult(tensors={action.tensor.name: result.cpu()}, generated_scale_names=[])


register_transform(Fp4ToBf16Transform())

