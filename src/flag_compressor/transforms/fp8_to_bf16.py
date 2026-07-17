from __future__ import annotations

import torch

from flag_compressor.backends.base import BackendRunContext, QuantBackend
from flag_compressor.core.plan import TensorAction
from flag_compressor.transforms.base import Transform, TransformResult, register_transform


class Fp8ToBf16Transform(Transform):
    name = "fp8_to_bf16"

    def apply(
        self,
        action: TensorAction,
        weight: torch.Tensor,
        scale: torch.Tensor | None,
        backend: QuantBackend,
        context: BackendRunContext,
    ) -> TransformResult:
        if scale is None:
            raise ValueError(f"{action.tensor.name} requires a scale tensor for fp8_to_bf16")
        block_size = int(action.params.get("block_size", 128))
        result = backend.run("fp8_dequant", weight, scale, block_size=block_size, context=context)
        return TransformResult(tensors={action.tensor.name: result.cpu()}, generated_scale_names=[])


register_transform(Fp8ToBf16Transform())

