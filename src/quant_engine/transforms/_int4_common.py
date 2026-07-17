"""Shared helpers for transforms that end in symmetric groupwise INT4."""

from __future__ import annotations

import torch

from quant_engine.backends.base import BackendRunContext, QuantBackend
from quant_engine.core.plan import TensorAction
from quant_engine.transforms.base import TransformResult


def quantize_to_int4(
    action: TensorAction,
    dequant: torch.Tensor,
    backend: QuantBackend,
    context: BackendRunContext,
) -> TransformResult:
    quantizer = action.quantizer or {"name": "mse"}
    quantizer_name = quantizer.get("name", "mse")
    if quantizer_name != "mse":
        raise NotImplementedError(
            f"Quantizer '{quantizer_name}' is declared but not implemented in this MVP"
        )

    group_size = int(quantizer.get("group_size", action.params.get("group_size", 32)))
    n_candidates = int(quantizer.get("n_candidates", action.params.get("n_candidates", 200)))
    chunk_size = int(quantizer.get("chunk_size", action.params.get("chunk_size", 4096)))

    int4_values, int4_scale = backend.run(
        "mse_int4_quant",
        dequant,
        group_size=group_size,
        n_candidates=n_candidates,
        chunk_size=chunk_size,
        context=context,
    )
    int4 = backend.run("int4_pack", int4_values, context=context)

    scale_suffix = action.output.get("scale_suffix", ".scale")
    scale_name = action.tensor.name + scale_suffix
    return TransformResult(
        tensors={
            action.tensor.name: int4.cpu(),
            scale_name: int4_scale.cpu(),
        },
        generated_scale_names=[scale_name],
    )
