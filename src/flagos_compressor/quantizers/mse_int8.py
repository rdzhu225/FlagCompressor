from __future__ import annotations

import torch

from flagos_compressor.backends.op_registry import register_op
from flagos_compressor.quantizers.mse_int import mse_int_quantize


def mse_int8_quantize(
    weight: torch.Tensor,
    group_size: int = 128,
    n_candidates: int = 200,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D float weight to signed INT8 values and BF16 group scales."""
    return mse_int_quantize(
        weight,
        num_bits=8,
        group_size=group_size,
        n_candidates=n_candidates,
        chunk_size=chunk_size,
    )


register_op("mse_int8_quant", "cpu")(mse_int8_quantize)
register_op("mse_int8_quant", "torch")(mse_int8_quantize)
