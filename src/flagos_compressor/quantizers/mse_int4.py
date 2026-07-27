from __future__ import annotations

import torch

from flagos_compressor.backends.op_registry import register_op


def mse_int4_quantize(
    weight: torch.Tensor,
    group_size: int = 32,
    n_candidates: int = 200,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D float weight to signed INT4 values and BF16 group scales."""
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D weight tensor, got {weight.dim()}D")
    out_features, in_features = weight.shape
    if group_size <= 0 or in_features % group_size:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )
    if n_candidates <= 0 or chunk_size <= 0:
        raise ValueError("n_candidates and chunk_size must be positive")

    num_groups = in_features // group_size
    groups = weight.to(torch.float32).reshape(-1, group_size)
    if not torch.isfinite(groups).all().item():
        raise ValueError("Cannot quantize weights containing NaN or infinity")
    total_groups = groups.shape[0]
    multipliers = torch.linspace(
        0.5, 1.5, n_candidates, device=groups.device, dtype=torch.float32
    )
    best_scales = torch.empty(total_groups, device=groups.device, dtype=torch.float32)
    int4_values = torch.empty_like(groups, dtype=torch.int8)

    for start in range(0, total_groups, chunk_size):
        end = min(start + chunk_size, total_groups)
        chunk = groups[start:end]
        base_scale = (chunk.abs().amax(dim=1) / 8.0).clamp(min=1e-10)
        candidates = base_scale.unsqueeze(1) * multipliers.unsqueeze(0)
        raw = torch.round(chunk.unsqueeze(1) / candidates.unsqueeze(2)).clamp(-8, 7)
        mse = ((chunk.unsqueeze(1) - raw * candidates.unsqueeze(2)) ** 2).mean(dim=2)
        best_idx = mse.argmin(dim=1)
        rows = end - start
        selected = candidates[torch.arange(rows, device=groups.device), best_idx]
        best_scales[start:end] = selected
        int4_values[start:end] = torch.round(chunk / selected.unsqueeze(1)).clamp(-8, 7).to(torch.int8)

    return (
        int4_values.reshape(out_features, in_features),
        best_scales.reshape(out_features, num_groups).to(torch.bfloat16),
    )


register_op("mse_int4_quant", "cpu")(mse_int4_quantize)
register_op("mse_int4_quant", "torch")(mse_int4_quantize)
