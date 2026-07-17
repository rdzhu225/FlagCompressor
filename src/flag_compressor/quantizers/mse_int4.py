from __future__ import annotations

import torch

from flag_compressor.backends.op_registry import register_op
def mse_int4_quantize(
    weight: torch.Tensor,
    group_size: int = 32,
    n_candidates: int = 200,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D float weight to signed INT4 values + BF16 per-group scale."""
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D weight tensor, got {weight.dim()}D")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )

    num_groups = in_features // group_size
    groups = weight.float().reshape(-1, group_size).to(torch.bfloat16)
    total_groups = groups.shape[0]

    multipliers = torch.linspace(
        0.5,
        1.5,
        n_candidates,
        device=groups.device,
        dtype=torch.bfloat16,
    )
    best_scales = torch.empty(total_groups, device=groups.device, dtype=torch.bfloat16)
    int4_values = torch.empty_like(groups, dtype=torch.int8)

    for start in range(0, total_groups, chunk_size):
        end = min(start + chunk_size, total_groups)
        chunk = groups[start:end]
        base_scale = (chunk.abs().amax(dim=1) / 8.0).clamp(min=1e-10).to(torch.bfloat16)
        candidates = base_scale.unsqueeze(1) * multipliers.unsqueeze(0)

        expanded_chunk = chunk.unsqueeze(1)
        expanded_candidates = candidates.unsqueeze(2)
        raw = torch.round(expanded_chunk / expanded_candidates).clamp(-8, 7)
        recon = raw * expanded_candidates
        mse = ((expanded_chunk - recon) ** 2).mean(dim=2)

        best_idx = mse.argmin(dim=1)
        local_rows = end - start
        best_scales[start:end] = candidates[torch.arange(local_rows, device=groups.device), best_idx]
        int4_values[start:end] = torch.round(chunk / best_scales[start:end].unsqueeze(1)).clamp(-8, 7).to(torch.int8)

    int4_values = int4_values.reshape(out_features, in_features)
    scales = best_scales.reshape(out_features, num_groups)
    return int4_values, scales.to(torch.bfloat16)


register_op("mse_int4_quant", "cpu")(mse_int4_quantize)
register_op("mse_int4_quant", "torch")(mse_int4_quantize)
