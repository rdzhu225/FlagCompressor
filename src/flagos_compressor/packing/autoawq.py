"""Native AutoAWQ GEMM qweight/qzeros/scales serialization."""

from __future__ import annotations

from dataclasses import dataclass

import torch


_AWQ_GEMM_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


@dataclass(frozen=True)
class AutoAWQPacked:
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor


def _pack_awq_columns(codes: torch.Tensor) -> torch.Tensor:
    if codes.dim() != 2 or codes.shape[1] % 8:
        raise ValueError("AutoAWQ GEMM packing requires columns divisible by 8")
    groups = codes.to(torch.int64).reshape(codes.shape[0], -1, 8)
    ordered = groups[..., list(_AWQ_GEMM_ORDER)]
    shifts = (torch.arange(8, device=codes.device, dtype=torch.int64) * 4).reshape(
        1, 1, 8
    )
    return torch.sum((ordered & 0xF) << shifts, dim=2).to(torch.int32)


@torch.no_grad()
def pack_autoawq_gemm(
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    *,
    group_size: int,
    scale_dtype: torch.dtype = torch.float16,
) -> AutoAWQPacked:
    """Pack AutoAWQ fake-quantized ``[out, in]`` weights for GEMM kernels.

    ``scales`` and ``zeros`` use the natural quantizer layout
    ``[out_features, num_groups]``. The returned tensors use AutoAWQ GEMM
    layouts: qweight ``[in, out/8]``, qzeros ``[groups, out/8]`` and scales
    ``[groups, out]``.
    """
    if weight.dim() != 2 or scales.dim() != 2 or zeros.shape != scales.shape:
        raise ValueError("Invalid AutoAWQ weight/scale/zero shapes")
    out_features, in_features = weight.shape
    if group_size <= 0 or in_features % group_size:
        raise ValueError("AutoAWQ group_size must divide in_features")
    if out_features % 8:
        raise ValueError("AutoAWQ GEMM requires out_features divisible by 8")
    num_groups = in_features // group_size
    if scales.shape != (out_features, num_groups):
        raise ValueError("AutoAWQ qparams do not match weight shape")

    scales_t = scales.t().contiguous().float()
    zeros_t = zeros.t().contiguous().float()
    group_index = torch.arange(in_features, device=weight.device) // group_size
    codes = torch.round(
        weight.t().float() / scales_t[group_index] + zeros_t[group_index]
    ).clamp(0, 15)
    return AutoAWQPacked(
        qweight=_pack_awq_columns(codes).cpu(),
        qzeros=_pack_awq_columns(torch.round(zeros_t)).cpu(),
        scales=scales_t.to(scale_dtype).cpu(),
    )


__all__ = ["AutoAWQPacked", "pack_autoawq_gemm"]
