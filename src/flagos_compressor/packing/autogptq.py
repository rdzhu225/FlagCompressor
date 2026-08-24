"""Native AutoGPTQ qweight/qzeros/scales/g_idx serialization."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AutoGPTQPacked:
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor
    g_idx: torch.Tensor


def _pack_rows(codes: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    if codes.dim() != 2 or codes.shape[0] % pack_factor:
        raise ValueError("AutoGPTQ qweight rows must align to the pack factor")
    groups = codes.to(torch.int64).reshape(
        codes.shape[0] // pack_factor, pack_factor, codes.shape[1]
    )
    shifts = torch.arange(pack_factor, device=codes.device, dtype=torch.int64)
    shifts = (shifts * bits).reshape(1, pack_factor, 1)
    mask = (1 << bits) - 1
    return torch.sum((groups & mask) << shifts, dim=1).to(torch.int32)


def _pack_columns(codes: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    if codes.dim() != 2 or codes.shape[1] % pack_factor:
        raise ValueError("AutoGPTQ qzeros columns must align to the pack factor")
    groups = codes.to(torch.int64).reshape(
        codes.shape[0], codes.shape[1] // pack_factor, pack_factor
    )
    shifts = torch.arange(pack_factor, device=codes.device, dtype=torch.int64)
    shifts = (shifts * bits).reshape(1, 1, pack_factor)
    mask = (1 << bits) - 1
    return torch.sum((groups & mask) << shifts, dim=2).to(torch.int32)


@torch.no_grad()
def pack_autogptq(
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    g_idx: torch.Tensor,
    *,
    bits: int,
    scale_dtype: torch.dtype = torch.float16,
) -> AutoGPTQPacked:
    """Pack fake-quantized ``[out, in]`` weights using AutoGPTQ's ABI."""
    if bits not in {4, 8}:
        raise ValueError("Native AutoGPTQ packing currently supports 4 or 8 bits")
    if weight.dim() != 2 or scales.dim() != 2 or zeros.shape != scales.shape:
        raise ValueError("Invalid AutoGPTQ weight/scale/zero shapes")
    out_features, in_features = weight.shape
    if scales.shape[0] != out_features or g_idx.shape != (in_features,):
        raise ValueError("AutoGPTQ qparams do not match weight shape")
    pack_factor = 32 // bits
    if in_features % pack_factor or out_features % pack_factor:
        raise ValueError(
            f"AutoGPTQ {bits}-bit packing requires in/out features divisible by {pack_factor}"
        )

    scales_t = scales.t().contiguous().float()
    zeros_t = zeros.t().contiguous().float()
    group_scales = scales_t[g_idx.long()]
    group_zeros = zeros_t[g_idx.long()]
    codes = torch.round(weight.t().float() / group_scales + group_zeros)
    codes = codes.clamp(0, (1 << bits) - 1).to(torch.int32)
    qweight = _pack_rows(codes, bits)

    # AutoGPTQ stores (zero_point - 1) modulo 2**bits. This detail is part of
    # the native loader ABI and intentionally differs from AWQ.
    zero_codes = torch.round(zeros_t).to(torch.int64) - 1
    qzeros = _pack_columns(zero_codes, bits)
    return AutoGPTQPacked(
        qweight=qweight.cpu(),
        qzeros=qzeros.cpu(),
        scales=scales_t.to(scale_dtype).cpu(),
        g_idx=g_idx.to(torch.int32).cpu(),
    )


__all__ = ["AutoGPTQPacked", "pack_autogptq"]
