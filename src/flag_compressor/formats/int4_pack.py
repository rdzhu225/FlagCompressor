from __future__ import annotations

import torch

from flag_compressor.backends.op_registry import register_op


def pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 values as low-even/high-odd two's-complement nibbles."""
    if values.dim() != 2:
        raise ValueError(f"Expected a 2D INT4 tensor, got {values.dim()}D")
    if values.shape[1] % 2:
        raise ValueError(f"INT4 packing requires even in_features, got {tuple(values.shape)}")
    values = values.to(torch.int8)
    if values.numel() and (values.min().item() < -8 or values.max().item() > 7):
        raise ValueError("Signed INT4 values must be in [-8, 7]")
    even = values[:, 0::2]
    odd = values[:, 1::2]
    low = even.to(torch.uint8) & 0x0F
    high = (odd.to(torch.uint8) & 0x0F) << 4
    return (low | high).to(torch.uint8)


def unpack_signed_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack low-even/high-odd nibbles into signed INT4 values in int8."""
    if packed.dim() != 2:
        raise ValueError(f"Expected a 2D packed INT4 tensor, got {packed.dim()}D")
    raw = packed.to(torch.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    values = torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)
    signed = torch.where(values >= 8, values.to(torch.int16) - 16, values.to(torch.int16))
    return signed.to(torch.int8)


register_op("int4_pack", "cpu")(pack_signed_int4)
register_op("int4_pack", "torch")(pack_signed_int4)
register_op("int4_unpack", "cpu")(unpack_signed_int4)
register_op("int4_unpack", "torch")(unpack_signed_int4)
