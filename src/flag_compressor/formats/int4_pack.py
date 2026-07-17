from __future__ import annotations

import torch

from flag_compressor.backends.op_registry import register_op


def pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 values in [-8, 7] into uint8 low/high nibbles."""
    if values.dim() != 2:
        raise ValueError(f"Expected a 2D INT4 tensor, got {values.dim()}D")
    if values.shape[1] % 2 != 0:
        raise ValueError(f"INT4 packing requires even in_features, got shape {tuple(values.shape)}")
    values = values.to(torch.int8).clamp(-8, 7)
    even = values[:, 0::2]
    odd = values[:, 1::2]
    low = even.to(torch.uint8) & 0x0F
    high = (odd.to(torch.uint8) & 0x0F) << 4
    return (low | high).to(torch.uint8)


def unpack_signed_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 low/high nibbles into signed INT4 values in int8."""
    raw = packed.to(torch.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    values = torch.stack([low, high], dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)
    return torch.where(values >= 8, values.to(torch.int16) - 16, values.to(torch.int16)).to(torch.int8)


register_op("int4_pack", "cpu")(pack_signed_int4)
register_op("int4_pack", "torch")(pack_signed_int4)
register_op("int4_unpack", "cpu")(unpack_signed_int4)
register_op("int4_unpack", "torch")(unpack_signed_int4)

