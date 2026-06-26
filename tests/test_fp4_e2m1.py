import torch

from quant_engine.formats.fp4_e2m1 import unpack_fp4_e2m1


def test_unpack_fp4_e2m1_low_high_order():
    packed = torch.tensor([[0x10, 0x32]], dtype=torch.uint8)
    unpacked = unpack_fp4_e2m1(packed)
    expected = torch.tensor([[0.0, 0.5, 1.0, 1.5]], dtype=torch.bfloat16)
    assert torch.equal(unpacked, expected)

