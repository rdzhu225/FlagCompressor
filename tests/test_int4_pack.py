import torch

from quant_engine.formats.int4_pack import pack_signed_int4, unpack_signed_int4


def test_pack_unpack_signed_int4_roundtrip():
    values = torch.tensor([[-8, -1, 0, 7], [3, -4, 6, -7]], dtype=torch.int8)
    packed = pack_signed_int4(values)
    unpacked = unpack_signed_int4(packed)
    assert torch.equal(unpacked, values)

