import pytest
import torch

from flag_compressor.formats.int4_pack import pack_signed_int4, unpack_signed_int4
from flag_compressor.quantizers.mse_int4 import mse_int4_quantize


def test_pack_unpack_signed_int4_roundtrip():
    values = torch.tensor([[-8, -7, -1, 0, 1, 6, 7, 3]], dtype=torch.int8)
    packed = pack_signed_int4(values)
    assert packed.dtype == torch.uint8
    assert torch.equal(unpack_signed_int4(packed), values)


def test_pack_rejects_out_of_range_values():
    with pytest.raises(ValueError, match=r"\[-8, 7\]"):
        pack_signed_int4(torch.tensor([[8, 0]], dtype=torch.int8))


def test_mse_quantizer_shapes_and_zero_group():
    weight = torch.cat(
        [torch.zeros(1, 32), torch.linspace(-2, 2, 32).reshape(1, 32)], dim=0
    )
    values, scales = mse_int4_quantize(weight, group_size=32, n_candidates=20)
    assert values.shape == weight.shape
    assert scales.shape == (2, 1)
    assert scales.dtype == torch.bfloat16
    assert torch.count_nonzero(values[0]) == 0
    reconstructed = values.float() * scales.float()
    assert torch.mean((weight - reconstructed) ** 2).item() < 0.02


def test_mse_quantizer_rejects_non_finite_weights():
    weight = torch.zeros((1, 32))
    weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        mse_int4_quantize(weight)
