import math

import torch

from flagos_compressor.packing.autogptq import pack_autogptq
from flagos_compressor.quantizers.gptq import GPTQQuantizer


def _unpack_rows(packed: torch.Tensor, bits: int) -> torch.Tensor:
    factor = 32 // bits
    words = packed.to(torch.int64) & 0xFFFFFFFF
    return torch.stack(
        [(words >> (index * bits)) & ((1 << bits) - 1) for index in range(factor)],
        dim=1,
    ).reshape(packed.shape[0] * factor, packed.shape[1])


def _unpack_columns(packed: torch.Tensor, bits: int) -> torch.Tensor:
    factor = 32 // bits
    words = packed.to(torch.int64) & 0xFFFFFFFF
    return torch.stack(
        [(words >> (index * bits)) & ((1 << bits) - 1) for index in range(factor)],
        dim=2,
    ).reshape(packed.shape[0], packed.shape[1] * factor)


def test_gptq_hessian_matches_autogptq_running_normalization():
    weight = torch.zeros(8, 8)
    inputs = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    quantizer = GPTQQuantizer(weight)
    quantizer.add_batch(inputs)

    flattened = inputs.reshape(-1, 8).t()
    expected = (math.sqrt(2) * flattened).matmul(
        (math.sqrt(2) * flattened).t()
    )
    torch.testing.assert_close(quantizer.hessian, expected)


def test_gptq_quantize_and_native_pack_round_trip():
    torch.manual_seed(7)
    weight = torch.randn(8, 16, dtype=torch.float32)
    quantizer = GPTQQuantizer(weight, bits=4, symmetric=True)
    quantizer.add_batch(torch.randn(2, 6, 16))
    result = quantizer.quantize(
        group_size=8,
        block_size=8,
        damp_percent=0.01,
        desc_act=True,
        static_groups=True,
    )

    assert result.scales.shape == (8, 2)
    assert result.zeros.shape == (8, 2)
    assert result.g_idx.shape == (16,)
    packed = pack_autogptq(
        result.weight,
        result.scales,
        result.zeros,
        result.g_idx,
        bits=4,
    )
    assert packed.qweight.shape == (2, 8)
    assert packed.qzeros.shape == (2, 1)
    assert packed.scales.shape == (2, 8)

    codes = _unpack_rows(packed.qweight, 4)
    zero_codes = (_unpack_columns(packed.qzeros, 4) + 1) & 0xF
    scales = packed.scales.float()
    reconstructed = (
        scales[packed.g_idx.long()] * (codes - zero_codes[packed.g_idx.long()])
    ).t()
    torch.testing.assert_close(reconstructed, result.weight, atol=2e-3, rtol=2e-3)
