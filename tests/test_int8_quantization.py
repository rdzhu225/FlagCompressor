import pytest
import torch

import flagos_compressor.formats.register  # noqa: F401
import flagos_compressor.quantizers.register  # noqa: F401
from flagos_compressor.backends.base import BackendRunContext
from flagos_compressor.backends.registry import build_backend
from flagos_compressor.core.report import ConversionReport
from flagos_compressor.formats.base import get_weight_format
from flagos_compressor.formats.int8_pack import (
    pack_uint8b128_int32,
    unpack_uint8b128_int32,
)
from flagos_compressor.quantizers.mse_int8 import mse_int8_quantize


def test_compressed_tensors_int8_pack_roundtrip_with_padding():
    values = torch.tensor(
        [[-128, -127, -1, 0, 1, 126, 127]],
        dtype=torch.int8,
    )
    packed = pack_uint8b128_int32(values)
    assert packed.dtype == torch.int32
    assert packed.shape == (1, 2)
    assert torch.equal(
        unpack_uint8b128_int32(packed, in_features=values.shape[1]),
        values,
    )


def test_int8_pack_rejects_out_of_range_values():
    with pytest.raises(ValueError, match=r"\[-128, 127\]"):
        pack_uint8b128_int32(torch.tensor([[128]], dtype=torch.int16))


def test_mse_int8_quantizer_shapes_and_accuracy():
    weight = torch.cat(
        [torch.zeros(1, 128), torch.linspace(-2, 2, 128).reshape(1, 128)],
        dim=0,
    )
    values, scales = mse_int8_quantize(
        weight,
        group_size=128,
        n_candidates=20,
    )
    assert values.shape == weight.shape
    assert values.dtype == torch.int8
    assert scales.shape == (2, 1)
    assert scales.dtype == torch.bfloat16
    assert torch.count_nonzero(values[0]) == 0
    reconstructed = values.float() * scales.float()
    assert torch.mean((weight - reconstructed) ** 2).item() < 1e-4


def test_mse_int8_quantizer_rejects_non_finite_weights():
    weight = torch.zeros((1, 128))
    weight[0, 0] = float("inf")
    with pytest.raises(ValueError, match="infinity"):
        mse_int8_quantize(weight)


def test_channelwise_format_uses_one_scale_per_output_row():
    weight = torch.randn(3, 13, dtype=torch.bfloat16)
    backend = build_backend("cpu")
    context = BackendRunContext(
        report=ConversionReport(backend="cpu"),
    )
    result = get_weight_format(
        "compressed_tensors_int8_channelwise"
    ).from_canonical(
        "model.proj.weight",
        weight,
        backend,
        context,
        {
            "n_candidates": 8,
            "chunk_size": 4,
        },
    )
    assert result.tensors["model.proj.weight_packed"].shape == (3, 4)
    assert result.tensors["model.proj.weight_scale"].shape == (3, 1)
