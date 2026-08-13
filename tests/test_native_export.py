import json

import pytest
import torch
from safetensors.torch import save_file

from flagos_compressor.calibration.runner import NativeQuantizedLayer
from flagos_compressor.calibration.moe import LinearExperts2D
from flagos_compressor.core.validation import validate_artifact
from flagos_compressor.formats.native_quantized import save_native_quantized_model
from flagos_compressor.packing.autoawq import pack_autoawq_gemm
from flagos_compressor.packing.autogptq import pack_autogptq
from flagos_compressor.quantizers.awq import pseudo_quantize_awq


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False)
        self.untouched = torch.nn.Linear(8, 8, bias=False)


def _source_checkpoint(tmp_path, model):
    source = tmp_path / "source"
    source.mkdir()
    save_file(
        {name: tensor.detach() for name, tensor in model.state_dict().items()},
        source / "model.safetensors",
    )
    (source / "config.json").write_text(
        json.dumps({"model_type": "toy", "torch_dtype": "float32"}),
        encoding="utf-8",
    )
    return source


@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_native_export_is_sharded_configured_and_validated(tmp_path, method):
    torch.manual_seed(3)
    model = _ToyModel().eval()
    source = _source_checkpoint(tmp_path, model)
    group_size = 8
    if method == "awq":
        params = pseudo_quantize_awq(
            model.proj.weight,
            bits=4,
            group_size=group_size,
            zero_point=True,
        )
        model.proj.weight.data.copy_(params.weight)
        packed = pack_autoawq_gemm(
            params.weight,
            params.scales,
            params.zeros,
            group_size=group_size,
        )
    else:
        scales = torch.full((8, 1), 0.05)
        zeros = torch.full((8, 1), 8.0)
        codes = torch.clamp(
            torch.round(model.proj.weight / scales) + zeros,
            0,
            15,
        )
        fake = scales * (codes - zeros)
        model.proj.weight.data.copy_(fake)
        packed = pack_autogptq(
            fake,
            scales,
            zeros,
            torch.zeros(8, dtype=torch.int32),
            bits=4,
        )
    quantized = {"proj": NativeQuantizedLayer(method, packed)}
    output = tmp_path / method

    save_native_quantized_model(
        source,
        output,
        model,
        quantized,
        method=method,
        bits=4,
        group_size=group_size,
        max_shard_size=64,
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    quant_config = config["quantization_config"]
    assert quant_config["quant_method"] == method
    if method == "awq":
        assert "untouched" in quant_config["modules_to_not_convert"]
    else:
        assert quant_config["modules_in_block_to_quantize"] == [["proj"]]
        assert (output / "gptq_model-4bit-8g.safetensors.index.json").exists()

    result = validate_artifact(output)
    assert result["valid"], result["errors"]
    assert result["native_method"] == method
    assert result["native_quantized_tensors"] == 1


def test_native_awq_export_skips_unselected_fused_moe_unit(tmp_path):
    class ModelWithExperts(_ToyModel):
        def __init__(self):
            super().__init__()
            fused = torch.nn.Module()
            fused.hidden_dim = 8
            fused.gate_up_proj = torch.nn.Parameter(torch.randn(2, 8, 8))
            fused.down_proj = torch.nn.Parameter(torch.randn(2, 8, 4))
            self.experts = LinearExperts2D(fused)

    model = ModelWithExperts().eval()
    source = tmp_path / "source-moe"
    source.mkdir()
    save_file({"dummy": torch.ones(1)}, source / "model.safetensors")
    (source / "config.json").write_text(
        json.dumps({"model_type": "toy"}), encoding="utf-8"
    )
    params = pseudo_quantize_awq(
        model.proj.weight,
        bits=4,
        group_size=8,
        zero_point=True,
    )
    packed = pack_autoawq_gemm(
        params.weight,
        params.scales,
        params.zeros,
        group_size=8,
    )
    output = tmp_path / "awq-moe-skip"

    save_native_quantized_model(
        source,
        output,
        model,
        {"proj": NativeQuantizedLayer("awq", packed)},
        method="awq",
        bits=4,
        group_size=8,
    )

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert "experts" in config["quantization_config"]["modules_to_not_convert"]
    assert validate_artifact(output)["valid"]
