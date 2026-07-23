import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

import flag_compressor.formats.register  # noqa: F401
import flag_compressor.quantizers.register  # noqa: F401
from flag_compressor.backends.registry import build_backend
from flag_compressor.cli.main import main
from flag_compressor.core.executor import execute_plan
from flag_compressor.core.planner import build_convert_plan, build_quantize_plan
from flag_compressor.core.policy import QuantizationPolicy
from flag_compressor.core.validation import validate_artifact
from flag_compressor.inspect.checkpoint_scanner import scan_hf_safetensors


def _make_checkpoint(path: Path) -> dict[str, str]:
    path.mkdir()
    routed = "model.layers.0.mlp.experts.0.w1.weight"
    shared = "model.layers.0.mlp.shared_experts.w1.weight"
    attention = "model.layers.0.self_attn.o_proj.weight"
    state = {
        routed: torch.full((2, 16), 0x21, dtype=torch.uint8),
        routed[: -len(".weight")] + ".scale": torch.ones((2, 1), dtype=torch.float32),
        shared: torch.full((2, 16), 0x43, dtype=torch.uint8),
        shared[: -len(".weight")] + ".scale": torch.ones((2, 1), dtype=torch.float32),
        attention: torch.ones((128, 128), dtype=torch.float8_e4m3fn),
        attention[: -len(".weight")] + ".scale": torch.ones((1, 1), dtype=torch.float32),
        "model.embed_tokens.weight": torch.ones((4, 32), dtype=torch.bfloat16),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(state, str(path / shard))
    weight_map = {name: shard for name in state}
    total_size = sum(t.numel() * t.element_size() for t in state.values())
    with (path / "model.safetensors.index.json").open("w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f)
    with (path / "config.json").open("w") as f:
        json.dump(
            {
                "torch_dtype": "float8_e4m3fn",
                "expert_dtype": "fp4",
                "quantization_config": {"quant_method": "fp8"},
            },
            f,
        )
    return weight_map


def test_quantize_moe_end_to_end(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "quantized"
    _make_checkpoint(source)
    with (source / "quant_manifest.json").open("w") as f:
        json.dump({"schema": "legacy"}, f)
    profile = scan_hf_safetensors(source)
    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(selections=("moe",), n_candidates=8, chunk_size=16),
    )
    execute_plan(source, output, plan, build_backend("cpu"))

    with (output / "quantization_manifest.json").open() as f:
        manifest = json.load(f)
    routed = "model.layers.0.mlp.experts.0.w1.weight"
    routed_packed = routed.replace(".weight", ".weight_packed")
    routed_scale = routed.replace(".weight", ".weight_scale")
    routed_shape = routed.replace(".weight", ".weight_shape")
    spec = manifest["tensors"][routed_packed]
    assert spec["logical_shape"] == [2, 32]
    assert spec["storage_shape"] == [2, 4]
    assert spec["scale_shape"] == [2, 1]

    shard = load_file(output / "model-00001-of-00001.safetensors")
    assert shard[routed_packed].dtype == torch.int32
    assert shard[routed_scale].dtype == torch.bfloat16
    assert torch.equal(shard[routed_shape], torch.tensor([2, 32]))
    assert shard["model.layers.0.self_attn.o_proj.weight"].dtype == torch.bfloat16
    with (output / "config.json").open() as f:
        config = json.load(f)
    assert config["torch_dtype"] == "bfloat16"
    assert "expert_dtype" not in config
    assert config["quantization_config"]["quant_method"] == "compressed-tensors"
    assert config["quantization_config"]["format"] == "pack-quantized"
    assert not (output / "quant_manifest.json").exists()
    assert manifest["unselected_weights"] == {
        "strategy": "convert",
        "format": "bf16",
    }
    assert validate_artifact(output)["valid"]
    rescanned = scan_hf_safetensors(output)
    assert (
        rescanned.tensors[routed_packed].storage_format
        == "compressed-tensors-pack-quantized-int4"
    )


def test_convert_to_bf16_end_to_end(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "converted"
    _make_checkpoint(source)
    plan = build_convert_plan(scan_hf_safetensors(source))
    execute_plan(source, output, plan, build_backend("cpu"))
    shard = load_file(output / "model-00001-of-00001.safetensors")
    assert all(t.dtype != torch.uint8 for name, t in shard.items() if name.endswith(".weight"))
    with (output / "config.json").open() as f:
        config = json.load(f)
    assert config["expert_dtype"] == "bfloat16"
    assert validate_artifact(output)["valid"]


def test_cli_commands_and_implicit_convert_dry_run(tmp_path, capsys):
    source = tmp_path / "source"
    _make_checkpoint(source)
    main(["inspect", "--input", str(source)])
    main(
        [
            "quantize",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "quantized"),
            "--select",
            "moe",
            "--dry-run",
        ]
    )
    main(
        [
            "--input",
            str(source),
            "--output",
            str(tmp_path / "converted"),
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out
    assert "moe.routed" in output
    assert "compressed_tensors_int4_groupwise" in output
    assert "fp4_e2m1_e8m0" in output
    assert "bf16" in output
