import pytest

from flag_compressor.core.compressed_tensors import (
    build_compressed_tensors_config,
    compile_compressed_tensors_targets,
    validate_fusion_closure,
)


def test_compiles_complete_moe_categories_to_compact_targets():
    weights = {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.shared_experts.gate_proj.weight",
        "model.layers.0.mlp.shared_experts.up_proj.weight",
        "model.layers.0.mlp.shared_experts.down_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    }
    selected = {name for name in weights if ".mlp." in name}
    targets = compile_compressed_tensors_targets(weights, selected)
    assert len(targets) == 2
    assert any(r"\.experts\." in target for target in targets)
    assert any("shared_expert" in target for target in targets)

    config = build_compressed_tensors_config(
        weights,
        selected,
        group_size=32,
    )
    assert config["quant_method"] == "compressed-tensors"
    assert config["format"] == "pack-quantized"
    assert (
        config["config_groups"]["w4a16_g32"]["weights"]["group_size"]
        == 32
    )


def test_irregular_selection_falls_back_to_exact_module_path():
    weights = {
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.1.self_attn.o_proj.weight",
    }
    selected = {"model.layers.1.self_attn.o_proj.weight"}
    assert compile_compressed_tensors_targets(weights, selected) == [
        "model.layers.1.self_attn.o_proj"
    ]


def test_rejects_mixed_gate_up_pair():
    weights = {
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    }
    with pytest.raises(ValueError, match="fused pair"):
        validate_fusion_closure(
            weights,
            {"model.layers.0.mlp.gate_proj.weight"},
        )


def test_rejects_partial_routed_moe_bank():
    weights = {
        f"model.layers.0.mlp.experts.{expert}.{proj}.weight"
        for expert in range(2)
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    selected = {
        name for name in weights if ".experts.0." in name
    }
    with pytest.raises(ValueError, match="partially selected"):
        validate_fusion_closure(weights, selected)
