from flag_compressor.core.planner import (
    MOE_EXPERT_INCLUDE_PATTERN,
    build_plan,
    make_regex_selector,
)
from flag_compressor.core.profile import ModelProfile, TensorInfo


def _profile() -> ModelProfile:
    profile = ModelProfile(model_path="/tmp/model")
    profile.tensors = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": TensorInfo(
            name="model.layers.0.mlp.experts.0.gate_proj.weight",
            shape=(4, 2),
            dtype="uint8",
            shard="model-1.safetensors",
            element_size=1,
            scale_name="model.layers.0.mlp.experts.0.gate_proj.scale",
            source_format="fp4_e2m1_e8m0",
            module_kind="moe_mlp_linear",
        ),
        "model.layers.0.self_attn.q_proj.weight": TensorInfo(
            name="model.layers.0.self_attn.q_proj.weight",
            shape=(4, 4),
            dtype="uint8",
            shard="model-1.safetensors",
            element_size=1,
            scale_name="model.layers.0.self_attn.q_proj.scale",
            source_format="fp8_block_e8m0",
            module_kind="attn_linear",
        ),
        "layers.0.attn.wkv.weight": TensorInfo(
            name="layers.0.attn.wkv.weight",
            shape=(4, 4),
            dtype="uint8",
            shard="model-1.safetensors",
            element_size=1,
            scale_name="layers.0.attn.wkv.scale",
            source_format="fp8_block_e8m0",
            module_kind=None,
        ),
        "model.layers.0.mlp.gate_proj.weight": TensorInfo(
            name="model.layers.0.mlp.gate_proj.weight",
            shape=(4, 4),
            dtype="bfloat16",
            shard="model-1.safetensors",
            element_size=2,
            scale_name=None,
            source_format=None,
            module_kind="mlp_linear",
        ),
    }
    return profile


def test_planner_defaults_to_full_bf16():
    plan = build_plan(_profile())

    # Only FP8/FP4 tensors produce actions; the BF16 tensor is kept as-is.
    assert plan.transform_counts == {"fp4_to_bf16": 1, "fp8_to_bf16": 2}


def test_moe_int4_alias_still_routes_moe_fp4_to_int4():
    plan = build_plan(_profile(), target="moe-int4")

    assert plan.transform_counts == {"fp4_to_int4": 1, "fp8_to_bf16": 2}


def test_target_int4_without_selector_is_full_bf16_pass():
    plan = build_plan(_profile(), target="int4", int4_selector=None)

    assert "fp4_to_int4" not in plan.transform_counts
    assert "fp8_to_int4" not in plan.transform_counts
    assert "bf16_to_int4" not in plan.transform_counts


def test_selector_routes_fp8_and_bf16_to_int4_by_name():
    # User asks: quantize all *_proj.weight tensors that are attention or mlp.
    selector = make_regex_selector(
        include=[r".*\.(q_proj|k_proj|v_proj|gate_proj|up_proj|down_proj)\.weight$"],
    )
    plan = build_plan(_profile(), target="int4", int4_selector=selector)

    # q_proj (fp8) -> fp8_to_int4, mlp.gate_proj (bf16 unscaled) -> bf16_to_int4,
    # moe expert gate_proj (fp4) -> fp4_to_int4.
    assert plan.transform_counts.get("fp8_to_int4") == 1
    assert plan.transform_counts.get("bf16_to_int4") == 1
    assert plan.transform_counts.get("fp4_to_int4") == 1
    # wkv (fp8, not selected) still goes to BF16.
    assert plan.transform_counts.get("fp8_to_bf16") == 1


def test_moe_pattern_selects_expert_only():
    selector = make_regex_selector(include=[MOE_EXPERT_INCLUDE_PATTERN])
    plan = build_plan(_profile(), target="int4", int4_selector=selector)

    assert plan.transform_counts.get("fp4_to_int4") == 1
    # attn / non-expert FP8 tensors follow the default BF16 path.
    assert plan.transform_counts.get("fp8_to_bf16") == 2


def test_selector_exclude_wins():
    selector = make_regex_selector(
        include=[r".*\.weight$"],
        exclude=[r".*\.experts\..*"],
    )
    plan = build_plan(_profile(), target="int4", int4_selector=selector)

    # experts tensor stays on fp4->bf16 because of the exclude.
    assert plan.transform_counts.get("fp4_to_bf16") == 1
    assert plan.transform_counts.get("fp4_to_int4", 0) == 0
