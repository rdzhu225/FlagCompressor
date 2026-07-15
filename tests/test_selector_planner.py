from quant_engine.core.planner import build_plan
from quant_engine.core.profile import ModelProfile, TensorInfo


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
    }
    return profile


def test_planner_defaults_to_full_bf16():
    plan = build_plan(_profile())

    assert plan.transform_counts == {"fp4_to_bf16": 1, "fp8_to_bf16": 2}
    assert plan.summary()["module_kinds"] == {"moe_mlp_linear": 1, "attn_linear": 1, "unknown": 1}


def test_planner_can_route_moe_fp4_to_int4():
    plan = build_plan(_profile(), target="moe-int4")

    assert plan.transform_counts == {"fp4_to_int4": 1, "fp8_to_bf16": 2}
    assert plan.summary()["module_kinds"] == {"moe_mlp_linear": 1, "attn_linear": 1, "unknown": 1}
