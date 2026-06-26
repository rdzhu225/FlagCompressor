from pathlib import Path

from quant_engine.core.planner import build_plan
from quant_engine.core.profile import ModelProfile, TensorInfo
from quant_engine.core.recipe import BackendConfig, ModelConfig, Recipe, RuleConfig


def test_planner_uses_recipe_groups_without_model_hardcoding():
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
        ),
        "model.layers.0.self_attn.q_proj.weight": TensorInfo(
            name="model.layers.0.self_attn.q_proj.weight",
            shape=(4, 4),
            dtype="uint8",
            shard="model-1.safetensors",
            element_size=1,
            scale_name="model.layers.0.self_attn.q_proj.scale",
            source_format="fp8_block_e8m0",
        ),
    }
    recipe = Recipe(
        version=1,
        model=ModelConfig(input_path=Path("/tmp/model"), output_path=Path("/tmp/out")),
        backend=BackendConfig(),
        module_groups={
            "moe_experts": {
                "include": [r".*\.experts\.\d+\..*\.weight$"],
                "exclude": [r".*shared_experts.*"],
            },
            "scaled_dense": {"selector": {"has_scale": True}, "exclude_groups": ["moe_experts"]},
        },
        rules=[
            RuleConfig(
                name="moe_fp4_to_int4",
                group="moe_experts",
                when={"source_format": "fp4_e2m1_e8m0"},
                transform="fp4_to_int4",
            ),
            RuleConfig(
                name="fp8_to_bf16",
                group="scaled_dense",
                when={"has_scale": True},
                transform="fp8_to_bf16",
            ),
        ],
    )
    plan = build_plan(profile, recipe)
    assert plan.transform_counts == {"fp4_to_int4": 1, "fp8_to_bf16": 1}

