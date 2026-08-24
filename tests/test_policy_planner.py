import pytest

from flagos_compressor.core.planner import build_quantize_plan
from flagos_compressor.core.policy import QuantizationPolicy, UnselectedWeightsPolicy
from flagos_compressor.core.profile import ModelProfile, TensorInfo


def _tensor(name, *, storage_format=None, dtype="bfloat16", shape=(4, 32), scale_name=None, tags=()):
    logical_shape = (shape[0], shape[1] * 2) if storage_format == "fp4_e2m1_e8m0" else shape
    return TensorInfo(
        name=name,
        shape=shape,
        logical_shape=logical_shape,
        dtype=dtype,
        shard="model.safetensors",
        element_size=1 if storage_format else 2,
        scale_name=scale_name,
        storage_format=storage_format,
        tags=tags,
    )


def _profile():
    profile = ModelProfile(model_path="/tmp/model")
    routed = "model.layers.0.mlp.experts.0.w1.weight"
    shared = "model.layers.0.mlp.shared_experts.w1.weight"
    attention = "model.layers.0.self_attn.o_proj.weight"
    profile.tensors = {
        routed: _tensor(
            routed,
            storage_format="fp4_e2m1_e8m0",
            dtype="uint8",
            shape=(4, 16),
            scale_name=routed + ".source_scale",
            tags=("linear", "moe", "moe.routed"),
        ),
        shared: _tensor(shared, tags=("linear", "moe", "moe.shared")),
        attention: _tensor(attention, tags=("attention", "linear")),
    }
    return profile


def test_moe_selection_includes_routed_and_shared():
    plan = build_quantize_plan(_profile(), QuantizationPolicy(selections=("moe",)))
    assert plan.input_format_counts == {"fp4_e2m1_e8m0": 1, "bf16": 1}
    assert plan.output_format_counts == {
        "compressed_tensors_int4_groupwise": 2
    }
    assert len(plan.kept_tensors) == 1


def test_regex_can_select_arbitrary_linear():
    policy = QuantizationPolicy(include_names=(r"self_attn\.o_proj\.weight$",))
    plan = build_quantize_plan(_profile(), policy)
    assert plan.output_format_counts["compressed_tensors_int4_groupwise"] == 1
    assert plan.output_format_counts["bf16"] == 1


def test_builtin_exclude_removes_shared_experts():
    policy = QuantizationPolicy(selections=("moe",), exclude_selections=("moe.shared",))
    plan = build_quantize_plan(_profile(), policy)
    assert plan.output_format_counts["compressed_tensors_int4_groupwise"] == 1


def test_preserve_policy_has_an_explicit_runtime_contract_boundary():
    policy = QuantizationPolicy(
        selections=("moe",),
        unselected=UnselectedWeightsPolicy(strategy="preserve", format=None),
    )
    with pytest.raises(ValueError, match="runtime config exporter"):
        build_quantize_plan(_profile(), policy)


def test_selected_shape_must_align_to_group_size():
    profile = _profile()
    name = "model.layers.0.self_attn.o_proj.weight"
    profile.tensors[name] = _tensor(name, shape=(4, 30), tags=("attention", "linear"))
    with pytest.raises(ValueError, match="divisible"):
        build_quantize_plan(profile, QuantizationPolicy(selections=("attention",)))


def test_selected_shape_must_align_to_pack_word():
    profile = _profile()
    name = "model.layers.0.self_attn.o_proj.weight"
    profile.tensors[name] = _tensor(name, shape=(4, 6), tags=("attention", "linear"))
    with pytest.raises(ValueError, match="divisible by 8"):
        build_quantize_plan(
            profile,
            QuantizationPolicy(selections=("attention",), group_size=2),
        )


def test_int8_selection_uses_w8a16_format_without_int4_pack_constraint():
    profile = _profile()
    name = "model.layers.0.self_attn.o_proj.weight"
    profile.tensors[name] = _tensor(
        name,
        shape=(4, 12),
        tags=("attention", "linear"),
    )
    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(
            selections=("attention",),
            num_bits=8,
            group_size=4,
        ),
    )
    assert plan.output_format_counts["compressed_tensors_int8_groupwise"] == 1
    action = next(
        action
        for action in plan.actions
        if action.output_format.name == "compressed_tensors_int8_groupwise"
    )
    assert action.output_format.params["num_bits"] == 8


def test_int8_channel_selection_does_not_require_group_alignment():
    profile = _profile()
    name = "model.layers.0.self_attn.o_proj.weight"
    profile.tensors[name] = _tensor(
        name,
        shape=(4, 13),
        tags=("attention", "linear"),
    )
    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(
            selections=("attention",),
            num_bits=8,
            strategy="channel",
        ),
    )
    assert plan.output_format_counts == {
        "compressed_tensors_int8_channelwise": 1,
        "bf16": 1,
    }
    channel_action = next(
        action
        for action in plan.actions
        if action.output_format.name == "compressed_tensors_int8_channelwise"
    )
    assert channel_action.output_format.params["strategy"] == "channel"
    assert channel_action.output_format.params["group_size"] is None


def test_w8a8_plan_propagates_bf16_scale_dtype():
    profile = _profile()
    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(
            selections=("attention",),
            num_bits=8,
            activation_num_bits=8,
            strategy="channel",
            scale_dtype="bf16",
        ),
    )
    action = next(
        action
        for action in plan.actions
        if action.output_format.name == "compressed_tensors_w8a8_channelwise"
    )
    assert action.output_format.params["scale_dtype"] == "bfloat16"


def test_int8_channel_rejects_routed_moe():
    with pytest.raises(ValueError, match="MoE.*group strategy"):
        build_quantize_plan(
            _profile(),
            QuantizationPolicy(
                selections=("moe.routed",),
                num_bits=8,
                strategy="channel",
            ),
        )
