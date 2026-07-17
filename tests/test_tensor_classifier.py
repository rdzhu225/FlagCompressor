from flag_compressor.inspect.tensor_classifier import classify_weight_module, infer_source_format


def test_classifies_common_checkpoint_linear_weights_without_model_adapter():
    assert (
        classify_weight_module("model.layers.0.self_attn.q_proj.weight")
        == "attn_linear"
    )
    assert (
        classify_weight_module("model.layers.0.mlp.down_proj.weight")
        == "mlp_linear"
    )
    assert (
        classify_weight_module("model.layers.0.mlp.experts.12.gate_proj.weight")
        == "moe_mlp_linear"
    )
    assert (
        classify_weight_module("model.layers.0.mlp.shared_experts.down_proj.weight")
        == "shared_moe_mlp_linear"
    )
    assert classify_weight_module("model.embed_tokens.weight") == "embedding"


def test_moe_mlp_linear_scaled_byte_weight_is_inferred_as_fp4():
    assert (
        infer_source_format(
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            element_size=1,
            scale_name="model.layers.0.mlp.experts.0.gate_proj.scale",
            module_kind="moe_mlp_linear",
        )
        == "fp4_e2m1_e8m0"
    )
    assert (
        infer_source_format(
            "model.layers.0.self_attn.q_proj.weight",
            element_size=1,
            scale_name="model.layers.0.self_attn.q_proj.scale",
            module_kind="attn_linear",
        )
        == "fp8_block_e8m0"
    )
