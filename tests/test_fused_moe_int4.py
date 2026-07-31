import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import flagos_compressor.formats.register  # noqa: F401
import flagos_compressor.quantizers.register  # noqa: F401
from flagos_compressor.backends.base import BackendRunContext
from flagos_compressor.backends.registry import build_backend
from flagos_compressor.core.executor import execute_plan
from flagos_compressor.core.moe_layout import (
    Qwen35MoeLayout,
    select_moe_layout,
)
from flagos_compressor.core.planner import build_quantize_plan
from flagos_compressor.core.policy import QuantizationPolicy
from flagos_compressor.core.profile import ModelProfile, TensorInfo
from flagos_compressor.core.report import ConversionReport
from flagos_compressor.core.validation import validate_artifact
from flagos_compressor.formats.base import get_weight_format
from flagos_compressor.formats.compressed_tensors_moe import fused_expert_bank_prefix
from flagos_compressor.formats.int4_pack import unpack_uint4b8_int32
from flagos_compressor.inspect.checkpoint_scanner import scan_hf_safetensors
from flagos_compressor.inspect.tensor_classifier import classify_weight

LAYOUT = Qwen35MoeLayout()


def _fused_bank(name, shape):
    return TensorInfo(
        name=name,
        shape=shape,
        logical_shape=shape,
        dtype="bfloat16",
        shard="model.safetensors",
        element_size=2,
        scale_name=None,
        module_kind="moe_routed_fused",
        tags=("linear", "moe", "moe.routed"),
    )


def _profile():
    profile = ModelProfile(model_path="/tmp/model")
    gate_up = "model.layers.0.mlp.experts.gate_up_proj"
    down = "model.layers.0.mlp.experts.down_proj"
    profile.tensors = {
        gate_up: _fused_bank(gate_up, (8, 128, 64)),
        down: _fused_bank(down, (8, 64, 96)),
    }
    return profile


# --------------------------------------------------------------------------
# Classification / layout selection
# --------------------------------------------------------------------------
def test_classifier_tags_fused_expert_banks():
    kind, tags = classify_weight("m.mlp.experts.gate_up_proj")
    assert kind == "moe_routed_fused"
    assert "moe.routed" in tags
    assert classify_weight("m.mlp.shared_expert.gate_up_proj") == (None, ())


def test_prefix_extraction():
    assert fused_expert_bank_prefix("m.mlp.experts.gate_up_proj") == "m.mlp.experts"


def test_select_layout_by_model_type_and_arch():
    assert isinstance(select_moe_layout({"model_type": "qwen3_5_moe"}), Qwen35MoeLayout)
    assert isinstance(
        select_moe_layout({"text_config": {"model_type": "qwen3_5_moe_text"}}),
        Qwen35MoeLayout,
    )
    assert isinstance(
        select_moe_layout({"architectures": ["Qwen3_5MoeForConditionalGeneration"]}),
        Qwen35MoeLayout,
    )


def test_select_layout_refuses_unknown_model():
    # Unknown layouts must be rejected, not guessed (Qwen3-VL needs a transpose).
    with pytest.raises(ValueError, match="not supported"):
        select_moe_layout({"model_type": "qwen3_vl_moe"})


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------
def test_planner_selects_fused_banks_as_moe_int4():
    plan = build_quantize_plan(
        _profile(), QuantizationPolicy(selections=("moe.routed",)), LAYOUT
    )
    assert plan.output_format_counts == {"compressed_tensors_int4_moe_fused": 2}
    assert not plan.kept_tensors


def test_planner_selects_fused_banks_as_moe_int8():
    plan = build_quantize_plan(
        _profile(),
        QuantizationPolicy(
            selections=("moe.routed",),
            num_bits=8,
            group_size=32,
        ),
        LAYOUT,
    )
    assert plan.output_format_counts == {"compressed_tensors_int8_moe_fused": 2}
    assert not plan.kept_tensors


def test_planner_selects_fused_banks_as_w8a8():
    plan = build_quantize_plan(
        _profile(),
        QuantizationPolicy(
            selections=("moe.routed",),
            num_bits=8,
            activation_num_bits=8,
            strategy="channel",
        ),
        LAYOUT,
    )
    assert plan.output_format_counts == {
        "compressed_tensors_w8a8_int8_moe_fused": 2
    }
    assert not plan.kept_tensors


def test_planner_requires_layout_for_fused_experts():
    with pytest.raises(ValueError, match="requires a MoeLayout"):
        build_quantize_plan(_profile(), QuantizationPolicy(selections=("moe.routed",)))


def test_gate_up_requires_even_out_features():
    profile = ModelProfile(model_path="/tmp/model")
    name = "m.mlp.experts.gate_up_proj"
    profile.tensors = {name: _fused_bank(name, (2, 15, 64))}
    with pytest.raises(ValueError, match="odd|gate"):
        build_quantize_plan(
            profile, QuantizationPolicy(selections=("moe.routed",)), LAYOUT
        )


def test_in_features_must_align_to_group_size():
    profile = ModelProfile(model_path="/tmp/model")
    name = "m.mlp.experts.down_proj"
    profile.tensors = {name: _fused_bank(name, (2, 64, 48))}  # 48 % 32 != 0
    with pytest.raises(ValueError, match="group_size"):
        build_quantize_plan(
            profile, QuantizationPolicy(selections=("moe.routed",), group_size=32), LAYOUT
        )


def test_in_features_must_be_multiple_of_8():
    # group_size even but in_features % 8 != 0 must fail in the planner, not at
    # execution (int32 pack packs 8 nibbles per word).
    profile = ModelProfile(model_path="/tmp/model")
    name = "m.mlp.experts.down_proj"
    profile.tensors = {name: _fused_bank(name, (2, 64, 6))}  # 6 % 2 == 0, 6 % 8 != 0
    with pytest.raises(ValueError, match="divisible by 8"):
        build_quantize_plan(
            profile, QuantizationPolicy(selections=("moe.routed",), group_size=2), LAYOUT
        )


def test_partial_bank_selection_is_rejected():
    # Selecting gate_up_proj but not down_proj would split the FusedMoE unit;
    # the planner must refuse it (fusion closure over 3D banks).
    profile = _profile()
    policy = QuantizationPolicy(include_names=(r"experts\.gate_up_proj$",), group_size=32)
    with pytest.raises(ValueError, match="partially selected|splits fused"):
        build_quantize_plan(profile, policy, LAYOUT)


# --------------------------------------------------------------------------
# Format round-trip
# --------------------------------------------------------------------------
def test_fused_format_roundtrip_names_and_values():
    torch.manual_seed(0)
    num_experts, fused_out, in_features = 3, 16, 64
    bank = torch.randn(num_experts, fused_out, in_features, dtype=torch.bfloat16)
    backend = build_backend("cpu", None)
    ctx = BackendRunContext(report=ConversionReport(backend="cpu"))
    fmt = get_weight_format("compressed_tensors_int4_moe_fused")
    res = fmt.from_canonical(
        "m.mlp.experts.gate_up_proj",
        bank,
        backend,
        ctx,
        {
            "quantizer": "mse",
            "group_size": 32,
            "n_candidates": 100,
            "chunk_size": 4096,
            "proj_kind": "gate_up_proj",
            "layout": LAYOUT.name,
            "num_experts": num_experts,
        },
    )
    assert len(res.tensors) == num_experts * 2 * 3
    base = "m.mlp.experts.0.gate_proj"
    packed = res.tensors[base + ".weight_packed"]
    scale = res.tensors[base + ".weight_scale"]
    shape = res.tensors[base + ".weight_shape"]
    assert packed.dtype == torch.int32
    assert tuple(packed.shape) == (8, in_features // 8)
    assert scale.dtype == torch.bfloat16
    assert tuple(scale.shape) == (8, in_features // 32)
    assert shape.tolist() == [8, in_features]

    orig = bank[0, :8, :].float()
    q = unpack_uint4b8_int32(packed).float()
    deq = (q.reshape(-1, 32) * scale.float().reshape(-1, 1)).reshape(8, in_features)
    assert ((deq - orig).norm() / orig.norm()).item() < 0.2


def test_down_proj_maps_to_single_projection():
    torch.manual_seed(1)
    bank = torch.randn(2, 48, 96, dtype=torch.bfloat16)
    backend = build_backend("cpu", None)
    ctx = BackendRunContext(report=ConversionReport(backend="cpu"))
    fmt = get_weight_format("compressed_tensors_int4_moe_fused")
    res = fmt.from_canonical(
        "m.mlp.experts.down_proj",
        bank,
        backend,
        ctx,
        {
            "quantizer": "mse",
            "group_size": 32,
            "proj_kind": "down_proj",
            "layout": LAYOUT.name,
            "num_experts": 2,
        },
    )
    assert len(res.tensors) == 2 * 1 * 3
    assert "m.mlp.experts.1.down_proj.weight_packed" in res.tensors


def test_fused_int8_format_roundtrip_shape_and_values():
    torch.manual_seed(2)
    bank = torch.randn(2, 16, 64, dtype=torch.bfloat16)
    backend = build_backend("cpu", None)
    ctx = BackendRunContext(report=ConversionReport(backend="cpu"))
    fmt = get_weight_format("compressed_tensors_int8_moe_fused")
    res = fmt.from_canonical(
        "m.mlp.experts.gate_up_proj",
        bank,
        backend,
        ctx,
        {
            "quantizer": "mse",
            "group_size": 32,
            "n_candidates": 20,
            "proj_kind": "gate_up_proj",
            "layout": LAYOUT.name,
            "num_experts": 2,
        },
    )
    base = "m.mlp.experts.0.gate_proj"
    packed = res.tensors[base + ".weight_packed"]
    scale = res.tensors[base + ".weight_scale"]
    assert packed.shape == (8, 16)
    assert scale.shape == (8, 2)

    from flagos_compressor.formats.int8_pack import unpack_uint8b128_int32

    quantized = unpack_uint8b128_int32(packed, in_features=64).float()
    dequantized = (
        quantized.reshape(-1, 32) * scale.float().reshape(-1, 1)
    ).reshape(8, 64)
    original = bank[0, :8, :].float()
    assert ((dequantized - original).norm() / original.norm()).item() < 0.02


def test_fused_w8a8_format_writes_per_expert_raw_int8():
    torch.manual_seed(4)
    bank = torch.randn(2, 16, 64, dtype=torch.bfloat16)
    backend = build_backend("cpu", None)
    ctx = BackendRunContext(report=ConversionReport(backend="cpu"))
    fmt = get_weight_format("compressed_tensors_w8a8_int8_moe_fused")
    res = fmt.from_canonical(
        "m.mlp.experts.gate_up_proj",
        bank,
        backend,
        ctx,
        {
            "quantizer": "mse",
            "strategy": "channel",
            "n_candidates": 20,
            "chunk_size": 16,
            "proj_kind": "gate_up_proj",
            "layout": LAYOUT.name,
            "num_experts": 2,
        },
    )
    assert len(res.tensors) == 2 * 2 * 2
    base = "m.mlp.experts.0.gate_proj"
    quantized = res.tensors[base + ".weight"]
    scale = res.tensors[base + ".weight_scale"]
    assert quantized.dtype == torch.int8
    assert quantized.shape == (8, 64)
    assert scale.dtype == torch.float32
    assert scale.shape == (8, 1)
    reconstructed = quantized.float() * scale
    original = bank[0, :8, :].float()
    assert ((reconstructed - original).norm() / original.norm()).item() < 0.02


# --------------------------------------------------------------------------
# End-to-end + artifact re-scan
# --------------------------------------------------------------------------
def _make_fused_checkpoint(path: Path, model_type="qwen3_5_moe") -> None:
    path.mkdir()
    num_experts = 4
    state = {
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(
            num_experts, 16, 64, dtype=torch.bfloat16
        ),
        "model.layers.0.mlp.experts.down_proj": torch.randn(
            num_experts, 64, 32, dtype=torch.bfloat16
        ),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(
            64, 64, dtype=torch.bfloat16
        ),
        "model.layers.0.mlp.gate.weight": torch.randn(
            num_experts, 64, dtype=torch.bfloat16
        ),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(state, str(path / shard))
    weight_map = {name: shard for name in state}
    total_size = sum(t.numel() * t.element_size() for t in state.values())
    with (path / "model.safetensors.index.json").open("w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f)
    with (path / "config.json").open("w") as f:
        json.dump({"torch_dtype": "bfloat16", "model_type": model_type, "num_experts": num_experts}, f)


def test_fused_moe_end_to_end_validates(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "quantized"
    _make_fused_checkpoint(source)
    profile = scan_hf_safetensors(source)
    assert sum(
        1 for info in profile.tensors.values() if info.module_kind == "moe_routed_fused"
    ) == 2

    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(selections=("moe.routed",), n_candidates=8, chunk_size=16),
        LAYOUT,
    )
    assert plan.output_format_counts == {"compressed_tensors_int4_moe_fused": 2}

    execute_plan(source, output, plan, build_backend("cpu", None))

    result = validate_artifact(output)
    assert result["valid"], result["errors"]
    # 4 experts * (gate + up + down) = 12 quantized projections.
    assert result["int4_tensors"] == 12

    config = json.loads((output / "config.json").read_text())
    qc = config["quantization_config"]
    assert qc["quant_method"] == "compressed-tensors"
    assert qc["format"] == "pack-quantized"
    assert next(iter(qc["config_groups"].values()))["weights"]["num_bits"] == 4
    assert config["torch_dtype"] == "bfloat16"

    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    assert "model.layers.0.mlp.experts.gate_up_proj" not in index
    assert "model.layers.0.mlp.experts.0.gate_proj.weight_packed" in index
    assert "model.layers.0.mlp.experts.0.up_proj.weight_packed" in index
    assert "model.layers.0.mlp.experts.3.down_proj.weight_packed" in index
    assert "model.layers.0.self_attn.o_proj.weight" in index
    assert "model.layers.0.mlp.gate.weight" in index


def test_fused_artifact_is_recognized_on_rescan(tmp_path):
    # Finding 2 regression: the quantized artifact's per-expert packed tensors
    # must be recognized by the scanner via the manifest (not left as None).
    source = tmp_path / "source"
    output = tmp_path / "quantized"
    _make_fused_checkpoint(source)
    profile = scan_hf_safetensors(source)
    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(selections=("moe.routed",), n_candidates=8, chunk_size=16),
        LAYOUT,
    )
    execute_plan(source, output, plan, build_backend("cpu", None))

    rescan = scan_hf_safetensors(output)
    packed = [n for n in rescan.tensors if n.endswith(".weight_packed")]
    assert packed, "no packed tensors found on rescan"
    for name in packed:
        info = rescan.tensors[name]
        # The manifest gives every packed tensor a concrete int4 provenance,
        # so it is not silently misread (e.g. as an unknown/None format).
        assert info.storage_format is not None


def test_full_w8a8_moe_end_to_end(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "quantized-w8a8"
    _make_fused_checkpoint(source)
    profile = scan_hf_safetensors(source)
    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(
            selections=("linear",),
            num_bits=8,
            activation_num_bits=8,
            strategy="channel",
            n_candidates=8,
            chunk_size=16,
        ),
        LAYOUT,
    )
    assert plan.output_format_counts == {
        "compressed_tensors_w8a8_int8": 1,
        "compressed_tensors_w8a8_int8_moe_fused": 2,
    }

    execute_plan(source, output, plan, build_backend("cpu", None))

    config = json.loads((output / "config.json").read_text())
    qc = config["quantization_config"]
    assert qc["format"] == "int-quantized"
    scheme = qc["config_groups"]["w8a8_channel"]
    assert scheme["weights"]["strategy"] == "channel"
    assert scheme["input_activations"]["strategy"] == "token"
    assert scheme["input_activations"]["dynamic"] is True

    shard = load_file(output / "model-00001-of-00001.safetensors")
    attention = "model.layers.0.self_attn.o_proj"
    expert = "model.layers.0.mlp.experts.0.gate_proj"
    assert shard[attention + ".weight"].dtype == torch.int8
    assert shard[attention + ".weight_scale"].dtype == torch.float32
    assert shard[expert + ".weight"].dtype == torch.int8
    assert shard[expert + ".weight_scale"].dtype == torch.float32
    assert "model.layers.0.mlp.experts.gate_up_proj" not in shard
    assert shard["model.layers.0.mlp.gate.weight"].dtype == torch.bfloat16

    result = validate_artifact(output)
    assert result["valid"], result["errors"]
    assert result["int8_tensors"] == 13

    manifest = json.loads(
        (output / "quantization_manifest.json").read_text()
    )
    assert manifest["artifact"]["compression_format"] == "int-quantized"
    assert manifest["artifact"]["weight_encoding"] == "int8"
    assert "pack_dtype" not in manifest["artifact"]

    rescan = scan_hf_safetensors(output)
    assert (
        rescan.tensors[expert + ".weight"].storage_format
        == "compressed-tensors-int-quantized-int8"
    )


def test_partial_selection_via_selector_is_rejected_end_to_end(tmp_path):
    # Finding 1 regression through the real scan+plan path.
    source = tmp_path / "source"
    _make_fused_checkpoint(source)
    profile = scan_hf_safetensors(source)
    policy = QuantizationPolicy(
        include_names=(r"experts\.gate_up_proj$",), group_size=32
    )
    with pytest.raises(ValueError, match="partially selected|splits fused"):
        build_quantize_plan(profile, policy, LAYOUT)
