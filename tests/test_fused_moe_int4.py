import json
from pathlib import Path

import torch
from safetensors.torch import save_file

import flag_compressor.formats.register  # noqa: F401
import flag_compressor.quantizers.register  # noqa: F401
from flag_compressor.backends.base import BackendRunContext
from flag_compressor.backends.registry import build_backend
from flag_compressor.core.executor import execute_plan
from flag_compressor.core.planner import build_quantize_plan
from flag_compressor.core.policy import QuantizationPolicy
from flag_compressor.core.profile import ModelProfile, TensorInfo
from flag_compressor.core.report import ConversionReport
from flag_compressor.core.validation import validate_artifact
from flag_compressor.formats.base import get_weight_format
from flag_compressor.formats.compressed_tensors_moe import fused_expert_bank_prefix
from flag_compressor.formats.int4_pack import unpack_uint4b8_int32
from flag_compressor.inspect.checkpoint_scanner import scan_hf_safetensors
from flag_compressor.inspect.tensor_classifier import classify_weight


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


def test_classifier_tags_fused_expert_banks():
    kind, tags = classify_weight("m.mlp.experts.gate_up_proj")
    assert kind == "moe_routed_fused"
    assert "moe.routed" in tags
    # A shared expert stored fused is not routed.
    assert classify_weight("m.mlp.shared_expert.gate_up_proj") == (None, ())


def test_prefix_extraction():
    assert (
        fused_expert_bank_prefix("m.mlp.experts.gate_up_proj") == "m.mlp.experts"
    )


def test_planner_selects_fused_banks_as_moe_int4():
    plan = build_quantize_plan(_profile(), QuantizationPolicy(selections=("moe.routed",)))
    assert plan.output_format_counts == {"compressed_tensors_int4_moe_fused": 2}
    assert not plan.kept_tensors


def test_gate_up_requires_even_out_features():
    profile = ModelProfile(model_path="/tmp/model")
    name = "m.mlp.experts.gate_up_proj"
    profile.tensors = {name: _fused_bank(name, (2, 15, 64))}
    try:
        build_quantize_plan(profile, QuantizationPolicy(selections=("moe.routed",)))
    except ValueError as exc:
        assert "gate" in str(exc) or "odd" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected odd fused out_features to be rejected")


def test_in_features_must_align_to_group_size():
    profile = ModelProfile(model_path="/tmp/model")
    name = "m.mlp.experts.down_proj"
    profile.tensors = {name: _fused_bank(name, (2, 64, 30))}
    try:
        build_quantize_plan(profile, QuantizationPolicy(selections=("moe.routed",)))
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected misaligned in_features to be rejected")


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
        },
    )
    # gate + up per expert, each with packed/scale/shape.
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
    rel = (deq - orig).norm() / orig.norm()
    assert rel.item() < 0.2


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
        {"quantizer": "mse", "group_size": 32, "proj_kind": "down_proj"},
    )
    assert len(res.tensors) == 2 * 1 * 3
    assert "m.mlp.experts.1.down_proj.weight_packed" in res.tensors


def _make_fused_checkpoint(path: Path) -> None:
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
        json.dump(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map}, f
        )
    with (path / "config.json").open("w") as f:
        json.dump({"torch_dtype": "bfloat16", "num_experts": num_experts}, f)


def test_fused_moe_end_to_end_validates(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "quantized"
    _make_fused_checkpoint(source)
    profile = scan_hf_safetensors(source)
    # The fused banks must be classified as routed MoE.
    banks = [
        name
        for name, info in profile.tensors.items()
        if info.module_kind == "moe_routed_fused"
    ]
    assert len(banks) == 2

    plan = build_quantize_plan(
        profile,
        QuantizationPolicy(
            selections=("moe.routed",), n_candidates=8, chunk_size=16
        ),
    )
    assert plan.output_format_counts == {"compressed_tensors_int4_moe_fused": 2}

    backend = build_backend("cpu", None)
    execute_plan(source, output, plan, backend)

    result = validate_artifact(output)
    assert result["valid"], result["errors"]
    # 4 experts * (gate + up + down) = 12 quantized projections.
    assert result["int4_tensors"] == 12

    config = json.loads((output / "config.json").read_text())
    qc = config["quantization_config"]
    assert qc["quant_method"] == "compressed-tensors"
    assert qc["format"] == "pack-quantized"
    group = next(iter(qc["config_groups"].values()))
    assert group["weights"]["num_bits"] == 4
    # Non-expert weights stay bf16 (attention, router gate untouched by select).
    assert config["torch_dtype"] == "bfloat16"

    # The fused bank names are gone; per-expert packed tensors exist.
    index = json.loads(
        (output / "model.safetensors.index.json").read_text()
    )["weight_map"]
    assert "model.layers.0.mlp.experts.gate_up_proj" not in index
    assert "model.layers.0.mlp.experts.0.gate_proj.weight_packed" in index
    assert "model.layers.0.mlp.experts.0.up_proj.weight_packed" in index
    assert "model.layers.0.mlp.experts.3.down_proj.weight_packed" in index
    # Untouched weights preserved.
    assert "model.layers.0.self_attn.o_proj.weight" in index
    assert "model.layers.0.mlp.gate.weight" in index
