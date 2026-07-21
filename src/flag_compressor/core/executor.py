from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from flag_compressor.backends.base import BackendRunContext, QuantBackend
from flag_compressor.core.plan import ExecutionPlan, TensorAction
from flag_compressor.core.report import ConversionReport
from flag_compressor.formats.base import get_weight_format
from flag_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint


def _build_action_map(plan: ExecutionPlan) -> dict[str, TensorAction]:
    return {action.tensor.name: action for action in plan.actions}


def _build_scale_names(plan: ExecutionPlan) -> set[str]:
    return {action.tensor.scale_name for action in plan.actions if action.tensor.scale_name}


_STRIP_CONFIG_KEYS = (
    "quantization_config",
    "compression_config",
    "quant_method",
)

_INT4_FORMATS = {"int4_symmetric_groupwise"}
_BF16_FORMATS = {"bf16"}


def _patch_bf16_config(output_path: Path) -> None:
    """Rewrite the output ``config.json`` so it describes a plain BF16 model.

    After dequantization the artifact no longer carries FP8/FP4 weights, so
    the source quantization metadata must be stripped and ``torch_dtype`` set
    to ``bfloat16``. ``expert_dtype``, when present on MoE configs, is
    rewritten rather than dropped so downstream loaders keep the field.
    """
    config_path = output_path / "config.json"
    if not config_path.exists():
        return

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config["torch_dtype"] = "bfloat16"
    for key in _STRIP_CONFIG_KEYS:
        config.pop(key, None)
    if "expert_dtype" in config:
        config["expert_dtype"] = "bfloat16"

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _patch_mixed_config(output_path: Path) -> None:
    """Remove stale source quantization claims from a custom mixed artifact."""
    config_path = output_path / "config.json"
    if not config_path.exists():
        return
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["torch_dtype"] = "bfloat16"
    for key in (*_STRIP_CONFIG_KEYS, "expert_dtype"):
        config.pop(key, None)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _write_quant_manifest(
    output_path: Path,
    plan: ExecutionPlan,
    generated_scale_map: dict[str, str],
) -> None:
    tensors: dict[str, dict] = {}
    for action in plan.actions:
        tensor = action.tensor
        if action.output_format.name in _INT4_FORMATS:
            logical_shape = tensor.effective_logical_shape
            group_size = int(action.output_format.params.get("group_size", 32))
            tensors[tensor.name] = {
                "format": "int4_symmetric_groupwise",
                "input_format": action.input_format.name,
                "storage_dtype": "uint8",
                "storage_shape": [logical_shape[0], logical_shape[1] // 2],
                "logical_shape": list(logical_shape),
                "scale": generated_scale_map[tensor.name],
                "scale_shape": [logical_shape[0], logical_shape[1] // group_size],
                "group_size": group_size,
                "rule": action.rule_name,
            }
        elif action.output_format.name in _BF16_FORMATS:
            tensors[tensor.name] = {
                "format": "bf16",
                "input_format": action.input_format.name,
                "storage_dtype": "bfloat16",
                "storage_shape": list(tensor.effective_logical_shape),
                "logical_shape": list(tensor.effective_logical_shape),
                "rule": action.rule_name,
            }

    # When the user keeps non-selected source weights, include their source
    # layouts so a runtime adapter does not need to guess a mixed artifact.
    for tensor in plan.kept_tensors:
        if tensor.storage_format and tensor.scale_name:
            tensors[tensor.name] = {
                "format": tensor.storage_format,
                "storage_dtype": tensor.dtype,
                "storage_shape": list(tensor.shape),
                "logical_shape": list(tensor.effective_logical_shape),
                "scale": tensor.scale_name,
                "kept_from_source": True,
            }

    manifest = {
        "abi_version": "flag_compressor.artifact.v1",
        "artifact_kind": "mixed_int4",
        "formats": {
            "int4_symmetric_groupwise": {
                "bits": 4,
                "signed": True,
                "storage_dtype": "uint8",
                "pack_order": "low_even_high_odd",
                "scale_dtype": "bfloat16",
                "scale_layout": "row_group",
                "zero_point": False,
            },
            "bf16": {"storage_dtype": "bfloat16"},
            "fp4_e2m1_e8m0": {
                "storage_dtype": "uint8",
                "pack_order": "low_even_high_odd",
                "scale_dtype": "e8m0",
                "scale_layout": "row_group",
            },
            "fp8_block_e8m0": {
                "storage_dtype": "float8",
                "block_size": 128,
                "scale_dtype": "e8m0",
            },
        },
        "tensors": tensors,
    }
    with (output_path / "quant_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def execute_plan(
    input_path: str | Path,
    output_path: str | Path,
    plan: ExecutionPlan,
    backend: QuantBackend,
) -> ConversionReport:
    checkpoint = HfSafetensorsCheckpoint(input_path)
    output = Path(output_path)
    if Path(input_path).resolve() == output.resolve():
        raise ValueError("Input and output checkpoint directories must be different")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.copy_auxiliary_files(output)
    manifest_path = output / "quant_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    action_map = _build_action_map(plan)
    old_scale_names = _build_scale_names(plan)
    generated_tensor_locations: dict[str, str] = {}
    generated_scale_map: dict[str, str] = {}
    report = ConversionReport(backend=backend.name)
    context = BackendRunContext(report=report)

    loaded_shards: dict[str, dict] = {}
    total_size = 0

    def get_tensor(name: str):
        shard = checkpoint.weight_map[name]
        if shard not in loaded_shards:
            loaded_shards[shard] = checkpoint.load_shard(shard)
        return loaded_shards[shard][name]

    for shard_name, _ in tqdm(list(checkpoint.iter_shards()), desc="Converting"):
        current = checkpoint.load_shard(shard_name)
        loaded_shards[shard_name] = current
        new_state: dict = {}

        for tensor_name, tensor in current.items():
            if tensor_name in old_scale_names:
                report.skipped_scales += 1
                continue

            action = action_map.get(tensor_name)
            if action is None:
                new_state[tensor_name] = tensor
                report.kept += 1
                continue

            scale = get_tensor(action.tensor.scale_name) if action.tensor.scale_name else None
            input_format = get_weight_format(action.input_format.name)
            canonical_weight = input_format.to_canonical(
                tensor,
                scale,
                backend,
                context,
                action.input_format.params,
            )
            output_format = get_weight_format(action.output_format.name)
            result = output_format.from_canonical(
                tensor_name,
                canonical_weight,
                backend,
                context,
                action.output_format.params,
            )
            new_state.update(result.tensors)
            report.converted += 1
            for generated_name in result.generated_tensor_names:
                if generated_name in checkpoint.weight_map and generated_name not in old_scale_names:
                    raise ValueError(
                        f"Generated tensor {generated_name!r} collides with an existing tensor"
                    )
                generated_tensor_locations[generated_name] = shard_name
                generated_scale_map[tensor_name] = generated_name

        total_size += sum(item.numel() * item.element_size() for item in new_state.values())

        checkpoint.save_shard(output, shard_name, new_state)
        while len(loaded_shards) > 2:
            oldest = next(iter(loaded_shards))
            del loaded_shards[oldest]

    new_weight_map = {
        name: shard
        for name, shard in checkpoint.weight_map.items()
        if name not in old_scale_names
    }
    new_weight_map.update(generated_tensor_locations)
    checkpoint.write_index(output, new_weight_map, total_size=total_size)
    if plan.metadata.get("artifact_kind") == "mixed_int4":
        _patch_mixed_config(output)
        _write_quant_manifest(output, plan, generated_scale_map)
    else:
        _patch_bf16_config(output)
    report.output_tensors = len(new_weight_map)
    report.extras.update(
        {
            "artifact_kind": plan.metadata.get("artifact_kind"),
            "input_formats": dict(plan.input_format_counts),
            "output_formats": dict(plan.output_format_counts),
            "generated_tensors": len(generated_tensor_locations),
        }
    )
    report.finish()
    report_filename = (
        "quantization_report.json"
        if plan.metadata.get("artifact_kind") == "mixed_int4"
        else "conversion_report.json"
    )
    report.extras["report_file"] = report_filename
    report.save(output / report_filename)
    # Keep the original report filename for callers of the pre-subcommand CLI.
    report.save(output / "quant_report.json")
    return report
