from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from quant_engine.backends.base import BackendRunContext, QuantBackend
from quant_engine.core.plan import ExecutionPlan, TensorAction
from quant_engine.core.report import ConversionReport
from quant_engine.io.hf_checkpoint import HfSafetensorsCheckpoint
from quant_engine.transforms.base import get_transform


def _build_action_map(plan: ExecutionPlan) -> dict[str, TensorAction]:
    return {action.tensor.name: action for action in plan.actions}


def _build_scale_names(plan: ExecutionPlan) -> set[str]:
    return {action.tensor.scale_name for action in plan.actions if action.tensor.scale_name}


def _write_quant_manifest(
    output_path: Path,
    plan: ExecutionPlan,
    generated_scale_map: dict[str, str],
) -> None:
    formats = {
        "int4_symmetric_groupwise": {
            "bits": 4,
            "signed": True,
            "storage_dtype": "uint8",
            "scale_dtype": "bf16",
            "pack_order": "low_high",
            "scale_layout": "row_group",
        },
        "bf16": {"storage_dtype": "bf16"},
    }
    tensors = {}
    for action in plan.actions:
        if action.transform == "fp4_to_int4":
            group_size = int((action.quantizer or {}).get("group_size", action.params.get("group_size", 32)))
            tensors[action.tensor.name] = {
                "format": "int4_symmetric_groupwise",
                "scale": generated_scale_map.get(action.tensor.name),
                "group_size": group_size,
                "original_shape": list(action.tensor.shape),
                "source_format": action.tensor.source_format,
                "module_kind": action.tensor.module_kind,
                "rule": action.rule_name,
            }
        elif action.transform in {"fp8_to_bf16", "fp4_to_bf16"}:
            tensors[action.tensor.name] = {
                "format": "bf16",
                "original_shape": list(action.tensor.shape),
                "source_format": action.tensor.source_format,
                "module_kind": action.tensor.module_kind,
                "rule": action.rule_name,
            }

    manifest = {
        "abi_version": "quant_engine.artifact.v1",
        "formats": formats,
        "tensors": tensors,
    }
    with (output_path / "quant_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def execute_plan(
    input_path: str | Path,
    output_path: str | Path,
    plan: ExecutionPlan,
    backend: QuantBackend,
) -> ConversionReport:
    checkpoint = HfSafetensorsCheckpoint(input_path)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.copy_auxiliary_files(output)

    action_map = _build_action_map(plan)
    old_scale_names = _build_scale_names(plan)
    generated_scale_names: list[tuple[str, str]] = []
    generated_scale_map: dict[str, str] = {}
    report = ConversionReport(backend=backend.name)
    context = BackendRunContext(report=report)

    # A tiny CPU cache for cross-shard scale lookup.
    loaded_shards: dict[str, dict] = {}

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
            transform = get_transform(action.transform)
            result = transform.apply(action, tensor, scale, backend, context)
            new_state.update(result.tensors)
            report.converted += 1

            for scale_name in result.generated_scale_names:
                generated_scale_names.append((scale_name, shard_name))
                generated_scale_map[tensor_name] = scale_name

        checkpoint.save_shard(output, shard_name, new_state)
        while len(loaded_shards) > 2:
            oldest = next(iter(loaded_shards))
            del loaded_shards[oldest]

    new_weight_map = {
        name: shard
        for name, shard in checkpoint.weight_map.items()
        if name not in old_scale_names
    }
    for scale_name, shard_name in generated_scale_names:
        new_weight_map[scale_name] = shard_name

    checkpoint.write_index(output, new_weight_map)
    _write_quant_manifest(output, plan, generated_scale_map)
    report.output_tensors = len(new_weight_map)
    report.finish()
    report.save(output / "quant_report.json")
    return report
