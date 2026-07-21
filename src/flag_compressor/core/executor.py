from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from flag_compressor.backends.base import BackendRunContext, QuantBackend
from flag_compressor.core.plan import ExecutionPlan, TensorAction
from flag_compressor.core.report import ConversionReport
from flag_compressor.io.hf_checkpoint import HfSafetensorsCheckpoint
from flag_compressor.transforms.base import get_transform


def _build_action_map(plan: ExecutionPlan) -> dict[str, TensorAction]:
    return {action.tensor.name: action for action in plan.actions}


def _build_scale_names(plan: ExecutionPlan) -> set[str]:
    return {action.tensor.scale_name for action in plan.actions if action.tensor.scale_name}


_STRIP_CONFIG_KEYS = (
    "quantization_config",
    "compression_config",
    "quant_method",
)


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
    report = ConversionReport(backend=backend.name)
    context = BackendRunContext(report=report)

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

        checkpoint.save_shard(output, shard_name, new_state)
        while len(loaded_shards) > 2:
            oldest = next(iter(loaded_shards))
            del loaded_shards[oldest]

    new_weight_map = {
        name: shard
        for name, shard in checkpoint.weight_map.items()
        if name not in old_scale_names
    }
    checkpoint.write_index(output, new_weight_map)
    _patch_bf16_config(output)
    report.output_tensors = len(new_weight_map)
    report.finish()
    report.save(output / "quant_report.json")
    return report
