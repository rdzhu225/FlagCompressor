from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors.torch import save_file
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
    int4_transforms = {"fp4_to_int4", "fp8_to_int4", "bf16_to_int4"}
    bf16_transforms = {"fp4_to_bf16", "fp8_to_bf16"}
    tensors = {}
    for action in plan.actions:
        if action.transform in int4_transforms:
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
        elif action.transform in bf16_transforms:
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



_BF16_NORMALIZED_TRANSFORMS = {
    "fp4_to_bf16",
    "fp8_to_bf16",
    "fp4_to_int4",
    "fp8_to_int4",
    "bf16_to_int4",
}


def _needs_bf16_config_normalization(plan: ExecutionPlan) -> bool:
    """Whether the output config should be rewritten to look like a BF16 model.

    Both plain BF16 dequant plans and INT4 quantization plans emit artifacts
    whose non-INT4 residue is BF16 and whose INT4 layout is described by the
    quant manifest, so in both cases the source ``quantization_config`` must
    be stripped and ``torch_dtype`` normalized to ``bfloat16``.
    """
    return bool(plan.actions) and all(
        action.transform in _BF16_NORMALIZED_TRANSFORMS for action in plan.actions
    )


_BF16_STRIP_CONFIG_KEYS = (
    "quantization_config",
    "compression_config",
    "quant_method",
    "expert_dtype",
)


def _patch_bf16_config(output_path: Path) -> dict | None:
    """Rewrite the output ``config.json`` so it describes a plain BF16 model.

    After a BF16 conversion the artifact no longer carries FP8/FP4 weights, so
    the source quantization metadata must be stripped and ``torch_dtype`` set
    to ``bfloat16``. Returns the loaded config (for downstream family-specific
    patches) or ``None`` if there is no config to update.
    """
    config_path = output_path / "config.json"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config["torch_dtype"] = "bfloat16"
    for key in _BF16_STRIP_CONFIG_KEYS:
        config.pop(key, None)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return config


def _patch_deepseek_v4_indexer_defaults(
    output_path: Path,
    config: dict,
    weight_map: dict[str, str],
) -> None:
    """Append default ``attn.indexer.k_norm`` tensors for DeepSeek V4 BF16 artifacts.

    Some DeepSeek V4 implementations still need the ``attn.indexer.k_norm``
    LayerNorm tensors for sparse indexer layers even though the source
    checkpoint omits them. We synthesize the defaults and append them to the
    last shard, updating the HF index accordingly.
    """
    if config.get("model_type") != "deepseek_v4":
        return

    index_path = output_path / "model.safetensors.index.json"
    if not index_path.exists():
        return

    indexer_layers = sorted(
        {
            int(match.group(1))
            for name in weight_map
            for match in [re.match(r"layers\.(\d+)\.attn\.indexer\.wq_b\.weight$", name)]
            if match
        }
    )
    missing: dict[str, torch.Tensor] = {}
    head_dim = int(config.get("index_head_dim", 128))
    for layer_idx in indexer_layers:
        weight_name = f"layers.{layer_idx}.attn.indexer.k_norm.weight"
        bias_name = f"layers.{layer_idx}.attn.indexer.k_norm.bias"
        if weight_name not in weight_map:
            missing[weight_name] = torch.ones(head_dim, dtype=torch.bfloat16)
        if bias_name not in weight_map:
            missing[bias_name] = torch.zeros(head_dim, dtype=torch.bfloat16)
    if not missing:
        return

    shard_name = sorted(set(weight_map.values()))[-1]
    shard_path = output_path / shard_name
    state = HfSafetensorsCheckpoint(output_path).load_shard(shard_name)
    state.update(missing)
    save_file(state, str(shard_path))
    for name in missing:
        weight_map[name] = shard_name

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    index["weight_map"] = weight_map
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

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
    if _needs_bf16_config_normalization(plan):
        config = _patch_bf16_config(output)
        if config is not None:
            _patch_deepseek_v4_indexer_defaults(output, config, new_weight_map)
    _write_quant_manifest(output, plan, generated_scale_map)
    report.output_tensors = len(new_weight_map)
    report.finish()
    report.save(output / "quant_report.json")
    return report
