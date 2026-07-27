from __future__ import annotations

import json
import logging
from pathlib import Path

from flagos_compressor.backends.registry import build_backend
from flagos_compressor.cli.helpers import build_quantization_policy, ensure_no_unmatched, print_plan
from flagos_compressor.core.executor import execute_plan
from flagos_compressor.core.moe_layout import select_moe_layout
from flagos_compressor.core.planner import build_quantize_plan
from flagos_compressor.inspect.checkpoint_scanner import scan_hf_safetensors

logger = logging.getLogger(__name__)


def _selected_fused_expert(profile, policy) -> bool:
    """Whether the policy selects any fused 3D routed-expert bank."""
    return any(
        tensor.module_kind == "moe_routed_fused" and policy.selects(tensor)
        for tensor in profile.tensors.values()
    )


def _load_moe_layout(model_path: str, profile, policy):
    """Select a fused-expert layout only when fused experts are quantized."""
    if not _selected_fused_expert(profile, policy):
        return None
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "Quantizing fused routed experts requires config.json to select a "
            f"layout adapter, but {config_path} is missing"
        )
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return select_moe_layout(config)


def run(args) -> None:
    import flagos_compressor.formats.register  # noqa: F401
    import flagos_compressor.quantizers.register  # noqa: F401

    policy = build_quantization_policy(args)
    profile = scan_hf_safetensors(args.input)
    moe_layout = _load_moe_layout(args.input, profile, policy)
    plan = build_quantize_plan(profile, policy, moe_layout)
    ensure_no_unmatched(plan)
    int4_count = plan.output_format_counts.get(
        "compressed_tensors_int4_groupwise",
        0,
    )
    fused_moe_count = plan.output_format_counts.get(
        "compressed_tensors_int4_moe_fused",
        0,
    )
    if int4_count == 0 and fused_moe_count == 0:
        raise RuntimeError("The INT4 selectors did not match any supported tensors")
    if args.dry_run:
        print_plan(plan)
        return

    backend = build_backend(args.backend, args.device)
    if not backend.is_available():
        logger.warning("backend %r is unavailable; CPU fallback may be used.", backend.name)
    report = execute_plan(args.input, args.output, plan, backend)
    print("Done.")
    print(f"INT4 tensors: {int4_count}")
    print(f"INT4 fused MoE banks: {fused_moe_count}")
    print(f"Converted tensors: {report.converted}")
    print(f"Kept tensors: {report.kept}")
    print(f"Manifest: {args.output}/quantization_manifest.json")
    print(f"Report: {args.output}/quantization_report.json")
