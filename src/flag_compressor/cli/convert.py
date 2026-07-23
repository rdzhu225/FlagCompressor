from __future__ import annotations

import logging

from flag_compressor.backends.registry import build_backend
from flag_compressor.core.executor import execute_plan
from flag_compressor.cli.helpers import ensure_no_unmatched, print_plan
from flag_compressor.core.planner import build_convert_plan
from flag_compressor.inspect.checkpoint_scanner import scan_hf_safetensors

logger = logging.getLogger(__name__)


def run(args) -> None:
    import flag_compressor.formats.register  # noqa: F401

    profile = scan_hf_safetensors(args.input)
    plan = build_convert_plan(profile)
    ensure_no_unmatched(plan)
    if not plan.actions:
        raise RuntimeError("No supported FP8/FP4 tensors were found")
    if getattr(args, "dry_run", False):
        print_plan(plan)
        return

    backend = build_backend(args.backend, args.device)
    if not backend.is_available():
        message = f"backend {backend.name!r} is not available in this environment"
        if backend.fallback_policy == "error":
            raise RuntimeError(message)
        logger.warning("%s; op-level CPU fallback may be used.", message)

    report = execute_plan(args.input, args.output, plan, backend)
    print("Done.")
    print(f"Converted tensors: {report.converted}")
    print(f"Kept tensors: {report.kept}")
    print(f"Skipped source scale tensors: {report.skipped_scales}")
    print(f"BF16 output tensors: {plan.output_format_counts.get('bf16', 0)}")
    print(f"Report: {args.output}/conversion_report.json")
