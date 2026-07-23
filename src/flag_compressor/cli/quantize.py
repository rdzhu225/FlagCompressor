from __future__ import annotations

import logging

from flag_compressor.backends.registry import build_backend
from flag_compressor.cli.helpers import build_quantization_policy, ensure_no_unmatched, print_plan
from flag_compressor.core.executor import execute_plan
from flag_compressor.core.planner import build_quantize_plan
from flag_compressor.inspect.checkpoint_scanner import scan_hf_safetensors

logger = logging.getLogger(__name__)


def run(args) -> None:
    import flag_compressor.formats.register  # noqa: F401
    import flag_compressor.quantizers.register  # noqa: F401

    policy = build_quantization_policy(args)
    profile = scan_hf_safetensors(args.input)
    plan = build_quantize_plan(profile, policy)
    ensure_no_unmatched(plan)
    int4_count = plan.output_format_counts.get("int4_symmetric_groupwise", 0)
    if int4_count == 0:
        raise RuntimeError("The INT4 selectors did not match any supported tensors")
    if args.dry_run:
        print_plan(plan)
        return

    backend = build_backend(args.backend, args.device)
    if not backend.is_available():
        logger.warning("backend %r is unavailable; CPU fallback may be used.", backend.name)
    report = execute_plan(args.input, args.output, plan, backend)
    logger.info("Done.")
    logger.info("INT4 tensors: %d", int4_count)
    logger.info("Converted tensors: %d", report.converted)
    logger.info("Kept tensors: %d", report.kept)
    logger.info("Manifest: %s/quant_manifest.json", args.output)
    logger.info("Report: %s/quantization_report.json", args.output)
