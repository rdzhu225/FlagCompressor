from __future__ import annotations

import warnings

from flag_compressor.backends.registry import build_backend
from flag_compressor.core.executor import execute_plan
from flag_compressor.core.planner import build_plan, make_regex_selector
from flag_compressor.inspect.checkpoint_scanner import scan_hf_safetensors


def _build_int4_selector(args):
    include = getattr(args, "int4_include", None)
    exclude = getattr(args, "int4_exclude", None)
    if not include and not exclude:
        return None
    return make_regex_selector(include=include, exclude=exclude)


def run(args) -> None:
    import flag_compressor.formats.register  # noqa: F401
    import flag_compressor.transforms.register  # noqa: F401

    target = args.target
    if target == "moe-int4":
        warnings.warn(
            "--target moe-int4 is a compatibility alias for --target int4 with an MoE-expert "
            "selector; prefer --target int4 --int4-include '<regex>' for explicit control.",
            DeprecationWarning,
            stacklevel=2,
        )

    if target == "bf16" and (getattr(args, "int4_include", None) or getattr(args, "int4_exclude", None)):
        raise SystemExit("--int4-include/--int4-exclude are only valid with --target int4")

    profile = scan_hf_safetensors(args.input)
    int4_selector = _build_int4_selector(args) if target == "int4" else None
    plan = build_plan(profile, target=target, int4_selector=int4_selector)
    if not plan.actions:
        raise RuntimeError("No supported FP8/FP4 tensors or user-selected INT4 targets were found")

    backend = build_backend(args.backend, args.device)
    if not backend.is_available():
        message = f"Backend '{backend.name}' is not available in this environment"
        if backend.fallback_policy == "error":
            raise RuntimeError(message)
        print(f"Warning: {message}; op-level CPU fallback may be used.")

    report = execute_plan(args.input, args.output, plan, backend)
    print("Done.")
    print(f"Target: {target}")
    print(f"Converted tensors: {report.converted}")
    print(f"Kept tensors: {report.kept}")
    print(f"Skipped source scale tensors: {report.skipped_scales}")
    print(f"FP4 -> BF16 tensors: {plan.transform_counts.get('fp4_to_bf16', 0)}")
    print(f"FP8 -> BF16 tensors: {plan.transform_counts.get('fp8_to_bf16', 0)}")
    print(f"FP4 -> INT4 tensors: {plan.transform_counts.get('fp4_to_int4', 0)}")
    print(f"FP8 -> INT4 tensors: {plan.transform_counts.get('fp8_to_int4', 0)}")
    print(f"BF16 -> INT4 tensors: {plan.transform_counts.get('bf16_to_int4', 0)}")
    print(f"Report: {args.output}/quant_report.json")
