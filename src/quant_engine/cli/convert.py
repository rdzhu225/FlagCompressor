from __future__ import annotations

from quant_engine.backends.registry import build_backend
from quant_engine.core.executor import execute_plan
from quant_engine.core.planner import build_plan
from quant_engine.inspect.checkpoint_scanner import scan_hf_safetensors


def run(args) -> None:
    import quant_engine.formats.register  # noqa: F401
    import quant_engine.transforms.register  # noqa: F401

    profile = scan_hf_safetensors(args.input)
    plan = build_plan(profile, target=args.target)
    if not plan.actions:
        raise RuntimeError("No supported FP8 or FP4 tensors were found for conversion")

    backend = build_backend(args.backend, args.device)
    if not backend.is_available():
        message = f"Backend '{backend.name}' is not available in this environment"
        if backend.fallback_policy == "error":
            raise RuntimeError(message)
        print(f"Warning: {message}; op-level CPU fallback may be used.")

    report = execute_plan(args.input, args.output, plan, backend)
    print("Done.")
    print(f"Target: {args.target}")
    print(f"Converted tensors: {report.converted}")
    print(f"Kept tensors: {report.kept}")
    print(f"Skipped source scale tensors: {report.skipped_scales}")
    print(f"FP4 -> BF16 tensors: {plan.transform_counts.get('fp4_to_bf16', 0)}")
    print(f"FP4 MoE -> INT4 tensors: {plan.transform_counts.get('fp4_to_int4', 0)}")
    print(f"FP8 -> BF16 tensors: {plan.transform_counts.get('fp8_to_bf16', 0)}")
    print(f"Report: {args.output}/quant_report.json")
