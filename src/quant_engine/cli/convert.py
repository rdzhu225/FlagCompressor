from __future__ import annotations

from quant_engine.backends.registry import build_backend
from quant_engine.core.executor import execute_plan
from quant_engine.core.planner import build_plan
from quant_engine.core.recipe import Recipe
from quant_engine.inspect.checkpoint_scanner import scan_hf_safetensors


def run(args) -> None:
    import quant_engine.formats.register  # noqa: F401
    import quant_engine.transforms.register  # noqa: F401

    recipe = Recipe.load(args.recipe)
    profile = scan_hf_safetensors(recipe.model.input_path)
    plan = build_plan(profile, recipe)

    backend_config = recipe.backend
    if args.backend:
        from dataclasses import replace

        backend_config = replace(backend_config, name=args.backend, device=args.device or backend_config.device)
    backend = build_backend(backend_config)
    if not backend.is_available():
        message = f"Backend '{backend.name}' is not available in this environment"
        if backend.fallback_policy == "error":
            raise RuntimeError(message)
        print(f"Warning: {message}; op-level CPU fallback may be used.")

    output_path = args.output or recipe.model.output_path
    if output_path is None:
        raise ValueError("Output path must be set by recipe.model.output_path or --output")

    report = execute_plan(recipe.model.input_path, output_path, plan, backend)
    print("Done.")
    print(f"Converted tensors: {report.converted}")
    print(f"Kept tensors: {report.kept}")
    print(f"Skipped source scale tensors: {report.skipped_scales}")
    print(f"Output tensors: {report.output_tensors}")
    print(f"Report: {output_path}/quant_report.json")

