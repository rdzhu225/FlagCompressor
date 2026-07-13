from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from quant_engine.backends.registry import build_backend
from quant_engine.core.executor import execute_plan
from quant_engine.core.planner import build_plan
from quant_engine.core.recipe import Recipe
from quant_engine.inspect.checkpoint_scanner import scan_hf_safetensors
from quant_engine.presets.weight_only import fp4_moe_int4_fp8_linear_bf16_recipe


def run(args) -> None:
    import quant_engine.formats.register  # noqa: F401
    import quant_engine.transforms.register  # noqa: F401

    recipe_data = fp4_moe_int4_fp8_linear_bf16_recipe(
        args.input,
        args.output,
        profile_path=None,
    )
    recipe = Recipe.from_dict(recipe_data)

    profile = scan_hf_safetensors(recipe.model.input_path)
    plan = build_plan(profile, recipe)
    if not plan.actions:
        raise RuntimeError("No convertible scaled linear weights matched the built-in weight-only preset")

    backend_config = replace(recipe.backend, name=args.backend, device=args.device)
    backend = build_backend(backend_config)
    if not backend.is_available():
        message = f"Backend '{backend.name}' is not available in this environment"
        if backend.fallback_policy == "error":
            raise RuntimeError(message)
        print(f"Warning: {message}; op-level CPU fallback may be used.")

    output_path = Path(args.output)
    report = execute_plan(recipe.model.input_path, output_path, plan, backend)
    print("Done.")
    print(f"FP4 expert MLP -> INT4 tensors: {plan.transform_counts.get('fp4_to_int4', 0)}")
    print(f"FP8 linear -> BF16 tensors: {plan.transform_counts.get('fp8_to_bf16', 0)}")
    print(f"Kept tensors: {report.kept}")
    print(f"Skipped source scale tensors: {report.skipped_scales}")
    print(f"Output tensors: {report.output_tensors}")
    print(f"Report: {output_path}/quant_report.json")

