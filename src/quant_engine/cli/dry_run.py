from __future__ import annotations

import json
from pathlib import Path

from quant_engine.core.planner import build_plan
from quant_engine.core.recipe import Recipe
from quant_engine.inspect.checkpoint_scanner import scan_hf_safetensors


def run(args) -> None:
    recipe = Recipe.load(args.recipe)
    profile = scan_hf_safetensors(recipe.model.input_path)
    plan = build_plan(profile, recipe)

    if args.out:
        with Path(args.out).open("w", encoding="utf-8") as f:
            json.dump(plan.to_dict(), f, indent=2)

    print("Matched module groups:")
    for group_name, count in sorted(plan.group_counts.items()):
        print(f"  {group_name}: {count}")
    print()
    print("Planned transforms:")
    for transform, count in sorted(plan.transform_counts.items()):
        print(f"  {transform}: {count}")
    print(f"  keep: {len(plan.kept_tensors)}")
    if plan.unmatched_quantized_tensors:
        print()
        print("Warnings:")
        print(f"  {len(plan.unmatched_quantized_tensors)} scaled 1-byte tensors were not matched by any rule.")

    if args.out:
        print(f"\nWrote plan: {args.out}")

