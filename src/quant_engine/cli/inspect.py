from __future__ import annotations

import json
from pathlib import Path

import yaml

from quant_engine.inspect.checkpoint_scanner import scan_hf_safetensors
from quant_engine.presets.weight_only import fp4_moe_int4_fp8_linear_bf16_recipe


def suggest_recipe(profile_path: str, input_path: str, output_path: str | None = None) -> dict:
    return fp4_moe_int4_fp8_linear_bf16_recipe(input_path, output_path, profile_path)


def run(args) -> None:
    profile = scan_hf_safetensors(args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)

    if args.suggest_recipe:
        recipe = suggest_recipe(str(out), args.model, args.output_model)
        with Path(args.suggest_recipe).open("w", encoding="utf-8") as f:
            yaml.safe_dump(recipe, f, sort_keys=False)

    print(f"Wrote profile: {out}")
    print(f"Tensors: {len(profile.tensors)}")
    print(f"Shards: {profile.metadata.get('num_shards')}")
    print("Module kinds:")
    for name, count in sorted(profile.summary()["module_kinds"].items()):
        print(f"  {name}: {count}")
    print("Source formats:")
    for name, count in sorted(profile.summary()["source_formats"].items()):
        print(f"  {name}: {count}")
    if args.suggest_recipe:
        print(f"Wrote suggested recipe: {args.suggest_recipe}")
