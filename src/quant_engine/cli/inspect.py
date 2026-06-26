from __future__ import annotations

import json
from pathlib import Path

import yaml

from quant_engine.inspect.checkpoint_scanner import scan_hf_safetensors


def suggest_recipe(profile_path: str, input_path: str, output_path: str | None = None) -> dict:
    output = output_path or str(Path(input_path).with_name(Path(input_path).name + "-quantized"))
    return {
        "version": 1,
        "model": {
            "input_path": input_path,
            "output_path": output,
            "format": "hf_safetensors",
        },
        "backend": {
            "name": "cpu",
            "fallback_policy": "warn",
            "op_placement": {
                "fp4_dequant": "device",
                "fp8_dequant": "device",
                "mse_int4_quant": "device",
                "int4_pack": "auto",
                "safetensors_io": "cpu",
            },
        },
        "discover": {
            "profile": profile_path,
            "auto_pair_scales": True,
        },
        "module_groups": {
            "moe_experts": {
                "include": [
                    r".*\.experts\.\d+\.(gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$"
                ],
                "exclude": [r".*shared_experts.*"],
            },
            "moe_shared_experts": {
                "include": [
                    r".*shared_experts.*\.(gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$"
                ]
            },
            "scaled_dense": {
                "selector": {"has_scale": True},
                "exclude_groups": ["moe_experts", "moe_shared_experts"],
            },
        },
        "rules": [
            {
                "name": "moe_fp4_to_int4",
                "group": "moe_experts",
                "when": {"source_format": "fp4_e2m1_e8m0"},
                "transform": "fp4_to_int4",
                "quantizer": {"name": "mse", "group_size": 32, "n_candidates": 200},
                "output": {"scale_suffix": ".scale"},
            },
            {
                "name": "shared_fp4_to_bf16",
                "group": "moe_shared_experts",
                "when": {"source_format": "fp4_e2m1_e8m0"},
                "transform": "fp4_to_bf16",
            },
            {
                "name": "scaled_fp8_to_bf16",
                "group": "scaled_dense",
                "when": {"has_scale": True},
                "transform": "fp8_to_bf16",
                "params": {"block_size": 128},
            },
        ],
        "calibration": {
            "enabled": False,
            "dataset_jsonl": None,
            "output_dir": None,
            "max_length": 2048,
            "batch_size": 1,
            "collect": {"groups": ["moe_experts"], "stats": ["hessian", "act_scales"]},
        },
    }


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
    if args.suggest_recipe:
        print(f"Wrote suggested recipe: {args.suggest_recipe}")

