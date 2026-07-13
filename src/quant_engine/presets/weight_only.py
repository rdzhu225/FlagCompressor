from __future__ import annotations

from pathlib import Path


def fp4_moe_int4_fp8_linear_bf16_recipe(
    input_path: str,
    output_path: str | None = None,
    profile_path: str | None = None,
) -> dict:
    output = output_path or str(Path(input_path).with_name(Path(input_path).name + "-quantized"))
    discover = {"auto_pair_scales": True}
    if profile_path:
        discover["profile"] = profile_path

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
        "discover": discover,
        "module_groups": {
            "moe_mlp_linear": {
                "selector": {
                    "has_scale": True,
                    "module_kind": "moe_mlp_linear",
                },
            },
            "scaled_non_moe_linear": {
                "selector": {
                    "has_scale": True,
                    "module_kind": [
                        "attn_linear",
                        "mlp_linear",
                        "shared_moe_mlp_linear",
                    ],
                },
            },
        },
        "rules": [
            {
                "name": "moe_fp4_to_int4",
                "group": "moe_mlp_linear",
                "when": {"source_format": "fp4_e2m1_e8m0"},
                "transform": "fp4_to_int4",
                "quantizer": {
                    "name": "mse",
                    "group_size": 32,
                    "n_candidates": 200,
                },
                "output": {"scale_suffix": ".scale"},
            },
            {
                "name": "linear_fp8_to_bf16",
                "group": "scaled_non_moe_linear",
                "when": {"source_format": "fp8_block_e8m0"},
                "transform": "fp8_to_bf16",
                "params": {"block_size": 128},
            },
        ],
    }
