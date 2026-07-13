# Quant Engine

`quant_engine` is a lightweight FlagOS quantization deployment tool for turning
FP8/FP4 HuggingFace safetensors checkpoints into deployable mixed-precision
artifacts.

The default path is checkpoint-only and weight-only:

```text
inspect checkpoint -> classify common linear weights -> plan transforms -> convert weights -> write manifest/report
```

It does not import `transformers` or require calibration data unless a PTQ/QAT
workflow asks for model execution.

## What It Supports Now

- HF `safetensors` checkpoint scanning and shard-safe conversion.
- Checkpoint-only classification of common linear weights:
  - `attn_linear`
  - `mlp_linear`
  - `shared_moe_mlp_linear`
  - `moe_mlp_linear`
  - `embedding`
  - `lm_head`
- Recipe-based tensor selection.
- Dry-run plans with group, transform, source-format, and module-kind summaries.
- FP8 block + E8M0 scale to BF16.
- MXFP4 E2M1 + E8M0 scale to BF16.
- MoE expert FP4 to packed signed INT4 + BF16 per-group scale.
- MSE INT4 weight-only quantization.
- `quant_manifest.json` and `quant_report.json` outputs.

## One-Command Weight Conversion

This command mirrors the standalone `fp4_int4.py` flow:

```bash
quant-engine convert-weight-only \
  --input /path/to/fp8_fp4_model \
  --output /path/to/output_model \
  --backend cuda
```

Built-in routing:

```text
moe_mlp_linear              FP4 -> packed INT4 + BF16 scale
attn_linear                 FP8 -> BF16
mlp_linear                  FP8 -> BF16
shared_moe_mlp_linear       FP8 -> BF16
BF16/FP32 tensors           kept as-is
source scale tensors        removed after conversion
```

## Recipe Flow

Use this when you want to inspect, edit, or review the conversion plan before
running it:

```bash
quant-engine inspect \
  --model /path/to/fp8_fp4_model \
  --out profile.json \
  --suggest-recipe recipe.yaml \
  --output-model /path/to/output_model

quant-engine dry-run --recipe recipe.yaml --out plan.json
quant-engine convert --recipe recipe.yaml --backend cuda
```

Users adapt behavior in YAML first:

```yaml
module_groups:
  moe_mlp_linear:
    selector:
      has_scale: true
      module_kind: moe_mlp_linear

rules:
  - name: moe_fp4_to_int4
    group: moe_mlp_linear
    when:
      source_format: fp4_e2m1_e8m0
    transform: fp4_to_int4
    quantizer:
      name: mse
      group_size: 32
      n_candidates: 200
```

## Adaptation Boundary

Keep the lightweight path model-family agnostic:

- Do not add `qwen.py`, `deepseek.py`, `llama.py`, or other model-family
  adapters for checkpoint scanning.
- Add only generic tensor-name patterns to
  `quant_engine.inspect.tensor_classifier` when a new common linear naming
  convention appears.
- Prefer recipe selectors and built-in presets for policy changes.
- Import `transformers` only for calibration, PTQ, QAT, or hook-based workflows
  that need model execution.

## Artifact Boundary

Conversion output includes:

```text
quant_manifest.json
quant_report.json
model.safetensors.index.json
*.safetensors
```

`quant_manifest.json` is the stable artifact ABI for inference integration.
Inference repositories should only need a thin bridge that parses this manifest,
loads weights/scales, and calls their current kernel hooks.

## Optional Calibration

Calibration commands are available for PTQ/QAT work that needs activations or
Hessian-like statistics:

```bash
quant-engine build-calib --model-path /path/to/model --output calib.jsonl
quant-engine calibrate --recipe recipe.yaml --dataset-jsonl calib.jsonl --output calib_stats
```

These commands require the optional `calibration` dependencies and may import
`transformers`. They are not part of the default FP4/FP8 weight-only conversion
path.

## Current Limitations

- GPTQ/AWQ dispatch points are reserved, but the built-in one-command path uses
  MSE INT4 today.
- Real NPU/MLU/XPU kernels are not included yet; CPU/CUDA/generic torch-device
  execution and fallback are the current path.
