# FlagCompressor

FlagCompressor is a lightweight checkpoint conversion tool for FlagOS-style
mixed FP8/FP4 HuggingFace `safetensors` models.

The implementation package is `quant_engine`, and the installed CLI is
`quant-engine`.

## Supported Scope

The framework intentionally supports only the current deployment path:

```text
scan HF safetensors -> classify linear weights -> build fixed plan -> convert -> write manifest/report
```

Supported conversions:

- FP8 block weight + E8M0 scale -> BF16.
- FP4 E2M1 weight + E8M0 scale -> BF16.
- Optional MoE expert FP4 E2M1 weight -> packed signed INT4 + BF16 group scale.
- BF16/FP16/FP32 and unsupported tensors are copied through unchanged.
- Source scale tensors consumed by supported conversions are removed from output shards.

The lightweight path does not include YAML recipes, calibration, PTQ/QAT hooks,
or model-family-specific adapters.

## Usage

```bash
quant-engine convert   --input /path/to/fp8_fp4_model   --output /path/to/output_model   --backend cpu
```

`--backend cuda` can be used when CUDA is available. The CPU implementation is
kept as the reference path.

Built-in routing for `--target bf16`:

```text
scaled FP4 tensors          FP4 -> BF16
attn_linear                 FP8 -> BF16
mlp_linear                  FP8 -> BF16
shared_moe_mlp_linear       FP8 -> BF16
other tensors               kept as-is
source scale tensors        removed after conversion
```

Built-in routing for `--target moe-int4`:

```text
moe_mlp_linear              FP4 -> packed INT4 + BF16 scale
other scaled FP4 tensors    FP4 -> BF16
attn_linear                 FP8 -> BF16
mlp_linear                  FP8 -> BF16
shared_moe_mlp_linear       FP8 -> BF16
other tensors               kept as-is
source scale tensors        removed after conversion
```

## Outputs

Conversion output includes:

```text
quant_manifest.json
quant_report.json
model.safetensors.index.json
*.safetensors
```

`quant_manifest.json` is the artifact ABI for inference integration. Runtime
repositories should parse the manifest, load converted tensors/scales, and call
their own kernel hooks.

## Adaptation Boundary

Keep this repository model-family agnostic:

- Add generic tensor-name patterns to `quant_engine.inspect.tensor_classifier`
  when a new common linear naming convention appears.
- Keep conversion policy in `quant_engine.core.planner` unless new supported
  routes are explicitly added.
- Keep runtime/inference integration outside this package.

## Current Limitations

- Only HF `safetensors` checkpoints with `model.safetensors.index.json` are supported.
- Only CPU/CUDA torch execution paths are exposed.
- Runtime loading and inference execution of quantized artifacts are not included.
