# Quant Engine

`quant_engine` is a recipe-driven, backend-aware quantization and dtype conversion framework.

The project is designed around one boundary:

```text
Quantization algorithms are device-independent.
Execution backends decide where each low-level op runs and when to fallback.
```

This keeps model structure handling, low precision codecs, quantization algorithms, device execution, and inference artifacts separated.

## MVP Scope

Implemented in this skeleton:

- HF `safetensors` checkpoint inspection.
- Recipe-based module selection.
- Tensor-level dry-run planning.
- FP8 block + E8M0 scale to BF16.
- MXFP4 E2M1 + E8M0 scale to BF16.
- MoE FP4 to packed signed INT4 + BF16 per-group scale.
- MSE INT4 quantizer.
- CPU, CUDA, and generic torch-device backends.
- Placeholder backend entries for Ascend, Cambricon, Kunlun, MUSA, and DCU style devices.
- Calibration JSONL builder and hook-based activation stats collector.
- `quant_manifest.json` and `quant_report.json` artifact outputs.

## Commands

```bash
quant-engine inspect \
  --model /path/to/model \
  --out profile.json \
  --suggest-recipe recipe.yaml
```

```bash
quant-engine dry-run --recipe recipe.yaml --out plan.json
```

```bash
quant-engine convert --recipe recipe.yaml
```

Backend override:

```bash
quant-engine convert \
  --recipe recipe.yaml \
  --backend ascend \
  --device npu:0
```

## Recipe Boundary

Users describe module groups and transforms in YAML instead of editing Python:

```yaml
module_groups:
  moe_experts:
    include:
      - ".*\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj|w1|w2|w3)\\.weight$"
    exclude:
      - ".*shared_experts.*"

rules:
  - name: moe_fp4_to_int4
    group: moe_experts
    transform: fp4_to_int4
    quantizer:
      name: mse
      group_size: 32
```

The Python code does not hardcode a specific model family as the only supported structure.

## Backend Boundary

Quantization code should not call `.cuda()`, `.npu()`, or `.mlu()` directly.

All compute goes through:

```python
backend.run("fp4_dequant", weight, scale)
backend.run("mse_int4_quant", dequant_weight)
```

Low-level ops are registered by name. A backend can provide native implementations later:

```text
fp4_dequant.cpu
fp4_dequant.torch
fp4_dequant.ascend      # future
fp4_dequant.cambricon   # future
```

If a backend-specific op is absent, fallback behavior is controlled by recipe/backend policy.

## Inference Boundary

The conversion output includes:

```text
quant_manifest.json
quant_report.json
model.safetensors.index.json
*.safetensors
```

`quant_manifest.json` is the stable artifact ABI for inference integration. If vLLM-FL or SGLang-FL integration must live in their repositories, put only a small `quant_bridge` there:

```text
quant_bridge/
  artifact manifest parser
  weight/scale loader
  reference dequant
  thin ports for current vLLM-FL/SGLang-FL hooks
```

The bridge should consume this artifact ABI rather than re-implementing quantization format rules across unstable inference code.

## Current Limitations

- GPTQ and AWQ are declared as future quantizer slots but are not implemented yet.
- Real NPU/MLU/XPU kernels are not included; generic torch-device execution and CPU fallback are the initial path.
- `datasets` is optional and only needed for `build-calib`.

