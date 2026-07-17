# FlagCompressor

FlagCompressor is a lightweight, model-family-agnostic checkpoint conversion
tool for FlagOS-style mixed FP8/FP4 HuggingFace `safetensors` models.

The installed CLI is `flag-compressor`. The Python package is `flag_compressor`.

## Supported Scope

The framework intentionally supports only the current deployment path:

```text
scan HF safetensors -> classify linear weights -> build plan -> convert -> write manifest/report
```

Supported conversions:

- FP8 block weight + E8M0 scale -> BF16.
- FP4 E2M1 weight + E8M0 scale -> BF16.
- Any weight (FP8, FP4, or BF16/FP16/FP32) selected by the user -> symmetric
  groupwise INT4 + BF16 group scale. When the source is FP8/FP4, the tensor is
  first dequantized to BF16 and then quantized to INT4 (the BF16 pass is
  always the first stop for going to a lower bit width).
- BF16/FP16/FP32 tensors that are not selected are copied through unchanged.
- Source scale tensors consumed by supported conversions are removed from
  output shards.

The lightweight path does not include YAML recipes, calibration, PTQ/QAT
hooks, or model-family-specific adapters. Selecting which linear layers get
INT4 is entirely up to the user via name-based regex selectors.

## Usage

Plain FP8/FP4 -> BF16 (works for any FP8/FP4 checkpoint, not just DeepSeek):

```bash
flag-compressor convert \
  --input /path/to/fp8_fp4_model \
  --output /path/to/output_bf16_model \
  --target bf16 \
  --backend cpu
```

User-selected INT4 quantization. The `--int4-include` regex is matched
against tensor names using `re.search`; repeat the flag to add more patterns.
`--int4-exclude` subtracts from the selection.

```bash
# DeepSeek V4: only routed MoE expert linears to INT4
flag-compressor convert \
  --input /path/to/DSv4 --output /path/to/DSv4-int4 \
  --target int4 \
  --int4-include '.*\.experts\.\d+\.(gate|up|down)_proj\.weight$'

# Qwen dense: full attention + MLP to INT4
flag-compressor convert \
  --input /path/to/Qwen --output /path/to/Qwen-int4 \
  --target int4 \
  --int4-include '.*\.(q|k|v|o|gate|up|down)_proj\.weight$'
```

`--target moe-int4` is retained as a deprecated alias for
`--target int4 --int4-include '.*\.experts\.\d+\.(gate|up|down|w1|w2|w3)_proj\.weight$'`.

`--backend cuda` can be used when CUDA is available. The CPU implementation
is kept as the reference path.

## Conversion Rules

For a given plan:

- Any FP8 or FP4 scaled tensor is first mapped to BF16.
- If `--target int4` is set and the tensor's name matches the selector, the
  BF16 output is further quantized to symmetric groupwise INT4 (group size 32
  by default) with BF16 per-group scales. The transform picked is
  `fp4_to_int4`, `fp8_to_int4`, or `bf16_to_int4` depending on the source
  dtype, but they all end in the same INT4 layout.
- Every other tensor is copied through unchanged.
- Source E8M0/block scale tensors consumed by conversions are stripped from
  the output shards; new BF16 group scales for INT4 tensors are appended.
- The output `config.json` has its source `quantization_config`,
  `compression_config`, `quant_method`, and `expert_dtype` stripped, and
  `torch_dtype` set to `bfloat16`. The INT4 layout is described entirely by
  `quant_manifest.json`.

## Outputs

Conversion output includes:

```text
quant_manifest.json
quant_report.json
model.safetensors.index.json
*.safetensors
```

`quant_manifest.json` is the artifact ABI for inference integration. Runtime
repositories should parse the manifest, load converted tensors/scales, and
call their own kernel hooks.

## Adaptation Boundary

Keep this repository model-family agnostic:

- Add generic tensor-name patterns to `flag_compressor.inspect.tensor_classifier`
  when a new common linear naming convention appears; the classifier's output
  is metadata only and does not drive conversion policy.
- Which tensors go to INT4 is a user choice expressed via `--int4-include` /
  `--int4-exclude` regexes. Do not add hard-coded model-family branches to
  the planner.
- Model-family-specific artifact fixups (for example the DeepSeek V4 sparse
  indexer LayerNorm defaults) live behind an explicit `model_type` check in
  `flag_compressor.core.executor` and only run when the input actually is that
  family.
- Keep runtime/inference integration outside this package.

## Current Limitations

- Only HF `safetensors` checkpoints with `model.safetensors.index.json` are
  supported.
- Only CPU/CUDA torch execution paths are exposed.
- Runtime loading and inference execution of quantized artifacts are not
  included.
