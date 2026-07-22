# FlagCompressor

FlagCompressor converts HuggingFace `safetensors` checkpoints between weight
formats. It has separate commands for deterministic low-to-high precision
conversion and quality-affecting quantization, while sharing the same scanner,
planner, backend and checkpoint writer.

## Install

```bash
pip install -e .
```

## Inspect a model

```bash
flag-compressor inspect --input /path/to/model
```

The output reports input storage formats and selectable groups such as `moe`,
`moe.routed`, `moe.shared`, `attention`, `mlp`, and `linear`.

## Convert FP4/FP8 to BF16

```bash
flag-compressor convert \
  --input /path/to/fp8_or_fp4_model \
  --output /path/to/bf16_model \
  --to bf16 \
  --backend cpu
```

FP4 and FP8 weights are dequantized to BF16; existing BF16/FP16/FP32 tensors
are copied. The legacy invocation without the `convert` word remains accepted.

## Quantize selected weights to INT4

Quantize all routed and shared MoE expert linears, with all other low-precision
weights converted to BF16:

```bash
flag-compressor quantize \
  --input /path/to/model \
  --output /path/to/model-int4 \
  --select moe \
  --format int4 \
  --method mse \
  --group-size 32 \
  --other-weights bf16 \
  --backend cuda
```

Other useful selections:

```bash
--select moe.routed
--select moe.shared
--select attention
--select mlp
--select linear
--select-name '.*\.self_attn\.o_proj\.weight$'
--exclude moe.shared
--exclude-name '.*\.layers\.0\..*'
```

Use `--dry-run` to inspect the plan without writing an output checkpoint.

### Simple quantize recipe

```yaml
version: 1
format: int4
method: mse
group_size: 32

select:
  - moe
  - name: '.*\.self_attn\.o_proj\.weight$'

exclude:
  - name: '.*\.layers\.0\..*'

other_weights: bf16
```

Recipe fields:

- `version` (int): recipe schema version. Currently `1`; any other value is rejected.
- `format` (str): output weight format. Only `int4` is currently accepted.
- `method` (str): quantizer method. Only `mse` is currently accepted.
- `group_size` (int, default `32`): group size along the input-feature axis for
  INT4 scales. Same as CLI `--group-size`.
- `n_candidates` (int, default `200`): number of candidate scales searched per
  group by the MSE quantizer. Same as CLI `--n-candidates`.
- `chunk_size` (int, default `4096`): output-feature chunk size used to bound
  peak memory during search. Same as CLI `--chunk-size`.
- `select` (list): tensors to quantize. Each entry is either a built-in group
  name (`moe`, `moe.routed`, `moe.shared`, `attention`, `mlp`, `linear`) or a
  mapping `{name: 'REGEX'}`. Mirrors `--select` / `--select-name`.
- `exclude` (list): tensors to skip, same shape as `select`. Applied on top of
  the `select` set. Mirrors `--exclude` / `--exclude-name`.
- `other_weights` (str, default `bf16`): how to handle tensors not selected for
  INT4. `bf16` dequantizes low-precision weights to BF16 and passes plain FP
  weights through; `keep` leaves them in their original storage format.

CLI flags and recipe fields are additive: `select` / `exclude` entries from the
recipe are merged with the corresponding CLI flags, and scalar fields
(`method`, `group_size`, `n_candidates`, `chunk_size`, `other_weights`,
`format`) take the CLI value when provided, otherwise fall back to the recipe,
otherwise to the default. At least one selector (via CLI or recipe) is
required.

Run it with:

```bash
flag-compressor quantize \
  --input /path/to/model \
  --output /path/to/model-int4 \
  --recipe quantize.yaml
```

The framework detects whether a selected tensor starts as FP4, FP8, or a
floating-point weight and builds the appropriate dequantize/quantize/pack path.
Currently the supported quantizer is weight-only symmetric groupwise MSE INT4.

## Input and output weight formats

Conversion plans do not register combinations such as `fp4_to_int4`. Every
action names an input and output weight format:

```text
input_format -- to_canonical() --> canonical_weight -- from_canonical() --> output_format
```

Built-in input formats are `fp4_e2m1_e8m0`, `fp8_block_e8m0`, `bf16`, `fp16`,
and `fp32`.
Built-in output formats are `bf16` and `int4_symmetric_groupwise`; MSE is a
quantizer parameter of the INT4 format. A format may implement either or both
directions. Adding INT8 or an FP4 target requires one output format
implementation, not one implementation for every input/output pair.

## INT4 artifact ABI

INT4 weights are stored as signed two's-complement nibbles in `uint8`, with the
even K element in the low nibble and odd K element in the high nibble. BF16
scales use row-group layout `[out_features, in_features / group_size]`.

Quantized output includes `quant_manifest.json`, which records logical and
storage shapes, scale names, group size and packing layout. Inference runtimes
must understand this manifest or adapt it to their native INT4 MoE/Linear
loader; a generic Transformers loader cannot infer INT4 semantics from a
`uint8` tensor alone.

## Validate output

```bash
flag-compressor validate --input /path/to/output
```

Validation checks shard/index consistency, total size, manifest references,
INT4 storage shapes, and BF16 scale shapes/dtypes.

## Current scope

- Sharded HuggingFace safetensors with `model.safetensors.index.json`.
- MXFP4 E2M1 + E8M0 and block FP8 + E8M0 source layouts.
- CPU and CUDA torch execution.
- FP4/FP8 to BF16 conversion.
- FP4/FP8/BF16/FP16/FP32 weight-only MSE INT4 quantization.
