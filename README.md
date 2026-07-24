# FlagCompressor

FlagCompressor converts and quantizes HuggingFace `safetensors` checkpoints.
Its INT4 command supports module-level selection: selected weights are
quantized, while other low-precision weights are converted to BF16.

## Install

```bash
pip install -e .
```

## Inspect

```bash
flag-compressor inspect --input /path/to/model
```

This reports detected weight formats and selectable groups such as `moe`,
`moe.routed`, `moe.shared`, `attention`, `mlp`, and `linear`.

## Convert to BF16

```bash
flag-compressor convert \
  --input /path/to/model \
  --output /path/to/model-bf16 \
  --backend cpu
```

## Quantize selected weights

```bash
flag-compressor quantize \
  --input /path/to/model \
  --output /path/to/model-int4 \
  --select moe \
  --method mse \
  --group-size 32 \
  --backend cuda
```

`quantize` directly writes an inference-ready compressed-tensors
`pack-quantized` W4A16 checkpoint and updates `config.json`. There is no
post-quantization conversion step. Unselected source-quantized weights use
BF16 by default.

Selections can be combined:

```bash
--select moe.routed
--select attention
--select-name '.*\.self_attn\.o_proj\.weight$'
--exclude moe.shared
--exclude-name '.*\.layers\.0\..*'
```

Use `--dry-run` to check the selected tensors before writing weights.

A YAML recipe is also supported:

```yaml
version: 1
method: mse
group_size: 32
unselected:
  strategy: convert
  format: bf16
select:
  - moe
  - name: '.*\.self_attn\.o_proj\.weight$'
exclude:
  - name: '.*\.layers\.0\..*'
```

Recipe fields:

- `version` (int): recipe schema version. Currently `1`; any other value is rejected.
- `method` (str, default `mse`): quantizer method. Only `mse` is currently accepted.
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
- `unselected` (mapping): how source-quantized weights outside the selected set
  are handled. Currently only `strategy: convert` (default) with
  `format: bf16` is supported; it dequantizes low-precision weights to BF16.
  `strategy: preserve` is reserved for a future runtime-compatible mixed-format
  exporter and is rejected for now.

CLI flags and recipe fields are additive: `select` / `exclude` entries from the
recipe are merged with the corresponding CLI flags, and scalar fields
(`method`, `group_size`, `n_candidates`, `chunk_size`) take the CLI value when
provided, otherwise fall back to the recipe, otherwise to the default. At
least one selector (via CLI or recipe) is required.

```bash
flag-compressor quantize \
  --input /path/to/model \
  --output /path/to/model-int4 \
  --recipe quantize.yaml
```

## Validate

```bash
flag-compressor validate --input /path/to/model-int4
```

Validation checks the checkpoint index, stored tensors, INT4 metadata, and
runtime quantization config.

## Current scope

- Sharded HuggingFace safetensors.
- MXFP4, block FP8, and floating-point input weights.
- Weight-only symmetric groupwise MSE INT4.
- CPU and CUDA torch execution.
