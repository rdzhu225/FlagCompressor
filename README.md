# FlagOS-Compressor

FlagOS-Compressor converts and quantizes HuggingFace `safetensors` checkpoints.
Its INT4/INT8 command supports module-level selection: selected weights are
quantized, while other low-precision weights are converted to BF16.

## Install

```bash
pip install -e .
```

## Inspect

```bash
flagos-compressor inspect --input /path/to/model
```

This reports detected weight formats and selectable groups such as `moe`,
`moe.routed`, `moe.shared`, `attention`, `mlp`, and `linear`.

## Convert to BF16

```bash
flagos-compressor convert \
  --input /path/to/model \
  --output /path/to/model-bf16 \
  --backend cpu
```

## Quantize selected weights

INT4 (the backwards-compatible default):

```bash
flagos-compressor quantize \
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

INT8 weight-only W8A16:

```bash
flagos-compressor quantize \
  --input /path/to/model \
  --output /path/to/model-int8 \
  --select attention \
  --bits 8 \
  --backend cuda
```

INT8 uses symmetric groupwise MSE quantization, BF16 scales, and
compressed-tensors `pack-quantized` int32 storage. Its default group size is
128; override it with `--group-size` when the model shape or runtime requires
a different value. INT4 keeps its existing default group size of 32.

INT8 also supports one scale per output channel:

```bash
flagos-compressor quantize \
  --input /path/to/model \
  --output /path/to/model-int8-channel \
  --select attention \
  --bits 8 \
  --strategy channel \
  --backend cuda
```

Do not pass `--group-size` with `--strategy channel`. Channelwise export stores
scales as `[out_features, 1]` and declares `strategy: channel` in the
compressed-tensors config. vLLM's WNA16 routed-MoE path currently requires
group quantization, so channel strategy is supported for ordinary Linear and
shared-expert Linear weights, but rejected for routed experts.

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
bits: 8
strategy: group
method: mse
group_size: 128
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
- `bits` (int, default `4`): weight bit width, either `4` or `8`. Same as CLI
  `--bits`.
- `strategy` (str, default `group`): `group` or `channel`. Channel strategy is
  currently available for INT8 Linear weights and must not specify
  `group_size`.
- `method` (str, default `mse`): quantizer method. Only `mse` is currently accepted.
- `group_size` (int, default `32` for INT4 and `128` for INT8): group size
  along the input-feature axis for weight scales. Same as CLI `--group-size`.
- `n_candidates` (int, default `200`): number of candidate scales searched per
  group by the MSE quantizer. Same as CLI `--n-candidates`.
- `chunk_size` (int, default `4096` for INT4 and `1024` for INT8): group chunk
  size used to bound peak memory during MSE search. Same as CLI `--chunk-size`.
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
(`bits`, `strategy`, `method`, `group_size`, `n_candidates`, `chunk_size`)
take the CLI value when provided, otherwise fall back to the recipe, otherwise
to the bit-width-specific default. At least one selector (via CLI or recipe)
is required.

```bash
flagos-compressor quantize \
  --input /path/to/model \
  --output /path/to/model-int4 \
  --recipe quantize.yaml
```

## Validate

```bash
flagos-compressor validate --input /path/to/model-int4
```

Validation checks the checkpoint index, stored tensors, INT4/INT8 metadata,
and runtime quantization config.

## Current scope

- Sharded HuggingFace safetensors.
- MXFP4, block FP8, and floating-point input weights.
- Weight-only symmetric groupwise MSE INT4.
- Weight-only symmetric groupwise MSE INT8.
- Weight-only symmetric per-channel MSE INT8 for non-routed Linear weights.
- CPU and CUDA torch execution.
