# FlagOS-Compressor

FlagOS-Compressor converts and quantizes HuggingFace `safetensors` checkpoints.
Its INT4/INT8 command supports module-level selection: selected weights are
quantized, while other low-precision weights are converted to BF16.

## Install

```bash
pip install -e .
```

The official AutoRound package is not required for native quantization. Install
`pip install -e '.[official-autoround]'` only when running the optional official
reference implementation or parity checks.

## Inspect

```bash
flagos-compressor inspect --input /path/to/model
```

This reports detected weight formats and selectable groups such as `moe`,
`moe.routed`, `moe.shared`, `attention`, `mlp`, and `linear`.

## Heterogeneous quantization in one command

Assign different target bit widths to different parts of a checkpoint in one
execution. For example, quantize DSV4 attention to INT8 and MoE weights to
INT4:

```bash
flagos-compressor quantize \
  --input /path/to/DSV4-Flash \
  --output /path/to/DSV4-Flash-mixed \
  --select moe=int4 \
  --select attention=int8 \
  --backend cuda
```

Each formatted `--select` value is `SELECTOR=FORMAT[:GROUP_SIZE]`. Selectors
can be a built-in group (`attention`, `moe`, `moe.routed`, `moe.shared`, `mlp`,
or `linear`). Regular expressions continue to use `--select-name`, for example
`--select-name 'model\.layers\.0\..*=int8:64'`. Formats are explicit so
future `fp4`/`fp8` support cannot be confused with `int4`/`int8`; currently the
exporter accepts `int4` and `int8`. They default to group sizes 32 and 128
respectively. Name rules are applied after built-in target rules, and the last
matching rule wins. Global `--exclude` and `--exclude-name` rules are applied
after the formatted selections.

The legacy homogeneous form remains valid: `--select moe --bits 4`. Formatted
and unformatted selectors cannot be mixed in one command, so no selection can
silently inherit the wrong format.

The mapping is an artifact contract, not a runtime-compatibility hint. If a
DSV4 source FP8 attention weight matches `attention=int8`, it is converted to
INT8, including `attn.wo_a`; the planner does not silently preserve FP8 for a
DeepGEMM implementation. Runtime support for consuming that layout is a
separate concern.

The command writes one `pack-quantized` compressed-tensors checkpoint with a
config group for every requested format/group-size combination. DSV4 fused
attention and shared-expert runtime aliases are included. Source-quantized
weights that match no rule follow the existing `unselected` policy and are
converted to BF16 by default.

Per-selector mode currently supports MSE W4A16/W8A16. The formatted selectors
own `bits`, `activation_bits`, `scale_dtype`, `strategy`, `group_size`, and
`chunk_size`, so do not combine them with those global settings.
`--n-candidates`, exclusions, the unselected policy, and backend/device flags
remain configurable. Use `--dry-run` to inspect the resolved plan without
writing the output checkpoint.

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

Do not pass `--group-size` with `--strategy channel`. Channelwise W8A16 export
stores scales as `[out_features, 1]` and declares `strategy: channel` in the
compressed-tensors config. vLLM's WNA16 routed-MoE path requires group
quantization, so W8A16 channel strategy is supported for ordinary Linear and
shared-expert Linear weights, but rejected for routed experts.

Dynamic per-token W8A8, including routed MoE experts:

```bash
flagos-compressor quantize \
  --input /path/to/qwen3.5-moe \
  --output /path/to/qwen3.5-moe-w8a8 \
  --select linear \
  --bits 8 \
  --activation-bits 8 \
  --strategy channel \
  --scale-dtype bf16 \
  --backend cuda
```

W8A8 uses compressed-tensors `int-quantized` storage: raw signed INT8 weights,
FP32 (default) or BF16 per-output-channel scales, and dynamic symmetric
per-token INT8 activations. Fused routed-expert banks are expanded to the standard
`experts.<id>.<projection>.weight` and `weight_scale` names consumed by vLLM.
W8A8 requires `--bits 8 --strategy channel`; `--group-size` is not accepted.
Use `--scale-dtype bf16` to emit BF16 `weight_scale` tensors.

For fused MoE model types without a registered layout adapter, the CLI can
infer the 3D bank order from a consistent `gate_up_proj` / `down_proj` pair:
`[E, 2I, H]` plus `[E, H, I]` is treated as `[E, out, in]`, while
`[E, H, 2I]` plus `[E, I, H]` is treated as `[E, in, out]`. Every discovered
pair must be complete, valid, and agree on the same order; otherwise
quantization stops instead of guessing.

### GPTQ (AutoGPTQ-compatible)

```bash
flagos-compressor quantize \
  --input /path/to/model \
  --output /path/to/model-gptq \
  --select linear \
  --method gptq \
  --bits 4 \
  --group-size 128 \
  --calibration-data /path/to/calibration.jsonl \
  --backend cuda
```

GPTQ uses AutoGPTQ's running Hessian, Cholesky error feedback, activation
ordering, true-sequential projection groups, and native
`qweight/qzeros/scales/g_idx` packing. Defaults are `desc_act: true`,
`static_groups: false`, `true_sequential: true`, and 1% dampening. The output
contains the standard GPTQ quantization config and canonical GPTQ safetensors
filenames.

### AWQ (AutoAWQ-compatible)

```bash
flagos-compressor quantize \
  --input /path/to/model \
  --output /path/to/model-awq \
  --select linear \
  --method awq \
  --bits 4 \
  --group-size 128 \
  --calibration-data /path/to/calibration.jsonl \
  --backend cuda
```

AWQ uses AutoAWQ's activation/weight grid-search scaling, output-MSE clipping,
asymmetric zero points, and native GEMM packing order. The result has
`qweight/qzeros/scales` tensors and an AWQ quantization config. Native AWQ is
currently W4A16 GEMM with zero points.

### AutoRound (native PyTorch)

```bash
flagos-compressor quantize \
  --input /path/to/model \
  --output /path/to/model-autoround \
  --select linear \
  --method autoround \
  --bits 4 \
  --group-size 128 \
  --calibration-data /path/to/calibration.jsonl \
  --autoround-iters 200 \
  --backend npu
```

AutoRound is implemented natively with PyTorch and does not depend on the
official `auto-round` package. It supports symmetric group-wise W4A16 and
W8A16, learnable rounding offsets, optional min/max tuning, quantized-input
cascading, and single-device execution. Device extensions such as `torch_npu`,
`torch_mlu`, or `torch_musa` are imported only when their backend is selected.
The output uses the established GPTQ tensor ABI for broad loader compatibility,
while config provenance records `algorithm: autoround`; algorithm and packing
are separate internally.

An official AutoRound `config.json` can be imported without installing the
official package:

```bash
flagos-compressor quantize \
  --input /path/to/model \
  --output /path/to/model-autoround \
  --select linear \
  --autoround-config /path/to/official/config.json \
  --calibration-data /path/to/calibration.jsonl \
  --backend npu
```

The bridge recognizes the current public fields such as `scheme`, `bits`,
`group_size`, `iters`, `nsamples`, `seqlen`, and the official historical
spelling `enable_quanted_input`. CLI and recipe values take precedence over
imported values. Exported GPTQ metadata retains official AutoRound-compatible
field names while identifying FlagOS-Compressor as the provider. The optional
official Python entry point is lazy-loaded only for explicit reference/parity
work; normal installation and native execution do not import it.

All calibrated methods execute the original Transformers model definition and
discover decoder blocks through Transformers' no-split contract; they do not
maintain a per-model forward adapter. Transformers-v5 fused expert modules are
temporarily exposed as ordinary per-expert `nn.Linear` modules, so the same
hooks handle dense and routed-MoE models. Routed experts must be selected as a
complete gate/up/down set. Source FP4/FP8 checkpoints are staged as BF16 before
calibration.

Calibration data can be a local `.txt`, `.json`, or `.jsonl` file. A Hugging
Face dataset name is also accepted when the optional `datasets` package is
installed. The default is 128 examples packed into 512-token blocks; use
`--calibration-samples` and `--calibration-seq-length` to change it. Ready-made
recipes are in `examples/recipes/`.

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
activation_bits: 8
strategy: channel
scale_dtype: bf16
method: mse
unselected:
  strategy: convert
  format: bf16
select:
  - moe
  - name: '.*\.self_attn\.o_proj\.weight$'
exclude:
  - name: '.*\.layers\.0\..*'
```

The equivalent heterogeneous recipe is:

```yaml
version: 1
select:
  - target: moe
    format: int4
  - target: attention
    format: int8
  - name: '^model\.layers\.0\.'
    format: int8
    group_size: 64
```

Recipe fields:

- `version` (int): recipe schema version. Versions `1` and `2` are accepted;
  use `2` for calibrated GPTQ/AWQ recipes.
- `select` (list): tensors to quantize. Legacy entries are a built-in group or
  `{name: 'REGEX'}` and use the global `bits` setting. Heterogeneous entries
  have exactly one of `target` (a built-in group) or `name` (a regex), plus an
  explicit `format` (`int4` or `int8`). Optional per-entry `group_size` and
  `chunk_size` override format-specific defaults. Name entries override target
  entries when both match.
- `bits` (int, default `4`): weight bit width, either `4` or `8`. Same as CLI
  `--bits`.
- `activation_bits` (int, default `16`): activation bit width, either `8` or
  `16`. Setting it to `8` enables dynamic-token W8A8 and requires `bits: 8`
  with `strategy: channel`. Same as CLI `--activation-bits`.
- `scale_dtype` (str, default `fp32`): W8A8 weight scale dtype, either `fp32`
  or `bf16`. Same as CLI `--scale-dtype`.
- `strategy` (str, default `group`): `group` or `channel`. Channel strategy is
  available for INT8 Linear weights and for routed experts in W8A8 mode. It
  must not specify `group_size`.
- `method` (str, default `mse`): `mse`, `gptq`, `awq`, or `autoround`.
- `format` (str): output checkpoint ABI. It is inferred as
  `compressed-tensors`, `gptq`, or `awq` from `method` and must agree when set.
  AutoRound currently uses `gptq` packing.
- `group_size` (int, default `32` for INT4 and `128` for INT8): group size
  along the input-feature axis for weight scales. Same as CLI `--group-size`.
- `n_candidates` (int, default `200`): number of candidate scales searched per
  group by the MSE quantizer. Same as CLI `--n-candidates`.
- `chunk_size` (int, default `4096` for INT4 and `1024` for INT8): group chunk
  size used to bound peak memory during MSE search. Same as CLI `--chunk-size`.
- `calibration` (mapping): `data`, `samples`, `sequence_length`, `seed`,
  `split`, `text_column`, and `trust_remote_code` for GPTQ/AWQ.
- `gptq` (mapping): `block_size`, `damp_percent`, `desc_act`, `static_groups`,
  `true_sequential`, and `symmetric`.
- `awq` (mapping): `zero_point`, `version`, `duo_scaling`, `apply_clip`,
  `n_grid`, and `max_chunk_memory`.
- `autoround` (mapping): `iters`, `lr`, `minmax_lr`, `batch_size`,
  `gradient_accumulate_steps`, `momentum`, `enable_minmax_tuning`, and
  `enable_quantized_input`. `official_config` may point to an official
  AutoRound JSON config whose values are used as lower-priority defaults.
- `exclude` (list): tensors to skip, same shape as `select`. Applied on top of
  the `select` set. Mirrors `--exclude` / `--exclude-name`.
- `unselected` (mapping): how source-quantized weights outside the selected set
  are handled. Currently only `strategy: convert` (default) with
  `format: bf16` is supported; it dequantizes low-precision weights to BF16.
  `strategy: preserve` is reserved for a future runtime-compatible mixed-format
  exporter and is rejected for now.

Outside per-selector mode, CLI flags and recipe fields are additive: `select`
and `exclude` entries from the recipe are merged with the corresponding CLI
flags, and scalar fields
(`bits`, `activation_bits`, `scale_dtype`, `strategy`, `method`, `group_size`,
`n_candidates`, `chunk_size`)
take the CLI value when provided, otherwise fall back to the recipe, otherwise
to the bit-width-specific default. At least one selector (via
CLI or recipe) is required.

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
runtime quantization config, and native GPTQ/AWQ tensor layouts.

## Current scope

- Sharded HuggingFace safetensors.
- MXFP4, block FP8, and floating-point input weights.
- Weight-only symmetric groupwise MSE INT4.
- Weight-only symmetric groupwise MSE INT8.
- Weight-only symmetric per-channel MSE INT8 for non-routed Linear weights.
- Dynamic-token W8A8 with symmetric per-channel INT8 weights for Linear and
  supported fused routed-MoE weights.
- AutoGPTQ-compatible W4A16/W8A16 calibration and native packing.
- AutoAWQ-compatible W4A16 calibration and native GEMM packing.
- Transformers-v5 dense and generic fused-MoE execution without model-specific
  forward adapters.
- CPU and CUDA torch execution.
