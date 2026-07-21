# FlagCompressor

**A checkpoint compression toolkit for large language models.**

FlagCompressor is a quantization and compression toolkit for LLM
checkpoints. It takes a HuggingFace `safetensors` checkpoint and produces a
compressed checkpoint that the downstream deployment stack can consume
directly for inference across a wide range of accelerators.

## Quick Start

Install:

```bash
git clone <this-repo>
cd FlagCompressor
pip install -e .
```

Convert a checkpoint:

```bash
flag-compressor convert \
  --input  /path/to/fp8_or_fp4_model \
  --output /path/to/bf16_model \
  --backend cpu
```

Use `--backend cuda` on a GPU host.

## Roadmap

Planned directions:

- Additional low-precision output formats (e.g. int8, int4).
- PTQ / QAT quantization support.

## License

See [LICENSE](./LICENSE).
