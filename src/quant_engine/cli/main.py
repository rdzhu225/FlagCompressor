from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-engine")
    parser.add_argument("command", nargs="?", default="convert", choices=["convert"])
    parser.add_argument("--input", required=True, help="Input HuggingFace safetensors checkpoint directory.")
    parser.add_argument("--output", required=True, help="Output checkpoint directory.")
    parser.add_argument(
        "--target",
        default="bf16",
        choices=["bf16", "moe-int4"],
        help="bf16 converts all supported FP8/FP4 weights to BF16; moe-int4 converts MoE FP4 to MSE INT4.",
    )
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--device")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from quant_engine.cli import convert

    convert.run(args)


if __name__ == "__main__":
    main()
