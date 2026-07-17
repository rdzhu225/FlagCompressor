from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flag-compressor")
    parser.add_argument("command", nargs="?", default="convert", choices=["convert"])
    parser.add_argument("--input", required=True, help="Input HuggingFace safetensors checkpoint directory.")
    parser.add_argument("--output", required=True, help="Output checkpoint directory.")
    parser.add_argument(
        "--target",
        default="bf16",
        choices=["bf16", "int4", "moe-int4"],
        help=(
            "bf16 dequantizes all FP8/FP4 weights to BF16. "
            "int4 does the BF16 pass but also quantizes tensors matching --int4-include "
            "(minus --int4-exclude) to symmetric groupwise INT4. "
            "moe-int4 is a deprecated alias for int4 with a built-in MoE-expert selector."
        ),
    )
    parser.add_argument(
        "--int4-include",
        action="append",
        default=None,
        metavar="REGEX",
        help=(
            "Regex matched against tensor names (re.search). May be repeated. "
            "Only used when --target=int4. If omitted with --target=int4, no tensor is INT4-quantized."
        ),
    )
    parser.add_argument(
        "--int4-exclude",
        action="append",
        default=None,
        metavar="REGEX",
        help="Regex matched against tensor names to exclude from INT4 quantization. May be repeated.",
    )
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--device")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from flag_compressor.cli import convert

    convert.run(args)


if __name__ == "__main__":
    main()
