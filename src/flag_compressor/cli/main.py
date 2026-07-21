from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flag-compressor")
    parser.add_argument("command", nargs="?", default="convert", choices=["convert"])
    parser.add_argument("--input", required=True, help="Input HuggingFace safetensors checkpoint directory.")
    parser.add_argument("--output", required=True, help="Output checkpoint directory.")
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
