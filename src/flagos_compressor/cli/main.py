from __future__ import annotations

import argparse
import logging
import sys

from flagos_compressor.core.policy import BUILTIN_SELECTIONS


COMMANDS = {"convert", "quantize", "inspect", "validate"}


def _add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--device")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flagos-compressor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert FP4/FP8 checkpoint weights to BF16.")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--to", default="bf16", choices=["bf16"])
    convert.add_argument("--dry-run", action="store_true")
    _add_backend_args(convert)

    quantize = subparsers.add_parser("quantize", help="Quantize selected weights to INT4.")
    quantize.add_argument("--input", required=True)
    quantize.add_argument("--output", required=True)
    quantize.add_argument("--recipe")
    quantize.add_argument("--select", action="append", choices=sorted(BUILTIN_SELECTIONS))
    quantize.add_argument("--exclude", action="append", choices=sorted(BUILTIN_SELECTIONS))
    quantize.add_argument("--select-name", action="append", metavar="REGEX")
    quantize.add_argument("--exclude-name", action="append", metavar="REGEX")
    quantize.add_argument("--method", choices=["mse"], default=None)
    quantize.add_argument("--group-size", type=int, default=None)
    quantize.add_argument("--n-candidates", type=int, default=None)
    quantize.add_argument("--chunk-size", type=int, default=None)
    quantize.add_argument("--dry-run", action="store_true")
    _add_backend_args(quantize)

    inspect = subparsers.add_parser("inspect", help="Inspect formats and selectable weight groups.")
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--json", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate a converted or quantized artifact.")
    validate.add_argument("--input", required=True)
    validate.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Backwards compatibility: the old CLI accepted --input/--output without
    # an explicit command and always performed BF16 conversion.
    if raw and raw[0] not in COMMANDS and raw[0] not in {"-h", "--help"}:
        raw.insert(0, "convert")
    parser = build_parser()
    args = parser.parse_args(raw)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if args.command == "convert":
        from flagos_compressor.cli import convert
        convert.run(args)
    elif args.command == "quantize":
        from flagos_compressor.cli import quantize
        quantize.run(args)
    elif args.command == "inspect":
        from flagos_compressor.cli import inspect_model
        inspect_model.run(args)
    else:
        from flagos_compressor.cli import validate
        validate.run(args)


if __name__ == "__main__":
    main()
