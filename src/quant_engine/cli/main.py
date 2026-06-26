from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-engine")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_parser = sub.add_parser("inspect", help="Scan a HF safetensors checkpoint.")
    inspect_parser.add_argument("--model", required=True)
    inspect_parser.add_argument("--out", required=True)
    inspect_parser.add_argument("--suggest-recipe")
    inspect_parser.add_argument("--output-model")

    dry_parser = sub.add_parser("dry-run", help="Compile recipe into a tensor-level plan.")
    dry_parser.add_argument("--recipe", required=True)
    dry_parser.add_argument("--out")

    convert_parser = sub.add_parser("convert", help="Execute a quantization/conversion recipe.")
    convert_parser.add_argument("--recipe", required=True)
    convert_parser.add_argument("--output")
    convert_parser.add_argument("--backend")
    convert_parser.add_argument("--device")

    build_calib_parser = sub.add_parser("build-calib", help="Build a JSONL calibration dataset.")
    build_calib_parser.add_argument("--model-path", required=True)
    build_calib_parser.add_argument("--output", required=True)
    build_calib_parser.add_argument("--datasets", nargs="+", default=["squad", "nq", "triviaqa", "mmlu", "openbookqa", "boolq"])
    build_calib_parser.add_argument("--num-per-dataset", type=int, default=64)
    build_calib_parser.add_argument("--tokenizer-path")
    build_calib_parser.add_argument("--seed", type=int, default=42)
    build_calib_parser.add_argument("--max-length", type=int, default=2048)

    calibrate_parser = sub.add_parser("calibrate", help="Collect activation statistics via model hooks.")
    calibrate_parser.add_argument("--recipe", required=True)
    calibrate_parser.add_argument("--model-path")
    calibrate_parser.add_argument("--tokenizer-path")
    calibrate_parser.add_argument("--dataset-jsonl")
    calibrate_parser.add_argument("--output")
    calibrate_parser.add_argument("--device")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        from quant_engine.cli import inspect

        inspect.run(args)
    elif args.command == "dry-run":
        from quant_engine.cli import dry_run

        dry_run.run(args)
    elif args.command == "convert":
        from quant_engine.cli import convert

        convert.run(args)
    elif args.command == "build-calib":
        from quant_engine.calibration import dataset_builder

        dataset_builder.run_cli(args)
    elif args.command == "calibrate":
        from quant_engine.calibration import collect

        collect.run_cli(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

