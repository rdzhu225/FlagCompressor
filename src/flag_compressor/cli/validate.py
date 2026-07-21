from __future__ import annotations

from flag_compressor.cli.helpers import dump_json
from flag_compressor.core.validation import validate_artifact


def run(args) -> None:
    result = validate_artifact(args.input)
    if args.json:
        dump_json(result)
    else:
        print("Valid" if result["valid"] else "Invalid")
        print(f"Tensors: {result['tensors']}")
        print(f"Shards: {result['shards']}")
        print(f"INT4 tensors: {result['int4_tensors']}")
        for error in result["errors"]:
            print(f"  ERROR: {error}")
    if not result["valid"]:
        raise SystemExit(1)
