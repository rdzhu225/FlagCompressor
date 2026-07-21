from __future__ import annotations

from collections import Counter

from flag_compressor.cli.helpers import dump_json
from flag_compressor.inspect.checkpoint_scanner import scan_hf_safetensors


def run(args) -> None:
    profile = scan_hf_safetensors(args.input)
    groups = Counter(tag for tensor in profile.tensors.values() for tag in tensor.tags)
    data = {
        "model": args.input,
        "summary": profile.summary(),
        "selectable_groups": {
            key: groups.get(key, 0)
            for key in ("moe", "moe.routed", "moe.shared", "attention", "mlp", "linear")
        },
    }
    if args.json:
        dump_json(data)
        return
    print(f"Model: {args.input}")
    print("Selectable weight groups")
    for name, count in data["selectable_groups"].items():
        print(f"  {name}: {count}")
    print("Input storage formats")
    for name, count in sorted(profile.summary()["storage_formats"].items()):
        print(f"  {name}: {count}")
