from __future__ import annotations

import logging
from collections import Counter

from flag_compressor.cli.helpers import dump_json
from flag_compressor.inspect.checkpoint_scanner import scan_hf_safetensors

logger = logging.getLogger(__name__)


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
    logger.info("Model: %s", args.input)
    logger.info("Selectable weight groups")
    for name, count in data["selectable_groups"].items():
        logger.info("  %s: %d", name, count)
    logger.info("Input storage formats")
    for name, count in sorted(profile.summary()["storage_formats"].items()):
        logger.info("  %s: %d", name, count)
