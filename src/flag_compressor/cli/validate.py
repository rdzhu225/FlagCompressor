from __future__ import annotations

import logging

from flag_compressor.cli.helpers import dump_json
from flag_compressor.core.validation import validate_artifact

logger = logging.getLogger(__name__)


def run(args) -> None:
    result = validate_artifact(args.input)
    if args.json:
        dump_json(result)
    else:
        logger.info("Valid" if result["valid"] else "Invalid")
        logger.info("Tensors: %d", result["tensors"])
        logger.info("Shards: %d", result["shards"])
        logger.info("INT4 tensors: %d", result["int4_tensors"])
        for error in result["errors"]:
            logger.error("  %s", error)
    if not result["valid"]:
        raise SystemExit(1)
