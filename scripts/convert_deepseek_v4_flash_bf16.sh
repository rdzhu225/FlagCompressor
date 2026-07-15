#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate flagos

REPO_DIR="/share-evpfs/flagos/ready/edithz/quant-engine"
INPUT_DIR="${1:-/share-evpfs/flagos/ready/edithz/DeepSeek-V4-Flash}"
OUTPUT_DIR="${2:-/share-evpfs/flagos/ready/edithz/DeepSeek-V4-Flash-bf16}"
BACKEND="${BACKEND:-cpu}"

cd "$REPO_DIR"
PYTHONPATH=src python -m quant_engine.cli.main convert   --input "$INPUT_DIR"   --output "$OUTPUT_DIR"   --target bf16   --backend "$BACKEND"
