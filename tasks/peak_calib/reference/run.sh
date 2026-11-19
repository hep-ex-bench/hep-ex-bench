#!/bin/bash
# EVAL_CONTRACT entrypoint: ./run.sh <dataset_dir> <output_dir>
set -euo pipefail
cd "$(dirname "$0")"
exec python3 calib.py "$1" "$2"
