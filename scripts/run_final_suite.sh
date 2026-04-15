#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

python3 -m nic_vrptw run --config configs/greedy_c101_official.yaml
python3 -m nic_vrptw run --config configs/aco_final_multiseed.yaml
python3 -m nic_vrptw run --config configs/aco_sweep.yaml
python3 -m nic_vrptw run --config configs/aco_ablation.yaml
python3 scripts/generate_final_analysis.py
