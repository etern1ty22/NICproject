#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nic_vrptw.experiments.analysis import build_final_analysis


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate final experiment summaries and plots.")
    parser.add_argument(
        "--greedy-csv",
        default=str(ROOT / "output/greedy/c101_official/greedy_official_c101.csv"),
    )
    parser.add_argument(
        "--aco-csv",
        default=str(ROOT / "output/aco/final_multiseed_c101/final_multiseed.csv"),
    )
    parser.add_argument(
        "--sweep-csv",
        default=str(ROOT / "output/aco/sweep_c101/sweep.csv"),
    )
    parser.add_argument(
        "--ablation-csv",
        default=str(ROOT / "output/aco/ablation_local_search_c101/ablation.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "output/analysis/final"),
    )
    args = parser.parse_args()

    artifacts = build_final_analysis(
        greedy_csv=args.greedy_csv,
        aco_csv=args.aco_csv,
        sweep_csv=args.sweep_csv,
        ablation_csv=args.ablation_csv,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
