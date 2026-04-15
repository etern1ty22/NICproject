from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401
from nic_vrptw.experiments.analysis import build_final_analysis


FIELDNAMES = [
    "run_id",
    "config_id",
    "dataset",
    "instance_id",
    "instance_format",
    "role",
    "seed",
    "solver_id",
    "evaluator_id",
    "feasible",
    "vehicles_used",
    "distance",
    "official_cost",
    "runtime_s",
    "objective_mode",
    "params",
    "violations",
]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class FinalAnalysisTests(unittest.TestCase):
    def test_build_final_analysis_creates_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            greedy_csv = root / "greedy.csv"
            aco_csv = root / "aco.csv"
            sweep_csv = root / "sweep.csv"
            ablation_csv = root / "ablation.csv"
            output_dir = root / "analysis"

            _write_rows(
                greedy_csv,
                [
                    {
                        "run_id": "g1",
                        "config_id": "cfg-g",
                        "dataset": "solomon_c101_official",
                        "instance_id": "C101",
                        "instance_format": "solomon",
                        "role": "baseline",
                        "seed": "17",
                        "solver_id": "greedy_solver",
                        "evaluator_id": "default_evaluator",
                        "feasible": "True",
                        "vehicles_used": "11",
                        "distance": "890.0",
                        "official_cost": "890.0",
                        "runtime_s": "0.3",
                        "objective_mode": "hierarchical",
                        "params": "{}",
                        "violations": "[]",
                    }
                ],
            )
            _write_rows(
                aco_csv,
                [
                    {
                        "run_id": "a1",
                        "config_id": "cfg-a",
                        "dataset": "solomon_c101_official",
                        "instance_id": "C101",
                        "instance_format": "solomon",
                        "role": "final",
                        "seed": "7",
                        "solver_id": "aco_solver",
                        "evaluator_id": "default_evaluator",
                        "feasible": "True",
                        "vehicles_used": "10",
                        "distance": "845.0",
                        "official_cost": "845.0",
                        "runtime_s": "1.1",
                        "objective_mode": "hierarchical",
                        "params": "{'alpha': 1.0, 'beta': 2.0, 'rho': 0.1}",
                        "violations": "[]",
                    },
                    {
                        "run_id": "a2",
                        "config_id": "cfg-a",
                        "dataset": "solomon_c101_official",
                        "instance_id": "C101",
                        "instance_format": "solomon",
                        "role": "final",
                        "seed": "11",
                        "solver_id": "aco_solver",
                        "evaluator_id": "default_evaluator",
                        "feasible": "True",
                        "vehicles_used": "10",
                        "distance": "840.0",
                        "official_cost": "840.0",
                        "runtime_s": "1.0",
                        "objective_mode": "hierarchical",
                        "params": "{'alpha': 1.0, 'beta': 2.0, 'rho': 0.1}",
                        "violations": "[]",
                    },
                ],
            )
            _write_rows(
                sweep_csv,
                [
                    {
                        "run_id": "s1",
                        "config_id": "cfg-s",
                        "dataset": "solomon_c101_official",
                        "instance_id": "C101",
                        "instance_format": "solomon",
                        "role": "tuning",
                        "seed": "7",
                        "solver_id": "aco_solver",
                        "evaluator_id": "default_evaluator",
                        "feasible": "True",
                        "vehicles_used": "10",
                        "distance": "844.0",
                        "official_cost": "844.0",
                        "runtime_s": "1.2",
                        "objective_mode": "hierarchical",
                        "params": "{'alpha': 0.8, 'beta': 2.0, 'rho': 0.05}",
                        "violations": "[]",
                    },
                    {
                        "run_id": "s2",
                        "config_id": "cfg-s",
                        "dataset": "solomon_c101_official",
                        "instance_id": "C101",
                        "instance_format": "solomon",
                        "role": "tuning",
                        "seed": "11",
                        "solver_id": "aco_solver",
                        "evaluator_id": "default_evaluator",
                        "feasible": "True",
                        "vehicles_used": "10",
                        "distance": "850.0",
                        "official_cost": "850.0",
                        "runtime_s": "1.3",
                        "objective_mode": "hierarchical",
                        "params": "{'alpha': 1.2, 'beta': 3.0, 'rho': 0.1}",
                        "violations": "[]",
                    },
                ],
            )
            _write_rows(
                ablation_csv,
                [
                    {
                        "run_id": "l1",
                        "config_id": "cfg-l",
                        "dataset": "solomon_c101_official",
                        "instance_id": "C101",
                        "instance_format": "solomon",
                        "role": "ablation",
                        "seed": "7",
                        "solver_id": "aco_solver",
                        "evaluator_id": "default_evaluator",
                        "feasible": "True",
                        "vehicles_used": "10",
                        "distance": "860.0",
                        "official_cost": "860.0",
                        "runtime_s": "0.9",
                        "objective_mode": "hierarchical",
                        "params": "{'local_search_operators': []}",
                        "violations": "[]",
                    },
                    {
                        "run_id": "l2",
                        "config_id": "cfg-l",
                        "dataset": "solomon_c101_official",
                        "instance_id": "C101",
                        "instance_format": "solomon",
                        "role": "ablation",
                        "seed": "11",
                        "solver_id": "aco_solver",
                        "evaluator_id": "default_evaluator",
                        "feasible": "True",
                        "vehicles_used": "10",
                        "distance": "842.0",
                        "official_cost": "842.0",
                        "runtime_s": "1.2",
                        "objective_mode": "hierarchical",
                        "params": "{'local_search_operators': ['relocate', 'swap', 'two_opt']}",
                        "violations": "[]",
                    },
                ],
            )

            artifacts = build_final_analysis(
                greedy_csv=greedy_csv,
                aco_csv=aco_csv,
                sweep_csv=sweep_csv,
                ablation_csv=ablation_csv,
                output_dir=output_dir,
            )

            for path in artifacts.values():
                self.assertTrue(path.exists(), msg=f"Missing analysis output: {path}")

            summary = artifacts["summary_md"].read_text(encoding="utf-8")
            self.assertIn("Greedy vs ACO", summary)
            self.assertIn("Best Sweep Setting", summary)
            self.assertIn("Best Local-Search Ablation", summary)


if __name__ == "__main__":
    unittest.main()
