import csv
import json
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401
from nic_vrptw.experiments.runner import run_experiments
from nic_vrptw.solvers import list_solvers


ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def test_smoke_runner_writes_csv_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            records = run_experiments(ROOT / "configs/smoke_e2e.yaml", output_dir=output_dir)
            self.assertEqual(len(records), 2)

            csv_files = list(output_dir.glob("*.csv"))
            json_files = list(output_dir.glob("*.json"))
            self.assertEqual(len(csv_files), 1)
            self.assertEqual(len(json_files), 1)

            with csv_files[0].open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("dataset", rows[0])
            self.assertIn("official_cost", rows[0])
            self.assertIn("evaluator_id", rows[0])

            metadata = json.loads(json_files[0].read_text(encoding="utf-8"))
            self.assertEqual(metadata["records_count"], 2)
            self.assertEqual(metadata["evaluator_id"], "default_evaluator")

    def test_aco_solver_is_registered_and_runs_via_runner(self) -> None:
        self.assertIn("aco_solver", list_solvers())

        with tempfile.TemporaryDirectory() as tmpdir:
            records = run_experiments(
                ROOT / "configs/smoke_e2e.yaml",
                solver_id="aco_solver",
                output_dir=Path(tmpdir),
                param_overrides={"n_ants": 1, "n_iterations": 1, "max_workers": 1},
            )

        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.solver_id == "aco_solver" for record in records))
        self.assertTrue(all(record.feasible for record in records))


if __name__ == "__main__":
    unittest.main()
