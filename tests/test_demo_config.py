import csv
import json
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401
from nic_vrptw.experiments.runner import _load_run_config, run_experiments


ROOT = Path(__file__).resolve().parents[1]


class DemoConfigTests(unittest.TestCase):
    def test_demo_config_uses_committed_fixture_and_writes_stable_aliases(self) -> None:
        config_path = ROOT / "configs/aco_demo.yaml"
        config = _load_run_config(config_path)

        self.assertEqual(config.name, "aco_demo_c101_mini")
        self.assertEqual(config.solver_id, "aco_solver")
        self.assertEqual(config.output_basename, "demo")
        self.assertEqual(len(config.seed_set), 1)
        self.assertEqual(len(config.datasets), 1)
        self.assertEqual(
            config.datasets[0].path,
            (ROOT / "data/fixtures/solomon/C101-mini.txt").resolve(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            records = run_experiments(config_path, output_dir=output_dir)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].dataset_id, "solomon_c101_mini_demo")
            self.assertTrue(records[0].feasible)

            csv_files = sorted(output_dir.glob("*.csv"))
            json_files = sorted(output_dir.glob("*.json"))
            self.assertEqual(len(csv_files), 2)
            self.assertEqual(len(json_files), 2)

            stable_csv = output_dir / "demo.csv"
            stable_json = output_dir / "demo.json"
            self.assertTrue(stable_csv.exists())
            self.assertTrue(stable_json.exists())

            with stable_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset"], "solomon_c101_mini_demo")
            self.assertEqual(rows[0]["solver_id"], "aco_solver")

            metadata = json.loads(stable_json.read_text(encoding="utf-8"))
            self.assertEqual(metadata["records_count"], 1)
            self.assertEqual(metadata["output_basename"], "demo")
            self.assertEqual(metadata["solver_id"], "aco_solver")


if __name__ == "__main__":
    unittest.main()
