from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from nic_vrptw.core.models import ScoreRecord
from nic_vrptw.experiments.evaluators import register_evaluator, unregister_evaluator
from nic_vrptw.experiments.runner import run_experiments


ROOT = Path(__file__).resolve().parents[1]


class _AlwaysInfeasibleEvaluator:
    evaluator_id = "always_infeasible"

    def evaluate(self, instance, solution, objective_mode):  # noqa: ANN001
        return ScoreRecord(
            feasible=False,
            vehicles_used=0,
            distance=0.0,
            official_cost=0.0,
            objective_mode=objective_mode,
            objective_value=(999.0,),
            violations=("forced-evaluator",),
            metadata={"source": "test"},
        )


class EvaluatorRegistryTests(unittest.TestCase):
    def test_runner_accepts_custom_evaluator_plugin(self) -> None:
        register_evaluator(_AlwaysInfeasibleEvaluator())
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                records = run_experiments(
                    ROOT / "configs/smoke_e2e.yaml",
                    evaluator_id="always_infeasible",
                    output_dir=Path(tmpdir),
                )
        finally:
            unregister_evaluator("always_infeasible")

        self.assertEqual(len(records), 2)
        self.assertTrue(all(not record.feasible for record in records))
        self.assertTrue(all(record.evaluator_id == "always_infeasible" for record in records))


if __name__ == "__main__":
    unittest.main()
