from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401
from nic_vrptw.experiments.runner import run_experiments
from nic_vrptw.solvers import register_solver, unregister_solver


ROOT = Path(__file__).resolve().parents[1]


class _BadSolver:
    solver_id = "bad_solver"

    def solve(self, instance, seed, params):  # noqa: ANN001
        return {"routes": []}


class ContractTests(unittest.TestCase):
    def test_runner_rejects_invalid_solver_contract(self) -> None:
        register_solver(_BadSolver())
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaises(ValueError):
                    run_experiments(
                        ROOT / "configs/smoke_e2e.yaml",
                        solver_id="bad_solver",
                        output_dir=Path(tmpdir),
                    )
        finally:
            unregister_solver("bad_solver")


if __name__ == "__main__":
    unittest.main()
