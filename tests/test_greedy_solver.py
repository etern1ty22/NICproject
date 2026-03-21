from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401
from nic_vrptw.data.loader import load_instance
from nic_vrptw.experiments.evaluator import evaluate_solution
from nic_vrptw.experiments.runner import run_experiments
from nic_vrptw.solvers import get_solver, list_solvers


ROOT = Path(__file__).resolve().parents[1]


class GreedySolverTests(unittest.TestCase):
    def test_greedy_solver_is_registered_and_runs_via_runner(self) -> None:
        self.assertIn("greedy_solver", list_solvers())

        with tempfile.TemporaryDirectory() as tmpdir:
            records = run_experiments(
                ROOT / "configs/greedy_baseline.yaml",
                output_dir=Path(tmpdir),
            )

        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.solver_id == "greedy_solver" for record in records))
        self.assertTrue(all(record.feasible for record in records))

    def test_greedy_solver_is_deterministic_for_same_seed(self) -> None:
        instance = load_instance(ROOT / "data/fixtures/solomon/C101-mini.txt")
        solver = get_solver("greedy_solver")

        solution_a = solver.solve(instance, seed=17, params={})
        solution_b = solver.solve(instance, seed=17, params={})
        score_a = evaluate_solution(instance, solution_a, objective_mode="hierarchical")
        score_b = evaluate_solution(instance, solution_b, objective_mode="hierarchical")

        self.assertEqual(solution_a, solution_b)
        self.assertEqual(score_a, score_b)


if __name__ == "__main__":
    unittest.main()
