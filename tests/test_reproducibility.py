from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
from nic_vrptw.data.loader import load_instance
from nic_vrptw.experiments.evaluator import evaluate_solution
from nic_vrptw.solvers import get_solver


ROOT = Path(__file__).resolve().parents[1]


class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_produces_same_solution_and_score(self) -> None:
        instance = load_instance(ROOT / "data/fixtures/solomon/C101-mini.txt")
        solver = get_solver("reference_solver")
        params = {"distance_weight": 1.0, "wait_weight": 0.2, "slack_weight": 0.05}

        solution_a = solver.solve(instance, seed=17, params=params)
        solution_b = solver.solve(instance, seed=17, params=params)
        score_a = evaluate_solution(instance, solution_a, objective_mode="hierarchical")
        score_b = evaluate_solution(instance, solution_b, objective_mode="hierarchical")

        self.assertEqual(solution_a, solution_b)
        self.assertEqual(score_a, score_b)

    def test_aco_solver_is_reproducible_on_committed_fixture_with_single_worker(self) -> None:
        instance = load_instance(ROOT / "data/fixtures/solomon/C101-mini.txt")
        solver = get_solver("aco_solver")
        params = {
            "n_ants": 5,
            "n_iterations": 8,
            "alpha": 1.0,
            "beta": 2.0,
            "rho": 0.1,
            "q_const": 100.0,
            "vehicle_penalty": 10.0,
            "enable_local_search": True,
            "local_search_operators": ("relocate", "swap", "two_opt"),
            "local_search_max_passes": 1,
            "max_workers": 1,
        }

        solution_a = solver.solve(instance, seed=23, params=params)
        solution_b = solver.solve(instance, seed=23, params=params)
        score_a = evaluate_solution(instance, solution_a, objective_mode="hierarchical")
        score_b = evaluate_solution(instance, solution_b, objective_mode="hierarchical")

        self.assertEqual(solution_a, solution_b)
        self.assertEqual(score_a, score_b)


if __name__ == "__main__":
    unittest.main()
