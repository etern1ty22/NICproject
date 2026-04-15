from __future__ import annotations

import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from nic_vrptw.core.models import Customer, Route, RouteSolution, TimeWindow, VRPTWInstance, VehicleSpec
from nic_vrptw.solvers.aco import AcoSolver
from nic_vrptw.solvers.local_search import (
    evaluate_route,
    evaluate_solution,
    improve_solution,
    inter_route_relocate,
    inter_route_swap,
    intra_route_two_opt,
)


def _build_instance(
    *,
    name: str,
    distance_matrix: tuple[tuple[float, ...], ...],
    windows: dict[int, tuple[float, float]],
) -> VRPTWInstance:
    customers = tuple(
        Customer(
            customer_id=node_id,
            x=None,
            y=None,
            demand=0.0 if node_id == 0 else 1.0,
            time_window=TimeWindow(*windows[node_id]),
            service_time=0.0,
            is_depot=node_id == 0,
        )
        for node_id in range(len(distance_matrix))
    )
    return VRPTWInstance(
        name=name,
        source_format="unit-test",
        vehicle=VehicleSpec(count=4, capacity=10.0),
        depot_id=0,
        customers=customers,
        node_ids=tuple(range(len(distance_matrix))),
        distance_matrix=distance_matrix,
        metadata={},
    )


def _relaxed_two_opt_instance() -> VRPTWInstance:
    return _build_instance(
        name="relaxed-two-opt",
        distance_matrix=(
            (0.0, 1.0, 5.0, 5.0, 1.0),
            (1.0, 0.0, 1.0, 10.0, 10.0),
            (5.0, 1.0, 0.0, 1.0, 10.0),
            (5.0, 10.0, 1.0, 0.0, 1.0),
            (1.0, 10.0, 10.0, 1.0, 0.0),
        ),
        windows={node_id: (0.0, 100.0) for node_id in range(5)},
    )


def _time_window_constrained_instance() -> VRPTWInstance:
    return _build_instance(
        name="constrained-two-opt",
        distance_matrix=(
            (0.0, 1.0, 2.0, 50.0, 50.0),
            (1.0, 0.0, 1.0, 1.0, 20.0),
            (2.0, 50.0, 0.0, 10.0, 1.0),
            (50.0, 50.0, 1.0, 0.0, 1.0),
            (1.0, 50.0, 50.0, 1.0, 0.0),
        ),
        windows={
            0: (0.0, 100.0),
            1: (0.0, 1.0),
            2: (0.0, 2.0),
            3: (12.0, 12.0),
            4: (13.0, 13.0),
        },
    )


def _relocate_instance() -> VRPTWInstance:
    return _build_instance(
        name="relocate-instance",
        distance_matrix=(
            (0.0, 5.0, 1.0, 1.0),
            (5.0, 0.0, 1.0, 1.0),
            (1.0, 1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0, 0.0),
        ),
        windows={node_id: (0.0, 100.0) for node_id in range(4)},
    )


def _swap_instance() -> VRPTWInstance:
    return _build_instance(
        name="swap-instance",
        distance_matrix=(
            (0.0, 1.0, 1.0, 1.0, 1.0),
            (1.0, 0.0, 10.0, 1.0, 10.0),
            (1.0, 10.0, 0.0, 10.0, 1.0),
            (1.0, 1.0, 10.0, 0.0, 10.0),
            (1.0, 10.0, 1.0, 10.0, 0.0),
        ),
        windows={node_id: (0.0, 100.0) for node_id in range(5)},
    )


class AcoLocalSearchTests(unittest.TestCase):
    def test_intra_route_two_opt_improves_distance_when_feasible(self) -> None:
        instance = _relaxed_two_opt_instance()
        route = Route(stops=(1, 3, 2, 4))

        original_metrics = evaluate_route(instance, route)
        improved_route = intra_route_two_opt(instance, route)
        improved_metrics = evaluate_route(instance, improved_route)

        self.assertIsNotNone(original_metrics)
        self.assertIsNotNone(improved_metrics)
        assert original_metrics is not None
        assert improved_metrics is not None
        self.assertLess(improved_metrics.distance, original_metrics.distance)
        self.assertEqual(improved_metrics.distance, 5.0)

    def test_inter_route_relocate_can_merge_routes_when_feasible(self) -> None:
        instance = _relocate_instance()
        solution = RouteSolution(
            routes=(Route(stops=(1,)), Route(stops=(2, 3))),
            solver_id="aco_solver",
        )

        original_metrics = evaluate_solution(instance, solution)
        improved_solution = inter_route_relocate(instance, solution)
        improved_metrics = evaluate_solution(instance, improved_solution)

        self.assertIsNotNone(original_metrics)
        self.assertIsNotNone(improved_metrics)
        assert original_metrics is not None
        assert improved_metrics is not None
        self.assertEqual(len(improved_solution.routes), 1)
        self.assertLess(improved_metrics.cost, original_metrics.cost)
        self.assertEqual(improved_metrics.vehicles_used, 1)

    def test_inter_route_swap_improves_cross_route_assignment(self) -> None:
        instance = _swap_instance()
        solution = RouteSolution(
            routes=(Route(stops=(1, 2)), Route(stops=(3, 4))),
            solver_id="aco_solver",
        )

        original_metrics = evaluate_solution(instance, solution)
        improved_solution = inter_route_swap(instance, solution)
        improved_metrics = evaluate_solution(instance, improved_solution)

        self.assertIsNotNone(original_metrics)
        self.assertIsNotNone(improved_metrics)
        assert original_metrics is not None
        assert improved_metrics is not None
        self.assertLess(improved_metrics.distance, original_metrics.distance)
        self.assertEqual(
            {frozenset(route.stops) for route in improved_solution.routes},
            {frozenset((1, 3)), frozenset((2, 4))},
        )

    def test_intra_route_two_opt_rejects_infeasible_reversal(self) -> None:
        instance = _time_window_constrained_instance()
        route = Route(stops=(1, 2, 3, 4))
        infeasible_but_shorter = Route(stops=(1, 3, 2, 4))

        self.assertIsNone(evaluate_route(instance, infeasible_but_shorter))

        improved_route = intra_route_two_opt(instance, route)
        improved_metrics = evaluate_route(instance, improved_route)

        self.assertEqual(improved_route, route)
        self.assertIsNotNone(improved_metrics)

    def test_improve_solution_preserves_contract_and_metadata(self) -> None:
        instance = _relocate_instance()
        solution = RouteSolution(
            routes=(Route(stops=(1,)), Route(stops=(2, 3))),
            solver_id="aco_solver",
            metadata={"seed": 17},
        )

        improved_solution = improve_solution(
            instance,
            solution,
            ("relocate", "swap", "two_opt"),
            max_passes=2,
        )
        improved_metrics = evaluate_solution(instance, improved_solution)
        original_metrics = evaluate_solution(instance, solution)

        self.assertEqual(improved_solution.solver_id, solution.solver_id)
        self.assertEqual(improved_solution.metadata["seed"], 17)
        self.assertEqual(
            improved_solution.metadata["local_search_operators"],
            ("relocate", "swap", "two_opt"),
        )
        self.assertEqual(improved_solution.metadata["local_search_max_passes"], 2)
        self.assertGreaterEqual(improved_solution.metadata["local_search_passes_run"], 1)
        self.assertTrue(improved_solution.metadata["local_search_applied"])
        self.assertIsNotNone(improved_metrics)
        self.assertIsNotNone(original_metrics)
        assert improved_metrics is not None
        assert original_metrics is not None
        self.assertLess(improved_metrics.cost, original_metrics.cost)

    def test_aco_solver_applies_local_search_on_per_ant_path(self) -> None:
        instance = _relocate_instance()
        base_solution = RouteSolution(
            routes=(Route(stops=(1,)), Route(stops=(2, 3))),
            solver_id="aco_solver",
        )

        with patch.object(AcoSolver, "_build_ant_solution", return_value=base_solution):
            solution = AcoSolver().solve(
                instance=instance,
                seed=11,
                params={
                    "n_ants": 1,
                    "n_iterations": 1,
                    "max_workers": 1,
                    "local_search_operators": ("relocate", "swap", "two_opt"),
                    "local_search_max_passes": 1,
                },
            )

        original_metrics = evaluate_solution(instance, base_solution)
        improved_metrics = evaluate_solution(instance, solution)

        self.assertIsNotNone(original_metrics)
        self.assertIsNotNone(improved_metrics)
        assert original_metrics is not None
        assert improved_metrics is not None
        self.assertEqual(solution.metadata["local_search_scope"], "per_ant")
        self.assertEqual(
            solution.metadata["local_search_operators"],
            ("relocate", "swap", "two_opt"),
        )
        self.assertLess(improved_metrics.cost, original_metrics.cost)
        self.assertEqual(len(solution.routes), 1)

    def test_aco_solver_can_limit_local_search_to_iteration_best_ant(self) -> None:
        instance = _relocate_instance()
        base_solution = RouteSolution(
            routes=(Route(stops=(1,)), Route(stops=(2, 3))),
            solver_id="aco_solver",
        )

        with (
            patch.object(AcoSolver, "_build_ant_solution", return_value=base_solution),
            patch("nic_vrptw.solvers.aco.improve_solution", return_value=base_solution) as mocked_improve,
        ):
            solution = AcoSolver().solve(
                instance=instance,
                seed=11,
                params={
                    "n_ants": 3,
                    "n_iterations": 2,
                    "max_workers": 1,
                    "local_search_scope": "iteration_best",
                    "local_search_operators": ("relocate", "swap", "two_opt"),
                    "local_search_max_passes": 1,
                },
            )

        self.assertEqual(mocked_improve.call_count, 2)
        self.assertEqual(solution.metadata["local_search_scope"], "iteration_best")


if __name__ == "__main__":
    unittest.main()
