from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping
from concurrent.futures import ProcessPoolExecutor  # Use processes to bypass the Global Interpreter Lock (GIL)

from nic_vrptw.core.models import Route, RouteSolution, VRPTWInstance
from nic_vrptw.solvers.local_search import improve_solution

@dataclass
class AcoSolver:
    solver_id: str = "aco_solver"

    def solve(
        self,
        instance: VRPTWInstance,
        seed: int,
        params: Mapping[str, Any],
    ) -> RouteSolution:
        master_rng = random.Random(seed)
        
        # Hyperparameters for ACO
        n_ants = int(params.get("n_ants", 10))
        n_iterations = int(params.get("n_iterations", 50))
        alpha = float(params.get("alpha", 1.0))
        beta = float(params.get("beta", 2.0))
        rho = float(params.get("rho", 0.1))
        q_const = float(params.get("q_const", 100.0))
        vehicle_penalty = float(params.get("vehicle_penalty", 10.0))
        # Number of CPU cores for parallel execution
        max_workers = int(params.get("max_workers", 10))
        local_search_operators = self._resolve_local_search_operators(params)
        local_search_max_passes = self._resolve_local_search_max_passes(params)
        local_search_scope = self._resolve_local_search_scope(params)

        nodes = [instance.depot_id] + list(instance.non_depot_ids)
        # Initialize pheromones on all possible edges
        pheromones: dict[tuple[int, int], float] = {
            (i, j): 1.0 for i in nodes for j in nodes if i != j
        }

        best_solution: RouteSolution | None = None
        best_cost = float('inf')

        if max_workers <= 1:
            for _ in range(n_iterations):
                ant_seeds = [master_rng.randint(0, 1_000_000) for _ in range(n_ants)]
                solutions_in_iter = [
                    self._build_and_score_ant(
                        instance,
                        ant_seed,
                        pheromones,
                        alpha,
                        beta,
                        local_search_operators,
                        vehicle_penalty,
                        local_search_max_passes,
                        local_search_scope,
                    )
                    for ant_seed in ant_seeds
                ]
                solutions_in_iter = self._maybe_apply_iteration_best_local_search(
                    instance,
                    solutions_in_iter,
                    local_search_operators,
                    vehicle_penalty,
                    local_search_max_passes,
                    local_search_scope,
                )
                best_solution, best_cost = self._update_iteration_state(
                    instance,
                    pheromones,
                    solutions_in_iter,
                    q_const,
                    rho,
                    best_solution,
                    best_cost,
                )
        else:
            # Create a process pool once for the entire optimization cycle.
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for _ in range(n_iterations):
                    ant_seeds = [master_rng.randint(0, 1_000_000) for _ in range(n_ants)]
                    futures = [
                        executor.submit(
                            self._build_and_score_ant,
                            instance,
                            ant_seed,
                            pheromones,
                            alpha,
                            beta,
                            local_search_operators,
                            vehicle_penalty,
                            local_search_max_passes,
                            local_search_scope,
                        )
                        for ant_seed in ant_seeds
                    ]
                    solutions_in_iter = [future.result() for future in futures]
                    solutions_in_iter = self._maybe_apply_iteration_best_local_search(
                        instance,
                        solutions_in_iter,
                        local_search_operators,
                        vehicle_penalty,
                        local_search_max_passes,
                        local_search_scope,
                    )
                    best_solution, best_cost = self._update_iteration_state(
                        instance,
                        pheromones,
                        solutions_in_iter,
                        q_const,
                        rho,
                        best_solution,
                        best_cost,
                    )

        # Return the best found solution or trigger fallback if no solution was found
        return best_solution or self._fallback_solution(instance, seed, params)

    @staticmethod
    def _build_and_score_ant(
        instance,
        seed,
        pheromones,
        alpha,
        beta,
        local_search_operators,
        vehicle_penalty,
        local_search_max_passes,
        local_search_scope,
    ):
        if local_search_scope == "per_ant":
            return AcoSolver._build_and_score_ant_with_local_search(
                instance,
                seed,
                pheromones,
                alpha,
                beta,
                local_search_operators,
                vehicle_penalty,
                local_search_max_passes,
            )
        return AcoSolver._build_and_score_ant_without_local_search(
            instance,
            seed,
            pheromones,
            alpha,
            beta,
            vehicle_penalty,
        )

    @staticmethod
    def _build_and_score_ant_without_local_search(
        instance,
        seed,
        pheromones,
        alpha,
        beta,
        vehicle_penalty,
    ):
        ant_rng = random.Random(seed)
        solver = AcoSolver()
        sol = solver._build_ant_solution(instance, ant_rng, pheromones, alpha, beta)
        cost = solver._calculate_cost(instance, sol, vehicle_penalty)
        return sol, cost

    @staticmethod
    def _build_and_score_ant_with_local_search(
        instance,
        seed,
        pheromones,
        alpha,
        beta,
        local_search_operators,
        vehicle_penalty,
        local_search_max_passes,
    ):
        """Helper method for parallel execution inside a separate process."""
        ant_rng = random.Random(seed)
        solver = AcoSolver()
        sol = solver._build_ant_solution(instance, ant_rng, pheromones, alpha, beta)
        sol = improve_solution(
            instance,
            sol,
            local_search_operators,
            vehicle_penalty=vehicle_penalty,
            max_passes=local_search_max_passes,
        )
        if local_search_operators and local_search_max_passes > 0:
            metadata = dict(sol.metadata)
            metadata["local_search_scope"] = "per_ant"
            sol = RouteSolution(
                routes=sol.routes,
                solver_id=sol.solver_id,
                metadata=metadata,
            )
        cost = solver._calculate_cost(instance, sol, vehicle_penalty)
        return sol, cost

    def _build_ant_solution(self, instance, rng, pheromones, alpha, beta) -> RouteSolution:
        """Constructs a complete solution for one ant by iterating through customers."""
        remaining = set(instance.non_depot_ids)
        routes: list[Route] = []

        while remaining:
            current_node = instance.depot_id
            current_time = max(0.0, instance.customer(instance.depot_id).time_window.start)
            current_load = 0.0
            route_stops = []

            while True:
                candidates = []
                total_prob = 0.0
                
                for cid in remaining:
                    sched = self._check_feasibility(instance, current_node, current_time, current_load, cid)
                    if sched:
                        # Visibility heuristic: inversely proportional to distance
                        eta = 1.0 / (instance.travel_time(current_node, cid) + 1.0)
                        tau = pheromones.get((current_node, cid), 1.0)
                        # Probability formula: pheromone^alpha * heuristic^beta
                        prob = (tau ** alpha) * (eta ** beta)
                        candidates.append((cid, prob, sched))
                        total_prob += prob

                if not candidates:
                    break

                # Roulette wheel selection based on calculated probabilities
                pick = rng.random() * total_prob
                cumulative = 0.0
                chosen = candidates[-1] # Default fallback
                for cid, prob, sched in candidates:
                    cumulative += prob
                    if cumulative >= pick:
                        chosen = (cid, prob, sched)
                        break
                
                cid, _, sched = chosen
                route_stops.append(cid)
                remaining.remove(cid)
                current_node = cid
                current_time = sched["departure"]
                current_load = sched["load"]

            routes.append(Route(stops=tuple(route_stops)))

        return RouteSolution(routes=tuple(routes), solver_id=self.solver_id)

    def _check_feasibility(self, instance, curr, time, load, cid):
        """Validates capacity and time window constraints for a potential move."""
        cust = instance.customer(cid)
        new_load = load + cust.demand
        # Check supply needs (capacity constraint)
        if new_load > instance.vehicle.capacity:
            return None
        
        # Calculate arrival and service start time
        arrival = time + instance.travel_time(curr, cid)
        start = max(arrival, cust.time_window.start)
        # Check if arrival is before the customer's time window end
        if start > cust.time_window.end:
            return None
            
        departure = start + cust.service_time
        # Ensure the vehicle can return to the depot within its working hours
        depot_return = departure + instance.travel_time(cid, instance.depot_id)
        if depot_return > instance.customer(instance.depot_id).time_window.end:
            return None
            
        return {"departure": departure, "load": new_load}

    def _calculate_cost(
        self,
        instance,
        solution: RouteSolution,
        vehicle_penalty: float = 10.0,
    ) -> float:
        """Evaluates the total cost of a solution (Distance + Vehicle Penalty)."""
        total_dist = 0.0
        for route in solution.routes:
            prev = instance.depot_id
            for stop in route.stops:
                total_dist += instance.travel_time(prev, stop)
                prev = stop
            total_dist += instance.travel_time(prev, instance.depot_id)
        return total_dist + (len(solution.routes) * vehicle_penalty)

    def _fallback_solution(self, instance, seed, params):
        """Standard greedy fallback if ACO fails to produce a valid solution."""
        from .reference import ReferenceSolver
        return ReferenceSolver().solve(instance, seed, params)

    @staticmethod
    def _resolve_local_search_operators(params: Mapping[str, Any]) -> tuple[str, ...]:
        if not bool(params.get("enable_local_search", True)):
            return ()

        raw_operators = params.get("local_search_operators", ("relocate", "swap", "two_opt"))
        if raw_operators is None:
            return ()
        if isinstance(raw_operators, str):
            normalized = raw_operators.strip()
            if not normalized or normalized.lower() in {"false", "none", "off"}:
                return ()
            if "," in normalized:
                return tuple(
                    part.strip()
                    for part in normalized.split(",")
                    if part.strip()
                )
            return (normalized,)
        return tuple(str(operator_id) for operator_id in raw_operators if str(operator_id).strip())

    @staticmethod
    def _resolve_local_search_max_passes(params: Mapping[str, Any]) -> int:
        return max(0, int(params.get("local_search_max_passes", 1)))

    @staticmethod
    def _resolve_local_search_scope(params: Mapping[str, Any]) -> str:
        raw_scope = str(params.get("local_search_scope", "per_ant")).strip().lower()
        valid_scopes = {"per_ant", "iteration_best"}
        if raw_scope not in valid_scopes:
            available = ", ".join(sorted(valid_scopes))
            raise ValueError(
                f"Unknown local_search_scope '{raw_scope}'. Available scopes: {available}."
            )
        return raw_scope

    def _maybe_apply_iteration_best_local_search(
        self,
        instance: VRPTWInstance,
        solutions_in_iter: list[tuple[RouteSolution, float]],
        local_search_operators: tuple[str, ...],
        vehicle_penalty: float,
        local_search_max_passes: int,
        local_search_scope: str,
    ) -> list[tuple[RouteSolution, float]]:
        if local_search_scope != "iteration_best":
            return solutions_in_iter
        if not local_search_operators or local_search_max_passes <= 0:
            return solutions_in_iter
        if not solutions_in_iter:
            return solutions_in_iter

        best_index = min(
            range(len(solutions_in_iter)),
            key=lambda index: solutions_in_iter[index][1],
        )
        best_solution, best_cost = solutions_in_iter[best_index]
        improved_solution = improve_solution(
            instance,
            best_solution,
            local_search_operators,
            vehicle_penalty=vehicle_penalty,
            max_passes=local_search_max_passes,
        )
        metadata = dict(improved_solution.metadata)
        metadata["local_search_scope"] = "iteration_best"
        improved_solution = RouteSolution(
            routes=improved_solution.routes,
            solver_id=improved_solution.solver_id,
            metadata=metadata,
        )
        improved_cost = self._calculate_cost(instance, improved_solution, vehicle_penalty)
        if improved_cost > best_cost:
            return solutions_in_iter

        updated_solutions = list(solutions_in_iter)
        updated_solutions[best_index] = (improved_solution, improved_cost)
        return updated_solutions

    def _update_iteration_state(
        self,
        instance: VRPTWInstance,
        pheromones: dict[tuple[int, int], float],
        solutions_in_iter: list[tuple[RouteSolution, float]],
        q_const: float,
        rho: float,
        best_solution: RouteSolution | None,
        best_cost: float,
    ) -> tuple[RouteSolution | None, float]:
        for sol, cost in solutions_in_iter:
            if cost < best_cost:
                best_cost = cost
                best_solution = sol

        for edge in pheromones:
            pheromones[edge] *= (1.0 - rho)

        for sol, cost in solutions_in_iter:
            contribution = q_const / cost
            for route in sol.routes:
                prev = instance.depot_id
                for stop in route.stops:
                    pheromones[(prev, stop)] += contribution
                    prev = stop
                pheromones[(prev, instance.depot_id)] += contribution

        return best_solution, best_cost
