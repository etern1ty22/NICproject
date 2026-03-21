from __future__ import annotations

import random
import math
from dataclasses import dataclass
from typing import Any, Mapping
from concurrent.futures import ProcessPoolExecutor  # Use processes to bypass the Global Interpreter Lock (GIL)

from nic_vrptw.core.models import Route, RouteSolution, VRPTWInstance

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
        # Number of CPU cores for parallel execution
        max_workers = int(params.get("max_workers", 10))

        nodes = [instance.depot_id] + list(instance.non_depot_ids)
        # Initialize pheromones on all possible edges
        pheromones: dict[tuple[int, int], float] = {
            (i, j): 1.0 for i in nodes for j in nodes if i != j
        }

        best_solution: RouteSolution | None = None
        best_cost = float('inf')

        # Create a process pool once for the entire optimization cycle
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for _ in range(n_iterations):
                # Generate unique seeds for each ant in the current iteration
                ant_seeds = [master_rng.randint(0, 1_000_000) for _ in range(n_ants)]
                
                # Run ants in parallel processes
                # Pass instance data and the current pheromone map to each worker
                futures = [
                    executor.submit(
                        self._build_and_score_ant, 
                        instance, s, pheromones, alpha, beta
                    ) for s in ant_seeds
                ]
                
                # Gather results as they complete
                solutions_in_iter = [f.result() for f in futures]

                # Update the global best solution
                for sol, cost in solutions_in_iter:
                    if cost < best_cost:
                        best_cost = cost
                        best_solution = sol

                # Pheromone evaporation step
                for edge in pheromones:
                    pheromones[edge] *= (1.0 - rho)
                
                # Pheromone deposition step based on solution quality
                for sol, cost in solutions_in_iter:
                    contribution = q_const / cost
                    for route in sol.routes:
                        prev = instance.depot_id
                        for stop in route.stops:
                            pheromones[(prev, stop)] += contribution
                            prev = stop
                        pheromones[(prev, instance.depot_id)] += contribution

        # Return the best found solution or trigger fallback if no solution was found
        return best_solution or self._fallback_solution(instance, seed, params)

    @staticmethod
    def _build_and_score_ant(instance, seed, pheromones, alpha, beta):
        """Helper method for parallel execution inside a separate process."""
        # Each process must have its own independent random generator
        ant_rng = random.Random(seed)
        # Re-instantiate the solver logic for the worker process
        solver = AcoSolver() 
        sol = solver._build_ant_solution(instance, ant_rng, pheromones, alpha, beta)
        cost = solver._calculate_cost(instance, sol)
        return sol, cost

    def _build_ant_solution(self, instance, rng, pheromones, alpha, beta) -> RouteSolution:
        """Constructs a complete solution for one ant by iterating through customers."""
        remaining = set(instance.non_depot_ids)
        routes: list[Route] = []

        while remaining:
            current_node = instance.depot_id
            current_time = 0.0
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

    def _calculate_cost(self, instance, solution: RouteSolution) -> float:
        """Evaluates the total cost of a solution (Distance + Vehicle Penalty)."""
        total_dist = 0.0
        for route in solution.routes:
            prev = instance.depot_id
            for stop in route.stops:
                total_dist += instance.travel_time(prev, stop)
                prev = stop
            total_dist += instance.travel_time(prev, instance.depot_id)
        # Large penalty per vehicle to minimize fleet size (e.g., 10.0 per route)
        return total_dist + (len(solution.routes) * 10.0)

    def _fallback_solution(self, instance, seed, params):
        """Standard greedy fallback if ACO fails to produce a valid solution."""
        from .reference import ReferenceSolver
        return ReferenceSolver().solve(instance, seed, params)
