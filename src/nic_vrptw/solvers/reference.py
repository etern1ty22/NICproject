from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

from nic_vrptw.core.models import Route, RouteSolution, VRPTWInstance


@dataclass
class ReferenceSolver:
    solver_id: str = "reference_solver"

    def solve(
        self,
        instance: VRPTWInstance,
        seed: int,
        params: Mapping[str, Any],
    ) -> RouteSolution:
        # This solver exists only as a reference implementation of the contract
        # and a smoke-test target for the runner until plugging in the
        # actual optimization modules.
        rng = random.Random(seed)
        remaining = set(instance.non_depot_ids)
        routes: list[Route] = []

        distance_weight = float(params.get("distance_weight", 1.0))
        wait_weight = float(params.get("wait_weight", 0.2))
        slack_weight = float(params.get("slack_weight", 0.05))
        max_vehicles = int(params.get("max_vehicles", instance.vehicle.count))

        while remaining and len(routes) < max_vehicles:
            current_node = instance.depot_id
            current_time = max(0.0, instance.customer(instance.depot_id).time_window.start)
            current_load = 0.0
            route_stops: list[int] = []

            while True:
                candidates: list[tuple[float, int, dict[str, float]]] = []
                for customer_id in sorted(remaining):
                    schedule = _feasible_append(instance, current_node, current_time, current_load, customer_id)
                    if schedule is None:
                        continue
                    score = (
                        distance_weight * schedule["travel"]
                        + wait_weight * schedule["wait"]
                        + slack_weight * schedule["slack"]
                        + rng.random() * 1e-6
                    )
                    candidates.append((score, customer_id, schedule))

                if not candidates:
                    break

                _, chosen_id, chosen_schedule = min(candidates, key=lambda item: item[0])
                route_stops.append(chosen_id)
                remaining.remove(chosen_id)
                current_node = chosen_id
                current_time = chosen_schedule["departure"]
                current_load += instance.customer(chosen_id).demand

            if route_stops:
                routes.append(Route(stops=tuple(route_stops)))
            else:
                customer_id = min(remaining)
                remaining.remove(customer_id)
                routes.append(Route(stops=(customer_id,)))

        if remaining:
            routes.extend(Route(stops=(customer_id,)) for customer_id in sorted(remaining))

        return RouteSolution(
            routes=tuple(routes),
            solver_id=self.solver_id,
            metadata={"seed": seed, "params": dict(params)},
        )


def _feasible_append(
    instance: VRPTWInstance,
    current_node: int,
    current_time: float,
    current_load: float,
    customer_id: int,
) -> dict[str, float] | None:
    # The helper returns only the scheduling data this reference implementation
    # needs. Solvers can reuse the idea without matching this helper.
    candidate = instance.customer(customer_id)
    depot = instance.customer(instance.depot_id)
    next_load = current_load + candidate.demand
    if next_load > instance.vehicle.capacity:
        return None

    travel = instance.travel_time(current_node, customer_id)
    arrival = current_time + travel
    service_start = max(arrival, candidate.time_window.start)
    if service_start > candidate.time_window.end:
        return None

    departure = service_start + candidate.service_time
    return_home = departure + instance.travel_time(customer_id, instance.depot_id)
    if return_home > depot.time_window.end:
        return None

    return {
        "travel": travel,
        "arrival": arrival,
        "wait": service_start - arrival,
        "departure": departure,
        "slack": candidate.time_window.end - service_start,
        "next_load": next_load,
    }
