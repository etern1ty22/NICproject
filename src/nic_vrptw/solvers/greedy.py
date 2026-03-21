from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from nic_vrptw.core.models import Route, RouteSolution, VRPTWInstance


@dataclass
class GreedySolver:
    solver_id: str = "greedy_solver"

    def solve(
        self,
        instance: VRPTWInstance,
        seed: int,
        params: Mapping[str, Any],
    ) -> RouteSolution:
        remaining = set(instance.non_depot_ids)
        routes: list[Route] = []
        max_vehicles = int(params.get("max_vehicles", instance.vehicle.count))

        while remaining and len(routes) < max_vehicles:
            current_node = instance.depot_id
            current_time = max(0.0, instance.customer(instance.depot_id).time_window.start)
            current_load = 0.0
            route_stops: list[int] = []

            while True:
                candidates: list[tuple[float, float, float, int, dict[str, float]]] = []
                for customer_id in sorted(remaining):
                    schedule = _feasible_append(instance, current_node, current_time, current_load, customer_id)
                    if schedule is None:
                        continue
                    candidates.append(
                        (
                            schedule["travel"],
                            schedule["wait"],
                            schedule["slack"],
                            customer_id,
                            schedule,
                        )
                    )

                if not candidates:
                    break

                _, _, _, chosen_id, chosen_schedule = min(candidates, key=lambda item: item[:4])
                route_stops.append(chosen_id)
                remaining.remove(chosen_id)
                current_node = chosen_id
                current_time = chosen_schedule["departure"]
                current_load = chosen_schedule["next_load"]

            if route_stops:
                routes.append(Route(stops=tuple(route_stops)))
                continue

            fallback_customer = _select_fallback_customer(instance, remaining)
            remaining.remove(fallback_customer)
            routes.append(Route(stops=(fallback_customer,)))

        if remaining:
            routes.extend(Route(stops=(customer_id,)) for customer_id in sorted(remaining))

        return RouteSolution(
            routes=tuple(routes),
            solver_id=self.solver_id,
            metadata={
                "seed": seed,
                "params": dict(params),
                "heuristic": "nearest_feasible_neighbor",
            },
        )


def _feasible_append(
    instance: VRPTWInstance,
    current_node: int,
    current_time: float,
    current_load: float,
    customer_id: int,
) -> dict[str, float] | None:
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
        "wait": service_start - arrival,
        "slack": candidate.time_window.end - service_start,
        "departure": departure,
        "next_load": next_load,
    }


def _select_fallback_customer(instance: VRPTWInstance, remaining: set[int]) -> int:
    return min(
        remaining,
        key=lambda customer_id: (
            instance.travel_time(instance.depot_id, customer_id),
            instance.customer(customer_id).time_window.end,
            customer_id,
        ),
    )
