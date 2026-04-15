from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from nic_vrptw.core.models import Route, RouteSolution, VRPTWInstance

DEFAULT_VEHICLE_PENALTY = 10.0
EPSILON = 1e-9


@dataclass(frozen=True)
class RouteMetrics:
    distance: float
    finish_time: float
    load: float


@dataclass(frozen=True)
class SolutionMetrics:
    distance: float
    vehicles_used: int
    cost: float


RouteOperator = Callable[[VRPTWInstance, Route], Route]
SolutionOperator = Callable[[VRPTWInstance, RouteSolution, float], RouteSolution]


def evaluate_route(instance: VRPTWInstance, route: Route) -> RouteMetrics | None:
    depot = instance.customer(instance.depot_id)
    current_node = instance.depot_id
    current_time = max(0.0, depot.time_window.start)
    current_load = 0.0
    total_distance = 0.0

    for stop in route.stops:
        if stop == instance.depot_id:
            return None

        customer = instance.customer(stop)
        current_load += customer.demand
        if current_load > instance.vehicle.capacity:
            return None

        travel = instance.travel_time(current_node, stop)
        arrival = current_time + travel
        service_start = max(arrival, customer.time_window.start)
        if service_start > customer.time_window.end:
            return None

        total_distance += travel
        current_time = service_start + customer.service_time
        current_node = stop

    return_to_depot = instance.travel_time(current_node, instance.depot_id)
    finish_time = current_time + return_to_depot
    if finish_time > depot.time_window.end:
        return None

    return RouteMetrics(
        distance=total_distance + return_to_depot,
        finish_time=finish_time,
        load=current_load,
    )


def evaluate_solution(
    instance: VRPTWInstance,
    solution: RouteSolution,
    vehicle_penalty: float = DEFAULT_VEHICLE_PENALTY,
) -> SolutionMetrics | None:
    total_distance = 0.0
    vehicles_used = 0
    seen_stops: set[int] = set()

    for route in solution.routes:
        if not route.stops:
            continue

        route_metrics = evaluate_route(instance, route)
        if route_metrics is None:
            return None

        for stop in route.stops:
            if stop in seen_stops:
                return None
            seen_stops.add(stop)

        total_distance += route_metrics.distance
        vehicles_used += 1

    if vehicles_used > instance.vehicle.count:
        return None

    return SolutionMetrics(
        distance=total_distance,
        vehicles_used=vehicles_used,
        cost=total_distance + (vehicles_used * vehicle_penalty),
    )


def intra_route_two_opt(instance: VRPTWInstance, route: Route) -> Route:
    if len(route.stops) < 2:
        return route

    best_route = route
    best_metrics = evaluate_route(instance, route)
    if best_metrics is None:
        return route

    improved = True
    while improved:
        improved = False
        candidate_route = best_route
        candidate_metrics = best_metrics

        for start in range(len(best_route.stops) - 1):
            for end in range(start + 1, len(best_route.stops)):
                candidate_stops = (
                    best_route.stops[:start]
                    + tuple(reversed(best_route.stops[start : end + 1]))
                    + best_route.stops[end + 1 :]
                )
                tested_route = Route(stops=candidate_stops)
                tested_metrics = evaluate_route(instance, tested_route)
                if tested_metrics is None:
                    continue
                if tested_metrics.distance + 1e-9 >= candidate_metrics.distance:
                    continue

                candidate_route = tested_route
                candidate_metrics = tested_metrics
                improved = True

        best_route = candidate_route
        best_metrics = candidate_metrics

    return best_route


def inter_route_relocate(
    instance: VRPTWInstance,
    solution: RouteSolution,
    vehicle_penalty: float = DEFAULT_VEHICLE_PENALTY,
) -> RouteSolution:
    normalized_solution = _build_solution_like(solution, solution.routes)
    if len(normalized_solution.routes) < 2:
        return normalized_solution

    best_solution = normalized_solution
    best_metrics = evaluate_solution(instance, normalized_solution, vehicle_penalty)
    if best_metrics is None:
        return normalized_solution

    routes = normalized_solution.routes
    for source_idx, source_route in enumerate(routes):
        for source_pos, customer_id in enumerate(source_route.stops):
            reduced_source = (
                source_route.stops[:source_pos] + source_route.stops[source_pos + 1 :]
            )

            for target_idx, target_route in enumerate(routes):
                if target_idx == source_idx:
                    continue

                for insert_pos in range(len(target_route.stops) + 1):
                    candidate_routes = list(routes)
                    candidate_routes[source_idx] = Route(stops=reduced_source)
                    candidate_routes[target_idx] = Route(
                        stops=(
                            target_route.stops[:insert_pos]
                            + (customer_id,)
                            + target_route.stops[insert_pos:]
                        )
                    )
                    candidate_solution = _build_solution_like(solution, candidate_routes)
                    candidate_metrics = evaluate_solution(
                        instance,
                        candidate_solution,
                        vehicle_penalty,
                    )
                    if candidate_metrics is None:
                        continue
                    if candidate_metrics.cost + EPSILON >= best_metrics.cost:
                        continue

                    best_solution = candidate_solution
                    best_metrics = candidate_metrics

    return best_solution


def inter_route_swap(
    instance: VRPTWInstance,
    solution: RouteSolution,
    vehicle_penalty: float = DEFAULT_VEHICLE_PENALTY,
) -> RouteSolution:
    normalized_solution = _build_solution_like(solution, solution.routes)
    if len(normalized_solution.routes) < 2:
        return normalized_solution

    best_solution = normalized_solution
    best_metrics = evaluate_solution(instance, normalized_solution, vehicle_penalty)
    if best_metrics is None:
        return normalized_solution

    routes = normalized_solution.routes
    for left_idx, left_route in enumerate(routes[:-1]):
        for right_idx in range(left_idx + 1, len(routes)):
            right_route = routes[right_idx]

            for left_pos, left_customer in enumerate(left_route.stops):
                for right_pos, right_customer in enumerate(right_route.stops):
                    candidate_routes = list(routes)
                    candidate_routes[left_idx] = Route(
                        stops=(
                            left_route.stops[:left_pos]
                            + (right_customer,)
                            + left_route.stops[left_pos + 1 :]
                        )
                    )
                    candidate_routes[right_idx] = Route(
                        stops=(
                            right_route.stops[:right_pos]
                            + (left_customer,)
                            + right_route.stops[right_pos + 1 :]
                        )
                    )
                    candidate_solution = _build_solution_like(solution, candidate_routes)
                    candidate_metrics = evaluate_solution(
                        instance,
                        candidate_solution,
                        vehicle_penalty,
                    )
                    if candidate_metrics is None:
                        continue
                    if candidate_metrics.cost + EPSILON >= best_metrics.cost:
                        continue

                    best_solution = candidate_solution
                    best_metrics = candidate_metrics

    return best_solution


_ROUTE_OPERATORS: dict[str, RouteOperator] = {
    "two_opt": intra_route_two_opt,
}

_SOLUTION_OPERATORS: dict[str, SolutionOperator] = {
    "two_opt": lambda instance, solution, _: _apply_route_operator(
        instance,
        solution,
        intra_route_two_opt,
    ),
    "relocate": inter_route_relocate,
    "swap": inter_route_swap,
}


def improve_route(
    instance: VRPTWInstance,
    route: Route,
    operators: tuple[str, ...],
) -> Route:
    improved_route = route
    for operator_id in operators:
        try:
            operator = _ROUTE_OPERATORS[operator_id]
        except KeyError as exc:
            available = ", ".join(sorted(_ROUTE_OPERATORS))
            raise ValueError(
                f"Unknown local-search operator '{operator_id}'. Available operators: {available}."
            ) from exc
        improved_route = operator(instance, improved_route)
    return improved_route


def improve_solution(
    instance: VRPTWInstance,
    solution: RouteSolution,
    operators: tuple[str, ...],
    vehicle_penalty: float = DEFAULT_VEHICLE_PENALTY,
    max_passes: int = 1,
) -> RouteSolution:
    if not operators or max_passes < 1:
        return solution

    improved_solution = _build_solution_like(solution, solution.routes)
    passes_run = 0

    for _ in range(max_passes):
        passes_run += 1
        previous_routes = improved_solution.routes

        for operator_id in operators:
            try:
                operator = _SOLUTION_OPERATORS[operator_id]
            except KeyError as exc:
                available = ", ".join(sorted(_SOLUTION_OPERATORS))
                raise ValueError(
                    f"Unknown local-search operator '{operator_id}'. Available operators: {available}."
                ) from exc
            improved_solution = operator(instance, improved_solution, vehicle_penalty)

        if improved_solution.routes == previous_routes:
            break

    metadata = dict(solution.metadata)
    metadata["local_search_operators"] = operators
    metadata["local_search_max_passes"] = max_passes
    metadata["local_search_passes_run"] = passes_run
    metadata["local_search_applied"] = True
    metadata["local_search_improved"] = improved_solution.routes != solution.routes
    return RouteSolution(
        routes=improved_solution.routes,
        solver_id=solution.solver_id,
        metadata=metadata,
    )


def _apply_route_operator(
    instance: VRPTWInstance,
    solution: RouteSolution,
    operator: RouteOperator,
) -> RouteSolution:
    improved_routes = tuple(operator(instance, route) for route in solution.routes)
    return _build_solution_like(solution, improved_routes)


def _build_solution_like(
    solution: RouteSolution,
    routes: Iterable[Route],
) -> RouteSolution:
    normalized_routes = tuple(route for route in routes if route.stops)
    return RouteSolution(
        routes=normalized_routes,
        solver_id=solution.solver_id,
        metadata=solution.metadata,
    )
