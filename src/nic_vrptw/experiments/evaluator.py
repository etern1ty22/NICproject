from __future__ import annotations

from nic_vrptw.core.models import RouteSolution, ScoreRecord, VRPTWInstance


def evaluate_solution(
    instance: VRPTWInstance,
    solution: RouteSolution,
    objective_mode: str,
) -> ScoreRecord:
    customer_map = instance.customer_map()
    depot = customer_map[instance.depot_id]
    seen_counts: dict[int, int] = {customer_id: 0 for customer_id in instance.non_depot_ids}
    violations: list[str] = []
    total_distance = 0.0
    vehicles_used = 0

    for route_index, route in enumerate(solution.routes, start=1):
        if not route.stops:
            continue
        vehicles_used += 1
        current_node = instance.depot_id
        current_time = max(0.0, depot.time_window.start)
        current_load = 0.0

        for stop in route.stops:
            if stop == instance.depot_id:
                violations.append(f"route {route_index} contains depot as a customer stop")
                continue
            if stop not in customer_map:
                violations.append(f"route {route_index} references unknown customer {stop}")
                continue

            customer = customer_map[stop]
            travel = instance.travel_time(current_node, stop)
            arrival = current_time + travel
            service_start = max(arrival, customer.time_window.start)
            if service_start > customer.time_window.end:
                violations.append(f"customer {stop} is served outside its time window")
            current_load += customer.demand
            if current_load > instance.vehicle.capacity:
                violations.append(f"route {route_index} exceeds vehicle capacity")

            seen_counts[stop] += 1
            total_distance += travel
            current_time = service_start + customer.service_time
            current_node = stop

        total_distance += instance.travel_time(current_node, instance.depot_id)
        return_time = current_time + instance.travel_time(current_node, instance.depot_id)
        if return_time > depot.time_window.end:
            violations.append(f"route {route_index} returns to depot after its due time")

    missing = [customer_id for customer_id, count in seen_counts.items() if count == 0]
    duplicated = [customer_id for customer_id, count in seen_counts.items() if count > 1]
    if missing:
        violations.append(f"missing customers: {missing}")
    if duplicated:
        violations.append(f"duplicate customers: {duplicated}")
    if vehicles_used > instance.vehicle.count:
        violations.append("solution uses more vehicles than allowed")

    objective_mode = objective_mode.lower()
    if objective_mode == "hierarchical":
        objective_value = (float(vehicles_used), float(total_distance))
    elif objective_mode == "distance_only":
        objective_value = (float(total_distance),)
    else:
        raise ValueError(f"Unsupported objective mode: {objective_mode}")

    return ScoreRecord(
        feasible=not violations,
        vehicles_used=vehicles_used,
        distance=total_distance,
        official_cost=total_distance,
        objective_mode=objective_mode,
        objective_value=objective_value,
        violations=tuple(violations),
        metadata={"solver_id": solution.solver_id},
    )
