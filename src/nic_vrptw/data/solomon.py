from __future__ import annotations

import math
from pathlib import Path

from nic_vrptw.core.models import Customer, TimeWindow, VRPTWInstance, VehicleSpec


def parse_solomon_instance(path: Path) -> VRPTWInstance:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 5:
        raise ValueError(f"Solomon instance {path} is too short.")

    name = lines[0]
    vehicle_index = _find_line(lines, "NUMBER")
    if vehicle_index is None or vehicle_index + 1 >= len(lines):
        raise ValueError(f"Vehicle specification not found in {path}.")

    vehicle_tokens = lines[vehicle_index + 1].split()
    if len(vehicle_tokens) < 2:
        raise ValueError(f"Vehicle line is malformed in {path}.")

    vehicle = VehicleSpec(count=int(float(vehicle_tokens[0])), capacity=float(vehicle_tokens[1]))
    customer_index = _find_customer_header(lines)
    if customer_index is None:
        raise ValueError(f"Customer section not found in {path}.")

    customers: list[Customer] = []
    for raw_line in lines[customer_index + 1 :]:
        tokens = raw_line.split()
        if len(tokens) < 7:
            continue
        try:
            values = [float(token) for token in tokens[:7]]
        except ValueError:
            continue

        customer_id = int(values[0])
        customer = Customer(
            customer_id=customer_id,
            x=values[1],
            y=values[2],
            demand=values[3],
            time_window=TimeWindow(start=values[4], end=values[5]),
            service_time=values[6],
            is_depot=customer_id == 0,
        )
        customers.append(customer)

    if not customers:
        raise ValueError(f"No customers parsed from {path}.")

    customers = sorted(customers, key=lambda customer: customer.customer_id)
    node_ids = tuple(customer.customer_id for customer in customers)
    distance_matrix = tuple(
        tuple(_euclidean_distance(source, target) for target in customers)
        for source in customers
    )
    depot_id = next((customer.customer_id for customer in customers if customer.is_depot), customers[0].customer_id)

    return VRPTWInstance(
        name=name,
        source_format="solomon",
        vehicle=vehicle,
        depot_id=depot_id,
        customers=tuple(customers),
        node_ids=node_ids,
        distance_matrix=distance_matrix,
        metadata={
            "path": str(path),
            "distance_mode": "euclidean",
            "asymmetric_allowed": False,
        },
    )


def _find_line(lines: list[str], marker: str) -> int | None:
    marker = marker.upper()
    for index, line in enumerate(lines):
        if marker in line.upper():
            return index
    return None


def _find_customer_header(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        upper = line.upper()
        if "CUST" in upper or "CUSTOMER" in upper:
            return index
    return None


def _euclidean_distance(source: Customer, target: Customer) -> float:
    if source.x is None or source.y is None or target.x is None or target.y is None:
        raise ValueError("Solomon instances require coordinates for every node.")
    return math.hypot(source.x - target.x, source.y - target.y)
