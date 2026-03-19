from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from nic_vrptw.core.models import Customer, TimeWindow, VRPTWInstance, VehicleSpec

try:
    import vrplib as _vrplib  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _vrplib = None


def parse_vrplib_instance(path: Path) -> VRPTWInstance:
    # The repository works even when vrplib is unavailable in the local environment.
    # A lightweight built-in reader keeps tests runnable, while the optional dependency
    # remains available for larger external instances.
    return _parse_minimal_vrplib(path)


def _parse_minimal_vrplib(path: Path) -> VRPTWInstance:
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata: dict[str, Any] = {"path": str(path), "parser": "builtin", "vrplib_available": _vrplib is not None}
    scalars: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.upper() == "EOF":
            break
        upper = stripped.upper()
        if upper.endswith("_SECTION"):
            current_section = upper
            sections.setdefault(current_section, [])
            continue
        if current_section is not None:
            looks_like_scalar = ":" in stripped and stripped.split(":", 1)[0].strip().upper() not in {"COMMENT"}
            if looks_like_scalar:
                current_section = None
            else:
                sections[current_section].append(stripped)
                continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            scalars[key.strip().upper()] = value.strip()
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isupper():
            scalars[parts[0].upper()] = parts[1].strip()

    name = scalars.get("NAME", path.stem)
    dimension = int(float(scalars["DIMENSION"]))
    vehicle_count = int(float(scalars.get("VEHICLES", "1")))
    capacity = float(scalars["CAPACITY"])
    vehicle = VehicleSpec(count=vehicle_count, capacity=capacity)

    depot_id = _parse_depot(sections.get("DEPOT_SECTION", []))
    coords = _parse_keyed_pairs(sections.get("NODE_COORD_SECTION", []), minimum_values=2)
    demands = _parse_keyed_pairs(sections.get("DEMAND_SECTION", []), minimum_values=1)
    service_times = _parse_keyed_pairs(sections.get("SERVICE_TIME_SECTION", []), minimum_values=1)
    time_windows = _parse_keyed_pairs(sections.get("TIME_WINDOW_SECTION", []), minimum_values=2)

    global_service_time = float(scalars.get("SERVICE_TIME", "0"))

    customers: list[Customer] = []
    node_ids = tuple(range(1, dimension + 1))
    for node_id in node_ids:
        point = coords.get(node_id)
        customer = Customer(
            customer_id=node_id,
            x=None if point is None else point[0],
            y=None if point is None else point[1],
            demand=demands.get(node_id, [0.0])[0],
            time_window=TimeWindow(
                start=time_windows.get(node_id, [0.0, float("inf")])[0],
                end=time_windows.get(node_id, [0.0, float("inf")])[1],
            ),
            service_time=service_times.get(node_id, [global_service_time])[0],
            is_depot=node_id == depot_id,
        )
        customers.append(customer)

    matrix = _parse_edge_weight_matrix(sections.get("EDGE_WEIGHT_SECTION", []), dimension)
    if matrix is None:
        if not coords:
            raise ValueError(f"VRPLIB instance {path} has neither explicit edge weights nor coordinates.")
        matrix = tuple(
            tuple(_euclidean_distance(source, target) for target in customers)
            for source in customers
        )
        metadata["distance_mode"] = "euclidean"
    else:
        metadata["distance_mode"] = "explicit"

    metadata["asymmetric_allowed"] = True
    metadata["edge_weight_format"] = scalars.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX")
    metadata["type"] = scalars.get("TYPE", "VRPTW")

    return VRPTWInstance(
        name=name,
        source_format="vrplib",
        vehicle=vehicle,
        depot_id=depot_id,
        customers=tuple(customers),
        node_ids=node_ids,
        distance_matrix=matrix,
        metadata=metadata,
    )


def _parse_depot(lines: list[str]) -> int:
    for line in lines:
        if line == "-1":
            break
        value = int(float(line.split()[0]))
        return value
    return 1


def _parse_keyed_pairs(lines: list[str], minimum_values: int) -> dict[int, list[float]]:
    parsed: dict[int, list[float]] = {}
    for line in lines:
        tokens = line.split()
        if len(tokens) < 1 + minimum_values:
            continue
        key = int(float(tokens[0]))
        parsed[key] = [float(token) for token in tokens[1 : 1 + minimum_values]]
    return parsed


def _parse_edge_weight_matrix(lines: list[str], dimension: int) -> tuple[tuple[float, ...], ...] | None:
    if not lines:
        return None
    tokens: list[float] = []
    for line in lines:
        tokens.extend(float(value) for value in line.split())
    expected = dimension * dimension
    if len(tokens) != expected:
        raise ValueError(f"EDGE_WEIGHT_SECTION expected {expected} values, got {len(tokens)}.")
    rows = []
    for row_start in range(0, expected, dimension):
        rows.append(tuple(tokens[row_start : row_start + dimension]))
    return tuple(rows)


def _euclidean_distance(source: Customer, target: Customer) -> float:
    if source.x is None or source.y is None or target.x is None or target.y is None:
        raise ValueError("Coordinates are required to compute Euclidean distance.")
    return math.hypot(source.x - target.x, source.y - target.y)
