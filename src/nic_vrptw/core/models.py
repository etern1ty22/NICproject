from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TimeWindow:
    start: float
    end: float


@dataclass(frozen=True)
class Customer:
    customer_id: int
    x: float | None
    y: float | None
    demand: float
    time_window: TimeWindow
    service_time: float
    is_depot: bool = False


@dataclass(frozen=True)
class VehicleSpec:
    count: int
    capacity: float


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    path: Path
    format: str | None = None
    role: str = "tuning"


@dataclass(frozen=True)
class Route:
    # Stops contain customer ids only. Depot insertion is handled by the evaluator.
    stops: tuple[int, ...]


@dataclass(frozen=True)
class RouteSolution:
    routes: tuple[Route, ...]
    solver_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreRecord:
    # `official_cost` is kept separate from the hierarchical project objective so
    # recent validation datasets can report their native metric without changing
    # the rest of the experiment pipeline.
    feasible: bool
    vehicles_used: int
    distance: float
    official_cost: float
    objective_mode: str
    objective_value: tuple[float, ...]
    violations: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_dir: Path
    solver_id: str
    evaluator_id: str
    objective_mode: str
    seed_set: tuple[int, ...]
    datasets: tuple[DatasetSpec, ...]
    solver_params: Mapping[str, Any]
    sweeps: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    config_id: str
    dataset_id: str
    instance_name: str
    instance_format: str
    role: str
    seed: int
    solver_id: str
    evaluator_id: str
    feasible: bool
    vehicles_used: int
    distance: float
    official_cost: float
    runtime_s: float
    objective_mode: str
    params: Mapping[str, Any]
    violations: tuple[str, ...]

    def to_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_id": self.config_id,
            "dataset": self.dataset_id,
            "instance_id": self.instance_name,
            "instance_format": self.instance_format,
            "role": self.role,
            "seed": self.seed,
            "solver_id": self.solver_id,
            "evaluator_id": self.evaluator_id,
            "feasible": self.feasible,
            "vehicles_used": self.vehicles_used,
            "distance": round(self.distance, 6),
            "official_cost": round(self.official_cost, 6),
            "runtime_s": round(self.runtime_s, 6),
            "objective_mode": self.objective_mode,
            "params": dict(self.params),
            "violations": list(self.violations),
        }


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    fingerprint: str | None = None


@dataclass(frozen=True)
class VRPTWInstance:
    # All dataset adapters must normalize their raw files into this schema.
    # Downstream solver code should not care whether the source was Solomon,
    # Homberger, or ORTEC/VRPLIB once this object is constructed.
    name: str
    source_format: str
    vehicle: VehicleSpec
    depot_id: int
    customers: tuple[Customer, ...]
    node_ids: tuple[int, ...]
    distance_matrix: tuple[tuple[float, ...], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def customer_map(self) -> dict[int, Customer]:
        return {customer.customer_id: customer for customer in self.customers}

    def index_map(self) -> dict[int, int]:
        return {node_id: idx for idx, node_id in enumerate(self.node_ids)}

    def customer(self, customer_id: int) -> Customer:
        return self.customer_map()[customer_id]

    def travel_time(self, from_customer: int, to_customer: int) -> float:
        index = self.index_map()
        return float(self.distance_matrix[index[from_customer]][index[to_customer]])

    @property
    def non_depot_ids(self) -> tuple[int, ...]:
        return tuple(node_id for node_id in self.node_ids if node_id != self.depot_id)

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload
