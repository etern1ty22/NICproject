from __future__ import annotations

from typing import Any, Mapping, Protocol

from .models import RouteSolution, ScoreRecord, VRPTWInstance


class SolverProtocol(Protocol):
    solver_id: str

    def solve(
        self,
        instance: VRPTWInstance,
        seed: int,
        params: Mapping[str, Any],
    ) -> RouteSolution:
        ...


class EvaluatorProtocol(Protocol):
    def evaluate(
        self,
        instance: VRPTWInstance,
        solution: RouteSolution,
        objective_mode: str,
    ) -> ScoreRecord:
        ...
