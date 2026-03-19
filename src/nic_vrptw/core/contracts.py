from __future__ import annotations

from typing import Any, Mapping, Protocol

from .models import RouteSolution, ScoreRecord, VRPTWInstance


class SolverProtocol(Protocol):
    # New solvers should implement only this surface.
    # The runner treats everything else as internal solver-specific logic.
    solver_id: str

    def solve(
        self,
        instance: VRPTWInstance,
        seed: int,
        params: Mapping[str, Any],
    ) -> RouteSolution:
        ...


class EvaluatorProtocol(Protocol):
    # This contract exists so the current scoring shim can later be replaced
    # by the dedicated evaluator module without changing solver integrations.
    def evaluate(
        self,
        instance: VRPTWInstance,
        solution: RouteSolution,
        objective_mode: str,
    ) -> ScoreRecord:
        ...
