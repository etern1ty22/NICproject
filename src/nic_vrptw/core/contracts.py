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
    # Evaluators stay behind a small protocol so scoring rules can evolve
    # without changing solver integrations or experiment wiring.
    def evaluate(
        self,
        instance: VRPTWInstance,
        solution: RouteSolution,
        objective_mode: str,
    ) -> ScoreRecord:
        ...
