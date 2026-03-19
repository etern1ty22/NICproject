from __future__ import annotations

from nic_vrptw.core.contracts import SolverProtocol

from .baseline import NearestFeasibleSolver

_SOLVERS: dict[str, SolverProtocol] = {}


def register_solver(solver: SolverProtocol) -> None:
    _SOLVERS[solver.solver_id] = solver


def unregister_solver(solver_id: str) -> None:
    _SOLVERS.pop(solver_id, None)


def get_solver(solver_id: str) -> SolverProtocol:
    if solver_id not in _SOLVERS:
        available = ", ".join(sorted(_SOLVERS)) or "<none>"
        raise ValueError(f"Unknown solver '{solver_id}'. Available solvers: {available}.")
    return _SOLVERS[solver_id]


def list_solvers() -> tuple[str, ...]:
    return tuple(sorted(_SOLVERS))


register_solver(NearestFeasibleSolver())
