from __future__ import annotations

from dataclasses import dataclass

from nic_vrptw.core.contracts import EvaluatorProtocol
from nic_vrptw.core.models import RouteSolution, ScoreRecord, VRPTWInstance

from .evaluator import evaluate_solution

_EVALUATORS: dict[str, EvaluatorProtocol] = {}


@dataclass
class DefaultEvaluator:
    evaluator_id: str = "default_evaluator"

    def evaluate(
        self,
        instance: VRPTWInstance,
        solution: RouteSolution,
        objective_mode: str,
    ) -> ScoreRecord:
        return evaluate_solution(instance, solution, objective_mode)


def register_evaluator(evaluator: EvaluatorProtocol) -> None:
    evaluator_id = getattr(evaluator, "evaluator_id", None)
    if not evaluator_id:
        raise ValueError("Evaluator must define evaluator_id.")
    _EVALUATORS[evaluator_id] = evaluator


def unregister_evaluator(evaluator_id: str) -> None:
    _EVALUATORS.pop(evaluator_id, None)


def get_evaluator(evaluator_id: str) -> EvaluatorProtocol:
    if evaluator_id not in _EVALUATORS:
        available = ", ".join(sorted(_EVALUATORS)) or "<none>"
        raise ValueError(f"Unknown evaluator '{evaluator_id}'. Available evaluators: {available}.")
    return _EVALUATORS[evaluator_id]


def list_evaluators() -> tuple[str, ...]:
    return tuple(sorted(_EVALUATORS))


register_evaluator(DefaultEvaluator())
