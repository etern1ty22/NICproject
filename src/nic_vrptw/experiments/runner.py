from __future__ import annotations

import csv
import itertools
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import yaml

from nic_vrptw.core.models import DatasetSpec, RouteSolution, RunConfig, RunRecord
from nic_vrptw.core.utils import make_json_safe, stable_digest
from nic_vrptw.data.loader import load_instance
from nic_vrptw.data.validation import validate_instance
from nic_vrptw.experiments.evaluators import get_evaluator
from nic_vrptw.solvers import get_solver


def run_experiments(
    config_path: str | Path,
    solver_id: str | None = None,
    evaluator_id: str | None = None,
    output_dir: str | Path | None = None,
    param_overrides: Mapping[str, Any] | None = None,
) -> list[RunRecord]:
    # This is the integration point for the whole team: loader -> solver ->
    # evaluator -> CSV/JSON artifacts. New solver modules should plug in here
    # through the registry instead of adding custom experiment entrypoints.
    config_path = Path(config_path)
    config = _load_run_config(
        config_path,
        solver_id=solver_id,
        evaluator_id=evaluator_id,
        output_dir=output_dir,
    )
    resolved_params = dict(config.solver_params)
    if param_overrides:
        resolved_params.update(param_overrides)
    config = RunConfig(
        name=config.name,
        output_dir=config.output_dir,
        output_basename=config.output_basename,
        solver_id=config.solver_id,
        evaluator_id=config.evaluator_id,
        objective_mode=config.objective_mode,
        seed_set=config.seed_set,
        datasets=config.datasets,
        solver_params=resolved_params,
        sweeps=config.sweeps,
        metadata=config.metadata,
    )

    solver = get_solver(config.solver_id)
    evaluator = get_evaluator(config.evaluator_id)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    config_id = stable_digest(make_json_safe(asdict(config)))
    parameter_sets = list(_iter_parameter_sets(config.solver_params, config.sweeps))
    records: list[RunRecord] = []

    for dataset in config.datasets:
        # Validation happens before the expensive part so malformed external
        # datasets fail fast and do not pollute experiment logs.
        instance = load_instance(dataset.path, format=dataset.format)
        report = validate_instance(instance)
        if not report.valid:
            details = "; ".join(report.errors)
            raise ValueError(f"Instance {dataset.path} is invalid: {details}")

        for params in parameter_sets:
            for seed in config.seed_set:
                start = perf_counter()
                solution = solver.solve(instance=instance, seed=seed, params=params)
                _ensure_solution_contract(solution)
                score = evaluator.evaluate(instance, solution, config.objective_mode)
                runtime_s = perf_counter() - start

                run_payload = {
                    "config_id": config_id,
                    "dataset": dataset.dataset_id,
                    "seed": seed,
                    "solver_id": config.solver_id,
                    "params": make_json_safe(params),
                }
                run_id = stable_digest(run_payload)[:12]
                records.append(
                    RunRecord(
                        run_id=run_id,
                        config_id=config_id,
                        dataset_id=dataset.dataset_id,
                        instance_name=instance.name,
                        instance_format=instance.source_format,
                        role=dataset.role,
                        seed=seed,
                        solver_id=config.solver_id,
                        evaluator_id=config.evaluator_id,
                        feasible=score.feasible,
                        vehicles_used=score.vehicles_used,
                        distance=score.distance,
                        official_cost=score.official_cost,
                        runtime_s=runtime_s,
                        objective_mode=score.objective_mode,
                        params=params,
                        violations=score.violations,
                    )
                )

    _write_outputs(config, config_id, records)
    return records


def _load_run_config(
    config_path: Path,
    solver_id: str | None = None,
    evaluator_id: str | None = None,
    output_dir: str | Path | None = None,
) -> RunConfig:
    # Config paths are resolved relative to the repository root so anyone can
    # move configs around inside `configs/` without changing runtime code.
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    datasets = tuple(
        DatasetSpec(
            dataset_id=item["id"],
            path=(config_path.parent.parent / item["path"]).resolve(),
            format=item.get("format"),
            role=item.get("role", "tuning"),
        )
        for item in payload.get("datasets", [])
    )
    sweeps = {
        key: tuple(values)
        for key, values in (payload.get("sweeps") or {}).items()
    }

    return RunConfig(
        name=payload["name"],
        output_dir=(Path(output_dir) if output_dir is not None else (config_path.parent.parent / payload["output_dir"]).resolve()),
        output_basename=payload.get("output_basename"),
        solver_id=solver_id or payload["solver"]["name"],
        evaluator_id=evaluator_id or payload.get("evaluator", {}).get("name", "default_evaluator"),
        objective_mode=payload.get("objective_mode", "hierarchical"),
        seed_set=tuple(int(seed) for seed in payload.get("seed_set", [0])),
        datasets=datasets,
        solver_params=dict(payload["solver"].get("params", {})),
        sweeps=sweeps,
        metadata=dict(payload.get("metadata", {})),
    )


def _iter_parameter_sets(
    base_params: Mapping[str, Any],
    sweeps: Mapping[str, tuple[Any, ...]],
) -> list[dict[str, Any]]:
    if not sweeps:
        return [dict(base_params)]
    keys = list(sweeps.keys())
    combinations = itertools.product(*(sweeps[key] for key in keys))
    parameter_sets: list[dict[str, Any]] = []
    for combination in combinations:
        params = dict(base_params)
        params.update(dict(zip(keys, combination, strict=True)))
        parameter_sets.append(params)
    return parameter_sets


def _ensure_solution_contract(solution: RouteSolution) -> None:
    if not isinstance(solution, RouteSolution):
        raise ValueError("Solver contract violation: solve() must return RouteSolution.")
    # Contract checks stay intentionally narrow: free to attach
    # extra solver metadata as long as the route structure remains stable.
    for route in solution.routes:
        if not all(isinstance(stop, int) for stop in route.stops):
            raise ValueError("Solver contract violation: route stops must be integer customer ids.")


def _write_outputs(config: RunConfig, config_id: str, records: list[RunRecord]) -> None:
    artifact_paths = _resolve_output_paths(config, config_id)
    csv_path = artifact_paths["csv"]
    json_path = artifact_paths["json"]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(records[0].to_row().keys()) if records else [
            "run_id",
            "config_id",
            "dataset",
            "instance_id",
            "instance_format",
            "role",
            "seed",
            "solver_id",
            "evaluator_id",
            "feasible",
            "vehicles_used",
            "distance",
            "official_cost",
            "runtime_s",
            "objective_mode",
            "params",
            "violations",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())

    metadata = {
        "name": config.name,
        "config_id": config_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "solver_id": config.solver_id,
        "evaluator_id": config.evaluator_id,
        "objective_mode": config.objective_mode,
        "seed_set": list(config.seed_set),
        "records_count": len(records),
        "output_basename": config.output_basename,
        "datasets": [make_json_safe(asdict(dataset)) for dataset in config.datasets],
        "solver_params": make_json_safe(config.solver_params),
        "sweeps": make_json_safe(config.sweeps),
        "metadata": make_json_safe(config.metadata),
    }
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    stable_csv_path = artifact_paths.get("stable_csv")
    stable_json_path = artifact_paths.get("stable_json")
    if stable_csv_path is not None:
        shutil.copyfile(csv_path, stable_csv_path)
    if stable_json_path is not None:
        shutil.copyfile(json_path, stable_json_path)


def _resolve_output_paths(config: RunConfig, config_id: str) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "csv": config.output_dir / f"{config.name}-{config_id[:12]}.csv",
        "json": config.output_dir / f"{config.name}-{config_id[:12]}.json",
    }
    if config.output_basename:
        paths["stable_csv"] = config.output_dir / f"{config.output_basename}.csv"
        paths["stable_json"] = config.output_dir / f"{config.output_basename}.json"
    return paths
