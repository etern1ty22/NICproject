from __future__ import annotations

import argparse
import json
from pathlib import Path

from nic_vrptw.data.downloads import download_dataset
from nic_vrptw.data.loader import load_instance
from nic_vrptw.data.validation import validate_instance
from nic_vrptw.experiments.runner import run_experiments
from nic_vrptw.solvers import list_solvers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nic-vrptw")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute experiments from a YAML config.")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--solver")
    run_parser.add_argument("--output-dir")
    run_parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Override solver params, for example distance_weight=1.2",
    )

    validate_parser = subparsers.add_parser("validate-instance", help="Validate a single instance file.")
    validate_parser.add_argument("--path", required=True)
    validate_parser.add_argument("--format")

    download_parser = subparsers.add_parser("download", help="Download a dataset from a manifest.")
    download_parser.add_argument("--manifest", required=True)
    download_parser.add_argument("--dataset", required=True)
    download_parser.add_argument("--output-dir", required=True)

    solvers_parser = subparsers.add_parser("list-solvers", help="List available solver plugins.")

    args = parser.parse_args(argv)

    if args.command == "run":
        overrides = {_key: _parse_scalar(_value) for _key, _value in (_split_param(item) for item in args.param)}
        records = run_experiments(
            config_path=args.config,
            solver_id=args.solver,
            output_dir=args.output_dir,
            param_overrides=overrides,
        )
        print(json.dumps({"records": len(records)}, indent=2))
        return 0

    if args.command == "validate-instance":
        instance = load_instance(args.path, format=args.format)
        report = validate_instance(instance)
        print(json.dumps({
            "name": instance.name,
            "format": instance.source_format,
            "valid": report.valid,
            "errors": report.errors,
            "warnings": report.warnings,
            "fingerprint": report.fingerprint,
        }, indent=2))
        return 0 if report.valid else 1

    if args.command == "download":
        output_path = download_dataset(args.manifest, args.dataset, args.output_dir)
        print(json.dumps({"path": str(output_path)}, indent=2))
        return 0

    if args.command == "list-solvers":
        print(json.dumps({"solvers": list(list_solvers())}, indent=2))
        return 0

    return 1


def _split_param(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Invalid --param value: {raw}")
    key, value = raw.split("=", 1)
    return key, value


def _parse_scalar(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


if __name__ == "__main__":
    raise SystemExit(main())
