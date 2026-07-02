# Hybrid ACO Solver for VRPTW

Research and optimization project for the Vehicle Routing Problem with Time Windows (VRPTW). The codebase implements reproducible data loaders, benchmark download helpers, a greedy baseline, and a hybrid Ant Colony Optimization (ACO) solver with local search operators (`two_opt`, `relocate`, `swap`).

The final reported experiment scope is centered on the official Solomon `C101` benchmark. Committed mini fixtures and the ORTEC mini holdout support smoke tests, demo validation, and auxiliary sanity checks.

## Portfolio focus

- End-to-end Python package with CLI entry point: `nic-vrptw`.
- Reproducible experiment runner driven by YAML configs.
- Greedy baseline and ACO solver behind a common solver contract.
- Local-search ablation and parameter sweep scripts.
- Dataset manifest/download layer with checksum validation.
- Unit, reproducibility, loader, evaluator, and smoke-test coverage.

## Tech stack

- Python 3.12+
- NumPy
- PyYAML
- Optional: pandas, matplotlib, vrplib
- unittest/pytest-compatible test suite

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,analysis]"
```

Run a smoke experiment:

```bash
nic-vrptw run --config configs/smoke_e2e.yaml
```

Run the greedy baseline:

```bash
nic-vrptw run --config configs/greedy_baseline.yaml
```

Run tests:

```bash
python3 -m unittest discover -s tests
```

If you need to run modules without installing the package first, prefix commands with `PYTHONPATH=src`.

## Final experiment commands

```bash
nic-vrptw run --config configs/greedy_c101_official.yaml
nic-vrptw run --config configs/aco_final_multiseed.yaml
nic-vrptw run --config configs/aco_sweep.yaml
nic-vrptw run --config configs/aco_ablation.yaml
python3 scripts/generate_final_analysis.py
```

Or run the full suite:

```bash
bash scripts/run_final_suite.sh
```

Generated final artifacts are expected under `output/analysis/final/`.

## Supported datasets

- Solomon / Homberger-style coordinate instances.
- ORTEC / VRPLIB-style instances with explicit matrices, including asymmetric travel times.

The full benchmark files are intentionally not committed. The repository ships a download utility with checksum validation and small local fixtures for testing.

Download examples:

```bash
nic-vrptw download --manifest data/manifests/fixtures.yaml --dataset solomon_c101_mini --output-dir data/downloads
nic-vrptw download --manifest data/manifests/benchmarks.yaml --dataset solomon_c101_official --output-dir data/downloads
```

## Project structure

```text
src/nic_vrptw/              package source
src/nic_vrptw/solvers/      greedy, ACO, local search, reference solver
src/nic_vrptw/data/         loaders, validation, downloads, parsers
src/nic_vrptw/experiments/  runner, evaluators, analysis
configs/                    experiment definitions
data/fixtures/              tiny committed fixtures
data/manifests/             portable benchmark manifests
scripts/                    final/demo automation
tests/                      unit, smoke, and reproducibility tests
```

## Report scope

- Primary final benchmark: `solomon_c101_official`.
- Demo path: `configs/aco_demo.yaml` via `scripts/run_aco_demo.sh`.
- Auxiliary validation: mini fixtures and ORTEC mini.
- Homberger download support exists in the manifest, but Homberger scalability is not presented as a completed final experiment in this repository.

Detailed final-submission notes live in `docs/FINAL_SUBMISSION_NOTES.md`.
