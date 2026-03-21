## Quick start

1. Create a virtual environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

2. Run the smoke experiment after installation:

```bash
nic-vrptw run --config configs/smoke_e2e.yaml
```

Greedy baseline is available via:

```bash
nic-vrptw run --config configs/greedy_baseline.yaml
```

3. Validate a fixture instance:

```bash
nic-vrptw validate-instance --path data/fixtures/solomon/C101-mini.txt
```

4. Run tests from the repository root:

```bash
python3 -m unittest discover -s tests
```

If you need to run modules without installing the package first, prefix commands with `PYTHONPATH=src`.

5. Download portable benchmark data examples:

```bash
nic-vrptw download --manifest data/manifests/fixtures.yaml --dataset solomon_c101_mini --output-dir data/downloads
nic-vrptw download --manifest data/manifests/benchmarks.yaml --dataset solomon_c101_official --output-dir data/downloads
```

## Layout

- `src/nic_vrptw/`: the only project package with application code.
- `configs/`: experiment definitions.
- `data/fixtures/`: tiny committed fixtures for both supported formats.
- `data/manifests/`: portable fixture manifest plus official benchmark download manifest.
- `tests/`: unit, reproducibility, and smoke tests.

## Supported datasets

- Solomon / Homberger-style coordinate instances.
- ORTEC / VRPLIB-style instances with explicit matrices, including asymmetric travel times.

The full benchmark files are intentionally not committed. The repository ships a download utility with checksum validation and small local fixtures for testing.

Official benchmark sources wired into the manifest:
- Solomon 100-customer benchmark from SINTEF TOP.
- Homberger 200-customer benchmark from SINTEF TOP.
- ORTEC VRPTW instance from the EURO Meets NeurIPS 2022 quickstart repository.

Integration notes:
- New solvers should register through `src/nic_vrptw/solvers/__init__.py`.
- New evaluators should register through `src/nic_vrptw/experiments/evaluators.py`.
- `greedy_solver` and `aco_solver` are integrated project solvers; `reference_solver` remains a smoke-test fallback.
