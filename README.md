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

3. Validate a fixture instance:

```bash
nic-vrptw validate-instance --path data/fixtures/solomon/C101-mini.txt
```

4. Run tests from the repository root:

```bash
python3 -m unittest discover -s tests
```

If you need to run modules without installing the package first, prefix commands with `PYTHONPATH=src`.

## Layout

- `src/nic_vrptw/`: the only project package with application code.
- `configs/`: experiment definitions.
- `data/fixtures/`: tiny committed fixtures for both supported formats.
- `data/manifests/`: download manifest examples and test fixtures.
- `tests/`: unit, reproducibility, and smoke tests.

## Supported datasets

- Solomon / Homberger-style coordinate instances.
- ORTEC / VRPLIB-style instances with explicit matrices, including asymmetric travel times.

The full benchmark files are intentionally not committed. The repository ships a download utility with checksum validation and small local fixtures for testing.
