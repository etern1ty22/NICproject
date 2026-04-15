# Final Submission Notes

## Scope Alignment

- Primary final benchmark claim: official Solomon `solomon_c101_official`.
- Final reported method: `aco_solver` with local search (`two_opt`, `relocate`, `swap`).
- Baseline for direct comparison: `greedy_solver` on the same official `C101` benchmark via `configs/greedy_c101_official.yaml`.
- Homberger support remains available through the downloader/loader stack, but it should be described as optional or future work in the final report and slides unless separate artifacts are generated and cited.

## Holdout Story

- `configs/holdout_validation.yaml` is a reference-solver smoke validation path, not evidence for the final ACO method.
- `configs/holdout_validation_aco.yaml` is the ACO-based auxiliary holdout sanity-check path if a small cross-format validation result is needed.
- The main performance claim for the report should still stay on Solomon `C101`.

## Evaluation Metric

- The repository evaluator uses the project hierarchical objective:
- First objective: minimize vehicles used.
- Second objective: minimize travel distance among feasible solutions with the same vehicle count.
- `official_cost` in CSV outputs stores the route distance value that can be cited directly in tables.

## Reproducible Final Pipeline

Run the full final experiment suite from the repository root:

```bash
bash scripts/run_final_suite.sh
```

This will populate stable source-of-truth artifacts at:

- `output/greedy/c101_official/greedy_official_c101.csv`
- `output/aco/final_multiseed_c101/final_multiseed.csv`
- `output/aco/sweep_c101/sweep.csv`
- `output/aco/ablation_local_search_c101/ablation.csv`
- `output/analysis/final/final_analysis_summary.md`

Runtime note:

- The final sweep is intentionally a focused 4-setting grid so the full final suite stays within a practical local turnaround window instead of expanding into a long overnight run.

## Files To Cite In The Report

- Use `output/analysis/final/final_analysis_summary.md` for the concise written summary.
- Use `output/analysis/final/greedy_vs_aco_summary.csv` for the baseline comparison table.
- Use `output/analysis/final/final_multiseed_summary.csv` for the final multi-seed result table.
- Use `output/analysis/final/parameter_sweep_summary.csv` for the tuning section.
- Use `output/analysis/final/local_search_ablation_summary.csv` for the ablation section.

## Verified In This Repository State

- `README.md` now follows the GitHub maintenance rubric with title + one project paragraph.
- `scripts/run_aco_demo.sh` remains the single supported demo path.
- `tests/test_reproducibility.py` now covers `aco_solver` reproducibility on a committed fixture with `max_workers=1`.
- The final analysis step is scripted in `scripts/generate_final_analysis.py` and implemented in `src/nic_vrptw/experiments/analysis.py`.
