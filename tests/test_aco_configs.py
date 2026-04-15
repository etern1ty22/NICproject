from pathlib import Path
import sys
import tempfile
import unittest

import yaml

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import _bootstrap  # noqa: F401
from nic_vrptw.experiments.runner import _load_run_config, run_experiments


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "data/downloads/c101.txt"
CONFIG_EXPECTATIONS = {
    "configs/aco_final_multiseed.yaml": {
        "name": "aco_final_multiseed_c101",
        "output_dir": "output/aco/final_multiseed_c101",
        "output_basename": "final_multiseed",
        "seed_count": 5,
        "sweeps": set(),
    },
    "configs/aco_sweep.yaml": {
        "name": "aco_sweep_c101",
        "output_dir": "output/aco/sweep_c101",
        "output_basename": "sweep",
        "seed_count": 3,
        "sweeps": {"alpha", "beta", "rho"},
        "grid_size": 4,
    },
    "configs/aco_ablation.yaml": {
        "name": "aco_ablation_local_search_c101",
        "output_dir": "output/aco/ablation_local_search_c101",
        "output_basename": "ablation",
        "seed_count": 3,
        "sweeps": {"local_search_operators"},
        "grid_size": 3,
    },
}


class AcoConfigTests(unittest.TestCase):
    def test_final_aco_configs_parse_and_point_to_manifest_benchmark(self) -> None:
        manifest = yaml.safe_load((ROOT / "data/manifests/benchmarks.yaml").read_text(encoding="utf-8"))
        benchmark_filenames = {
            spec["filename"]
            for spec in manifest["datasets"].values()
        }

        for relative_path, expected in CONFIG_EXPECTATIONS.items():
            config_path = ROOT / relative_path
            config = _load_run_config(config_path)

            self.assertEqual(config.name, expected["name"])
            self.assertEqual(config.solver_id, "aco_solver")
            self.assertEqual(config.evaluator_id, "default_evaluator")
            self.assertEqual(config.output_dir, (ROOT / expected["output_dir"]).resolve())
            self.assertEqual(config.output_basename, expected["output_basename"])
            self.assertEqual(len(config.seed_set), expected["seed_count"])
            self.assertEqual(set(config.sweeps.keys()), expected["sweeps"])
            self.assertEqual(config.solver_params["local_search_scope"], "iteration_best")
            if "grid_size" in expected:
                grid_size = 1
                for values in config.sweeps.values():
                    grid_size *= len(values)
                self.assertEqual(grid_size, expected["grid_size"])
            self.assertEqual(len(config.datasets), 1)

            dataset = config.datasets[0]
            self.assertEqual(dataset.dataset_id, "solomon_c101_official")
            self.assertEqual(dataset.path, BENCHMARK_PATH.resolve())
            self.assertEqual(dataset.path.name, "c101.txt")
            self.assertIn(dataset.path.name, benchmark_filenames)
            self.assertEqual(dataset.format, "solomon")

    @unittest.skipUnless(BENCHMARK_PATH.exists(), "official Solomon C101 benchmark is not downloaded")
    def test_final_multiseed_config_smoke_runs_when_benchmark_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            records = run_experiments(
                ROOT / "configs/aco_final_multiseed.yaml",
                output_dir=output_dir,
                param_overrides={
                    "n_ants": 1,
                    "n_iterations": 1,
                    "max_workers": 1,
                    "local_search_max_passes": 1,
                },
            )

        self.assertEqual(len(records), 5)
        self.assertTrue(all(record.solver_id == "aco_solver" for record in records))
        self.assertTrue(all(record.dataset_id == "solomon_c101_official" for record in records))

    def test_greedy_official_config_targets_same_benchmark_for_baseline_comparison(self) -> None:
        config = _load_run_config(ROOT / "configs/greedy_c101_official.yaml")

        self.assertEqual(config.name, "greedy_c101_official")
        self.assertEqual(config.solver_id, "greedy_solver")
        self.assertEqual(config.output_basename, "greedy_official_c101")
        self.assertEqual(config.output_dir, (ROOT / "output/greedy/c101_official").resolve())
        self.assertEqual(config.seed_set, (17,))
        self.assertEqual(len(config.datasets), 1)
        self.assertEqual(config.datasets[0].dataset_id, "solomon_c101_official")
        self.assertEqual(config.datasets[0].path, BENCHMARK_PATH.resolve())


if __name__ == "__main__":
    unittest.main()
