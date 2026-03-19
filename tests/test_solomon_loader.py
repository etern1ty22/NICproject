from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
from nic_vrptw.data.loader import fingerprint_instance, load_instance
from nic_vrptw.data.validation import validate_instance


ROOT = Path(__file__).resolve().parents[1]


class SolomonLoaderTests(unittest.TestCase):
    def test_solomon_fixture_parses_and_validates(self) -> None:
        instance = load_instance(ROOT / "data/fixtures/solomon/C101-mini.txt")
        report = validate_instance(instance)

        self.assertEqual(instance.source_format, "solomon")
        self.assertEqual(instance.depot_id, 0)
        self.assertEqual(instance.vehicle.count, 2)
        self.assertTrue(report.valid, msg=report.errors)
        self.assertEqual(instance.non_depot_ids, (1, 2, 3, 4))

    def test_solomon_fingerprint_is_deterministic(self) -> None:
        left = load_instance(ROOT / "data/fixtures/solomon/C101-mini.txt")
        right = load_instance(ROOT / "data/fixtures/solomon/C101-mini.txt")
        self.assertEqual(fingerprint_instance(left), fingerprint_instance(right))


if __name__ == "__main__":
    unittest.main()
