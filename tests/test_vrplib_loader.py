from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
from nic_vrptw.data.loader import load_instance
from nic_vrptw.data.validation import validate_instance


ROOT = Path(__file__).resolve().parents[1]


class VrplibLoaderTests(unittest.TestCase):
    def test_ortec_fixture_parses_and_validates(self) -> None:
        instance = load_instance(ROOT / "data/fixtures/ortec/ORTEC-mini.vrp")
        report = validate_instance(instance)

        self.assertEqual(instance.source_format, "vrplib")
        self.assertEqual(instance.depot_id, 1)
        self.assertTrue(report.valid, msg=report.errors)
        self.assertNotEqual(instance.travel_time(1, 2), instance.travel_time(2, 1))


if __name__ == "__main__":
    unittest.main()
