import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "preprocessing" / "data_audit.py"
SPEC = importlib.util.spec_from_file_location("data_audit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DataAuditTest(unittest.TestCase):
    def test_haversine_equator(self):
        distance = MODULE.haversine_m(np.array([[0.0, 0.0], [0.001, 0.0]]))[0]
        self.assertAlmostEqual(distance, 111.195, places=2)

    def test_empty_quantiles(self):
        self.assertIsNone(MODULE.quantiles([])["p50"])

    def test_quantiles(self):
        self.assertEqual(MODULE.quantiles([1, 2, 3])["p50"], 2.0)


if __name__ == "__main__":
    unittest.main()
