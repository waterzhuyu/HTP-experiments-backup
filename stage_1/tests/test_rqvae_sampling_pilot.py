import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "evaluation" / "rqvae_sampling_pilot.py"
SPEC = importlib.util.spec_from_file_location("pilot", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MetricsTest(unittest.TestCase):
    def test_identical_clusterings(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        self.assertAlmostEqual(MODULE.normalized_mutual_info(labels, labels), 1.0)
        self.assertAlmostEqual(MODULE.adjusted_rand_index(labels, labels), 1.0)

    def test_macro_f1(self):
        labels = np.array([0, 1, 1, 0])
        self.assertEqual(MODULE.macro_f1(labels, labels), 1.0)

    def test_view_indices_keep_ends(self):
        rng = np.random.default_rng(0)
        indices = MODULE.view_indices(20, 1, missing_rate=0.9, rng=rng)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 19)


if __name__ == "__main__":
    unittest.main()
