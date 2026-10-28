"""
Tests that the gap_fill_metrics helpers accept NumPy scalars as numbers.
Expected values are derived by hand (derivation in the comment above each
one), never by running the code. Where an expected value is written with a
math call, the standard library, not the code under test, computes it; the
comment gives the decimal checked by hand.

Run from the repository root:  python -m unittest -v tests.test_gap_fill_numpy_types
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gap_fill_metrics as gfm  # noqa: E402

PLACES = 12


class TestNumpyScalarsAccepted(unittest.TestCase):
    def test_ratio_term_float32(self):
        # float32 holds 10 and 5 exactly, so R = 10 / 5 = 2.0 exactly; (1 - 2)^2 = 1
        self.assertAlmostEqual(gfm.ratio_term(np.float32(10), np.float32(5)), 1.0,
                               places=PLACES)

    def test_ratio_term_int64(self):
        # R = 10 / 5 = 2; (1 - 2)^2 = 1
        self.assertAlmostEqual(gfm.ratio_term(np.int64(10), np.int64(5)), 1.0, places=PLACES)

    def test_ratio_term_float64(self):
        # R = 5 / 10 = 0.5; (1 - 0.5)^2 = 0.25
        self.assertAlmostEqual(gfm.ratio_term(np.float64(5), np.float64(10)), 0.25,
                               places=PLACES)

    def test_normalize_weights_numpy_weights(self):
        # k = 2, S = 1 + 3 = 4; a = 1 * 2 / 4 = 0.5, b = 3 * 2 / 4 = 1.5
        result = gfm.normalize_weights(["a", "b"], {"a": np.float32(1), "b": np.int64(3)})
        self.assertEqual(list(result), ["a", "b"])
        self.assertAlmostEqual(result["a"], 0.5, places=PLACES)
        self.assertAlmostEqual(result["b"], 1.5, places=PLACES)

    def test_combine_terms_numpy_terms(self):
        # weights 1 and 1; sum 1 + 4 = 5; score 1 - sqrt(5).
        # math (not the code under test) computes it: sqrt(5) = 2.236068..., score = -1.236068...
        self.assertAlmostEqual(gfm.combine_terms({"a": np.float32(1), "b": np.float64(4)}),
                               1 - math.sqrt(5), places=PLACES)


class TestNonNumbersStillRejected(unittest.TestCase):
    def test_bool_raises_type_error(self):
        with self.assertRaises(TypeError):
            gfm.ratio_term(True, 5)

    def test_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            gfm.ratio_term("10", 5)


if __name__ == "__main__":
    unittest.main()
