"""
Tests for the two safeguards of gap_fill_metrics.jdkge: the ts_factor check
and the identical-values guard of _jsd_between. Expected values are derived by
hand (derivation in the comment above each one), never by running the code.

Run from the repository root:  python -m unittest -v tests.test_jdkge_safeguards
"""

import math
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gap_fill_metrics as gfm  # noqa: E402

PLACES = 12
OBS = [1.0, 2.0, 3.0, 4.0, 5.0]
POW2 = [1.0, 2.0, 4.0, 8.0]


def frame(sim, obs, sim_col="Qsim", obs_col="Qobs"):
    return pd.DataFrame({sim_col: sim, obs_col: obs})


def quiet_jdkge(df, **kwargs):
    """jdkge with the short-record warning suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gfm.jdkge(df, **kwargs)


class TestTsFactorCheck(unittest.TestCase):
    def test_zero_negative_nan_infinity_raise_value_error(self):
        for ts_factor in (0, -1.0, math.nan, math.inf):
            with self.subTest(ts_factor=ts_factor):
                with self.assertRaises(ValueError) as cm:
                    quiet_jdkge(frame(OBS, OBS), ts_factor=ts_factor)
                self.assertIn("ts_factor", str(cm.exception))

    def test_string_and_bool_raise_type_error(self):
        for ts_factor in ("1", True):
            with self.subTest(ts_factor=ts_factor):
                with self.assertRaises(TypeError):
                    quiet_jdkge(frame(OBS, OBS), ts_factor=ts_factor)

    def test_ts_factor_checked_first(self):
        # obs all 3 would raise the zero-variance ValueError; the TypeError comes first
        with self.assertRaises(TypeError):
            quiet_jdkge(frame(OBS, [3.0] * 5), ts_factor=True)

    def test_valid_ts_factor_works(self):
        # sim = obs: r = beta = alpha = 1; identical histograms, JSD = 0; score 1
        self.assertAlmostEqual(quiet_jdkge(frame(OBS, OBS), ts_factor=2.0), 1.0, places=PLACES)


class TestIdenticalValuesGuard(unittest.TestCase):
    def test_all_values_identical_raises(self):
        # x_min = x_max = ln 2: every bin would have zero width
        with self.assertRaises(ValueError):
            gfm._jsd_between(np.array([2.0, 2.0]), np.array([2.0, 2.0]), 1.0)

    def test_normal_call_unaffected(self):
        # obs = sim = [1, 2, 4, 8]: identical histograms, p = q, every log is 0; JSD 0
        self.assertAlmostEqual(gfm._jsd_between(np.array(POW2), np.array(POW2), 1.0), 0.0,
                               places=PLACES)


if __name__ == "__main__":
    unittest.main()
