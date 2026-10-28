"""
Tests for gap_fill_metrics.percentile_nse_index. Expected values are derived
by hand (derivation in the comment above each one), never by running the code.
Where an expected value is written with a math call, the standard library, not
the code under test, computes it; the comment gives the decimal checked by hand.

Percentiles use numpy's default linear interpolation: for n = 5 sorted values,
percentile p sits at position p/100 * 4 (0-based) between neighbors.

Run from the repository root:  python -m unittest -v tests.test_percentile_nse
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gap_fill_metrics as gfm  # noqa: E402

PLACES = 12
OBS = [1.0, 2.0, 3.0, 4.0, 5.0]
DOUBLED = [2.0, 4.0, 6.0, 8.0, 10.0]


def frame(sim, obs, sim_col="Qsim", obs_col="Qobs"):
    return pd.DataFrame({sim_col: sim, obs_col: obs})


# Doubled model, linear (obs = [1..5], sim = 2 * obs):
#   p5 at position 0.05 * 4 = 0.2: obs 1 + 0.2 * (2 - 1) = 1.2, sim 2.4, R = 2, term (1 - 2)^2 = 1
#   p95 at position 0.95 * 4 = 3.8: obs 4 + 0.8 * (5 - 4) = 4.8, sim 9.6, R = 2, term 1
#   NSE: mean obs 3, sum (obs - 3)^2 = 4 + 1 + 0 + 1 + 4 = 10; sim - obs = obs, so
#   sum (sim - obs)^2 = 1 + 4 + 9 + 16 + 25 = 55; NSE = 1 - 55 / 10 = -4.5;
#   term (1 - (-4.5))^2 = 5.5^2 = 30.25
#   unweighted (k = 3): sum 1 + 1 + 30.25 = 32.25; score 1 - sqrt(32.25).
#   math (not the code under test) computes it: sqrt(32.25) = 5.678908..., score = -4.678908...
DOUBLED_SCORE = 1 - math.sqrt(32.25)


class TestPercentileNseTerms(unittest.TestCase):
    def test_term_names_in_order(self):
        self.assertEqual(gfm.PERCENTILE_NSE_TERMS, ("low", "high", "nse"))


class TestPercentileNseValues(unittest.TestCase):
    def test_perfect_model(self):
        # sim = obs: every ratio 1, so both ratio terms 0; NSE = 1, term 0; score 1 - 0 = 1
        self.assertAlmostEqual(gfm.percentile_nse_index(frame(OBS, OBS)), 1.0, places=PLACES)

    def test_doubled_linear(self):
        # derivation above DOUBLED_SCORE: 1 - sqrt(32.25) = -4.678908...
        self.assertAlmostEqual(gfm.percentile_nse_index(frame(DOUBLED, OBS)), DOUBLED_SCORE,
                               places=PLACES)

    def test_doubled_log(self):
        # R = 2 at p5 and p95, so each ratio term is (ln 2)^2; NSE term 30.25 (as above).
        # score 1 - sqrt(2 * (ln 2)^2 + 30.25). math computes it: (ln 2)^2 = 0.480453...,
        # sum = 31.210906..., sqrt = 5.586672..., score = -4.586672...
        expected = 1 - math.sqrt(2 * math.log(2) ** 2 + 30.25)
        self.assertAlmostEqual(gfm.percentile_nse_index(frame(DOUBLED, OBS), ratio="log"),
                               expected, places=PLACES)

    def test_weighted_nse_half(self):
        # raw weights low 1, high 1, nse 0.5; k = 3, S = 2.5; normalized 1 * 3 / 2.5 = 1.2,
        # 1.2, 0.5 * 3 / 2.5 = 0.6; sum 1.2 * 1 + 1.2 * 1 + 0.6 * 30.25 = 2.4 + 18.15 = 20.55;
        # score 1 - sqrt(20.55). math computes it: sqrt(20.55) = 4.533211..., score = -3.533211...
        self.assertAlmostEqual(gfm.percentile_nse_index(frame(DOUBLED, OBS), weights={"nse": 0.5}),
                               1 - math.sqrt(20.55), places=PLACES)

    def test_zero_weight_drops_nse(self):
        # nse dropped; active low and high, k = 2, weights 1 and 1; sum 1 + 1 = 2;
        # score 1 - sqrt(2). math computes it: sqrt(2) = 1.414214..., score = -0.414214...
        self.assertAlmostEqual(gfm.percentile_nse_index(frame(DOUBLED, OBS), weights={"nse": 0}),
                               1 - math.sqrt(2), places=PLACES)

    def test_custom_percentiles(self):
        # obs = [1..5], sim = [1, 2, 3, 4, 10]. p25 at position 1, p75 at position 3:
        # obs 2 and 4, sim 2 and 4; ratios 1, terms 0. NSE: only the last pair differs
        # (10 - 5 = 5), sum (sim - obs)^2 = 25; sum (obs - 3)^2 = 10; NSE = 1 - 2.5 = -1.5;
        # term (1 - (-1.5))^2 = 2.5^2 = 6.25; score 1 - sqrt(6.25) = 1 - 2.5 = -1.5
        df = frame([1.0, 2.0, 3.0, 4.0, 10.0], OBS)
        self.assertAlmostEqual(gfm.percentile_nse_index(df, low_p=25, high_p=75), -1.5,
                               places=PLACES)

    def test_default_percentiles_uneven_model(self):
        # obs = [1..5], sim = [1, 2, 3, 4, 10]. p5: obs 1.2, sim 1 + 0.2 * (2 - 1) = 1.2,
        # ratio 1, term 0. p95: obs 4.8, sim 4 + 0.8 * (10 - 4) = 8.8, R = 8.8 / 4.8 = 11/6,
        # term (1 - 11/6)^2 = (5/6)^2 = 25/36. NSE term 6.25 = 225/36 (as above).
        # sum 250/36; score 1 - sqrt(250/36) = 1 - sqrt(250) / 6.
        # math computes it: sqrt(250) = 15.811388..., / 6 = 2.635231..., score = -1.635231...
        df = frame([1.0, 2.0, 3.0, 4.0, 10.0], OBS)
        self.assertAlmostEqual(gfm.percentile_nse_index(df), 1 - math.sqrt(250) / 6,
                               places=PLACES)

    def test_nan_pairs_dropped(self):
        # the last pair has obs NaN and is dropped; the rest is the doubled model:
        # score 1 - sqrt(32.25) = -4.678908... (derivation above DOUBLED_SCORE)
        df = frame(DOUBLED + [7.0], OBS + [np.nan])
        self.assertAlmostEqual(gfm.percentile_nse_index(df), DOUBLED_SCORE, places=PLACES)

    def test_custom_column_names(self):
        # the doubled model under other column names: score 1 - sqrt(32.25) = -4.678908...
        df = frame(DOUBLED, OBS, sim_col="simulated", obs_col="observed")
        self.assertAlmostEqual(
            gfm.percentile_nse_index(df, sim_col="simulated", obs_col="observed"),
            DOUBLED_SCORE, places=PLACES)


class TestPercentileNseZeroFlows(unittest.TestCase):
    def test_keep_zero_percentile_raises(self):
        # obs = sim = [0, 0, 0, 0, 5]: p5 at position 0.2 between 0 and 0 is 0, so the
        # low ratio term gets a value <= 0 and raises ValueError
        df = frame([0.0, 0.0, 0.0, 0.0, 5.0], [0.0, 0.0, 0.0, 0.0, 5.0])
        with self.assertRaises(ValueError):
            gfm.percentile_nse_index(df, nonpositive="keep")

    def test_offset_makes_ratios_work(self):
        # offset: obs clipped at 0 (unchanged), eps = 0.01 * mean([0, 0, 0, 0, 5]) = 0.01 * 1
        # = 0.01 added to both series, so sim = obs stays equal and every ratio is 1 (terms 0).
        # NSE (no offset for it: neither log nor positive) of sim = obs is 1, since obs has
        # non-zero variance; term 0. score 1 - 0 = 1
        df = frame([0.0, 0.0, 0.0, 0.0, 5.0], [0.0, 0.0, 0.0, 0.0, 5.0])
        self.assertAlmostEqual(gfm.percentile_nse_index(df, nonpositive="offset"), 1.0,
                               places=PLACES)

    def test_skip_drops_zero_rows(self):
        # skip: the row with obs = 0 is dropped (for the percentiles and for NSE); the rest
        # [1..5] has sim = obs, so ratio terms 0, NSE 1, term 0; score 1
        values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(gfm.percentile_nse_index(frame(values, values), nonpositive="skip"),
                               1.0, places=PLACES)


class TestPercentileNseErrors(unittest.TestCase):
    def setUp(self):
        self.df = frame(DOUBLED, OBS)

    def test_low_not_below_high(self):
        for low_p, high_p in ((95, 5), (50, 50)):
            with self.subTest(low_p=low_p, high_p=high_p):
                with self.assertRaises(ValueError):
                    gfm.percentile_nse_index(self.df, low_p=low_p, high_p=high_p)

    def test_low_p_zero(self):
        with self.assertRaises(ValueError):
            gfm.percentile_nse_index(self.df, low_p=0)

    def test_high_p_hundred(self):
        with self.assertRaises(ValueError):
            gfm.percentile_nse_index(self.df, high_p=100)

    def test_string_percentile(self):
        with self.assertRaises(TypeError):
            gfm.percentile_nse_index(self.df, low_p="5")

    def test_unknown_ratio(self):
        with self.assertRaises(ValueError):
            gfm.percentile_nse_index(self.df, ratio="square")

    def test_unknown_weights_key(self):
        with self.assertRaises(ValueError) as cm:
            gfm.percentile_nse_index(self.df, weights={"middle": 1})
        self.assertIn("valid term names", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
