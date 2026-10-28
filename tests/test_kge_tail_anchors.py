"""
Tests for gap_fill_metrics.kge_tail_anchors. Expected values are derived by
hand (derivation in the comment above each one), never by running the code.
Where an expected value is written with a math call, the standard library, not
the code under test, computes it; the comment gives the decimal checked by hand.

KGE components: r = Pearson correlation, alpha = std(sim) / std(obs) (population
std, ddof 0, as numpy's default), beta = mean(sim) / mean(obs).
Percentiles use numpy's default linear interpolation: for n = 5 sorted values,
percentile p sits at position p/100 * 4 (0-based) between neighbors.

Run from the repository root:  python -m unittest -v tests.test_kge_tail_anchors
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
UNEVEN = [1.0, 2.0, 3.0, 4.0, 10.0]


def frame(sim, obs, sim_col="Qsim", obs_col="Qobs"):
    return pd.DataFrame({sim_col: sim, obs_col: obs})


# Doubled model, linear (obs = [1..5], sim = 2 * obs):
#   r = 1 (sim is a positive multiple of obs), term 0; alpha = std(2 obs) / std(obs) = 2,
#   term (1 - 2)^2 = 1; beta = 6 / 3 = 2, term 1.
#   p5 at position 0.2: obs 1 + 0.2 * 1 = 1.2, sim 2.4, R = 2, term 1;
#   p95 at position 3.8: obs 4 + 0.8 * 1 = 4.8, sim 9.6, R = 2, term 1.
#   sum 0 + 1 + 1 + 1 + 1 = 4; score 1 - sqrt(4) = -1
DOUBLED_SCORE = -1.0

# Uneven model (obs = [1..5], sim = [1, 2, 3, 4, 10]):
#   mean(sim) = 20 / 5 = 4; sim deviations -3, -2, -1, 0, 6, squares 9 + 4 + 1 + 0 + 36 = 50;
#   obs deviations -2, -1, 0, 1, 2, squares 10; cross-products 6 + 2 + 0 + 0 + 12 = 20.
#   r = 20 / sqrt(50 * 10) = 2 / sqrt(5) = 0.894427..., term (1 - r)^2 = 0.011146...
#   alpha = sqrt(50 / 10) = sqrt(5) = 2.236068..., term (1 - sqrt 5)^2 = 6 - 2 sqrt 5 = 1.527864...
#   beta = 4 / 3, term (1 - 4/3)^2 = 1/9 = 0.111111...
#   S_kge = 0.011146 + 1.527864 + 0.111111 = 1.650121... (math computes the square roots)
S_KGE_UNEVEN = (1 - 2 / math.sqrt(5)) ** 2 + (1 - math.sqrt(5)) ** 2 + 1 / 9


class TestKgeTailAnchorTerms(unittest.TestCase):
    def test_term_names_in_order(self):
        self.assertEqual(gfm.KGE_TAIL_ANCHOR_TERMS, ("r", "alpha", "beta", "low", "high"))


class TestKgeTailAnchorValues(unittest.TestCase):
    def test_perfect_model(self):
        # sim = obs: r = alpha = beta = 1 and every ratio 1, so all five terms 0; score 1
        self.assertAlmostEqual(gfm.kge_tail_anchors(frame(OBS, OBS)), 1.0, places=PLACES)

    def test_doubled_linear(self):
        # derivation above DOUBLED_SCORE: sum 4, score -1
        self.assertAlmostEqual(gfm.kge_tail_anchors(frame(DOUBLED, OBS)), DOUBLED_SCORE,
                               places=PLACES)

    def test_doubled_log(self):
        # KGE terms unchanged (0, 1, 1); anchors (ln 2)^2 each.
        # score 1 - sqrt(2 + 2 (ln 2)^2). math computes it: (ln 2)^2 = 0.480453...,
        # sum = 2.960906..., sqrt = 1.720728..., score = -0.720728...
        expected = 1 - math.sqrt(2 + 2 * math.log(2) ** 2)
        self.assertAlmostEqual(gfm.kge_tail_anchors(frame(DOUBLED, OBS), ratio="log"), expected,
                               places=PLACES)

    def test_weighted_alpha_three(self):
        # raw weights r 1, alpha 3, beta 1, low 1, high 1; k = 5, S = 7; normalized w * 5 / 7:
        # r 5/7, alpha 15/7, beta 5/7, low 5/7, high 5/7. Doubled terms 0, 1, 1, 1, 1:
        # sum 15/7 + 5/7 + 5/7 + 5/7 = 30/7; score 1 - sqrt(30/7).
        # math computes it: 30/7 = 4.285714..., sqrt = 2.070197..., score = -1.070197...
        self.assertAlmostEqual(gfm.kge_tail_anchors(frame(DOUBLED, OBS), weights={"alpha": 3}),
                               1 - math.sqrt(30 / 7), places=PLACES)

    def test_zero_weight_on_zero_term(self):
        # r dropped (its term is 0 anyway); active alpha, beta, low, high, k = 4, weights 1;
        # sum 1 + 1 + 1 + 1 = 4; score 1 - 2 = -1 (same as unweighted)
        self.assertAlmostEqual(gfm.kge_tail_anchors(frame(DOUBLED, OBS), weights={"r": 0}),
                               -1.0, places=PLACES)

    def test_zero_weight_drops_alpha(self):
        # alpha dropped; active r, beta, low, high, k = 4, weights 1; sum 0 + 1 + 1 + 1 = 3;
        # score 1 - sqrt(3). math computes it: sqrt(3) = 1.732051..., score = -0.732051...
        self.assertAlmostEqual(gfm.kge_tail_anchors(frame(DOUBLED, OBS), weights={"alpha": 0}),
                               1 - math.sqrt(3), places=PLACES)

    def test_reversed_series_anchors_blind_to_timing(self):
        # sim = [5, 4, 3, 2, 1]: r = -1, term (1 - (-1))^2 = 4; same std, alpha 1, term 0;
        # same mean, beta 1, term 0; sorted sim = sorted obs, so every ratio 1, anchors 0.
        # sum 4; score 1 - 2 = -1
        self.assertAlmostEqual(gfm.kge_tail_anchors(frame([5.0, 4.0, 3.0, 2.0, 1.0], OBS)),
                               -1.0, places=PLACES)

    def test_uneven_model_mid_percentiles(self):
        # p25 at position 1, p75 at position 3: obs 2 and 4, sim 2 and 4; anchors 0.
        # score 1 - sqrt(S_kge) (S_kge above). math computes it: sqrt(1.650121) = 1.284571...,
        # score = -0.284571...
        self.assertAlmostEqual(gfm.kge_tail_anchors(frame(UNEVEN, OBS), low_p=25, high_p=75),
                               1 - math.sqrt(S_KGE_UNEVEN), places=PLACES)

    def test_uneven_model_default_percentiles(self):
        # p5: obs 1.2, sim 1 + 0.2 * (2 - 1) = 1.2, term 0. p95: obs 4.8, sim 4 + 0.8 * (10 - 4)
        # = 8.8, R = 8.8 / 4.8 = 11/6, term (5/6)^2 = 25/36 = 0.694444...
        # score 1 - sqrt(S_kge + 25/36). math computes it: 1.650121 + 0.694444 = 2.344566...,
        # sqrt = 1.531198..., score = -0.531198...
        self.assertAlmostEqual(gfm.kge_tail_anchors(frame(UNEVEN, OBS)),
                               1 - math.sqrt(S_KGE_UNEVEN + 25 / 36), places=PLACES)

    def test_nan_pairs_dropped(self):
        # the last pair has obs NaN and is dropped; the rest is the doubled model: score -1
        df = frame(DOUBLED + [7.0], OBS + [np.nan])
        self.assertAlmostEqual(gfm.kge_tail_anchors(df), DOUBLED_SCORE, places=PLACES)

    def test_custom_column_names(self):
        # the doubled model under other column names: score -1
        df = frame(DOUBLED, OBS, sim_col="simulated", obs_col="observed")
        self.assertAlmostEqual(gfm.kge_tail_anchors(df, sim_col="simulated", obs_col="observed"),
                               DOUBLED_SCORE, places=PLACES)


class TestKgeTailAnchorZeroFlows(unittest.TestCase):
    ZEROS = [0.0, 0.0, 0.0, 0.0, 5.0]

    def test_keep_zero_percentile_raises(self):
        # obs = sim = [0, 0, 0, 0, 5]: p5 at position 0.2 between 0 and 0 is 0 -> ValueError
        with self.assertRaises(ValueError):
            gfm.kge_tail_anchors(frame(self.ZEROS, self.ZEROS), nonpositive="keep")

    def test_offset_makes_ratios_work(self):
        # KGE terms use positive=False, so no shift: sim = obs, r = alpha = beta = 1, terms 0.
        # Percentile pairing: eps = 0.01 * mean([0, 0, 0, 0, 5]) = 0.01 added to both series,
        # so sim = obs stays equal and every ratio is 1. score 1
        self.assertAlmostEqual(
            gfm.kge_tail_anchors(frame(self.ZEROS, self.ZEROS), nonpositive="offset"),
            1.0, places=PLACES)

    def test_skip_drops_zero_rows(self):
        # the obs = 0 row is dropped; the rest [1..5] has sim = obs, all terms 0; score 1
        values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(gfm.kge_tail_anchors(frame(values, values), nonpositive="skip"),
                               1.0, places=PLACES)


class TestKgeTailAnchorErrors(unittest.TestCase):
    def setUp(self):
        self.df = frame(DOUBLED, OBS)

    def test_low_not_below_high(self):
        for low_p, high_p in ((95, 5), (50, 50)):
            with self.subTest(low_p=low_p, high_p=high_p):
                with self.assertRaises(ValueError):
                    gfm.kge_tail_anchors(self.df, low_p=low_p, high_p=high_p)

    def test_low_p_zero(self):
        with self.assertRaises(ValueError):
            gfm.kge_tail_anchors(self.df, low_p=0)

    def test_high_p_hundred(self):
        with self.assertRaises(ValueError):
            gfm.kge_tail_anchors(self.df, high_p=100)

    def test_string_percentile(self):
        with self.assertRaises(TypeError):
            gfm.kge_tail_anchors(self.df, high_p="95")

    def test_unknown_ratio(self):
        with self.assertRaises(ValueError):
            gfm.kge_tail_anchors(self.df, ratio="square")

    def test_unknown_weights_key(self):
        with self.assertRaises(ValueError) as cm:
            gfm.kge_tail_anchors(self.df, weights={"nse": 1})
        self.assertIn("valid term names", str(cm.exception))

    def test_constant_observed_series(self):
        # obs std 0
        with self.assertRaises(ValueError) as cm:
            gfm.kge_tail_anchors(frame(OBS, [3.0] * 5))
        self.assertIn("observed", str(cm.exception))

    def test_constant_simulated_series(self):
        # sim std 0
        with self.assertRaises(ValueError) as cm:
            gfm.kge_tail_anchors(frame([3.0] * 5, OBS))
        self.assertIn("simulated", str(cm.exception))

    def test_zero_observed_mean(self):
        # obs = [-1, 0, 1]: mean 0, std > 0; the KGE terms are computed before the
        # percentile terms, so the zero-mean check raises first
        with self.assertRaises(ValueError) as cm:
            gfm.kge_tail_anchors(frame([1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]), nonpositive="keep")
        self.assertIn("mean", str(cm.exception))

    def test_fewer_than_two_pairs(self):
        # the second pair has obs NaN and is dropped, leaving one pair
        with self.assertRaises(ValueError) as cm:
            gfm.kge_tail_anchors(frame([1.0, 2.0], [1.0, np.nan]))
        self.assertIn("at least 2", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
