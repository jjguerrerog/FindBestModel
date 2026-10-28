"""
Tests for gap_fill_metrics.jdkge and its JSD helpers. Expected values are
derived by hand (derivation in the comment above each one), never by running
the code. Where an expected value is written with a math call, the standard
library, not the code under test, computes it; the comment gives the decimal
checked by hand.

JDKGE (Ficchi et al. 2026, Eq. 2): 1 - sqrt((r-1)^2 + (beta-1)^2 + (alpha-1)^2
+ JSD^2), with r Pearson, beta = mean ratio, alpha = cv_sim / cv_obs, JSD the
base-2 Jensen-Shannon divergence of the log-flow histograms. Percentiles use
numpy's default linear interpolation: for n sorted values, percentile p sits at
position p/100 * (n - 1) (0-based) between neighbors.

Run from the repository root:  python -m unittest -v tests.test_jdkge
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
POW2_SIM = [1.0, 1.0, 4.0, 8.0]
EPSILON = 1e-6


def frame(sim, obs, sim_col="Qsim", obs_col="Qobs"):
    return pd.DataFrame({sim_col: sim, obs_col: obs})


def quiet_jdkge(df, **kwargs):
    """jdkge with the short-record warning suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gfm.jdkge(df, **kwargs)


# JSD of the example obs = [1, 2, 4, 8], sim = [1, 1, 4, 8] (see test_small_example):
#   0.375 * (2 - log2 3); math computes log2(3) = 1.5849625..., JSD = 0.155639...
JSD_POW2 = 0.375 * (2 - math.log2(3))

# KGE' terms of the same example (math computes the square roots):
#   means obs 15/4 = 3.75, sim 14/4 = 3.5; beta = 14/15 = 0.933333..., term (1/15)^2 = 0.004444...
#   obs deviations -2.75, -1.75, 0.25, 4.25, squares 7.5625 + 3.0625 + 0.0625 + 18.0625 = 28.75;
#   sim deviations -2.5, -2.5, 0.5, 4.5, squares 6.25 + 6.25 + 0.25 + 20.25 = 33;
#   cross-products 6.875 + 4.375 + 0.125 + 19.125 = 30.5.
#   r = 30.5 / sqrt(28.75 * 33) = 30.5 / sqrt(948.75) = 30.5 / 30.80179 = 0.990202...,
#   term 0.009798^2 = 0.000096...
#   alpha = (std_sim / 3.5) / (std_obs / 3.75) = sqrt(33 / 28.75) * 3.75 / 3.5
#   = 1.071366 * 1.071429 = 1.147892... (the ddof cancels), term 0.147892^2 = 0.021872...
R_TERM = (1 - 30.5 / math.sqrt(28.75 * 33)) ** 2
BETA_TERM = (1 / 15) ** 2
ALPHA_TERM = (1 - math.sqrt(33 / 28.75) * 3.75 / 3.5) ** 2


class TestJdkgeConstants(unittest.TestCase):
    def test_term_names_in_order(self):
        self.assertEqual(gfm.JDKGE_TERMS, ("r", "beta", "alpha", "jsd"))

    def test_bin_limits_and_minimum_years(self):
        self.assertEqual(gfm.JSD_MIN_BINS, 25)
        self.assertEqual(gfm.JSD_MAX_BINS, 100)
        self.assertEqual(gfm.JSD_MIN_YEARS, 4)


class TestJsd(unittest.TestCase):
    def test_identical_vectors_give_zero(self):
        # m = p, every log2(p / m) = 0
        self.assertAlmostEqual(gfm._jsd([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]), 0.0, places=PLACES)

    def test_disjoint_vectors_give_one(self):
        # p = [1, 0], q = [0, 1]: m = [0.5, 0.5]; zero summands contribute 0;
        # JSD = 0.5 * 1 * log2(1 / 0.5) + 0.5 * 1 * log2(1 / 0.5) = 0.5 + 0.5 = 1
        self.assertAlmostEqual(gfm._jsd([1.0, 0.0], [0.0, 1.0]), 1.0, places=PLACES)

    def test_half_and_one(self):
        # p = [0.5, 0.5], q = [1, 0]: m = [0.75, 0.25];
        # JSD = 0.5 * (0.5 log2(0.5/0.75) + 0.5 log2(0.5/0.25)) + 0.5 * (1 * log2(1/0.75))
        # = 0.5 * (0.5 (1 - log2 3) + 0.5) + 0.5 (2 - log2 3) = 1.5 - 0.75 log2 3.
        # math computes it: log2(3) = 1.5849625..., JSD = 0.311278...
        self.assertAlmostEqual(gfm._jsd([0.5, 0.5], [1.0, 0.0]), 1.5 - 0.75 * math.log2(3),
                               places=PLACES)


class TestBinIndices(unittest.TestCase):
    EDGES = np.array([0.0, 1.0, 2.0, 3.0])

    def test_paper_closure(self):
        # first bin [0, 1] closed on both sides: 0, 0.5 and 1 in bin 0; (1, 2]: 1.5 and 2 in
        # bin 1; (2, 3]: 3 in bin 2
        x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
        self.assertEqual(gfm._bin_indices(x, self.EDGES).tolist(), [0, 0, 0, 1, 1, 2])

    def test_value_above_last_edge_is_clipped(self):
        # 3.0000000001 is above the last edge 3: clipped into the last bin, 2
        self.assertEqual(gfm._bin_indices(np.array([3.0000000001]), self.EDGES).tolist(), [2])


class TestNumberOfBins(unittest.TestCase):
    # epsilon 1e-6, so h_min = min(100 * 1e-6, 0.1) = 1e-4 in every case below
    SPREAD = [0.0, 5.0, 5.0, 5.0, 5.4, 5.4, 5.4, 5.4, 10.0]

    def test_zero_iqr_gives_exactly_min_bins(self):
        # x_obs = [0, 0, 0, 0, 1]: percentiles 25 and 75 at positions 1 and 3, both 0, IQR 0;
        # h = (1 - 0) / 25 = 0.04, range / h = 25 (25.000000000000004 in floating point,
        # rounded to 9 decimals: 25); ceiling 25, not 26
        x_obs = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(gfm._number_of_bins(x_obs, 0.0, 1.0, EPSILON), 25)

    def test_freedman_diaconis_between_limits(self):
        # n = 9: positions 2 and 6 give percentiles 5 and 5.4, IQR 0.4;
        # 9^(-1/3) = 0.480750; h = 2 * 0.4 * 0.480750 = 0.384600;
        # range / h = 10 / 0.384600 = 26.001; ceiling 27
        self.assertEqual(gfm._number_of_bins(np.array(self.SPREAD), 0.0, 10.0, EPSILON), 27)

    def test_time_step_factor(self):
        # same data, ts_factor 8: 8^(1/3) = 2, so 20 / 0.384600 = 52.002; ceiling 53
        self.assertEqual(gfm._number_of_bins(np.array(self.SPREAD), 0.0, 10.0, EPSILON,
                                             ts_factor=8.0), 53)

    def test_upper_limit(self):
        # IQR = 5.1 - 5 = 0.1; h = 0.2 * 0.480750 = 0.096150; range / h = 104.004;
        # ceiling 105, limited to 100
        x_obs = np.array([0.0, 5.0, 5.0, 5.0, 5.1, 5.1, 5.1, 5.1, 10.0])
        self.assertEqual(gfm._number_of_bins(x_obs, 0.0, 10.0, EPSILON), 100)

    def test_lower_limit(self):
        # x_obs = ln [1, 2, 4, 8] (math.log builds the inputs): n = 4; the 25th percentile at
        # position 0.75: 0.75 ln 2 = 0.519860; the 75th at position 2.25: ln 4 + 0.25 ln 2
        # = 1.559581; IQR = 1.039721; h = 2 * 1.039721 * 4^(-1/3) = 2.079442 * 0.629961
        # = 1.309968; range / h = 2.079442 / 1.309968 = 1.587401; ceiling 2, raised to 25
        x_obs = np.array([math.log(v) for v in POW2])
        self.assertEqual(gfm._number_of_bins(x_obs, 0.0, math.log(8), EPSILON), 25)


class TestJsdBetween(unittest.TestCase):
    def test_small_example(self):
        # obs = [1, 2, 4, 8], sim = [1, 1, 4, 8]: epsilon = min(1e-6, 0.1 * 1) = 1e-6;
        # x from 0 to ln 8, 25 bins (test_lower_limit). Width w = ln 8 / 25: ln 1 = 0 in bin 0;
        # ln 2 at 25/3 = 8.33 widths, bin 8; ln 4 at 16.67 widths, bin 16; ln 8 on the last
        # edge, bin 24. p = 0.25 in bins 0, 8, 16, 24; q = 0.5 in bin 0, 0 in bin 8, 0.25 in
        # bins 16 and 24. Without smoothing: m = 0.375, 0.125, 0.25, 0.25;
        # JSD = 0.5 * (0.25 log2(0.25/0.375) + 0.25 log2(0.25/0.125)) + 0.5 * (0.5 log2(0.5/0.375))
        # = 0.375 * (2 - log2 3) = 0.155639... (JSD_POW2). The smoothing (25 * 1e-6 = 2.5e-5
        # against a density sum 1/w = 12.02) changes JSD by less than 1e-5: places=4
        jsd = gfm._jsd_between(np.array(POW2), np.array(POW2_SIM), 1.0)
        self.assertAlmostEqual(jsd, JSD_POW2, places=4)


class TestCombinationAgainstPaper(unittest.TestCase):
    # Ficchi et al. 2026, Fig. 7 (p. 29): (KGE', JSD, JDKGE), printed to 3 decimals.
    # This checks only how the terms are combined, not the JSD pipeline.
    # Hand values of 1 - sqrt((1 - KGE')^2 + JSD^2):
    #   0.010404 + 0.001225 = 0.011629, sqrt 0.107838, 0.892162 -> 0.892
    #   0.009604 + 0.004489 = 0.014093, sqrt 0.118714, 0.881286 -> 0.881
    #   0.003969 + 0.000121 = 0.004090, sqrt 0.063953, 0.936047 -> 0.936
    #   0.002809 + 0.010609 = 0.013418, sqrt 0.115836, 0.884164 -> 0.884
    #   0.005929 + 0.000841 = 0.006770, sqrt 0.082280, 0.917720 -> 0.918
    #   0.003481 + 0.029241 = 0.032722, sqrt 0.180893, 0.819107 -> 0.819
    TRIPLES = ((0.898, 0.035, 0.892), (0.902, 0.067, 0.881), (0.937, 0.011, 0.936),
               (0.947, 0.103, 0.884), (0.923, 0.029, 0.918), (0.941, 0.171, 0.819))

    def test_figure_seven_triples(self):
        for kge_prime, jsd, printed in self.TRIPLES:
            with self.subTest(kge_prime=kge_prime, jsd=jsd):
                score = gfm.combine_terms({"kge_prime": (1 - kge_prime) ** 2, "jsd": jsd ** 2})
                self.assertAlmostEqual(score, printed, places=3)


class TestJdkgeValues(unittest.TestCase):
    def test_perfect_model(self):
        # sim = obs: r = beta = alpha = 1; identical histograms so p = q and JSD = 0; score 1
        self.assertAlmostEqual(quiet_jdkge(frame(OBS, OBS)), 1.0, places=PLACES)

    def test_full_pipeline_example(self):
        # terms (constants above): 0.000096 + 0.004444 + 0.021872 + JSD^2 0.024223 = 0.050636;
        # score 1 - sqrt(0.050636) = 1 - 0.225025 = 0.774975 (places=3, JSD smoothing < 1e-5)
        expected = 1 - math.sqrt(R_TERM + BETA_TERM + ALPHA_TERM + JSD_POW2 ** 2)
        self.assertAlmostEqual(quiet_jdkge(frame(POW2_SIM, POW2)), expected, places=3)

    def test_zero_weight_drops_jsd(self):
        # active r, beta, alpha, k = 3, weights 1: 0.000096 + 0.004444 + 0.021872 = 0.026413;
        # score 1 - sqrt(0.026413) = 1 - 0.162521 = 0.837479. math computes it (places=3)
        expected = 1 - math.sqrt(R_TERM + BETA_TERM + ALPHA_TERM)
        self.assertAlmostEqual(quiet_jdkge(frame(POW2_SIM, POW2), weights={"jsd": 0}),
                               expected, places=3)

    def test_negative_observed_pair_dropped(self):
        # the pair with obs -1 is dropped; the rest has sim = obs: score 1
        df = frame(POW2 + [5.0], POW2 + [-1.0])
        self.assertAlmostEqual(quiet_jdkge(df), 1.0, places=PLACES)

    def test_missing_observed_pair_dropped(self):
        # the pair with obs NaN is dropped; the rest has sim = obs: score 1
        df = frame(POW2 + [3.0], POW2 + [np.nan])
        self.assertAlmostEqual(quiet_jdkge(df), 1.0, places=PLACES)

    def test_negative_simulated_pair_dropped(self):
        # the pair with sim -2 is dropped; the rest has sim = obs: score 1
        df = frame(POW2 + [-2.0], POW2 + [3.0])
        self.assertAlmostEqual(quiet_jdkge(df), 1.0, places=PLACES)

    def test_zeros_become_epsilon_in_both(self):
        # obs = sim = [0, 1, 2, 4]: the zero becomes epsilon in both, identical histograms,
        # JSD = 0; r = beta = alpha = 1; score 1
        values = [0.0, 1.0, 2.0, 4.0]
        self.assertAlmostEqual(quiet_jdkge(frame(values, values)), 1.0, places=PLACES)


class TestJdkgeShortRecord(unittest.TestCase):
    def test_short_record_warns(self):
        # 5 pairs: 5 / 365.25 years, below 4
        with self.assertWarns(UserWarning):
            gfm.jdkge(frame(OBS, OBS))

    def test_long_record_does_not_warn(self):
        # 1500 pairs: 1500 / 365.25 = 4.107 years, not below 4
        values = [1.0 + (i % 7) for i in range(1500)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gfm.jdkge(frame(values, values))
        self.assertEqual(caught, [])


class TestJdkgeErrors(unittest.TestCase):
    def test_constant_observed_series(self):
        with self.assertRaises(ValueError) as cm:
            quiet_jdkge(frame(OBS, [3.0] * 5))
        self.assertIn("observed", str(cm.exception))

    def test_constant_simulated_series(self):
        with self.assertRaises(ValueError) as cm:
            quiet_jdkge(frame([3.0] * 5, OBS))
        self.assertIn("simulated", str(cm.exception))

    def test_observed_all_zeros(self):
        # obs = [0, 0, 0]: observed std 0
        with self.assertRaises(ValueError) as cm:
            quiet_jdkge(frame([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]))
        self.assertIn("observed", str(cm.exception))

    def test_fewer_than_two_pairs(self):
        # the second pair has obs NaN and is dropped, leaving one pair
        with self.assertRaises(ValueError) as cm:
            quiet_jdkge(frame([1.0, 2.0], [1.0, np.nan]))
        self.assertIn("at least 2", str(cm.exception))

    def test_unknown_weights_key(self):
        with self.assertRaises(ValueError) as cm:
            quiet_jdkge(frame(POW2_SIM, POW2), weights={"alpha_np": 1})
        self.assertIn("valid term names", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
