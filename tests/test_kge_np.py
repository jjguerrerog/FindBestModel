"""
Tests for gap_fill_metrics.kge_np. Expected values are derived by hand
(derivation in the comment above each one), never by running the code.
Where an expected value is written with a math call, the standard library, not
the code under test, computes it; the comment gives the decimal checked by hand.

KGE_NP components (Pool et al. 2018): r_s = Pearson correlation of the ranks
(ties get average ranks); alpha_np = 1 - 0.5 * sum |sim_sorted / (n * mean_sim)
- obs_sorted / (n * mean_obs)|, each series sorted descending on its own;
beta = mean(sim) / mean(obs).

Run from the repository root:  python -m unittest -v tests.test_kge_np
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


# Doubled model (obs = [1..5], sim = 2 * obs):
#   r_s = 1 (same ranks), term 0. Sorted descending: obs [5, 4, 3, 2, 1] / (5 * 3 = 15),
#   sim [10, 8, 6, 4, 2] / (5 * 6 = 30); both [1/3, 4/15, 1/5, 2/15, 1/15], so the sum of
#   differences is 0, alpha_np = 1, term 0. beta = 6 / 3 = 2, term 1.
#   score 1 - sqrt(0 + 0 + 1) = 0
DOUBLED_SCORE = 0.0


class TestKgeNpTerms(unittest.TestCase):
    def test_term_names_in_order(self):
        self.assertEqual(gfm.KGE_NP_TERMS, ("r_s", "alpha_np", "beta"))


class TestKgeNpValues(unittest.TestCase):
    def test_perfect_model(self):
        # sim = obs: r_s = 1, identical sorted normalized series so alpha_np = 1, beta = 1;
        # all terms 0, score 1
        self.assertAlmostEqual(gfm.kge_np(frame(OBS, OBS)), 1.0, places=PLACES)

    def test_doubled_model_scaling_cancels_in_alpha_np(self):
        # derivation above DOUBLED_SCORE: score 0
        self.assertAlmostEqual(gfm.kge_np(frame(DOUBLED, OBS)), DOUBLED_SCORE, places=PLACES)

    def test_reversed_series(self):
        # sim = [5, 4, 3, 2, 1]: ranks exactly opposite, r_s = -1, term (1 - (-1))^2 = 4;
        # sorted normalized series identical, alpha_np = 1, term 0; equal means, beta = 1,
        # term 0. score 1 - sqrt(4) = -1
        self.assertAlmostEqual(gfm.kge_np(frame([5.0, 4.0, 3.0, 2.0, 1.0], OBS)), -1.0,
                               places=PLACES)

    def test_ties_and_partial_alpha_np(self):
        # sim = [1, 1, 1, 1, 11]: means obs 3, sim 3, beta = 1, term 0.
        # Ranks: obs [1, 2, 3, 4, 5]; sim's four tied 1s share ranks 1..4, average 2.5, and
        # 11 gets 5: [2.5, 2.5, 2.5, 2.5, 5]. Both rank means 3. Deviations: obs -2, -1, 0,
        # 1, 2 (squares 10); sim -0.5 four times and 2 (squares 4 * 0.25 + 4 = 5);
        # cross-products 1 + 0.5 + 0 - 0.5 + 4 = 5. r_s = 5 / sqrt(10 * 5) = 1 / sqrt(2),
        # term (1 - 1/sqrt 2)^2 = 1.5 - sqrt(2) = 0.085786...
        # alpha_np: both normalizers 5 * 3 = 15. Sorted descending obs [5, 4, 3, 2, 1],
        # sim [11, 1, 1, 1, 1]; |differences| * 15: 6, 3, 2, 1, 0, sum 12, / 15 = 0.8;
        # alpha_np = 1 - 0.5 * 0.8 = 0.6, term 0.16.
        # score 1 - sqrt(1.5 - sqrt(2) + 0.16) = 1 - sqrt(1.66 - sqrt(2)). math computes it:
        # 1.66 - 1.414214 = 0.245786..., sqrt = 0.495768..., score = 0.504232...
        self.assertAlmostEqual(gfm.kge_np(frame([1.0, 1.0, 1.0, 1.0, 11.0], OBS)),
                               1 - math.sqrt(1.66 - math.sqrt(2)), places=PLACES)

    def test_weighted_beta_three(self):
        # raw weights r_s 1, alpha_np 1, beta 3; k = 3, S = 5; normalized w * 3 / 5:
        # 0.6, 0.6, 1.8. Doubled terms 0, 0, 1: sum 1.8; score 1 - sqrt(1.8).
        # math computes it: sqrt(1.8) = 1.341641..., score = -0.341641...
        self.assertAlmostEqual(gfm.kge_np(frame(DOUBLED, OBS), weights={"beta": 3}),
                               1 - math.sqrt(1.8), places=PLACES)

    def test_zero_weight_drops_beta(self):
        # beta dropped; active r_s and alpha_np, both terms 0 for the doubled model; score 1
        self.assertAlmostEqual(gfm.kge_np(frame(DOUBLED, OBS), weights={"beta": 0}), 1.0,
                               places=PLACES)

    def test_nan_pairs_dropped(self):
        # the last pair has obs NaN and is dropped; the rest is the doubled model: score 0
        df = frame(DOUBLED + [7.0], OBS + [np.nan])
        self.assertAlmostEqual(gfm.kge_np(df), DOUBLED_SCORE, places=PLACES)

    def test_custom_column_names(self):
        # the doubled model under other column names: score 0
        df = frame(DOUBLED, OBS, sim_col="simulated", obs_col="observed")
        self.assertAlmostEqual(gfm.kge_np(df, sim_col="simulated", obs_col="observed"),
                               DOUBLED_SCORE, places=PLACES)


class TestKgeNpZeroFlows(unittest.TestCase):
    ZEROS = [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_keep_works_with_a_zero(self):
        # sim = obs = [0, 1, 2, 3, 4]: no percentiles needed; r_s = 1, alpha_np = 1,
        # beta = 1 (mean 2 > 0); score 1
        self.assertAlmostEqual(gfm.kge_np(frame(self.ZEROS, self.ZEROS), nonpositive="keep"),
                               1.0, places=PLACES)

    def test_skip_drops_zero_row(self):
        # the obs = 0 row is dropped; the rest [1..4] has sim = obs; score 1
        self.assertAlmostEqual(gfm.kge_np(frame(self.ZEROS, self.ZEROS), nonpositive="skip"),
                               1.0, places=PLACES)

    def test_offset_does_not_shift(self):
        # log and positive are False, so 'offset' leaves the pairing as is; sim = obs; score 1
        self.assertAlmostEqual(gfm.kge_np(frame(self.ZEROS, self.ZEROS), nonpositive="offset"),
                               1.0, places=PLACES)


class TestKgeNpErrors(unittest.TestCase):
    def test_constant_observed_series(self):
        # obs all 3: mean 3 > 0, ranks all equal
        with self.assertRaises(ValueError) as cm:
            gfm.kge_np(frame(OBS, [3.0] * 5))
        self.assertIn("observed", str(cm.exception))

    def test_constant_simulated_series(self):
        # sim all 3: ranks all equal
        with self.assertRaises(ValueError) as cm:
            gfm.kge_np(frame([3.0] * 5, OBS))
        self.assertIn("simulated", str(cm.exception))

    def test_observed_mean_not_positive(self):
        # obs = [-1, 0, 1]: mean 0
        with self.assertRaises(ValueError) as cm:
            gfm.kge_np(frame([1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]))
        self.assertIn("observed", str(cm.exception))

    def test_simulated_mean_not_positive(self):
        # sim = [-1, 0, 1]: mean 0
        with self.assertRaises(ValueError) as cm:
            gfm.kge_np(frame([-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]))
        self.assertIn("simulated", str(cm.exception))

    def test_fewer_than_two_pairs(self):
        # the second pair has obs NaN and is dropped, leaving one pair
        with self.assertRaises(ValueError) as cm:
            gfm.kge_np(frame([1.0, 2.0], [1.0, np.nan]))
        self.assertIn("at least 2", str(cm.exception))

    def test_unknown_weights_key(self):
        with self.assertRaises(ValueError) as cm:
            gfm.kge_np(frame(DOUBLED, OBS), weights={"r": 1})
        self.assertIn("valid term names", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
