"""
Tests for gap_fill_metrics.kge_percentile_blocks and its block aggregation.
Expected values are derived by hand (derivation in the comment above each
one), never by running the code. Where an expected value is written with a
math call, the standard library, not the code under test, computes it; the
comment gives the decimal checked by hand.

KGE components: r = Pearson correlation, alpha = std(sim) / std(obs) (ddof 0),
beta = mean(sim) / mean(obs). Percentiles use numpy's default linear
interpolation: for n = 5 sorted values, percentile p sits at position
p/100 * 4 (0-based) between neighbors.

Run from the repository root:  python -m unittest -v tests.test_kge_percentile_blocks
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
SMALL_LOW = [25, 50]
SMALL_HIGH = [75, 90]


def frame(sim, obs, sim_col="Qsim", obs_col="Qobs"):
    return pd.DataFrame({sim_col: sim, obs_col: obs})


# Uneven model (obs = [1..5], sim = [1, 2, 3, 4, 10]), KGE terms:
#   mean(sim) = 4; sim deviation squares 9 + 4 + 1 + 0 + 36 = 50; obs 10; cross-products 20.
#   r = 20 / sqrt(500) = 2 / sqrt(5), term 0.011146...; alpha = sqrt(5), term 6 - 2 sqrt 5
#   = 1.527864...; beta = 4/3, term 1/9 = 0.111111...; S_kge = 1.650121...
#   (math computes the square roots)
S_KGE_UNEVEN = (1 - 2 / math.sqrt(5)) ** 2 + (1 - math.sqrt(5)) ** 2 + 1 / 9

# Uneven model, high block [75, 90, 95] (positions 3, 3.6, 3.8):
#   p75: obs 4, sim 4, term 0
#   p90: obs 4 + 0.6 * 1 = 4.6, sim 4 + 0.6 * 6 = 7.6, R = 7.6 / 4.6 = 38/23,
#        term (1 - 38/23)^2 = (15/23)^2 = 225/529 = 0.425331...
#   p95: obs 4.8, sim 4 + 0.8 * 6 = 8.8, R = 11/6, term (5/6)^2 = 25/36 = 0.694444...
#   sum 0 + 225/529 + 25/36 = 1.119775...
P90_TERM = 225 / 529
P95_TERM = 25 / 36


class TestAggregateBlock(unittest.TestCase):
    def test_four_values(self):
        # [0, 0, 0, 1]: mean 1/4 = 0.25; median of sorted 0, 0, 0, 1 is (0 + 0) / 2 = 0;
        # max 1; sum 1
        expected = {"mean": 0.25, "median": 0.0, "max": 1.0, "sum": 1.0}
        for block_agg, value in expected.items():
            with self.subTest(block_agg=block_agg):
                self.assertAlmostEqual(gfm._aggregate_block([0.0, 0.0, 0.0, 1.0], block_agg),
                                       value, places=PLACES)

    def test_single_value(self):
        # one value 3: mean, median, max and sum are all 3
        for block_agg in gfm.BLOCK_AGG_OPTIONS:
            with self.subTest(block_agg=block_agg):
                self.assertAlmostEqual(gfm._aggregate_block([3.0], block_agg), 3.0, places=PLACES)

    def test_unknown_block_agg_lists_options(self):
        with self.assertRaises(ValueError) as cm:
            gfm._aggregate_block([1.0], "mode")
        for option in gfm.BLOCK_AGG_OPTIONS:
            self.assertIn(option, str(cm.exception))


class TestConstants(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(gfm.DEFAULT_LOW_BLOCK, (1, 5, 10, 15))
        self.assertEqual(gfm.DEFAULT_HIGH_BLOCK, (85, 90, 95, 99))
        self.assertEqual(gfm.BLOCK_AGG_OPTIONS, ("mean", "median", "max", "sum"))
        self.assertEqual(gfm.KGE_PERCENTILE_BLOCK_TERMS,
                         ("r", "alpha", "beta", "low_block", "high_block"))


class TestKgePercentileBlockValues(unittest.TestCase):
    def test_perfect_model_every_aggregation(self):
        # sim = obs: KGE terms 0 and every ratio 1, so every percentile term and block is 0;
        # score 1 for every block_agg
        for block_agg in gfm.BLOCK_AGG_OPTIONS:
            with self.subTest(block_agg=block_agg):
                self.assertAlmostEqual(gfm.kge_percentile_blocks(frame(OBS, OBS),
                                                                 block_agg=block_agg),
                                       1.0, places=PLACES)

    def test_doubled_small_blocks_mean_median_max(self):
        # sim = 2 obs: every percentile of sim is twice that of obs (linear interpolation
        # scales), so R = 2 and each percentile term is (1 - 2)^2 = 1. KGE terms 0, 1, 1.
        # Blocks [1, 1] give 1 for mean, median and max; sum 0 + 1 + 1 + 1 + 1 = 4; score -1
        for block_agg in ("mean", "median", "max"):
            with self.subTest(block_agg=block_agg):
                score = gfm.kge_percentile_blocks(frame(DOUBLED, OBS), low_percentiles=SMALL_LOW,
                                                  high_percentiles=SMALL_HIGH,
                                                  block_agg=block_agg)
                self.assertAlmostEqual(score, -1.0, places=PLACES)

    def test_doubled_small_blocks_sum(self):
        # block sum: low_block 1 + 1 = 2, high_block 2; terms 0, 1, 1, 2, 2, sum 6;
        # score 1 - sqrt(6). math computes it: sqrt(6) = 2.449490..., score = -1.449490...
        score = gfm.kge_percentile_blocks(frame(DOUBLED, OBS), low_percentiles=SMALL_LOW,
                                          high_percentiles=SMALL_HIGH, block_agg="sum")
        self.assertAlmostEqual(score, 1 - math.sqrt(6), places=PLACES)

    def test_doubled_small_blocks_log_mean(self):
        # each percentile term (ln 2)^2, so each block (mean) is (ln 2)^2; KGE terms 0, 1, 1.
        # score 1 - sqrt(2 + 2 (ln 2)^2). math computes it: sum 2.960906..., sqrt 1.720728...,
        # score = -0.720728...
        score = gfm.kge_percentile_blocks(frame(DOUBLED, OBS), low_percentiles=SMALL_LOW,
                                          high_percentiles=SMALL_HIGH, ratio="log")
        self.assertAlmostEqual(score, 1 - math.sqrt(2 + 2 * math.log(2) ** 2), places=PLACES)

    def test_uneven_model_distinct_aggregations(self):
        # low block [25, 50] (positions 1 and 2): obs 2 and 3, sim 2 and 3, terms 0 and 0,
        # so low_block = 0 for every aggregation. High block terms [0, 225/529, 25/36]
        # (derivation above P90_TERM):
        #   mean   = (225/529 + 25/36) / 3 = 0.373258...
        #   median = 225/529 = 0.425331... (middle of sorted 0, 0.425331, 0.694444)
        #   max    = 25/36 = 0.694444...
        #   sum    = 225/529 + 25/36 = 1.119775...
        # score = 1 - sqrt(S_kge + 0 + high_block); with S_kge = 1.650121 (math computes the
        # square roots): mean -0.422455..., median -0.440643..., max -0.531198..., sum -0.664300...
        high_block = {
            "mean": (0 + P90_TERM + P95_TERM) / 3,
            "median": P90_TERM,
            "max": P95_TERM,
            "sum": 0 + P90_TERM + P95_TERM,
        }
        for block_agg, block in high_block.items():
            with self.subTest(block_agg=block_agg):
                score = gfm.kge_percentile_blocks(frame(UNEVEN, OBS), low_percentiles=[25, 50],
                                                  high_percentiles=[75, 90, 95],
                                                  block_agg=block_agg)
                self.assertAlmostEqual(score, 1 - math.sqrt(S_KGE_UNEVEN + block), places=PLACES)

    def test_zero_weight_drops_block(self):
        # low_block dropped; active r, alpha, beta, high_block, k = 4, weights 1;
        # sum 0 + 1 + 1 + 1 = 3; score 1 - sqrt(3). math computes it: score = -0.732051...
        score = gfm.kge_percentile_blocks(frame(DOUBLED, OBS), low_percentiles=SMALL_LOW,
                                          high_percentiles=SMALL_HIGH,
                                          weights={"low_block": 0})
        self.assertAlmostEqual(score, 1 - math.sqrt(3), places=PLACES)

    def test_default_blocks_equal_explicit_lists(self):
        # the defaults are (1, 5, 10, 15) and (85, 90, 95, 99), so both calls compute the same
        # thing by construction; this test only guards the default values
        df = frame(UNEVEN, OBS)
        self.assertAlmostEqual(
            gfm.kge_percentile_blocks(df),
            gfm.kge_percentile_blocks(df, low_percentiles=(1, 5, 10, 15),
                                      high_percentiles=(85, 90, 95, 99)),
            places=PLACES)

    def test_nan_pairs_dropped(self):
        # the last pair has obs NaN and is dropped; the rest is the doubled model with the
        # small blocks: score -1
        df = frame(DOUBLED + [7.0], OBS + [np.nan])
        score = gfm.kge_percentile_blocks(df, low_percentiles=SMALL_LOW,
                                          high_percentiles=SMALL_HIGH)
        self.assertAlmostEqual(score, -1.0, places=PLACES)

    def test_custom_column_names(self):
        # the doubled model with the small blocks under other column names: score -1
        df = frame(DOUBLED, OBS, sim_col="simulated", obs_col="observed")
        score = gfm.kge_percentile_blocks(df, sim_col="simulated", obs_col="observed",
                                          low_percentiles=SMALL_LOW, high_percentiles=SMALL_HIGH)
        self.assertAlmostEqual(score, -1.0, places=PLACES)


class TestKgePercentileBlockZeroFlows(unittest.TestCase):
    ZEROS = [0.0, 0.0, 0.0, 0.0, 5.0]

    def test_keep_zero_percentile_raises(self):
        # obs = sim = [0, 0, 0, 0, 5]: p25 at position 1 between 0 and 0 is 0 -> ValueError
        with self.assertRaises(ValueError):
            gfm.kge_percentile_blocks(frame(self.ZEROS, self.ZEROS), low_percentiles=SMALL_LOW,
                                      high_percentiles=SMALL_HIGH, nonpositive="keep")

    def test_offset_makes_ratios_work(self):
        # KGE terms unshifted (positive=False): sim = obs, terms 0. Percentile pairing shifted
        # by eps = 0.01 * mean([0, 0, 0, 0, 5]) = 0.01 on both series, all ratios 1. score 1
        score = gfm.kge_percentile_blocks(frame(self.ZEROS, self.ZEROS), low_percentiles=SMALL_LOW,
                                          high_percentiles=SMALL_HIGH, nonpositive="offset")
        self.assertAlmostEqual(score, 1.0, places=PLACES)


class TestKgePercentileBlockErrors(unittest.TestCase):
    def setUp(self):
        self.df = frame(DOUBLED, OBS)

    def call(self, **kwargs):
        return gfm.kge_percentile_blocks(self.df, **kwargs)

    def test_empty_low_list(self):
        with self.assertRaises(ValueError):
            self.call(low_percentiles=[])

    def test_empty_high_list(self):
        with self.assertRaises(ValueError):
            self.call(high_percentiles=())

    def test_string_instead_of_list(self):
        with self.assertRaises(TypeError):
            self.call(low_percentiles="5")

    def test_duplicate_percentile(self):
        with self.assertRaises(ValueError):
            self.call(low_percentiles=[5, 10, 5])

    def test_percentile_zero_or_hundred(self):
        for kwargs in ({"low_percentiles": [0, 5]}, {"high_percentiles": [90, 100]}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    self.call(**kwargs)

    def test_bool_or_string_percentile(self):
        for kwargs in ({"low_percentiles": [True, 5]}, {"high_percentiles": [90, "95"]}):
            with self.subTest(**kwargs):
                with self.assertRaises(TypeError):
                    self.call(**kwargs)

    def test_low_not_below_high(self):
        # max(low) = 80 >= min(high) = 70
        with self.assertRaises(ValueError):
            self.call(low_percentiles=[50, 80], high_percentiles=[70, 90])

    def test_unknown_block_agg(self):
        with self.assertRaises(ValueError):
            self.call(block_agg="mode")

    def test_unknown_ratio(self):
        with self.assertRaises(ValueError):
            self.call(ratio="square")

    def test_unknown_weights_key(self):
        with self.assertRaises(ValueError) as cm:
            self.call(weights={"low": 1})
        self.assertIn("valid term names", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
