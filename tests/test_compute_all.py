"""
Tests for gap_fill_metrics.compute_all, metric_columns and DEFAULT_CONFIGS.
Expected values are derived by hand (derivation in the comment above each one),
never by running the code. Where an expected value is written with a math call,
the standard library, not the code under test, computes it; the comment gives
the decimal checked by hand. jdkge warns about short records for every
five-value series: the warning is suppressed wherever the default configs run.

Run from the repository root:  python -m unittest -v tests.test_compute_all
"""

import math
import sys
import unittest
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gap_fill_metrics as gfm  # noqa: E402

PLACES = 12
OBS = [1.0, 2.0, 3.0, 4.0, 5.0]
DOUBLED = [2.0, 4.0, 6.0, 8.0, 10.0]
ZEROS = [0.0, 0.0, 0.0, 0.0, 5.0]
COLUMNS = ["KGE_TA_p5_p95", "KGE_TA_p5_p95_log", "KGE_TA_p2_p98", "KGE_TA_p2_p98_log",
           "KGE_PB_pdi", "KGE_PB_pdi_log", "PNSE_p5_p95", "PNSE_p5_p95_log", "KGE_NP", "JDKGE"]
PERCENTILE_COLUMNS = COLUMNS[:8]


def frame(sim, obs, sim_col="Qsim", obs_col="Qobs"):
    return pd.DataFrame({sim_col: sim, obs_col: obs})


def quiet_compute_all(df, **kwargs):
    """compute_all with the jdkge short-record warning suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gfm.compute_all(df, **kwargs)


def raises_runtime_error(df, sim_col, obs_col, nonpositive="keep", offset_frac=0.01):
    raise RuntimeError("bug")


def raises_value_error(df, sim_col, obs_col, nonpositive="keep", offset_frac=0.01):
    raise ValueError("undefined")


def returns_half(df, sim_col, obs_col, nonpositive="keep", offset_frac=0.01):
    return 0.5


def three_arguments_only(df, sim_col, obs_col):
    return 0.25


class TestDefaultConfigs(unittest.TestCase):
    def test_metric_columns_in_order(self):
        self.assertEqual(gfm.metric_columns(), COLUMNS)

    def test_ten_default_configs(self):
        self.assertEqual(len(gfm.DEFAULT_CONFIGS), 10)


class TestComputeAllValues(unittest.TestCase):
    def test_perfect_model_all_ones(self):
        # sim = obs: every ratio 1, KGE terms 0, NSE 1, rank correlation 1, identical
        # histograms; every column 1
        result = quiet_compute_all(frame(OBS, OBS))
        for column in COLUMNS:
            with self.subTest(column=column):
                self.assertAlmostEqual(result[column], 1.0, places=PLACES)

    def test_doubled_model(self):
        # sim = 2 * obs: R = 2 at every percentile, linear term 1, log term (ln 2)^2
        # (math computes it: 0.480453...). KGE terms r 0, alpha 1, beta 1.
        # Linear tail columns: 1 - sqrt(0 + 1 + 1 + 1 + 1) = -1.
        # Log tail columns: 1 - sqrt(2 + 2 (ln 2)^2) = 1 - sqrt(2.960906) = -0.720728...
        # (the block mean of equal terms equals the term).
        # PNSE: NSE = 1 - 55/10 = -4.5, term 5.5^2 = 30.25; linear 1 - sqrt(32.25)
        # = 1 - 5.678908 = -4.678908...; log 1 - sqrt(2 (ln 2)^2 + 30.25)
        # = 1 - sqrt(31.210906) = 1 - 5.586672 = -4.586672...
        # KGE_NP: r_s 1, alpha_np 1 (scaling cancels), beta 2: 1 - sqrt(1) = 0.
        log2_sq = math.log(2) ** 2
        expected = {
            "KGE_TA_p5_p95": -1.0, "KGE_TA_p2_p98": -1.0, "KGE_PB_pdi": -1.0,
            "KGE_TA_p5_p95_log": 1 - math.sqrt(2 + 2 * log2_sq),
            "KGE_TA_p2_p98_log": 1 - math.sqrt(2 + 2 * log2_sq),
            "KGE_PB_pdi_log": 1 - math.sqrt(2 + 2 * log2_sq),
            "PNSE_p5_p95": 1 - math.sqrt(32.25),
            "PNSE_p5_p95_log": 1 - math.sqrt(2 * log2_sq + 30.25),
            "KGE_NP": 0.0,
        }
        result = quiet_compute_all(frame(DOUBLED, OBS))
        for column, value in expected.items():
            with self.subTest(column=column):
                self.assertAlmostEqual(result[column], value, places=PLACES)

    def test_doubled_model_jdkge_at_most_zero(self):
        # beta term (1 - 2)^2 = 1, so the sum of terms is at least 1 and the score at most 0
        self.assertLessEqual(quiet_compute_all(frame(DOUBLED, OBS))["JDKGE"], 0.0)

    def test_custom_column_names(self):
        # the doubled model under other column names: KGE_NP 0
        df = frame(DOUBLED, OBS, sim_col="simulated", obs_col="observed")
        result = quiet_compute_all(df, sim_col="simulated", obs_col="observed")
        self.assertAlmostEqual(result["KGE_NP"], 0.0, places=PLACES)

    def test_result_follows_config_order(self):
        self.assertEqual(list(quiet_compute_all(frame(OBS, OBS))), COLUMNS)


class TestComputeAllZeroFlows(unittest.TestCase):
    def test_value_error_gives_nan_and_errors_entry(self):
        # obs = sim = [0, 0, 0, 0, 5], keep: the observed 5th (and 2nd, 1st) percentile is 0,
        # so the eight percentile-based columns raise ValueError and are NaN.
        # KGE_NP: obs mean 1 > 0, ranks not constant, sim = obs: 1.
        # JDKGE: identical arrays, zeros become epsilon in both, JSD 0: 1.
        errors = {}
        result = quiet_compute_all(frame(ZEROS, ZEROS), nonpositive="keep", errors=errors)
        for column in PERCENTILE_COLUMNS:
            with self.subTest(column=column):
                self.assertTrue(math.isnan(result[column]))
        self.assertEqual(sorted(errors), sorted(PERCENTILE_COLUMNS))
        self.assertAlmostEqual(result["KGE_NP"], 1.0, places=PLACES)
        self.assertAlmostEqual(result["JDKGE"], 1.0, places=PLACES)

    def test_without_errors_dict_same_values(self):
        # same data, errors None: the same NaN columns and the same two values 1
        result = quiet_compute_all(frame(ZEROS, ZEROS), nonpositive="keep")
        for column in PERCENTILE_COLUMNS:
            with self.subTest(column=column):
                self.assertTrue(math.isnan(result[column]))
        self.assertAlmostEqual(result["KGE_NP"], 1.0, places=PLACES)
        self.assertAlmostEqual(result["JDKGE"], 1.0, places=PLACES)

    def test_offset_removes_nan(self):
        # offset: KGE terms unshifted and perfect; the percentile pairing shifts both series
        # by 0.01 * mean(clipped obs) = 0.01 * 1 = 0.01, so every ratio is 1; pm.nse of
        # sim = obs is 1; KGE_NP and JDKGE as above. Every column 1, no errors
        errors = {}
        result = quiet_compute_all(frame(ZEROS, ZEROS), nonpositive="offset", errors=errors)
        for column in COLUMNS:
            with self.subTest(column=column):
                self.assertAlmostEqual(result[column], 1.0, places=PLACES)
        self.assertEqual(errors, {})


class TestComputeAllExceptions(unittest.TestCase):
    def test_other_exception_propagates(self):
        configs = (gfm.MetricConfig("bug", raises_runtime_error),)
        with self.assertRaises(RuntimeError):
            gfm.compute_all(frame(OBS, OBS), configs=configs)

    def test_value_error_column_nan_others_computed(self):
        configs = (gfm.MetricConfig("undefined", raises_value_error),
                   gfm.MetricConfig("half", returns_half))
        errors = {}
        result = gfm.compute_all(frame(OBS, OBS), configs=configs, errors=errors)
        self.assertTrue(math.isnan(result["undefined"]))
        self.assertEqual(result["half"], 0.5)
        self.assertEqual(errors, {"undefined": "undefined"})

    def test_accepts_nonpositive_false_skips_arguments(self):
        configs = (gfm.MetricConfig("plain", three_arguments_only, accepts_nonpositive=False),)
        self.assertEqual(gfm.compute_all(frame(OBS, OBS), configs=configs), {"plain": 0.25})

    def test_accepts_nonpositive_true_passes_arguments(self):
        # the function does not take nonpositive: the TypeError propagates
        configs = (gfm.MetricConfig("plain", three_arguments_only),)
        with self.assertRaises(TypeError):
            gfm.compute_all(frame(OBS, OBS), configs=configs)


class TestConfigValidation(unittest.TestCase):
    DUPLICATES = (gfm.MetricConfig("same", returns_half), gfm.MetricConfig("same", returns_half))

    def test_duplicate_columns_raise(self):
        with self.assertRaises(ValueError):
            gfm.metric_columns(self.DUPLICATES)
        with self.assertRaises(ValueError):
            gfm.compute_all(frame(OBS, OBS), configs=self.DUPLICATES)

    def test_empty_configs_raise(self):
        with self.assertRaises(ValueError):
            gfm.metric_columns(())
        with self.assertRaises(ValueError):
            gfm.compute_all(frame(OBS, OBS), configs=())

    def test_errors_not_a_dict_raises(self):
        with self.assertRaises(TypeError):
            gfm.compute_all(frame(OBS, OBS), configs=(gfm.MetricConfig("half", returns_half),),
                            errors=[])


if __name__ == "__main__":
    unittest.main()
