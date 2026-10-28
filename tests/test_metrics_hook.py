"""
Tests for the metrics-hook in the copy find_best_model_holdout.py: compute_metrics
appends the gap_fill_metrics columns and metrics_error after the 14 original
columns, and SUMMARY_METRICS is extended. Expected values are derived by hand
(derivation in the comment above each one), never by running the code. Where an
expected value is written with a math call, the standard library, not the code
under test, computes it; the comment gives the decimal checked by hand.

jdkge warns about short records (fewer than 4 years): the warning is suppressed
in every call below.

Run from the repository root:  python -m unittest -v tests.test_metrics_hook
"""

import math
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import find_best_model as original  # noqa: E402
import find_best_model_holdout as fbm  # noqa: E402
import gap_fill_metrics as gm  # noqa: E402
from model_parameter_space import PARAM_SPACE  # noqa: E402

PLACES = 12
ORIGINAL_COLUMNS = ["NSE", "KGE", "KGE_r", "KGE_alpha", "KGE_beta", "PBIAS", "correlation",
                    "mean_annual_peak_flow_diff", "pdi_ave", "pdi_ave2", "pdi_ave3", "pdi_low",
                    "pdi_high", "log_NSE"]
NEW_COLUMNS = ["KGE_TA_p5_p95", "KGE_TA_p5_p95_log", "KGE_TA_p2_p98", "KGE_TA_p2_p98_log",
               "KGE_PB_pdi", "KGE_PB_pdi_log", "PNSE_p5_p95", "PNSE_p5_p95_log", "KGE_NP", "JDKGE"]
PERCENTILE_COLUMNS = NEW_COLUMNS[:8]
ZEROS = [0.0, 0.0, 0.0, 0.0, 5.0]


def frame(q, qobs):
    """Frame with the copy's column names and a daily DatetimeIndex."""
    return pd.DataFrame({"Q": q, "Qobs": qobs},
                        index=pd.date_range("2020-01-01", periods=len(qobs), freq="D"))


def long_frame():
    # A fixture, not an expected value: 1100 daily rows (just over 3 years) with
    # varying positive flows, Qobs = 1 + (i % 7) + 0.001 i and Q = 1.1 * Qobs, so
    # every original metric is defined. Both modules are compared on it.
    i = np.arange(1100)
    qobs = 1.0 + (i % 7) + 0.001 * i
    return frame(1.1 * qobs, qobs)


def quiet(fn, *args, **kwargs):
    """Call fn with every warning suppressed (jdkge's short-record warning)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


class OriginalColumnsAssertions(unittest.TestCase):
    def assert_same_value(self, a, b, column):
        """Equal floats, or both NaN."""
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return
        self.assertEqual(a, b, column)

    def assert_original_columns_match(self, metrics, sim, **kwargs):
        expected = quiet(original.compute_metrics, sim, **kwargs)
        self.assertEqual(list(expected), ORIGINAL_COLUMNS)
        for column in ORIGINAL_COLUMNS:
            with self.subTest(column=column):
                self.assert_same_value(metrics[column], expected[column], column)


class TestComputeMetricsKeys(unittest.TestCase):
    def test_keys_original_then_new_then_metrics_error(self):
        metrics = quiet(fbm.compute_metrics, long_frame())
        self.assertEqual(list(metrics), ORIGINAL_COLUMNS + gm.metric_columns() + ["metrics_error"])


class TestOriginalColumnsUnchanged(OriginalColumnsAssertions):
    def test_matches_original_module(self):
        # independent code: the original script's compute_metrics on the same frame
        sim = long_frame()
        self.assert_original_columns_match(quiet(fbm.compute_metrics, sim), sim)


class TestNewColumnsWired(unittest.TestCase):
    def test_doubled_model(self):
        # Q = 2 * Qobs, Qobs = [1..5]: every percentile ratio 2, linear term 1, log term
        # (ln 2)^2 (math computes it: 0.480453...); KGE terms r 0, alpha 1, beta 1.
        # Linear tail columns: 1 - sqrt(0 + 1 + 1 + 1 + 1) = -1.
        # Log tail columns: 1 - sqrt(2 + 2 (ln 2)^2) = 1 - sqrt(2.960906) = -0.720728...
        # PNSE: NSE = 1 - 55/10 = -4.5, term 30.25; linear 1 - sqrt(32.25) = -4.678908...;
        # log 1 - sqrt(2 (ln 2)^2 + 30.25) = 1 - sqrt(31.210906) = -4.586672...
        # KGE_NP: r_s 1, alpha_np 1, beta 2: 1 - sqrt(1) = 0.
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
        metrics = quiet(fbm.compute_metrics, frame([2.0, 4.0, 6.0, 8.0, 10.0],
                                                   [1.0, 2.0, 3.0, 4.0, 5.0]))
        for column, value in expected.items():
            with self.subTest(column=column):
                self.assertAlmostEqual(metrics[column], value, places=PLACES)
        # JDKGE: its beta term is (1 - 2)^2 = 1, so the score is at most 0
        self.assertLessEqual(metrics["JDKGE"], 0.0)
        self.assertIsNone(metrics["metrics_error"])


class TestFailuresStayLocal(OriginalColumnsAssertions):
    def test_value_error_handled_inside_compute_all(self):
        # Q = Qobs = [0, 0, 0, 0, 5] (a fixture; the real Q is floored), keep: the observed
        # 5th, 2nd and 1st percentiles are 0, so the eight percentile-based columns are NaN;
        # KGE_NP: obs mean 1 > 0, ranks not constant, sim = obs: 1; JDKGE: identical arrays,
        # zeros become epsilon in both, JSD 0: 1. The ValueErrors are handled in compute_all.
        metrics = quiet(fbm.compute_metrics, frame(ZEROS, ZEROS))
        for column in PERCENTILE_COLUMNS:
            with self.subTest(column=column):
                self.assertTrue(math.isnan(metrics[column]))
        self.assertAlmostEqual(metrics["KGE_NP"], 1.0, places=PLACES)
        self.assertAlmostEqual(metrics["JDKGE"], 1.0, places=PLACES)
        self.assertIsNone(metrics["metrics_error"])

    def test_unexpected_exception_gives_nan_and_message(self):
        sim = long_frame()
        with mock.patch.object(gm, "compute_all", side_effect=RuntimeError("boom")):
            metrics = quiet(fbm.compute_metrics, sim)
        self.assert_original_columns_match(metrics, sim)
        for column in NEW_COLUMNS:
            with self.subTest(column=column):
                self.assertTrue(math.isnan(metrics[column]))
        self.assertEqual(metrics["metrics_error"], "RuntimeError: boom")


class TestSwitchReachesNewMetrics(unittest.TestCase):
    def test_offset_makes_all_new_columns_one(self):
        # offset: KGE terms unshifted and perfect; percentile pairing shifted by
        # 0.01 * mean(clipped obs) = 0.01 on both series, every ratio 1; pm.nse of sim = obs
        # is 1; KGE_NP and JDKGE 1 as in the keep case. All ten columns 1
        metrics = quiet(fbm.compute_metrics, frame(ZEROS, ZEROS), nonpositive_obs="offset")
        for column in NEW_COLUMNS:
            with self.subTest(column=column):
                self.assertAlmostEqual(metrics[column], 1.0, places=PLACES)

    def test_keep_leaves_percentile_columns_nan(self):
        metrics = quiet(fbm.compute_metrics, frame(ZEROS, ZEROS), nonpositive_obs="keep")
        for column in PERCENTILE_COLUMNS:
            with self.subTest(column=column):
                self.assertTrue(math.isnan(metrics[column]))


class TestSummaryMetrics(unittest.TestCase):
    def test_original_five_then_new_columns(self):
        self.assertEqual(fbm.SUMMARY_METRICS,
                         ["NSE", "KGE", "log_NSE", "pdi_ave", "pdi_low"] + gm.metric_columns())

    def test_fifteen_entries_no_duplicates(self):
        self.assertEqual(len(fbm.SUMMARY_METRICS), 15)
        self.assertEqual(len(set(fbm.SUMMARY_METRICS)), 15)


class TestNoNameClashes(unittest.TestCase):
    NEW_NAMES = set(gm.metric_columns()) | {"metrics_error"}

    def test_no_clash_with_original_columns(self):
        self.assertEqual(self.NEW_NAMES & set(ORIGINAL_COLUMNS), set())

    def test_no_clash_with_error_and_model(self):
        self.assertEqual(self.NEW_NAMES & {"error", "model"}, set())

    def test_no_clash_with_model_parameters(self):
        # every parameter name of every model in PARAM_SPACE (plain, Frac and Derived)
        parameters = {name for space in PARAM_SPACE.values() for name in space}
        self.assertEqual(self.NEW_NAMES & parameters, set())


class TestBuildTableAcceptsNewColumns(unittest.TestCase):
    def test_row_with_new_columns(self):
        metrics = quiet(fbm.compute_metrics, long_frame())
        samples = pd.DataFrame([{"Smax": 100.0}])
        table = fbm._build_table(samples, [(metrics, None)])
        for column in NEW_COLUMNS + ["metrics_error"]:
            with self.subTest(column=column):
                self.assertIn(column, table.columns)


if __name__ == "__main__":
    unittest.main()
