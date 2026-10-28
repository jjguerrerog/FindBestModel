"""
Tests for the scoring side of the hold-out (holdout_blocks.py: holdout_mask,
mask_hidden, check_results_folder) and for its effect on the metrics of
performance_metrics.py. Expected values are derived by hand (derivation in the
comment above each one), never by running the code.

Run from the repository root:  python -m unittest -v tests.test_holdout_masking
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import find_best_model_holdout as fbm  # noqa: E402
import holdout_blocks as hb  # noqa: E402
import performance_metrics as pm  # noqa: E402

T = pd.Timestamp


class TestHoldoutMask(unittest.TestCase):
    def test_inclusive_end(self):
        dates = pd.date_range("2010-09-28", "2010-10-03", freq="D")
        mask = hb.holdout_mask(dates, [(T("2009-10-01"), T("2010-09-30"))])
        # the interval ends Sep 30 inclusive: Sep 28, 29, 30 inside; Oct 1, 2, 3 outside
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(mask.tolist(), [True, True, True, False, False, False])


class TestMaskHidden(unittest.TestCase):
    def test_only_hidden_rows_become_nan(self):
        df = pd.DataFrame({"Q": [1.0, 2.0, 3.0], "Qobs": [4.0, 5.0, 6.0]},
                          index=pd.date_range("2010-01-01", periods=3, freq="D"))
        before = df.copy()
        out = hb.mask_hidden(df, np.array([False, True, False]))
        # row 2 hidden: NaN in both columns; rows 1 and 3 unchanged
        self.assertTrue(out.iloc[1].isna().all())
        self.assertEqual(out.iloc[[0, 2]].to_dict("list"), {"Q": [1.0, 3.0], "Qobs": [4.0, 6.0]})
        # the input is not modified
        pd.testing.assert_frame_equal(df, before)


class TestMetricsOnMaskedData(unittest.TestCase):
    def test_nse_ignores_hidden_day(self):
        df = pd.DataFrame({"Qsim": [1.0, 2.0, 3.0, 4.0, 100.0],
                           "Qobs": [1.0, 2.0, 3.0, 4.0, 5.0]})
        masked = hb.mask_hidden(df, np.array([False, False, False, False, True]))
        # first four days match exactly: NSE = 1 - 0 / (sum of squares > 0) = 1.0
        self.assertEqual(pm.nse(masked, "Qsim", "Qobs"), 1.0)
        # unmasked: errors 0, 0, 0, 0, 95 -> 9025; obs mean 3, sum of squares
        # 4 + 1 + 0 + 1 + 4 = 10 -> NSE = 1 - 9025 / 10 = -901.5
        self.assertAlmostEqual(pm.nse(df, "Qsim", "Qobs"), -901.5)

    def test_peak_does_not_leak(self):
        dates = pd.date_range("2008-09-01", "2008-12-31", freq="D")
        df = pd.DataFrame({"Qsim": 1.0, "Qobs": 1.0}, index=dates)
        df.loc[T("2008-11-15"), "Qsim"] = 50.0
        hidden = hb.holdout_mask(dates, [(T("2008-10-01"), T("2009-09-30"))])
        # both masked: only Sep 2008 is left in calendar year 2008; both peaks 1.0
        # -> difference 0
        masked = hb.mask_hidden(df, hidden)
        self.assertEqual(pm.mean_annual_peak_flow_difference(masked, "Qsim", "Qobs"), 0.0)
        # unmasked: 2008 peaks 50.0 (Nov 15) and 1.0 -> difference 49
        self.assertEqual(pm.mean_annual_peak_flow_difference(df, "Qsim", "Qobs"), 49.0)
        # only the observed flow masked: the hidden Nov 15 simulated peak (50.0) is
        # still the 2008 peak, against an observed peak of 1.0 -> 49 (the leak)
        obs_only = df.copy()
        obs_only.loc[hidden, "Qobs"] = np.nan
        self.assertEqual(pm.mean_annual_peak_flow_difference(obs_only, "Qsim", "Qobs"), 49.0)


def ones(*nan_days):
    obs = pd.Series(1.0, index=pd.date_range("2008-01-01", "2013-12-31", freq="D"))
    for d in nan_days:
        obs[T(d)] = np.nan
    return obs


class TestLowCoverage(unittest.TestCase):
    WY2009 = [(T("2008-10-01"), T("2009-09-30"))]
    WY2012 = [(T("2011-10-01"), T("2012-09-30"))]

    def test_common_year_threshold(self):
        # water year 2009 has 365 days; 99% of 365 = 361.35
        # 3 NaN -> 362 observed (362/365 = 99.18%) -> OK
        self.assertEqual(hb.low_coverage_water_years(
            ones("2008-11-01", "2009-02-01", "2009-05-01"), self.WY2009), [])
        # 4 NaN -> 361 observed (361/365 = 98.90%) -> reported
        low = hb.low_coverage_water_years(
            ones("2008-11-01", "2009-02-01", "2009-05-01", "2009-08-01"), self.WY2009)
        self.assertEqual([row[:3] for row in low], [(2009, 361, 365)])
        self.assertAlmostEqual(low[0][3], 361 / 365)

    def test_leap_year_threshold(self):
        # water year 2012 has 366 days (Feb 29, 2012); 99% of 366 = 362.34
        # 3 NaN (one on Feb 29) -> 363 observed (99.18%) -> OK
        self.assertEqual(hb.low_coverage_water_years(
            ones("2011-11-01", "2012-02-29", "2012-05-01"), self.WY2012), [])
        # 4 NaN -> 362 observed (362/366 = 98.91%) -> reported
        low = hb.low_coverage_water_years(
            ones("2011-11-01", "2012-02-29", "2012-05-01", "2012-08-01"), self.WY2012)
        self.assertEqual([row[:3] for row in low], [(2012, 362, 366)])
        self.assertAlmostEqual(low[0][3], 362 / 366)

    def test_only_short_year_reported(self):
        # block = water years 2009-2011; only 2010 (Oct 1, 2009 to Sep 30, 2010,
        # 365 days) has 5 NaN -> 360 observed (98.63%); 2009 and 2011 complete
        obs = ones("2009-10-15", "2009-12-15", "2010-02-15", "2010-04-15", "2010-06-15")
        low = hb.low_coverage_water_years(obs, [(T("2008-10-01"), T("2011-09-30"))])
        self.assertEqual([row[:3] for row in low], [(2010, 360, 365)])
        self.assertAlmostEqual(low[0][3], 360 / 365)

    def test_complete_series(self):
        # no NaN: every water year 100% observed -> nothing reported
        self.assertEqual(hb.low_coverage_water_years(
            ones(), [(T("2008-10-01"), T("2011-09-30"))]), [])


class TestCheckResultsFolder(unittest.TestCase):
    def test_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "02232000"
            # the folder does not exist -> passes
            hb.check_results_folder(out_dir)
            # it exists without run_info.json -> passes
            out_dir.mkdir()
            (out_dir / "summary.parquet").write_text("")
            hb.check_results_folder(out_dir)
            # it has run_info.json -> raises, naming the file and the flag
            (out_dir / "run_info.json").write_text("{}")
            with self.assertRaises(FileExistsError) as ctx:
                hb.check_results_folder(out_dir)
            self.assertIn("run_info.json", str(ctx.exception))
            self.assertIn("--overwrite-results", str(ctx.exception))
            # the same folder with overwrite=True -> passes
            hb.check_results_folder(out_dir, overwrite=True)


class _GuardPassed(Exception):
    """Raised by the stub of check_parameter_space: main() got past the guard."""


class TestMainOverwriteGuard(unittest.TestCase):
    # main() calls check_parameter_space() right after the guard; a stub that
    # raises _GuardPassed stops main() there, so no data is prepared and no
    # model runs. PERFORMANCE_DIR points to a temporary folder.
    GAUGE = "02232000"

    def run_main(self, tmp, overwrite=False):
        argv = ["find_best_model_holdout.py", "--gauge", self.GAUGE, "--n", "1",
                "--start-year", "2020", "--end-year", "2025", "--n-procs", "1"]
        if overwrite:
            argv.append("--overwrite-results")
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(fbm, "PERFORMANCE_DIR", Path(tmp)), \
                mock.patch.object(sys, "argv", argv), \
                mock.patch.object(fbm, "check_parameter_space",
                                  side_effect=_GuardPassed) as stub, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                fbm.main()
            finally:
                self.stub, self.out, self.err = stub, out.getvalue(), err.getvalue()

    def make_gauge_folder(self, tmp, with_run_info):
        gauge_dir = Path(tmp) / self.GAUGE
        gauge_dir.mkdir()
        if with_run_info:
            (gauge_dir / "run_info.json").write_text("{}")

    # <gauge>/run_info.json exists and no --overwrite-results: the guard raises
    # FileExistsError, main() catches it, prints the message and calls sys.exit(1)
    def test_blocked_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_gauge_folder(tmp, with_run_info=True)
            with self.assertRaises(SystemExit) as ctx:
                self.run_main(tmp)
        # the catch calls sys.exit(1) -> code 1 (non-zero)
        self.assertEqual(ctx.exception.code, 1)
        self.assertNotEqual(ctx.exception.code, 0)
        # the printed message is the FileExistsError text, which names the flag
        self.assertIn("--overwrite-results", self.err)
        # main() stopped at the guard: check_parameter_space never called, so the
        # sentinel was not raised
        self.stub.assert_not_called()
        # only the message is printed, no traceback
        self.assertNotIn("Traceback", self.err + self.out)

    # same folder with --overwrite-results: the guard does not raise, so main()
    # reaches the stub (sentinel raised, no SystemExit)
    def test_overwrite_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_gauge_folder(tmp, with_run_info=True)
            with self.assertRaises(_GuardPassed):
                self.run_main(tmp, overwrite=True)
        self.stub.assert_called_once()

    # no <gauge> folder (a gauge never run before): no run_info.json, so the
    # guard passes and main() reaches the stub
    def test_missing_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(_GuardPassed):
                self.run_main(tmp)
        self.stub.assert_called_once()

    # <gauge> exists but is empty: no run_info.json, so the guard passes and
    # main() reaches the stub
    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_gauge_folder(tmp, with_run_info=False)
            with self.assertRaises(_GuardPassed):
                self.run_main(tmp)
        self.stub.assert_called_once()


if __name__ == "__main__":
    unittest.main()
