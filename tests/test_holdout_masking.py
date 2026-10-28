"""
Tests for the scoring side of the hold-out (holdout_blocks.py: holdout_mask,
mask_hidden, check_results_folder) and for its effect on the metrics of
performance_metrics.py. Expected values are derived by hand (derivation in the
comment above each one), never by running the code.

Run from the repository root:  python -m unittest -v tests.test_holdout_masking
"""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
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
                "--start-year", "2020", "--end-year", "2025", "--n-procs", "1",
                "--run-label", "test_label"]
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

    # builds the per-run folder <tmp>/<gauge>/test_label, with run_info.json if requested
    def make_gauge_folder(self, tmp, with_run_info):
        gauge_dir = Path(tmp) / self.GAUGE / "test_label"
        gauge_dir.mkdir(parents=True)
        if with_run_info:
            (gauge_dir / "run_info.json").write_text("{}")

    # <gauge>/test_label/run_info.json exists, no --overwrite-results: the guard raises
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

    # <gauge>/test_label exists but is empty: no run_info.json, so the guard passes and
    # main() reaches the stub
    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_gauge_folder(tmp, with_run_info=False)
            with self.assertRaises(_GuardPassed):
                self.run_main(tmp)
        self.stub.assert_called_once()


# 2026-10-27 22:13:05 local time at a fixed UTC offset of -4 hours (EDT)
START = datetime(2026, 10, 27, 22, 13, 5, tzinfo=timezone(timedelta(hours=-4)))
# the 18 keys run_info.json had before run-log, and the 9 added by run-log
OLD_KEYS = ["gauge", "start_year", "end_year", "warmup_years", "n_realizations", "n_procs",
            "seed", "precip_source", "nasa_sampling", "et_method", "nonpositive_obs",
            "obs_offset_frac", "initial_storage_mm", "temperature_available",
            "models_evaluated", "models_removed", "models_skipped_no_temperature",
            "elapsed_seconds"]
NEW_KEYS = ["run_label", "status", "started_at", "holdout_mode", "holdout_years",
            "holdout_position", "holdout_intervals", "extreme_date", "extreme_value"]
# the 4 keys added by sim-helper (early write)
SIM_HELPER_KEYS = ["q_floor", "time_step", "git_commit", "git_dirty"]
# hold-out information returned by the prepare_data stub (02232000-like values)
STUB_HOLDOUT = {"intervals": [["2021-10-01", "2024-09-30"]],
                "extreme_date": "2022-10-03", "extreme_value": 7.163}
NO_HOLDOUT = {"intervals": None, "extreme_date": None, "extreme_value": None}


class TestRunLog(unittest.TestCase):
    # Tests for the per-run folder performance_holdout/<gauge>/<run_label>/ and
    # the two writes of run_info.json. Tests that call main() patch sys.argv,
    # PERFORMANCE_DIR (temporary folder), _now_local (START), prepare_data (a
    # minimal data dict), run_search (a stub) and check_parameter_space (no-op),
    # so no data is loaded or downloaded and no model runs.
    GAUGE = "02232000"

    def argv(self, *extra):
        return ["find_best_model_holdout.py", "--gauge", self.GAUGE, "--n", "50",
                "--start-year", "2020", "--end-year", "2025", "--n-procs", "1", *extra]

    def run_main(self, tmp, argv, holdout=STUB_HOLDOUT, search=None):
        data = {"T": np.zeros(1), "Qobs": np.array([1.0]), "holdout": holdout}
        if search is None:
            def search(out_dir, models, n, n_procs, data, seed):
                return pd.DataFrame(), {}
        self.out, self.err = io.StringIO(), io.StringIO()
        with mock.patch.object(fbm, "PERFORMANCE_DIR", Path(tmp)), \
                mock.patch.object(sys, "argv", argv), \
                mock.patch.object(fbm, "_now_local", return_value=START), \
                mock.patch.object(fbm, "prepare_data", return_value=data), \
                mock.patch.object(fbm, "run_search", side_effect=search) as stub, \
                mock.patch.object(fbm, "check_parameter_space"), \
                contextlib.redirect_stdout(self.out), contextlib.redirect_stderr(self.err):
            self.search_stub = stub
            fbm.main()

    def read_run_info(self, run_dir):
        return json.loads((run_dir / "run_info.json").read_text())

    # mode none: tokens none, n50, stamp 20261027_221305 (START as YYYYMMDD_HHMMSS)
    def test_label_without_holdout(self):
        self.assertEqual(hb.auto_run_label("none", 3, None, 50, START),
                         "none_n50_20261027_221305")

    # max_event, 3 years, position not given -> y3, mid; then n50 and the stamp
    def test_label_with_default_position(self):
        self.assertEqual(hb.auto_run_label("max_event", 3, None, 50, START),
                         "max_event_y3_mid_n50_20261027_221305")

    # min_flow, 3 years, position 2 given -> y3, p2; then n50 and the stamp
    def test_label_with_explicit_position(self):
        self.assertEqual(hb.auto_run_label("min_flow", 3, 2, 50, START),
                         "min_flow_y3_p2_n50_20261027_221305")

    # "my_run-1" has only letters, digits, underscore and hyphen -> accepted, and
    # the folder is <gauge>/my_run-1 exactly (no timestamp added)
    def test_manual_label_accepted(self):
        self.assertTrue(hb.is_valid_run_label("my_run-1"))
        with tempfile.TemporaryDirectory() as tmp:
            self.run_main(tmp, self.argv("--run-label", "my_run-1"))
            gauge_dir = Path(tmp) / self.GAUGE
            self.assertEqual([p.name for p in gauge_dir.iterdir()], ["my_run-1"])
            self.assertEqual(self.read_run_info(gauge_dir / "my_run-1")["run_label"], "my_run-1")

    # "" (no character), "a/b" (slash), "a b" (space), "a:b" (colon), ".." (dots)
    # each break the rule -> rejected; parse_args uses parser.error, which
    # exits with code 2 (argparse's usage-error code)
    def test_manual_label_rejected(self):
        for label in ["", "a/b", "a b", "a:b", ".."]:
            with self.subTest(label=label):
                self.assertFalse(hb.is_valid_run_label(label))
                with mock.patch.object(sys, "argv", self.argv("--run-label", label)), \
                        contextlib.redirect_stderr(io.StringIO()) as err:
                    with self.assertRaises(SystemExit) as ctx:
                        fbm.parse_args()
                self.assertEqual(ctx.exception.code, 2)
                self.assertIn("--run-label", err.getvalue())

    # the folder is PERFORMANCE_DIR / gauge / label, PERFORMANCE_DIR read at call time
    def test_run_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(fbm, "PERFORMANCE_DIR", Path(tmp)):
                self.assertEqual(fbm._run_folder(self.GAUGE, "lbl"),
                                 Path(tmp) / self.GAUGE / "lbl")

    # the run_search stub reads run_info.json when it is called, i.e. after the
    # early write and before any model would run
    def test_early_write(self):
        seen = {}

        def search(out_dir, models, n, n_procs, data, seed):
            seen["out_dir"] = out_dir
            seen["exists"] = (out_dir / "run_info.json").exists()
            seen["info"] = self.read_run_info(out_dir)
            return pd.DataFrame(), {}

        with tempfile.TemporaryDirectory() as tmp:
            self.run_main(tmp, self.argv("--holdout-mode", "max_event", "--holdout-years", "3"),
                          search=search)
        info = seen["info"]
        self.assertTrue(seen["exists"])
        self.assertEqual(info["status"], "running")
        # run_label is the name of the folder run_search was given
        self.assertEqual(info["run_label"], seen["out_dir"].name)
        # START in ISO 8601 with its -04:00 offset
        self.assertEqual(info["started_at"], "2026-10-27T22:13:05-04:00")
        # the label's last 15 characters are START as YYYYMMDD_HHMMSS, the same
        # instant as started_at (2026-10-27 22:13:05)
        self.assertEqual(info["run_label"][-15:], "20261027_221305")
        # hold-out keys: mode and years from argv; position not given -> resolved
        # to 3 // 2 = 1; the rest are the prepare_data stub's values
        self.assertEqual(info["holdout_mode"], "max_event")
        self.assertEqual(info["holdout_years"], 3)
        self.assertEqual(info["holdout_position"], 1)
        self.assertEqual(info["holdout_intervals"], [["2021-10-01", "2024-09-30"]])
        self.assertEqual(info["extreme_date"], "2022-10-03")
        self.assertEqual(info["extreme_value"], 7.163)
        # keys known only after the models ran are absent
        self.assertNotIn("models_evaluated", info)
        self.assertNotIn("elapsed_seconds", info)

    # after main() finishes: status "finished" and exactly the 18 old + 9 run-log
    # + 4 sim-helper keys (18 + 9 + 4 = 31)
    def test_final_write(self):
        expected = ["gauge", "start_year", "end_year", "warmup_years", "n_realizations",
                    "n_procs", "seed", "precip_source", "nasa_sampling", "et_method",
                    "nonpositive_obs", "obs_offset_frac", "initial_storage_mm",
                    "temperature_available", "models_evaluated", "models_removed",
                    "models_skipped_no_temperature", "elapsed_seconds",
                    "run_label", "status", "started_at", "holdout_mode", "holdout_years",
                    "holdout_position", "holdout_intervals", "extreme_date", "extreme_value",
                    "q_floor", "time_step", "git_commit", "git_dirty"]
        self.assertEqual(len(expected), 31)
        self.assertEqual(expected, OLD_KEYS + NEW_KEYS + SIM_HELPER_KEYS)
        with tempfile.TemporaryDirectory() as tmp:
            self.run_main(tmp, self.argv("--holdout-mode", "max_event", "--run-label", "final"))
            info = self.read_run_info(Path(tmp) / self.GAUGE / "final")
        self.assertEqual(info["status"], "finished")
        self.assertEqual(sorted(info), sorted(expected))

    # mode none: holdout_mode "none", the other five hold-out keys null; the
    # automatic label is none_n50_20261027_221305 (see test_label_without_holdout)
    def test_no_holdout_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_main(tmp, self.argv(), holdout=NO_HOLDOUT)
            info = self.read_run_info(Path(tmp) / self.GAUGE / "none_n50_20261027_221305")
        self.assertEqual(info["holdout_mode"], "none")
        for key in ["holdout_years", "holdout_position", "holdout_intervals",
                    "extreme_date", "extreme_value"]:
            self.assertIsNone(info[key], key)

    # run_search raises: the exception reaches the caller, and the early file
    # (status "running") is still there, since the final write never happened
    def test_crashed_run(self):
        def search(out_dir, models, n, n_procs, data, seed):
            raise RuntimeError("model crashed")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                self.run_main(tmp, self.argv("--run-label", "crash"), search=search)
            run_info = Path(tmp) / self.GAUGE / "crash" / "run_info.json"
            self.assertTrue(run_info.exists())
            self.assertEqual(self.read_run_info(run_info.parent)["status"], "running")

    # the first run leaves <gauge>/again/run_info.json; a second run with the same
    # label and no --overwrite-results hits the guard -> sys.exit(1)
    def test_second_run_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_main(tmp, self.argv("--run-label", "again"))
            with self.assertRaises(SystemExit) as ctx:
                self.run_main(tmp, self.argv("--run-label", "again"))
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("--overwrite-results", self.err.getvalue())
            # the second run stopped at the guard, before run_search
            self.search_stub.assert_not_called()

    # no temperature (data stub T = None) and only snow models selected: models 6
    # (Alpine model v1) and 12 (Alpine model v2) have 'T' in their forcing_vars
    # (marrmot_models.MODEL_REGISTRY), so both are skipped, no model is left, and
    # main() calls sys.exit("Nothing left to evaluate: ...") before the early write
    def test_stops_before_writing(self):
        data = {"T": None, "Qobs": np.array([1.0]), "holdout": NO_HOLDOUT}
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(fbm, "PERFORMANCE_DIR", Path(tmp)), \
                    mock.patch.object(sys, "argv", self.argv("--models", "6,12",
                                                             "--run-label", "snow_only")), \
                    mock.patch.object(fbm, "_now_local", return_value=START), \
                    mock.patch.object(fbm, "prepare_data", return_value=data), \
                    mock.patch.object(fbm, "run_search") as search_stub, \
                    mock.patch.object(fbm, "check_parameter_space"), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    fbm.main()
            # sys.exit(message) puts the message in SystemExit.code (the
            # interpreter prints it and exits with status 1)
            self.assertEqual(ctx.exception.code,
                             "Nothing left to evaluate: every selected model needs temperature")
            # main() stopped before run_search
            self.assertEqual(search_stub.call_count, 0)
            # no run folder and no run_info.json anywhere in the temporary folder
            self.assertFalse((Path(tmp) / self.GAUGE / "snow_only").exists())
            self.assertEqual(list(Path(tmp).rglob("run_info.json")), [])


def sim_data():
    # 5 simulated days: day 1 is warm-up (eval_mask False), days 2-5 are scored
    return {"P": np.zeros(5), "Ep": np.zeros(5), "T": None, "t": np.arange(1.0, 6.0),
            "eval_mask": np.array([False, True, True, True, True])}


class TestSimulateSeries(unittest.TestCase):
    # marrmot.simulate is stubbed with a known Q, so no model runs. Model 1
    # (Collie River Basin 1) has n_stores = 1 in marrmot_models.MODEL_REGISTRY.

    # stub Q = [5.0, 0.0, 2e-5, 3.0, -1.0]; eval_mask drops day 1 -> [0.0, 2e-5, 3.0, -1.0];
    # floor at Q_FLOOR = 1e-4: 0.0 -> 1e-4, 2e-5 -> 1e-4, 3.0 stays, -1.0 -> 1e-4
    def test_floor_and_eval_days(self):
        with mock.patch.object(fbm.marrmot, "simulate",
                               return_value={"Q": [5.0, 0.0, 2e-5, 3.0, -1.0]}) as stub:
            out = fbm.simulate_series(1, {"Smax": 100.0}, sim_data())
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.tolist(), [1e-4, 1e-4, 3.0, 1e-4])
        # S0 = n_stores (1) stores at INITIAL_STORAGE (1.0 mm) -> [1.0]; dt = TIME_STEP = 1.0
        kwargs = stub.call_args.kwargs
        self.assertEqual(kwargs["S0"].tolist(), [1.0])
        self.assertEqual(kwargs["dt"], 1.0)

    # inf on day 3 (an evaluation day) -> FloatingPointError
    def test_nonfinite_raises(self):
        with mock.patch.object(fbm.marrmot, "simulate",
                               return_value={"Q": [1.0, 1.0, np.inf, 1.0, 1.0]}):
            with self.assertRaises(FloatingPointError):
                fbm.simulate_series(1, {"Smax": 100.0}, sim_data())

    # NaN only on day 1 (warm-up, eval_mask False) -> dropped before the check,
    # no error; days 2-5 = 2.0 (above the floor) -> [2.0, 2.0, 2.0, 2.0]
    def test_nonfinite_warmup_ignored(self):
        with mock.patch.object(fbm.marrmot, "simulate",
                               return_value={"Q": [np.nan, 2.0, 2.0, 2.0, 2.0]}):
            out = fbm.simulate_series(1, {"Smax": 100.0}, sim_data())
        self.assertEqual(out.tolist(), [2.0, 2.0, 2.0, 2.0])

    # _run_one keeps its error handling: the FloatingPointError of the helper
    # becomes (None, "<type>: <message>") = (None, "FloatingPointError: simulated
    # flow is not finite")
    def test_run_one_reports_error(self):
        data = sim_data()
        data.update({"Qobs": np.ones(4), "dates": pd.date_range("2020-01-02", periods=4),
                     "hidden": np.zeros(4, dtype=bool), "nonpositive_obs": "keep",
                     "obs_offset_frac": 0.01})
        with mock.patch.object(fbm, "_DATA", data), \
                mock.patch.object(fbm.marrmot, "simulate",
                                  return_value={"Q": [1.0, 1.0, np.inf, 1.0, 1.0]}):
            result = fbm._run_one(1, {"Smax": 100.0})
        self.assertEqual(result, (None, "FloatingPointError: simulated flow is not finite"))


def git_stub(commit_out, status_out):
    def run(cmd, **kwargs):
        stdout = commit_out if cmd[1] == "rev-parse" else status_out
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
    return run


class TestGitState(unittest.TestCase):
    # subprocess.run is stubbed, so git is never called

    # rev-parse prints "abc123\n" (stripped -> "abc123"); empty porcelain -> clean (False)
    def test_clean(self):
        with mock.patch.object(fbm.subprocess, "run", side_effect=git_stub("abc123\n", "")):
            self.assertEqual(fbm._git_state(), ("abc123", False))

    # one porcelain line (" M holdout_blocks.py") -> not empty -> dirty (True)
    def test_dirty(self):
        with mock.patch.object(fbm.subprocess, "run",
                               side_effect=git_stub("abc123\n", " M holdout_blocks.py\n")):
            self.assertEqual(fbm._git_state(), ("abc123", True))

    # git not installed: subprocess.run raises FileNotFoundError -> (None, None)
    def test_git_missing(self):
        with mock.patch.object(fbm.subprocess, "run", side_effect=FileNotFoundError("git")):
            self.assertEqual(fbm._git_state(), (None, None))

    # not a repository: git exits 128, check=True raises CalledProcessError -> (None, None)
    def test_not_a_repository(self):
        with mock.patch.object(fbm.subprocess, "run",
                               side_effect=subprocess.CalledProcessError(128, ["git"])):
            self.assertEqual(fbm._git_state(), (None, None))


class TestRunInfoProvenance(unittest.TestCase):
    # main() with the same patches as TestRunLog, plus _git_state; the run_search
    # stub reads the early run_info.json
    GAUGE = "02232000"

    def early_run_info(self, git_state):
        seen = {}

        def search(out_dir, models, n, n_procs, data, seed):
            seen["info"] = json.loads((out_dir / "run_info.json").read_text())
            return pd.DataFrame(), {}

        argv = ["find_best_model_holdout.py", "--gauge", self.GAUGE, "--n", "50",
                "--start-year", "2020", "--end-year", "2025", "--n-procs", "1",
                "--run-label", "provenance"]
        data = {"T": np.zeros(1), "Qobs": np.array([1.0]), "holdout": NO_HOLDOUT}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(fbm, "PERFORMANCE_DIR", Path(tmp)), \
                mock.patch.object(sys, "argv", argv), \
                mock.patch.object(fbm, "_now_local", return_value=START), \
                mock.patch.object(fbm, "_git_state", return_value=git_state), \
                mock.patch.object(fbm, "prepare_data", return_value=data), \
                mock.patch.object(fbm, "run_search", side_effect=search), \
                mock.patch.object(fbm, "check_parameter_space"), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fbm.main()
        return seen["info"]

    # early file: q_floor = Q_FLOOR = 1e-4, time_step = TIME_STEP = 1.0, and the
    # stubbed git state ("abc123", True)
    def test_early_keys(self):
        info = self.early_run_info(("abc123", True))
        self.assertEqual(info["q_floor"], 1e-4)
        self.assertEqual(info["time_step"], 1.0)
        self.assertEqual(info["git_commit"], "abc123")
        self.assertIs(info["git_dirty"], True)

    # without git, _git_state gives (None, None) -> both keys null in JSON
    def test_no_git(self):
        info = self.early_run_info((None, None))
        self.assertIsNone(info["git_commit"])
        self.assertIsNone(info["git_dirty"])


if __name__ == "__main__":
    unittest.main()
