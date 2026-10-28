"""
Tests for holdout_results.py (reading a hold-out run and picking winners).
Expected values are derived by hand (derivation in the comment above each one),
never by running the code.

Run from the repository root:  python -m unittest -v tests.test_holdout_results
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import holdout_results as hr  # noqa: E402

NAN = np.nan


def table(rows, columns=("p", "error", "M")):
    return pd.DataFrame(rows, columns=list(columns))


class TestBestRow(unittest.TestCase):
    def test_skips_failed_and_nan_rows(self):
        # ok values 0.2, 0.5, 0.9 (p = 1, 2, 3); 0.99 is a failed row, NaN is ignored: p = 3
        t = table([(1, None, 0.2), (2, None, 0.5), (3, None, 0.9), (4, "boom", 0.99),
                   (5, None, NAN)])
        self.assertEqual(hr.best_row(t, "M")["p"], 3)

    def test_tie_goes_to_first_row(self):
        # both 0.9: the first in table order, p = 1
        t = table([(1, None, 0.9), (2, None, 0.9)])
        self.assertEqual(hr.best_row(t, "M")["p"], 1)

    def test_no_qualifying_row_gives_none(self):
        t = table([(1, "boom", 0.5), (2, None, NAN)])
        self.assertIsNone(hr.best_row(t, "M"))

    def test_missing_metric_raises(self):
        with self.assertRaises(KeyError):
            hr.best_row(table([(1, None, 0.5)]), "N")


class TestBestModelPerMetric(unittest.TestCase):
    SUMMARY = pd.DataFrame({"best_A": [0.1, 0.9, 0.5], "best_B": [0.7, 0.7, NAN],
                            "best_C": [NAN, NAN, 0.4], "best_D": [NAN, NAN, NAN]},
                           index=pd.Index([1, 2, 3], name="model"))

    def test_winner_and_second(self):
        # A: 0.9 (model 2) then 0.5 (model 3)
        self.assertEqual(hr.best_model_per_metric(self.SUMMARY, ["A"])["A"],
                         {"model": 2, "value": 0.9, "second_model": 3, "second_value": 0.5})

    def test_tie_goes_to_lowest_model(self):
        # B: models 1 and 2 both 0.7, model 3 NaN: winner 1, second 2
        self.assertEqual(hr.best_model_per_metric(self.SUMMARY, ["B"])["B"],
                         {"model": 1, "value": 0.7, "second_model": 2, "second_value": 0.7})

    def test_single_value_has_no_second(self):
        # C: only model 3 has a value
        self.assertEqual(hr.best_model_per_metric(self.SUMMARY, ["C"])["C"],
                         {"model": 3, "value": 0.4, "second_model": None, "second_value": None})

    def test_all_nan_has_no_winner(self):
        result = hr.best_model_per_metric(self.SUMMARY, ["D"])["D"]
        self.assertIsNone(result["model"])
        self.assertIsNone(result["value"])

    def test_missing_column_raises(self):
        with self.assertRaises(KeyError) as cm:
            hr.best_model_per_metric(self.SUMMARY, ["E"])
        self.assertIn("best_E", str(cm.exception))


class TestSharedWinnerGroups(unittest.TestCase):
    def test_groups_in_input_order_without_none(self):
        self.assertEqual(hr.shared_winner_groups({"A": 2, "B": 2, "C": 3, "D": None}),
                         {2: ["A", "B"], 3: ["C"]})


class TestNanShare(unittest.TestCase):
    def test_share_over_ok_rows(self):
        # ok rows: [1, NaN, 3] and [NaN, 5] = 5 rows, 2 NaN; the failed NaN row does not count:
        # 2 / 5 = 0.4
        tables = {1: table([(1, None, 1.0), (2, None, NAN), (3, None, 3.0)]),
                  2: table([(1, None, NAN), (2, None, 5.0), (3, "boom", NAN)])}
        self.assertAlmostEqual(hr.nan_share(tables, "M"), 0.4, places=12)

    def test_no_ok_rows_gives_nan(self):
        tables = {1: table([(1, "boom", 1.0)])}
        self.assertTrue(math.isnan(hr.nan_share(tables, "M")))


class TestMetricRankCorrelation(unittest.TestCase):
    def test_identical_and_reversed_ranks(self):
        # ok rows A = [1, 2, 3], B = [10, 20, 30], C = [3, 2, 1]; the failed row (9, 9, 9)
        # is excluded. Ranks: A and B identical (1), C reversed (-1); diagonal 1
        cols = ("error", "A", "B", "C")
        tables = {1: table([(None, 1.0, 10.0, 3.0), (None, 2.0, 20.0, 2.0)], cols),
                  2: table([(None, 3.0, 30.0, 1.0), ("boom", 9.0, 9.0, 9.0)], cols)}
        corr = hr.metric_rank_correlation(tables, ["A", "B", "C"])
        expected = {("A", "B"): 1.0, ("A", "C"): -1.0, ("B", "C"): -1.0,
                    ("A", "A"): 1.0, ("B", "B"): 1.0, ("C", "C"): 1.0}
        for (a, b), value in expected.items():
            with self.subTest(pair=(a, b)):
                self.assertAlmostEqual(corr.loc[a, b], value, places=12)

    def test_all_nan_metric_gives_nan(self):
        cols = ("error", "A", "Z")
        tables = {1: table([(None, 1.0, NAN), (None, 2.0, NAN), (None, 3.0, NAN)], cols)}
        corr = hr.metric_rank_correlation(tables, ["A", "Z"])
        self.assertTrue(math.isnan(corr.loc["A", "Z"]))
        self.assertTrue(math.isnan(corr.loc["Z", "A"]))


class TestLoadRun(unittest.TestCase):
    def write_run(self, folder, with_summary=True):
        (folder / "run_info.json").write_text(json.dumps({"status": "finished"}))
        files = {1: "model_01_Collie_River_Basin_1.parquet", 7: "model_07_GR4J.parquet"}
        for model, name in files.items():
            table([(1, None, 0.5)]).to_parquet(folder / name)
        if with_summary:
            summary = pd.DataFrame({"name": ["Collie River Basin 1", "GR4J"],
                                    "file": list(files.values())},
                                   index=pd.Index(list(files), name="model"))
            summary.to_parquet(folder / "summary.parquet")

    def test_loads_info_summary_and_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_run(Path(tmp))
            run = hr.load_run(tmp)
        self.assertEqual(set(run), {"run_info", "summary", "tables"})
        self.assertEqual(sorted(run["tables"]), [1, 7])
        self.assertEqual(run["run_info"]["status"], "finished")

    def test_missing_summary_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_run(Path(tmp), with_summary=False)
            with self.assertRaises(FileNotFoundError):
                hr.load_run(tmp)


if __name__ == "__main__":
    unittest.main()
