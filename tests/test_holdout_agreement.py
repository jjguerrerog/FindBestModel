"""
Tests for winner_agreement_counts in holdout_results.py (cross-gauge agreement
of the winning models). Expected values are derived by hand (derivation in the
comment above each one), never by running the code.

Run from the repository root:  python -m unittest -v tests.test_holdout_agreement
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import holdout_results as hr  # noqa: E402

METRICS = ["A", "B", "C"]


class TestWinnerAgreementCounts(unittest.TestCase):
    WINNERS = {"g1": {"A": {"model": 2}, "B": {"model": 2}, "C": {"model": 3}},
               "g2": {"A": {"model": 1}, "B": {"model": 4}, "C": {"model": None}}}

    def test_counts_pairs_with_winners_and_same_model(self):
        agree, both = hr.winner_agreement_counts(self.WINNERS, METRICS)
        # g1 has winners for A, B, C; g2 for A and B only (C is None).
        # A-B: g1 2 and 2 agree, g2 1 and 4 differ: agree 1, both 2.
        # A-C: only g1 has both, 2 and 3 differ: agree 0, both 1.
        # B-C: only g1 has both, 2 and 3 differ: agree 0, both 1.
        # Diagonal: A has a winner in 2 gauges, B in 2, C in 1 (same for agree and both).
        expected_agree = {("A", "A"): 2, ("B", "B"): 2, ("C", "C"): 1,
                          ("A", "B"): 1, ("A", "C"): 0, ("B", "C"): 0}
        expected_both = {("A", "A"): 2, ("B", "B"): 2, ("C", "C"): 1,
                         ("A", "B"): 2, ("A", "C"): 1, ("B", "C"): 1}
        for (a, b), value in expected_agree.items():
            with self.subTest(matrix="agree", pair=(a, b)):
                self.assertEqual(agree.loc[a, b], value)
                self.assertEqual(agree.loc[b, a], value)  # symmetric
        for (a, b), value in expected_both.items():
            with self.subTest(matrix="both", pair=(a, b)):
                self.assertEqual(both.loc[a, b], value)
                self.assertEqual(both.loc[b, a], value)  # symmetric

    def test_missing_metric_raises_naming_gauge_and_metric(self):
        winners = {"g1": {"A": {"model": 2}, "B": {"model": 2}}}
        with self.assertRaises(KeyError) as cm:
            hr.winner_agreement_counts(winners, METRICS)
        self.assertIn("g1", str(cm.exception))
        self.assertIn("C", str(cm.exception))

    def test_empty_input_gives_zeros(self):
        agree, both = hr.winner_agreement_counts({}, METRICS)
        # no gauge: every count is 0, shape 3 x 3
        for frame in (agree, both):
            self.assertEqual(frame.shape, (3, 3))
            self.assertEqual(int(frame.to_numpy().sum()), 0)


if __name__ == "__main__":
    unittest.main()
