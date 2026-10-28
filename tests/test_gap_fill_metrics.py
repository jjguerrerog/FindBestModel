"""
Tests for gap_fill_metrics.py (helpers only). Expected values are derived by
hand (derivation in the comment above each one), never by running the code.
Where an expected value is written with a math call, the standard library, not
the code under test, computes it; the comment gives the decimal checked by hand.

Run from the repository root:  python -m unittest -v tests.test_gap_fill_metrics
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gap_fill_metrics as gfm  # noqa: E402

PLACES = 12


class WeightsAssertions(unittest.TestCase):
    def assertWeights(self, result, expected):
        # same names in the same order, values equal to 12 places
        self.assertEqual(list(result), list(expected))
        for name, value in expected.items():
            self.assertAlmostEqual(result[name], value, places=PLACES)


class TestRatioTermLinear(unittest.TestCase):
    def test_double_gives_one(self):
        # R = 10 / 5 = 2; (1 - 2)^2 = 1
        self.assertAlmostEqual(gfm.ratio_term(10, 5), 1.0, places=PLACES)

    def test_half_gives_quarter(self):
        # R = 5 / 10 = 0.5; (1 - 0.5)^2 = 0.25
        self.assertAlmostEqual(gfm.ratio_term(5, 10), 0.25, places=PLACES)

    def test_triple_gives_four(self):
        # R = 15 / 5 = 3; (1 - 3)^2 = 4
        self.assertAlmostEqual(gfm.ratio_term(15, 5), 4.0, places=PLACES)

    def test_equal_gives_zero(self):
        # R = 7 / 7 = 1; (1 - 1)^2 = 0
        self.assertAlmostEqual(gfm.ratio_term(7, 7), 0.0, places=PLACES)


class TestRatioTermLog(unittest.TestCase):
    def test_double_and_half_symmetric(self):
        # R = 2: (ln 2)^2; R = 0.5: (ln 0.5)^2 = (-ln 2)^2 = (ln 2)^2, the same value
        # on both sides. math (not the code under test) computes it:
        # ln 2 = 0.693147..., squared = 0.480453...
        expected = math.log(2) ** 2
        self.assertAlmostEqual(gfm.ratio_term(10, 5, ratio="log"), expected, places=PLACES)
        self.assertAlmostEqual(gfm.ratio_term(5, 10, ratio="log"), expected, places=PLACES)

    def test_equal_gives_zero(self):
        # R = 1; (ln 1)^2 = 0
        self.assertAlmostEqual(gfm.ratio_term(3, 3, ratio="log"), 0.0, places=PLACES)


class TestRatioTermErrors(unittest.TestCase):
    def assertValueErrorNames(self, sim, obs, argument):
        # both ratio options raise, and the message names the wrong argument and its value
        for ratio in gfm.RATIO_OPTIONS:
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError) as cm:
                    gfm.ratio_term(sim, obs, ratio=ratio)
                message = str(cm.exception)
                self.assertIn(argument, message)
                bad = sim if argument == "sim_value" else obs
                self.assertIn(repr(bad), message)

    def test_zero_obs(self):
        self.assertValueErrorNames(5, 0, "obs_value")

    def test_negative_obs(self):
        self.assertValueErrorNames(5, -2.0, "obs_value")

    def test_zero_sim(self):
        self.assertValueErrorNames(0, 5, "sim_value")

    def test_negative_sim(self):
        self.assertValueErrorNames(-1.5, 5, "sim_value")

    def test_nan(self):
        self.assertValueErrorNames(float("nan"), 5, "sim_value")
        self.assertValueErrorNames(5, float("nan"), "obs_value")

    def test_infinity(self):
        self.assertValueErrorNames(float("inf"), 5, "sim_value")
        self.assertValueErrorNames(5, float("inf"), "obs_value")

    def test_unknown_ratio(self):
        # the message lists the valid options
        with self.assertRaises(ValueError) as cm:
            gfm.ratio_term(10, 5, ratio="square")
        self.assertIn("linear", str(cm.exception))
        self.assertIn("log", str(cm.exception))

    def test_string_argument(self):
        with self.assertRaises(TypeError) as cm:
            gfm.ratio_term("10", 5)
        self.assertIn("sim_value", str(cm.exception))


class TestNormalizeWeights(WeightsAssertions):
    def test_no_weights(self):
        # k = 2, S = 1 + 1 = 2; each 1 * 2 / 2 = 1
        self.assertWeights(gfm.normalize_weights(["a", "b"]), {"a": 1.0, "b": 1.0})

    def test_equal_weights_match_no_weights(self):
        # k = 2, S = 7 + 7 = 14; each 7 * 2 / 14 = 1, same as no weights
        self.assertWeights(gfm.normalize_weights(["a", "b"], {"a": 7, "b": 7}),
                           {"a": 1.0, "b": 1.0})

    def test_hand_derived_case(self):
        # k = 2, S = 1 + 3 = 4; a = 1 * 2 / 4 = 0.5, b = 3 * 2 / 4 = 1.5 (sum 2 = k)
        self.assertWeights(gfm.normalize_weights(["a", "b"], {"a": 1, "b": 3}),
                           {"a": 0.5, "b": 1.5})

    def test_missing_weight_counts_as_one(self):
        # a and c missing -> weight 1; k = 3, S = 1 + 2 + 1 = 4;
        # a = 1 * 3 / 4 = 0.75, b = 2 * 3 / 4 = 1.5, c = 0.75 (sum 3 = k)
        self.assertWeights(gfm.normalize_weights(["a", "b", "c"], {"b": 2}),
                           {"a": 0.75, "b": 1.5, "c": 0.75})

    def test_zero_weight_drops_term(self):
        # b dropped; active a and c (weight 1 each), k = 2, S = 2; each 1 * 2 / 2 = 1
        result = gfm.normalize_weights(["a", "b", "c"], {"b": 0})
        self.assertWeights(result, {"a": 1.0, "c": 1.0})
        self.assertNotIn("b", result)


class TestNormalizeWeightsErrors(unittest.TestCase):
    def test_unknown_name_lists_valid_names(self):
        with self.assertRaises(ValueError) as cm:
            gfm.normalize_weights(["a", "b", "c"], {"d": 1})
        message = str(cm.exception)
        self.assertIn("'d'", message)
        self.assertIn("valid term names", message)
        for name in ("'a'", "'b'", "'c'"):
            self.assertIn(name, message)

    def test_negative_weight(self):
        with self.assertRaises(ValueError):
            gfm.normalize_weights(["a", "b"], {"a": -1})

    def test_nan_weight(self):
        with self.assertRaises(ValueError):
            gfm.normalize_weights(["a", "b"], {"a": float("nan")})

    def test_infinite_weight(self):
        with self.assertRaises(ValueError):
            gfm.normalize_weights(["a", "b"], {"a": float("inf")})

    def test_all_weights_zero(self):
        with self.assertRaises(ValueError):
            gfm.normalize_weights(["a", "b"], {"a": 0, "b": 0})

    def test_empty_names(self):
        with self.assertRaises(ValueError):
            gfm.normalize_weights([])

    def test_duplicate_names(self):
        with self.assertRaises(ValueError):
            gfm.normalize_weights(["a", "b", "a"])

    def test_weights_list_type_error(self):
        with self.assertRaises(TypeError):
            gfm.normalize_weights(["a", "b"], [1, 3])


class TestCombineTerms(unittest.TestCase):
    def test_unweighted(self):
        # weights 1 and 1; sum 1 + 4 = 5; score 1 - sqrt(5).
        # math (not the code under test) computes it: sqrt(5) = 2.236068..., score = -1.236068...
        self.assertAlmostEqual(gfm.combine_terms({"a": 1, "b": 4}), 1 - math.sqrt(5),
                               places=PLACES)

    def test_perfect(self):
        # every term 0; sum 0; score 1 - 0 = 1
        self.assertAlmostEqual(gfm.combine_terms({"a": 0, "b": 0, "c": 0}), 1.0, places=PLACES)

    def test_all_ones_weights_equal_unweighted(self):
        # {a: 1, b: 1} and {a: 7, b: 7} both normalize to 1 and 1 (k = 2); sum 1 + 4 = 5;
        # score 1 - sqrt(5) = -1.236068... (sqrt computed by math, not the code under test)
        for weights in ({"a": 1, "b": 1}, {"a": 7, "b": 7}):
            with self.subTest(weights=weights):
                self.assertAlmostEqual(gfm.combine_terms({"a": 1, "b": 4}, weights),
                                       1 - math.sqrt(5), places=PLACES)

    def test_weighted(self):
        # weights {a: 1, b: 3}: k = 2, S = 4, normalized 0.5 and 1.5;
        # sum 0.5 * 1 + 1.5 * 4 = 0.5 + 6 = 6.5; score 1 - sqrt(6.5).
        # math computes it: sqrt(6.5) = 2.549510..., score = -1.549510...
        self.assertAlmostEqual(gfm.combine_terms({"a": 1, "b": 4}, {"a": 1, "b": 3}),
                               1 - math.sqrt(6.5), places=PLACES)

    def test_zero_weight_drops_term(self):
        # b dropped; active a and c, k = 2, weights 1 and 1; sum 1 + 9 = 10; score 1 - sqrt(10).
        # math computes it: sqrt(10) = 3.162278..., score = -2.162278...
        self.assertAlmostEqual(gfm.combine_terms({"a": 1, "b": 4, "c": 9}, {"b": 0}),
                               1 - math.sqrt(10), places=PLACES)

    def test_zero_weight_leaves_one_term(self):
        # a dropped; active b, k = 1, weight 1 * 1 / 1 = 1; sum 4; score 1 - sqrt(4) = 1 - 2 = -1
        self.assertAlmostEqual(gfm.combine_terms({"a": 1, "b": 4}, {"a": 0}), -1.0,
                               places=PLACES)


class TestCombineTermsErrors(unittest.TestCase):
    def test_negative_term(self):
        with self.assertRaises(ValueError):
            gfm.combine_terms({"a": 1, "b": -0.5})

    def test_nan_term(self):
        with self.assertRaises(ValueError):
            gfm.combine_terms({"a": 1, "b": float("nan")})

    def test_empty_dict(self):
        with self.assertRaises(ValueError):
            gfm.combine_terms({})

    def test_list_type_error(self):
        with self.assertRaises(TypeError):
            gfm.combine_terms([1, 4])

    def test_unknown_weight_name(self):
        with self.assertRaises(ValueError) as cm:
            gfm.combine_terms({"a": 1, "b": 4}, {"c": 2})
        message = str(cm.exception)
        self.assertIn("valid term names", message)
        self.assertIn("'a'", message)
        self.assertIn("'b'", message)


if __name__ == "__main__":
    unittest.main()
