"""
Tests for the compute_all safeguard: a config must not set the nonpositive
switch (nonpositive or offset_frac) in its kwargs, because compute_all passes
it. The check runs before any metric is computed.

Run from the repository root:  python -m unittest -v tests.test_compute_all_safeguards
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gap_fill_metrics as gfm  # noqa: E402

OBS = [1.0, 2.0, 3.0, 4.0, 5.0]


def frame(sim, obs, sim_col="Qsim", obs_col="Qobs"):
    return pd.DataFrame({sim_col: sim, obs_col: obs})


class RecordingMetric:
    """Stub metric that counts its calls and returns 0.5."""

    def __init__(self):
        self.calls = 0

    def __call__(self, df, sim_col, obs_col, nonpositive="keep", offset_frac=0.01, **kwargs):
        self.calls += 1
        return 0.5


class TestSwitchInKwargsRejected(unittest.TestCase):
    def check_rejected(self, key, value):
        configs = (gfm.MetricConfig("own_switch", RecordingMetric(), {key: value}),)
        with self.assertRaises(TypeError) as cm:
            gfm.compute_all(frame(OBS, OBS), configs=configs)
        self.assertIn("own_switch", str(cm.exception))
        self.assertIn(key, str(cm.exception))
        with self.assertRaises(TypeError) as cm:
            gfm.metric_columns(configs)
        self.assertIn("own_switch", str(cm.exception))
        self.assertIn(key, str(cm.exception))

    def test_nonpositive_in_kwargs(self):
        self.check_rejected("nonpositive", "skip")

    def test_offset_frac_in_kwargs(self):
        self.check_rejected("offset_frac", 0.05)

    def test_nothing_called_when_validation_fails(self):
        # the valid config comes first; validation still runs before any metric
        first, second = RecordingMetric(), RecordingMetric()
        configs = (gfm.MetricConfig("valid", first),
                   gfm.MetricConfig("own_switch", second, {"nonpositive": "skip"}))
        with self.assertRaises(TypeError):
            gfm.compute_all(frame(OBS, OBS), configs=configs)
        self.assertEqual(first.calls, 0)
        self.assertEqual(second.calls, 0)


class TestValidConfigsUnaffected(unittest.TestCase):
    def test_default_configs_validate(self):
        self.assertEqual(len(gfm.metric_columns(gfm.DEFAULT_CONFIGS)), 10)

    def test_other_kwargs_unaffected(self):
        # other keys pass through: the stub is called once and returns 0.5
        metric = RecordingMetric()
        configs = (gfm.MetricConfig("other", metric, {"low_p": 5}),)
        self.assertEqual(gfm.metric_columns(configs), ["other"])
        self.assertEqual(gfm.compute_all(frame(OBS, OBS), configs=configs), {"other": 0.5})
        self.assertEqual(metric.calls, 1)


if __name__ == "__main__":
    unittest.main()
