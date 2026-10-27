"""
Tests for holdout_blocks.py. Expected values are derived by hand from calendar
logic (derivation in the comment above each one), never by running the code.

Synthetic series: daily, 2000-01-01 to 2025-12-31, constant flow 5.0 unless a
test changes it. start_year=2001, end_year=2025, so the usable water years are
2002 (Oct 1, 2001 to Sep 30, 2002) to 2025 (Oct 1, 2024 to Sep 30, 2025).

Run from the repository root:  python -m unittest -v tests.test_holdout_blocks
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import holdout_blocks as hb  # noqa: E402

START_YEAR, END_YEAR = 2001, 2025
T = pd.Timestamp


def base_series():
    days = pd.date_range("2000-01-01", "2025-12-31", freq="D")
    return pd.Series(5.0, index=days)


def block(obs, mode="max_event", **kwargs):
    return hb.build_holdout_block(obs, mode, START_YEAR, END_YEAR, **kwargs)


def spike(*dates, value=100.0):
    obs = base_series()
    for d in dates:
        obs[T(d)] = value
    return obs


class TestWaterYear(unittest.TestCase):
    def test_water_year(self):
        # Sep 30 is the last day of the water year ending that year: 2009
        self.assertEqual(hb.water_year(T("2009-09-30")), 2009)
        # Oct 1, 2009 starts the water year ending Sep 30, 2010: 2010
        self.assertEqual(hb.water_year(T("2009-10-01")), 2010)
        # March is before October, so the water year is the calendar year: 2010
        self.assertEqual(hb.water_year(T("2010-03-12")), 2010)
        # leap day, February, so the calendar year: 2012
        self.assertEqual(hb.water_year(T("2012-02-29")), 2012)

    def test_usable_water_years(self):
        # Oct 1, 2015 to Sep 30, 2025 holds water years 2016 ... 2025
        self.assertEqual(hb.usable_water_years(2015, 2025), list(range(2016, 2026)))


class TestMaxEvent(unittest.TestCase):
    def test_spike_mid_period(self):
        r = block(spike("2010-03-12"))
        # spike in water year 2010; 3 years with it in the middle: 2009, 2010, 2011
        # -> Oct 1, 2008 to Sep 30, 2011
        self.assertEqual(r["intervals"], [(T("2008-10-01"), T("2011-09-30"))])
        self.assertEqual(r["extreme_date"], T("2010-03-12"))
        self.assertEqual(r["extreme_value"], 100.0)
        self.assertEqual(r["mode"], "max_event")
        self.assertEqual((r["first_water_year"], r["last_water_year"]), (2009, 2011))

    def test_water_year_boundaries(self):
        # Oct 5, 2009 is in water year 2010 -> same block as the Mar 12, 2010 spike
        self.assertEqual(block(spike("2009-10-05"))["intervals"],
                         [(T("2008-10-01"), T("2011-09-30"))])
        # Sep 30, 2010 is the last day of water year 2010 -> same block
        self.assertEqual(block(spike("2010-09-30"))["intervals"],
                         [(T("2008-10-01"), T("2011-09-30"))])
        # Oct 1, 2010 starts water year 2011 -> years 2010, 2011, 2012
        # -> Oct 1, 2009 to Sep 30, 2012
        self.assertEqual(block(spike("2010-10-01"))["intervals"],
                         [(T("2009-10-01"), T("2012-09-30"))])

    def test_edges_shift_block(self):
        # Jan 15, 2002 is in water year 2002, the first usable one; centered would be
        # 2001-2003, but 2001 is not usable, so shift by one: 2002-2004
        # -> Oct 1, 2001 to Sep 30, 2004
        r = block(spike("2002-01-15"))
        self.assertEqual(r["intervals"], [(T("2001-10-01"), T("2004-09-30"))])
        self.assertEqual(r["extreme_date"], T("2002-01-15"))
        # May 1, 2025 is in water year 2025, the last usable one; centered would be
        # 2024-2026, shift back by one: 2023-2025 -> Oct 1, 2022 to Sep 30, 2025
        self.assertEqual(block(spike("2025-05-01"))["intervals"],
                         [(T("2022-10-01"), T("2025-09-30"))])

    def test_tie_takes_first(self):
        # equal spikes; the first, Jun 1, 2005, is in water year 2005 -> 2004-2006
        # -> Oct 1, 2003 to Sep 30, 2006
        r = block(spike("2005-06-01", "2010-03-12"))
        self.assertEqual(r["extreme_date"], T("2005-06-01"))
        self.assertEqual(r["intervals"], [(T("2003-10-01"), T("2006-09-30"))])

    def test_position(self):
        obs = spike("2010-03-12")
        # position 0: water year 2010 is the first of the block: 2010-2012
        # -> Oct 1, 2009 to Sep 30, 2012
        self.assertEqual(block(obs, position=0)["intervals"],
                         [(T("2009-10-01"), T("2012-09-30"))])
        # position 2: water year 2010 is the last of the block: 2008-2010
        # -> Oct 1, 2007 to Sep 30, 2010
        self.assertEqual(block(obs, position=2)["intervals"],
                         [(T("2007-10-01"), T("2010-09-30"))])


class TestMinFlow(unittest.TestCase):
    def test_lowest_7day_mean(self):
        obs = base_series()
        obs[T("2012-07-10"):T("2012-07-16")] = 1.0
        # an isolated low day: any 7-day window with it has mean
        # (0.1 + 6 x 5.0) / 7 = 30.1 / 7 = 4.3 > 1.0, so it must not win
        obs[T("2018-02-10")] = 0.1
        r = block(obs, "min_flow")
        # window Jul 10-16, 2012, mean 1.0; middle (4th) day Jul 13; water year 2012
        # -> 2011-2013 -> Oct 1, 2010 to Sep 30, 2013
        self.assertEqual(r["extreme_date"], T("2012-07-13"))
        self.assertEqual(r["extreme_value"], 1.0)
        self.assertEqual(r["intervals"], [(T("2010-10-01"), T("2013-09-30"))])
        self.assertEqual(r["mode"], "min_flow")

    def test_zero_stretch_not_eligible(self):
        obs = base_series()
        obs[T("2012-07-10"):T("2012-07-16")] = 1.0
        obs[T("2018-02-01"):T("2018-02-07")] = 0.0
        r = block(obs, "min_flow")
        # every window that touches the zero stretch has a zero day, so none is
        # eligible: neither the all-zero window (mean 0) nor the edge window of
        # 6 zeros + 1 day of 5.0 (mean 5/7 = 0.714, lower than the 1.0 dip).
        # The 1.0 dip wins as in test_lowest_7day_mean: middle day Jul 13, 2012,
        # water year 2012 -> 2011-2013 -> Oct 1, 2010 to Sep 30, 2013
        self.assertEqual(r["extreme_date"], T("2012-07-13"))
        self.assertEqual(r["extreme_value"], 1.0)
        self.assertEqual(r["intervals"], [(T("2010-10-01"), T("2013-09-30"))])

    def test_single_zero_day_breaks_window(self):
        obs = base_series()
        obs[T("2012-07-10"):T("2012-07-16")] = 1.0
        obs[T("2012-07-13")] = 0.0
        r = block(obs, "min_flow")
        # a window is eligible only if it avoids Jul 13, so it lies entirely before
        # (ends by Jul 12) or after (starts on Jul 14); each side holds at most 3 dip
        # days (Jul 10-12 or Jul 14-16), so the lowest mean is
        # (3 x 1.0 + 4 x 5.0) / 7 = 23/7, reached by Jul 6-12 (middle day Jul 9) and
        # Jul 14-20 (middle day Jul 17). Tie -> the first: Jul 9, 2012, water year
        # 2012 -> 2011-2013 -> Oct 1, 2010 to Sep 30, 2013
        self.assertEqual(r["extreme_date"], T("2012-07-09"))
        self.assertAlmostEqual(r["extreme_value"], 23 / 7)
        self.assertEqual(r["intervals"], [(T("2010-10-01"), T("2013-09-30"))])

    def test_nan_breaks_window(self):
        obs = base_series()
        obs[T("2012-07-10"):T("2012-07-16")] = 1.0
        obs[T("2012-07-13")] = np.nan
        obs[T("2016-08-01"):T("2016-08-07")] = 2.0
        r = block(obs, "min_flow")
        # windows touching Jul 13, 2012 are incomplete; the best complete window
        # there has mean (3 x 1.0 + 4 x 5.0) / 7 = 23/7 = 3.29 > 2.0, so the
        # 2.0 dip wins: middle day Aug 4, 2016, water year 2016 -> 2015-2017
        # -> Oct 1, 2014 to Sep 30, 2017
        self.assertEqual(r["extreme_date"], T("2016-08-04"))
        self.assertEqual(r["extreme_value"], 2.0)
        self.assertEqual(r["intervals"], [(T("2014-10-01"), T("2017-09-30"))])


class TestErrors(unittest.TestCase):
    def test_too_many_years(self):
        # 30 years > the 24 usable water years (2002-2025)
        with self.assertRaises(ValueError):
            block(spike("2010-03-12"), holdout_years=30)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            block(spike("2010-03-12"), "wettest_month")

    def test_all_nan(self):
        obs = pd.Series(np.nan, index=base_series().index)
        for mode in hb.MODES:
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                block(obs, mode)


class TestBlocksEqual(unittest.TestCase):
    def test_blocks_equal(self):
        # Mar 12, 2010 and Oct 5, 2009 are both in water year 2010 -> same block
        a = block(spike("2010-03-12"))
        b = block(spike("2009-10-05"))
        # Oct 1, 2010 is in water year 2011 -> a different block
        c = block(spike("2010-10-01"))
        self.assertTrue(hb.blocks_equal(a, b))
        self.assertFalse(hb.blocks_equal(a, c))


if __name__ == "__main__":
    unittest.main()
