"""
Hold-out blocks for the gap-filling tests (find_best_model_holdout.py).

A hold-out block is a run of whole water years (Oct 1 to Sep 30) hidden from
the model selection. It is placed around an extreme of the observed daily
flow, inside the usable water years of the evaluation period:

- "max_event": the date of the largest observed daily flow;
- "min_flow": the middle day of the lowest 7-day mean flow over a window whose
  7 days are all observed and above zero (a single low day may be a rounding
  fluke, and a window with no-flow days is not a low-flow stretch).

The block is returned as a list of (start, end) date intervals (inclusive),
so a later version can hide several gaps with the same code.
"""

import pandas as pd

MODES = ("max_event", "min_flow")
MIN_FLOW_WINDOW = 7  # [days] window of the 7-day mean in "min_flow" mode


def water_year(ts):
    """Water year of a date, named by the year it ends in (Oct 1, 2009 is in 2010)."""
    ts = pd.Timestamp(ts)
    return ts.year + 1 if ts.month >= 10 else ts.year


def water_year_bounds(wy):
    """First and last day (Oct 1, Sep 30) of water year `wy`."""
    return pd.Timestamp(wy - 1, 10, 1), pd.Timestamp(wy, 9, 30)


def usable_water_years(start_year, end_year):
    """Water years fully inside Oct 1 of start_year to Sep 30 of end_year."""
    return list(range(start_year + 1, end_year + 1))


def _usable_daily(obs, start_year, end_year):
    """`obs` on a complete daily index over the usable water years (missing days NaN)."""
    years = usable_water_years(start_year, end_year)
    if not years:
        raise ValueError(f"no usable water year between {start_year} and {end_year}")
    first, _ = water_year_bounds(years[0])
    _, last = water_year_bounds(years[-1])
    days = pd.date_range(first, last, freq="D")
    obs = obs.sort_index()
    return obs[~obs.index.duplicated()].reindex(days).astype(float)


def find_extreme_date(obs, mode, start_year, end_year):
    """Date and value of the extreme that a hold-out block is built around.

    obs is a daily pd.Series with a DatetimeIndex (NaN = missing). Only the
    usable water years are searched. On a tie the first date wins.
    "min_flow" considers only 7-day windows whose days are all observed and
    above zero, and returns the middle (4th) day of the one with the lowest
    mean, and that mean. Raises ValueError for an unknown mode or no eligible
    day.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    q = _usable_daily(obs, start_year, end_year)

    if mode == "max_event":
        values = q.dropna()
        if values.empty:
            raise ValueError("no observed day in the usable water years")
        date = values.idxmax()
        return date, float(values[date])

    # centered window: the label is the 4th of the 7 days; every day must be
    # observed (min_periods) and above zero (the window's minimum)
    windows = q.rolling(MIN_FLOW_WINDOW, center=True, min_periods=MIN_FLOW_WINDOW)
    means = windows.mean()[windows.min() > 0]
    if means.empty:
        raise ValueError("no 7-day window with every day observed and above zero")
    date = means.idxmin()
    return date, float(means[date])


def build_holdout_block(obs, mode, start_year, end_year, holdout_years=3, position=None):
    """Hold-out block of `holdout_years` whole water years around an extreme.

    The extreme's water year is the year with index `position` of the block
    (zero-based; default: the middle, holdout_years // 2). A block that would
    leave the usable water years is shifted by whole years just enough to fit.

    Returns a dict: intervals (list of inclusive (start, end) Timestamps),
    extreme_date, extreme_value, mode, first_water_year, last_water_year.
    """
    years = usable_water_years(start_year, end_year)
    if holdout_years < 1 or holdout_years > len(years):
        raise ValueError(f"holdout_years={holdout_years} does not fit in the "
                         f"{len(years)} usable water years ({start_year + 1}-{end_year})")
    if position is None:
        position = holdout_years // 2
    if not 0 <= position < holdout_years:
        raise ValueError(f"position={position} is outside the block (0 to {holdout_years - 1})")

    extreme_date, extreme_value = find_extreme_date(obs, mode, start_year, end_year)
    first_wy = water_year(extreme_date) - position
    first_wy = min(max(first_wy, years[0]), years[-1] - holdout_years + 1)
    last_wy = first_wy + holdout_years - 1

    start, _ = water_year_bounds(first_wy)
    _, end = water_year_bounds(last_wy)
    return {
        "intervals": [(start, end)],
        "extreme_date": extreme_date,
        "extreme_value": extreme_value,
        "mode": mode,
        "first_water_year": first_wy,
        "last_water_year": last_wy,
    }


def blocks_equal(a, b):
    """True if two results of build_holdout_block hide the same days."""
    return list(a["intervals"]) == list(b["intervals"])
