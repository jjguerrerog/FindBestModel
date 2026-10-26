"""
Goodness-of-fit metrics for comparing simulated and observed discharge.

Every function takes a DataFrame with a simulated and an observed discharge
column (any units, as long as both columns use the same one) and returns a
single performance statistic. `summary()` runs all of them at once and
returns a dict, which is the easiest entry point for most notebooks.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


NONPOSITIVE_MODES = ("keep", "offset", "skip")


def _paired_values(df, sim_col="Qsim", obs_col="Qobs", log=False, positive=False,
                   nonpositive="keep", offset_frac=0.01):
    """
    Pulls the simulated/observed columns out of `df` as aligned numpy arrays,
    dropping any row where either value is missing.

    Zero or negative observed flows (e.g. flow reversals at low-gradient or
    tidal gauges) make log and ratio-based metrics undefined. `nonpositive`
    sets how they are handled:
        'keep'   -- use them as they are (log metrics then fail, and
                    percentile ratios can be distorted).
        'offset' -- only for metrics that need positive values (`log` or
                    `positive`): clip observed flows at 0, then add
                    eps = offset_frac * mean(clipped obs) to both series
                    (Pushpalatha et al., 2012). Other metrics use them as they are.
        'skip'   -- drop the rows where the observed flow is <= 0, for every metric.

    Parameters:
        df (pd.DataFrame): table containing `sim_col` and `obs_col`.
        sim_col (str): name of the simulated-discharge column.
        obs_col (str): name of the observed-discharge column.
        log (bool): if True, return the natural log of both series instead
            of the raw values. Requires all paired values to be > 0.
        positive (bool): the metric needs positive values even without `log`
            (e.g. ratios of percentiles); only matters for 'offset'.
        nonpositive (str): 'keep', 'offset' or 'skip' (see above).
        offset_frac (float): eps as a fraction of the mean observed flow ('offset').

    Returns:
        tuple[np.ndarray, np.ndarray]: (sim, obs), same length, no NaNs.
    """
    if nonpositive not in NONPOSITIVE_MODES:
        raise ValueError(f"nonpositive must be one of {NONPOSITIVE_MODES}, not {nonpositive!r}")
    paired = df[[sim_col, obs_col]].dropna()
    if nonpositive == "skip":
        paired = paired[paired[obs_col] > 0]
    if paired.empty:
        raise ValueError(
            f"No overlapping, non-NaN data between '{sim_col}' and '{obs_col}'"
            + (" with positive observed values" if nonpositive == "skip" else "")
        )
    sim, obs = paired[sim_col].to_numpy(dtype=float), paired[obs_col].to_numpy(dtype=float)
    if nonpositive == "offset" and (log or positive):
        obs = np.maximum(obs, 0.0)
        eps = offset_frac * obs.mean()
        sim, obs = sim + eps, obs + eps
    if log:
        if np.any(sim <= 0) or np.any(obs <= 0):
            raise ValueError(
                f"'{sim_col}' and '{obs_col}' must be strictly positive to compute log-scale values"
            )
        sim, obs = np.log(sim), np.log(obs)
    return sim, obs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def nse(df, sim_col="Qsim", obs_col="Qobs", log=False, nonpositive="keep", offset_frac=0.01):
    """
    Nash-Sutcliffe Efficiency (Nash & Sutcliffe, 1970).

    1 = perfect fit, 0 = no better than predicting the observed mean,
    negative = worse than the observed mean.

    Parameters:
        df (pd.DataFrame): table containing `sim_col` and `obs_col`.
        sim_col (str): name of the simulated-discharge column.
        obs_col (str): name of the observed-discharge column.
        log (bool): if True, compute NSE on the natural log of both series
            (log-NSE), which downweights high flows and emphasizes low-flow
            fit. Requires all values to be > 0.

        nonpositive (str): how observed flows <= 0 are handled: 'keep',
            'offset' or 'skip' (see `_paired_values`).
        offset_frac (float): eps as a fraction of the mean observed flow
            ('offset' only).
    Returns:
        float: NSE (or log-NSE), unitless.
    """
    sim, obs = _paired_values(df, sim_col, obs_col, log=log,
                              nonpositive=nonpositive, offset_frac=offset_frac)
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        raise ValueError("Observed series has zero variance; NSE is undefined")
    return 1.0 - np.sum((sim - obs) ** 2) / denom


def kge(df, sim_col="Qsim", obs_col="Qobs", log=False, nonpositive="keep", offset_frac=0.01):
    """
    Kling-Gupta Efficiency (Gupta et al., 2009) and its three components.

    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2), where r is the
    Pearson correlation, alpha = std(sim)/std(obs) is the variability
    ratio, and beta = mean(sim)/mean(obs) is the bias ratio.
    1 = perfect fit.

    Parameters:
        df (pd.DataFrame): table containing `sim_col` and `obs_col`.
        sim_col (str): name of the simulated-discharge column.
        obs_col (str): name of the observed-discharge column.
        log (bool): if True, compute KGE on the natural log of both series
            (log-KGE), which downweights high flows and emphasizes low-flow
            fit. Requires all values to be > 0.

        nonpositive (str): how observed flows <= 0 are handled: 'keep',
            'offset' or 'skip' (see `_paired_values`).
        offset_frac (float): eps as a fraction of the mean observed flow
            ('offset' only).
    Returns:
        dict: {'KGE', 'r', 'alpha', 'beta'}, all unitless floats.
    """
    sim, obs = _paired_values(df, sim_col, obs_col, log=log,
                              nonpositive=nonpositive, offset_frac=offset_frac)
    if obs.mean() == 0 or obs.std() == 0:
        raise ValueError("Observed series has zero mean or zero variance; KGE is undefined")

    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    kge_value = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {"KGE": kge_value, "r": r, "alpha": alpha, "beta": beta}


def pbias(df, sim_col="Qsim", obs_col="Qobs", nonpositive="keep", offset_frac=0.01):
    """
    Percent bias: average tendency of simulated values to be larger or
    smaller than observed. 0% = no bias, positive = overestimation,
    negative = underestimation.

    Parameters:
        df (pd.DataFrame): table containing `sim_col` and `obs_col`.
        sim_col (str): name of the simulated-discharge column.
        obs_col (str): name of the observed-discharge column.

        nonpositive (str): how observed flows <= 0 are handled: 'keep',
            'offset' or 'skip' (see `_paired_values`).
        offset_frac (float): eps as a fraction of the mean observed flow
            ('offset' only).
    Returns:
        float: PBIAS in percent.
    """
    sim, obs = _paired_values(df, sim_col, obs_col,
                              nonpositive=nonpositive, offset_frac=offset_frac)
    obs_sum = obs.sum()
    if obs_sum == 0:
        raise ValueError("Observed series sums to zero; PBIAS is undefined")
    return 100.0 * np.sum(sim - obs) / obs_sum


def correlation(df, sim_col="Qsim", obs_col="Qobs", nonpositive="keep", offset_frac=0.01):
    """
    Pearson linear correlation coefficient between simulated and observed
    discharge.

    Parameters:
        df (pd.DataFrame): table containing `sim_col` and `obs_col`.
        sim_col (str): name of the simulated-discharge column.
        obs_col (str): name of the observed-discharge column.

        nonpositive (str): how observed flows <= 0 are handled: 'keep',
            'offset' or 'skip' (see `_paired_values`).
        offset_frac (float): eps as a fraction of the mean observed flow
            ('offset' only).
    Returns:
        float: correlation coefficient, unitless, in [-1, 1].
    """
    sim, obs = _paired_values(df, sim_col, obs_col,
                              nonpositive=nonpositive, offset_frac=offset_frac)
    return np.corrcoef(sim, obs)[0, 1]


def mean_annual_peak_flow_difference(df, sim_col="Qsim", obs_col="Qobs", relative=False):
    """
    Mean, over all years present, of (annual peak simulated flow - annual
    peak observed flow). `df` must have a DatetimeIndex so annual peaks can
    be identified; a year is included only if both series have at least one
    non-NaN value in it.

    Parameters:
        df (pd.DataFrame): table containing `sim_col` and `obs_col`, indexed
            by a `pd.DatetimeIndex`.
        sim_col (str): name of the simulated-discharge column.
        obs_col (str): name of the observed-discharge column.
        relative (bool): if True, express each year's difference relative to
            that year's observed peak (%) before averaging, instead of an
            absolute difference in the series' own units.

    Returns:
        float: mean annual peak-flow difference (same units as `df`, or
            percent if `relative=True`).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex to compute annual peak flows")

    annual_sim = df[sim_col].groupby(df.index.year).max()
    annual_obs = df[obs_col].groupby(df.index.year).max()
    paired = pd.DataFrame({"sim": annual_sim, "obs": annual_obs}).dropna()
    if paired.empty:
        raise ValueError(
            f"No overlapping years between '{sim_col}' and '{obs_col}' to compute peak flows"
        )

    diff = paired["sim"] - paired["obs"]
    if relative:
        if (paired["obs"] == 0).any():
            raise ValueError("At least one year has a zero observed peak; relative diff is undefined")
        diff = 100.0 * diff / paired["obs"]
    return diff.mean()


def percentile_relative_difference(df, percentile, sim_col="Qsim", obs_col="Qobs",
                                   nonpositive="keep", offset_frac=0.01):
    """
    Relative difference between the simulated and observed flow at a given
    percentile of each series' own distribution (i.e. `np.percentile(sim, p)`
    vs. `np.percentile(obs, p)`), expressed as percent of the observed value.
    Use a low percentile (e.g. 5) to check low-flow performance and a high
    percentile (e.g. 95) to check high-flow performance.

    Parameters:
        df (pd.DataFrame): table containing `sim_col` and `obs_col`.
        percentile (float): percentile to compare, in [0, 100].
        sim_col (str): name of the simulated-discharge column.
        obs_col (str): name of the observed-discharge column.

        nonpositive (str): how observed flows <= 0 are handled: 'keep',
            'offset' or 'skip' (see `_paired_values`).
        offset_frac (float): eps as a fraction of the mean observed flow
            ('offset' only).
    Returns:
        float: percent relative difference, 100 * (sim_p - obs_p) / obs_p.
    """
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    sim, obs = _paired_values(df, sim_col, obs_col, positive=True,
                              nonpositive=nonpositive, offset_frac=offset_frac)
    sim_p = np.percentile(sim, percentile)
    obs_p = np.percentile(obs, percentile)
    if obs_p == 0:
        raise ValueError(f"Observed {percentile}th percentile is zero; relative diff is undefined")
    return 100.0 * (sim_p - obs_p) / obs_p


def percentiles_index(df, percentiles, sim_col="Qsim", obs_col="Qobs",
                      nonpositive="keep", offset_frac=0.01):
    """
    Composite index comparing simulated and observed flow across several
    percentiles of each series' own distribution:
    1 - sqrt(sum((1 - sim_p/obs_p) ** 2)), summed over all `percentiles`.
    1 = perfect fit across all requested percentiles.

    Parameters:
        df (pd.DataFrame): table containing `sim_col` and `obs_col`.
        percentiles (array-like): percentiles to compare, each in [0, 100].
        sim_col (str): name of the simulated-discharge column.
        obs_col (str): name of the observed-discharge column.

        nonpositive (str): how observed flows <= 0 are handled: 'keep',
            'offset' or 'skip' (see `_paired_values`).
        offset_frac (float): eps as a fraction of the mean observed flow
            ('offset' only).
    Returns:
        float: percentiles index, unitless.
    """
    sim, obs = _paired_values(df, sim_col, obs_col, positive=True,
                              nonpositive=nonpositive, offset_frac=offset_frac)
    percentiles_obs = np.percentile(obs, percentiles)
    percentiles_sim = np.percentile(sim, percentiles)
    if np.any(percentiles_obs == 0):
        raise ValueError("At least one observed percentile is zero; index is undefined")
    return 1 - np.sqrt(np.sum((1 - percentiles_sim / percentiles_obs) ** 2))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def summary(df, sim_col="Qsim", obs_col="Qobs", peak_percentile=None, peak_relative=False,
            nonpositive="keep", offset_frac=0.01):
    """
    Runs every metric in this module and returns the results in one dict.
    Requires a DatetimeIndex on `df` (needed for the annual peak-flow metric).

    Parameters:
        df (pd.DataFrame): table containing `sim_col` and `obs_col`, indexed
            by a `pd.DatetimeIndex`.
        sim_col (str): name of the simulated-discharge column.
        obs_col (str): name of the observed-discharge column.
        peak_percentile (float, optional): if given, also computes
            `percentile_relative_difference` at this percentile.
        peak_relative (bool): passed through to
            `mean_annual_peak_flow_difference` (see its docstring).
        nonpositive (str): how observed flows <= 0 are handled: 'keep',
            'offset' or 'skip' (see `_paired_values`).
        offset_frac (float): eps as a fraction of the mean observed flow
            ('offset' only).

    Returns:
        dict: {'NSE', 'KGE', 'r' (KGE component), 'alpha' (KGE component),
            'beta' (KGE component), 'PBIAS', 'correlation',
            'mean_annual_peak_flow_diff', and, if `peak_percentile` is
            given, 'percentile_relative_diff'}.
    """
    mode = {"nonpositive": nonpositive, "offset_frac": offset_frac}
    kge_result = kge(df, sim_col, obs_col, **mode)
    results = {
        "NSE": nse(df, sim_col, obs_col, **mode),
        "KGE": kge_result["KGE"],
        "r": kge_result["r"],
        "alpha": kge_result["alpha"],
        "beta": kge_result["beta"],
        "PBIAS": pbias(df, sim_col, obs_col, **mode),
        "correlation": correlation(df, sim_col, obs_col, **mode),
        "mean_annual_peak_flow_diff": mean_annual_peak_flow_difference(
            df, sim_col, obs_col, relative=peak_relative
        ),
        "pdi_ave": percentiles_index(df, [5, 25, 50, 75, 95], sim_col, obs_col, **mode),
        "pdi_low": percentiles_index(df, [1,5,10,15], sim_col, obs_col, **mode),
        "pdi_high": percentiles_index(df, [85,90,95,99], sim_col, obs_col, **mode)
    }
    if peak_percentile is not None:
        results["percentile_relative_diff"] = percentile_relative_difference(
            df, peak_percentile, sim_col, obs_col, **mode
        )
    return results
