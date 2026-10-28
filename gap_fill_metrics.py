"""
Selection metrics for the gap-filling tests (OUTLINE_gap_metrics.md, Part 1).

Shared helpers (ratio_term, normalize_weights, combine_terms) and the metrics
built on them; so far: percentile_nse_index, kge_tail_anchors,
kge_percentile_blocks, kge_np and jdkge. More metrics are added in later
steps.

Conventions for every metric of this module:

- Each term is a squared distance from its ideal point: a term whose ideal is 1
  enters as (1 - x) ** 2, a term whose ideal is 0 enters as x ** 2.
- The score is 1 minus the square root of the weighted sum of the terms:
  higher is better, ideal score 1.
- Weights are normalized: with k the number of active terms (weight > 0) and S
  the sum of their weights, each active weight becomes w * k / S, so all-ones
  weights give the unweighted score and the active weights sum to k.

Decisions (2026-10-28):

- Unknown weight names raise an error (decided by J).
- k counts only the terms with non-zero weight, so dropping a term gives the
  same score as a metric built without it (tutor's logic decision, J may
  veto). This reads "weight 0 removes a term" literally; the outline text says
  "k the number of terms".
- Negative, NaN or infinite weights, and all weights zero, raise errors.
- ratio_term raises for any sim or obs value that is not finite or is <= 0,
  for both ratio options.

Decisions for percentile_nse_index (tutor's logic decisions, 2026-10-28):

- Zero and non-positive flows follow the existing --nonpositive-obs switch
  (keep, offset, skip) like pm.percentiles_index, except that every
  percentile <= 0 or not finite, observed or simulated, raises ValueError
  (pm.percentiles_index raises only on an observed percentile of exactly 0);
  with the default keep such a percentile makes the metric raise, which the
  search wrapper turns into NaN; with offset the ratios work.
- The NSE term reuses pm.nse, with the same pairing as the existing NSE column.
- NSE is not clamped.

Decisions for kge_tail_anchors (tutor's logic decisions, 2026-10-28):

- The ratio option changes only the percentile terms, never r, alpha or beta.
- Zero variance (observed or simulated) and a zero observed mean raise
  ValueError.
- The percentile terms follow the nonpositive switch and raise for any
  percentile <= 0.

kge_percentile_blocks: each block is one term, its percentile terms
aggregated by block_agg. "mean" is the default: a block does not gain weight
from having more percentiles (settled by J, 2026-10-27). "median" ignores both
ends of a short block and hides the extreme percentile; it is an option to
show the effect, not the default.

Decisions for kge_percentile_blocks (tutor's logic decisions, 2026-10-28):

- Percentile lists must be unique, strictly between 0 and 100, and every low
  percentile below every high one.

kge_np: the non-parametric KGE of Pool et al. (2018), Hydrological Sciences
Journal 63(13-14), 1941-1953, Eqs. 1, 5, 6, 7: Spearman rank correlation r_s,
alpha_np from the normalized flow-duration curves, beta = mean ratio.

Decisions for kge_np (tutor's logic decisions, 2026-10-28):

- Ties get average ranks (the paper does not say how to rank ties).
- The pairing follows the nonpositive switch (positive=False).
- The observed and simulated means must be positive.
- A constant series (all ranks equal) raises ValueError.

jdkge: KGE' plus the squared Jensen-Shannon divergence of the log-flow
histograms, after Ficchi et al. (2026), preprint egusphere-2026-43, Eq. 2 and
Sections 2.1.2 and 2.1.3.

Decisions for jdkge (tutor's logic decisions, 2026-10-28):

- Natural log in the JSD pipeline.
- The IQR uses numpy's default percentile method.
- cv uses ddof 0 for both series.
- The number of bins is the ceiling of the ratio rounded to 9 decimals.
- One cleaned sample (no missing, non-finite or negative values) for all four
  terms.
- Zeros become epsilon only inside the JSD pipeline.
- Fewer than 4 years of pairs gives a warning, not an error (the paper only
  recommends 4 years).
- The fixed-reference option is not part of this step.
- ts_factor is checked first: a real number, finite and > 0 (TypeError or
  ValueError otherwise), because ts_factor ** (1/3) of 0 or a negative number
  would fail silently.
- _jsd_between raises ValueError when every value of both series is identical
  (all bin widths would be zero); jdkge cannot reach this because its std
  checks run first, the guard protects future callers.
"""

import math
import numbers
import warnings

import numpy as np
import pandas as pd

import performance_metrics as pm

RATIO_OPTIONS = ("linear", "log")
PERCENTILE_NSE_TERMS = ("low", "high", "nse")
KGE_TAIL_ANCHOR_TERMS = ("r", "alpha", "beta", "low", "high")
DEFAULT_LOW_BLOCK = (1, 5, 10, 15)      # same as the pdi_low list in performance_metrics
DEFAULT_HIGH_BLOCK = (85, 90, 95, 99)   # same as the pdi_high list in performance_metrics
BLOCK_AGG_OPTIONS = ("mean", "median", "max", "sum")
KGE_PERCENTILE_BLOCK_TERMS = ("r", "alpha", "beta", "low_block", "high_block")
KGE_NP_TERMS = ("r_s", "alpha_np", "beta")
JDKGE_TERMS = ("r", "beta", "alpha", "jsd")
JSD_MIN_BINS = 25
JSD_MAX_BINS = 100
JSD_MIN_YEARS = 4


def _check_real(name, value):
    """Raise TypeError unless `value` is a real number (any numbers.Real, including
    NumPy scalars such as float32, float64 and int64; not bool)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__} ({value!r})")


def ratio_term(sim_value, obs_value, ratio="linear"):
    """
    Squared distance of the ratio R = sim_value / obs_value from its ideal.

    ratio="linear": (1 - R) ** 2 (ideal R = 1). ratio="log": (ln R) ** 2
    (ideal ln R = 0; R and 1/R give the same term).

    Raises ValueError if either value is not finite or is <= 0, or for an
    unknown ratio option; TypeError if either value is not a real number.
    """
    if ratio not in RATIO_OPTIONS:
        raise ValueError(f"ratio must be one of {RATIO_OPTIONS}, not {ratio!r}")
    for name, value in (("sim_value", sim_value), ("obs_value", obs_value)):
        _check_real(name, value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    r = sim_value / obs_value
    if ratio == "linear":
        return (1.0 - r) ** 2
    return math.log(r) ** 2


def normalize_weights(term_names, weights=None):
    """
    Normalized weights of the active terms.

    term_names is a non-empty sequence of unique strings. weights is None or a
    dict from term name to weight; a term missing from it has weight 1. Terms
    with weight 0 are dropped. With k the number of active terms and S the sum
    of their weights, each active weight becomes w * k / S.

    Returns a dict from term name to normalized weight, only for the active
    terms, in the order of term_names (the values sum to k).
    """
    if isinstance(term_names, str):
        raise TypeError("term_names must be a sequence of names, not a single string")
    names = list(term_names)
    if not names:
        raise ValueError("term_names is empty")
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"term names must be strings, got {name!r}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate term names: {duplicates}")

    if weights is None:
        weights = {}
    elif not isinstance(weights, dict):
        raise TypeError(f"weights must be None or a dict, not {type(weights).__name__}")
    unknown = [name for name in weights if name not in names]
    if unknown:
        raise ValueError(f"unknown weight name(s) {unknown}; valid term names: {names}")

    active = {}
    for name in names:
        w = weights.get(name, 1)
        _check_real(f"weight of {name!r}", w)
        if not math.isfinite(w) or w < 0:
            raise ValueError(f"weight of {name!r} must be finite and >= 0, got {w!r}")
        if w > 0:
            active[name] = w
    if not active:
        raise ValueError("every weight is zero; at least one term needs a positive weight")

    k, total = len(active), sum(active.values())
    return {name: w * k / total for name, w in active.items()}


def combine_terms(terms, weights=None):
    """
    Score 1 - sqrt(sum of normalized weight * term) over the active terms.

    terms is a non-empty dict from term name to its squared distance (a finite
    number >= 0). weights as in normalize_weights; a term with weight 0 does
    not contribute, whatever its value.
    """
    if not isinstance(terms, dict):
        raise TypeError(f"terms must be a dict, not {type(terms).__name__}")
    if not terms:
        raise ValueError("terms is empty")
    for name, value in terms.items():
        _check_real(f"term {name!r}", value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"term {name!r} must be finite and >= 0, got {value!r}")
    normalized = normalize_weights(list(terms), weights)
    return 1.0 - math.sqrt(sum(w * terms[name] for name, w in normalized.items()))


def _check_percentile(name, value):
    """Raise unless `value` is a real number strictly between 0 and 100."""
    _check_real(name, value)
    if not 0 < value < 100:
        raise ValueError(f"{name} must be strictly between 0 and 100, got {value!r}")


def percentile_nse_index(df, sim_col="Qsim", obs_col="Qobs", low_p=5, high_p=95,
                         ratio="linear", weights=None, nonpositive="keep", offset_frac=0.01):
    """
    Percentile anchors plus NSE (the advisor's idea): 1 - sqrt of the weighted
    sum of three terms, higher is better, ideal 1.

    Terms (names in PERCENTILE_NSE_TERMS):
        "low":  ratio_term(sim_low, obs_low, ratio), with sim_low and obs_low
                the low_p percentile of each series (np.percentile, linear
                interpolation, as in pm.percentiles_index);
        "high": the same at high_p;
        "nse":  (1 - NSE) ** 2, NSE from pm.nse. NSE is not clamped: 1 - NSE
                is unbounded on poor models, so there it dominates the score.

    Parameters:
        df (pd.DataFrame): table containing `sim_col` and `obs_col`.
        sim_col, obs_col (str): simulated and observed discharge columns.
        low_p, high_p (real): percentiles strictly between 0 and 100,
            low_p < high_p (ValueError otherwise; TypeError if not a number).
        ratio (str): "linear" or "log" (see ratio_term).
        weights (dict or None): weights by term name (see normalize_weights);
            an unknown name raises ValueError.
        nonpositive (str), offset_frac (float): how observed flows <= 0 are
            handled ('keep', 'offset' or 'skip'; see pm._paired_values),
            passed to both the percentile pairing and pm.nse.

    Decisions (tutor's logic decisions, 2026-10-28):
        - Zero and non-positive flows follow the existing --nonpositive-obs
          switch (keep, offset, skip) like pm.percentiles_index, except that
          every percentile <= 0 or not finite, observed or simulated, raises
          ValueError (pm.percentiles_index raises only on an observed
          percentile of exactly 0); with the default keep such a percentile
          makes the metric raise, which the search wrapper turns into NaN;
          with offset the ratios work.
        - The NSE term reuses pm.nse, with the same pairing as the existing
          NSE column.
        - NSE is not clamped.

    Returns:
        float: the score, ideal 1.
    """
    _check_percentile("low_p", low_p)
    _check_percentile("high_p", high_p)
    if not low_p < high_p:
        raise ValueError(f"low_p must be below high_p, got low_p={low_p!r}, high_p={high_p!r}")

    sim, obs = pm._paired_values(df, sim_col, obs_col, log=False, positive=True,
                                 nonpositive=nonpositive, offset_frac=offset_frac)
    sim_low, sim_high = np.percentile(sim, [low_p, high_p])
    obs_low, obs_high = np.percentile(obs, [low_p, high_p])
    nse = pm.nse(df, sim_col, obs_col, nonpositive=nonpositive, offset_frac=offset_frac)

    terms = {
        "low": ratio_term(sim_low, obs_low, ratio),
        "high": ratio_term(sim_high, obs_high, ratio),
        "nse": (1.0 - nse) ** 2,
    }
    return combine_terms(terms, weights)


def _kge_terms(df, sim_col, obs_col, nonpositive, offset_frac):
    """
    Squared distances of the KGE components: {"r": (1 - r) ** 2,
    "alpha": (1 - alpha) ** 2, "beta": (1 - beta) ** 2}, with r the Pearson
    correlation, alpha = std(sim) / std(obs) and beta = mean(sim) / mean(obs),
    as in pm.kge. Pairs through pm._paired_values with positive=False.

    Raises ValueError for fewer than 2 paired values, a zero observed or
    simulated std, or a zero observed mean.
    """
    sim, obs = pm._paired_values(df, sim_col, obs_col, log=False, positive=False,
                                 nonpositive=nonpositive, offset_frac=offset_frac)
    if len(obs) < 2:
        raise ValueError(f"KGE needs at least 2 paired values, got {len(obs)}")
    obs_std, sim_std, obs_mean = obs.std(), sim.std(), obs.mean()
    if obs_std == 0:
        raise ValueError("observed series has zero variance; KGE is undefined")
    if sim_std == 0:
        raise ValueError("simulated series has zero variance; KGE is undefined")
    if obs_mean == 0:
        raise ValueError("observed series has zero mean; KGE is undefined")
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim_std / obs_std
    beta = sim.mean() / obs_mean
    return {"r": (1.0 - r) ** 2, "alpha": (1.0 - alpha) ** 2, "beta": (1.0 - beta) ** 2}


def _percentile_terms(df, sim_col, obs_col, percentiles, ratio, nonpositive, offset_frac):
    """
    One ratio_term(sim_p, obs_p, ratio) per percentile, in order, with sim_p
    and obs_p the percentiles of each series (np.percentile, linear
    interpolation). Pairs through pm._paired_values with positive=True.
    """
    sim, obs = pm._paired_values(df, sim_col, obs_col, log=False, positive=True,
                                 nonpositive=nonpositive, offset_frac=offset_frac)
    sim_p = np.percentile(sim, percentiles)
    obs_p = np.percentile(obs, percentiles)
    return [ratio_term(s, o, ratio) for s, o in zip(sim_p, obs_p)]


def kge_tail_anchors(df, sim_col="Qsim", obs_col="Qobs", low_p=5, high_p=95,
                     ratio="linear", weights=None, nonpositive="keep", offset_frac=0.01):
    """
    KGE plus one low and one high percentile anchor: 1 - sqrt of the weighted
    sum of five terms, higher is better, ideal 1.

    Terms (names in KGE_TAIL_ANCHOR_TERMS):
        "r", "alpha", "beta": (1 - r) ** 2, (1 - alpha) ** 2, (1 - beta) ** 2,
                with r the Pearson correlation, alpha = std(sim) / std(obs)
                and beta = mean(sim) / mean(obs), as in pm.kge;
        "low":  ratio_term(sim_low, obs_low, ratio), with sim_low and obs_low
                the low_p percentile of each series (np.percentile, linear
                interpolation);
        "high": the same at high_p.

    Pairing: the KGE terms pair through pm._paired_values with
    positive=False, the percentile terms with positive=True (so 'offset'
    shifts only the percentile terms).

    Parameters:
        df (pd.DataFrame): table containing `sim_col` and `obs_col`.
        sim_col, obs_col (str): simulated and observed discharge columns.
        low_p, high_p (real): percentiles strictly between 0 and 100,
            low_p < high_p (ValueError otherwise; TypeError if not a number).
        ratio (str): "linear" or "log" (see ratio_term); only the percentile
            terms use it.
        weights (dict or None): weights by term name (see normalize_weights);
            an unknown name raises ValueError.
        nonpositive (str), offset_frac (float): how observed flows <= 0 are
            handled ('keep', 'offset' or 'skip'; see pm._paired_values).

    Decisions (tutor's logic decisions, 2026-10-28):
        - The ratio option changes only the percentile terms, never r, alpha
          or beta.
        - Zero variance (observed or simulated) and a zero observed mean
          raise ValueError.
        - The percentile terms follow the nonpositive switch and raise for
          any percentile <= 0.

    Returns:
        float: the score, ideal 1.
    """
    _check_percentile("low_p", low_p)
    _check_percentile("high_p", high_p)
    if not low_p < high_p:
        raise ValueError(f"low_p must be below high_p, got low_p={low_p!r}, high_p={high_p!r}")

    terms = _kge_terms(df, sim_col, obs_col, nonpositive, offset_frac)
    terms["low"], terms["high"] = _percentile_terms(df, sim_col, obs_col, [low_p, high_p],
                                                    ratio, nonpositive, offset_frac)
    return combine_terms(terms, weights)


def _aggregate_block(values, block_agg):
    """Mean, median, max or sum of a block's percentile terms, as a Python float."""
    functions = {"mean": np.mean, "median": np.median, "max": np.max, "sum": np.sum}
    if block_agg not in BLOCK_AGG_OPTIONS:
        raise ValueError(f"block_agg must be one of {BLOCK_AGG_OPTIONS}, not {block_agg!r}")
    return float(functions[block_agg](values))


def _check_percentile_list(name, percentiles):
    """Raise unless `percentiles` is a non-empty list or tuple of unique percentiles in (0, 100)."""
    if not isinstance(percentiles, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, not {type(percentiles).__name__}")
    if not percentiles:
        raise ValueError(f"{name} is empty")
    for i, value in enumerate(percentiles):
        _check_percentile(f"{name}[{i}]", value)
    if len(set(percentiles)) != len(percentiles):
        raise ValueError(f"{name} has duplicate percentiles: {list(percentiles)}")


def kge_percentile_blocks(df, sim_col="Qsim", obs_col="Qobs", low_percentiles=DEFAULT_LOW_BLOCK,
                          high_percentiles=DEFAULT_HIGH_BLOCK, ratio="linear", block_agg="mean",
                          weights=None, nonpositive="keep", offset_frac=0.01):
    """
    KGE plus a low and a high block of percentile terms: 1 - sqrt of the
    weighted sum of five terms, higher is better, ideal 1.

    Terms (names in KGE_PERCENTILE_BLOCK_TERMS):
        "r", "alpha", "beta": as in kge_tail_anchors (pm.kge's components);
        "low_block":  the percentile terms of low_percentiles (each
                      ratio_term(sim_p, obs_p, ratio): (1 - R_p) ** 2 for
                      linear, (ln R_p) ** 2 for log), aggregated by block_agg
                      into ONE term;
        "high_block": the same for high_percentiles.

    "mean" is the default block_agg: a block does not gain weight from having
    more percentiles (settled by J, 2026-10-27). "median" ignores both ends of
    a short block and hides the extreme percentile; it is an option to show
    the effect, not the default.

    Parameters:
        df (pd.DataFrame): table containing `sim_col` and `obs_col`.
        sim_col, obs_col (str): simulated and observed discharge columns.
        low_percentiles, high_percentiles (list or tuple): percentiles of each
            block (default: the pdi_low and pdi_high lists).
        ratio (str): "linear" or "log" (see ratio_term); only the blocks use it.
        block_agg (str): one of BLOCK_AGG_OPTIONS ("mean", "median", "max",
            "sum").
        weights (dict or None): weights by term name (see normalize_weights);
            an unknown name raises ValueError.
        nonpositive (str), offset_frac (float): how observed flows <= 0 are
            handled ('keep', 'offset' or 'skip'; see pm._paired_values). The
            KGE terms pair with positive=False, the blocks with positive=True.

    Decisions (tutor's logic decisions, 2026-10-28):
        - Percentile lists must be unique, strictly between 0 and 100, and
          every low percentile below every high one.

    Returns:
        float: the score, ideal 1.
    """
    if block_agg not in BLOCK_AGG_OPTIONS:
        raise ValueError(f"block_agg must be one of {BLOCK_AGG_OPTIONS}, not {block_agg!r}")
    _check_percentile_list("low_percentiles", low_percentiles)
    _check_percentile_list("high_percentiles", high_percentiles)
    if not max(low_percentiles) < min(high_percentiles):
        raise ValueError(f"every low percentile must be below every high one, got "
                         f"low_percentiles={list(low_percentiles)}, "
                         f"high_percentiles={list(high_percentiles)}")

    terms = _kge_terms(df, sim_col, obs_col, nonpositive, offset_frac)
    for name, percentiles in (("low_block", low_percentiles), ("high_block", high_percentiles)):
        block = _percentile_terms(df, sim_col, obs_col, list(percentiles), ratio,
                                  nonpositive, offset_frac)
        terms[name] = _aggregate_block(block, block_agg)
    return combine_terms(terms, weights)


def _kge_np_terms(df, sim_col, obs_col, nonpositive, offset_frac):
    """
    Squared distances of the non-parametric KGE components (Pool et al. 2018):
    {"r_s": (1 - r_s) ** 2, "alpha_np": (1 - alpha_np) ** 2,
    "beta": (1 - beta) ** 2}. Pairs through pm._paired_values with
    positive=False.

    r_s: Pearson correlation of the ranks of the paired sim and obs (Eq. 6),
    ties get average ranks. alpha_np = 1 - 0.5 * sum_k |sim_sorted[k] /
    (n * mean_sim) - obs_sorted[k] / (n * mean_obs)| (Eq. 5), each series
    sorted separately in descending order and divided by n times its own mean.
    beta = mean_sim / mean_obs (Eq. 1).

    Raises ValueError for fewer than 2 paired values, an observed or simulated
    mean <= 0, or a series whose ranks are all equal (a constant series).
    """
    sim, obs = pm._paired_values(df, sim_col, obs_col, log=False, positive=False,
                                 nonpositive=nonpositive, offset_frac=offset_frac)
    n = len(obs)
    if n < 2:
        raise ValueError(f"KGE_NP needs at least 2 paired values, got {n}")
    obs_mean, sim_mean = obs.mean(), sim.mean()
    if obs_mean <= 0:
        raise ValueError(f"observed series has mean {obs_mean!r} <= 0; KGE_NP is undefined")
    if sim_mean <= 0:
        raise ValueError(f"simulated series has mean {sim_mean!r} <= 0; KGE_NP is undefined")
    obs_ranks = pd.Series(obs).rank(method="average").to_numpy()
    sim_ranks = pd.Series(sim).rank(method="average").to_numpy()
    if np.all(obs_ranks == obs_ranks[0]):
        raise ValueError("observed series is constant (all ranks equal); "
                         "the rank correlation is undefined")
    if np.all(sim_ranks == sim_ranks[0]):
        raise ValueError("simulated series is constant (all ranks equal); "
                         "the rank correlation is undefined")
    r_s = np.corrcoef(sim_ranks, obs_ranks)[0, 1]
    sim_sorted = np.sort(sim)[::-1] / (n * sim_mean)
    obs_sorted = np.sort(obs)[::-1] / (n * obs_mean)
    alpha_np = 1.0 - 0.5 * np.sum(np.abs(sim_sorted - obs_sorted))
    beta = sim_mean / obs_mean
    return {"r_s": (1.0 - r_s) ** 2, "alpha_np": (1.0 - alpha_np) ** 2,
            "beta": (1.0 - beta) ** 2}


def kge_np(df, sim_col="Qsim", obs_col="Qobs", weights=None, nonpositive="keep",
           offset_frac=0.01):
    """
    Non-parametric KGE (Pool et al. 2018, Hydrological Sciences Journal
    63(13-14), 1941-1953, Eq. 7): 1 - sqrt of the weighted sum of three terms,
    higher is better, ideal 1.

    Terms (names in KGE_NP_TERMS):
        "r_s":      (1 - r_s) ** 2, r_s the Spearman rank correlation: the
                    Pearson correlation of the ranks of the paired series
                    (Eq. 6);
        "alpha_np": (1 - alpha_np) ** 2, alpha_np = 1 - 0.5 * sum_k
                    |sim_sorted[k] / (n * mean_sim) - obs_sorted[k] /
                    (n * mean_obs)| over the n pairs, each series sorted
                    separately in descending order and divided by n times its
                    own mean (Eq. 5);
        "beta":     (1 - beta) ** 2, beta = mean_sim / mean_obs (Eq. 1).

    Parameters:
        df (pd.DataFrame): table containing `sim_col` and `obs_col`.
        sim_col, obs_col (str): simulated and observed discharge columns.
        weights (dict or None): weights by term name (see normalize_weights);
            an unknown name raises ValueError.
        nonpositive (str), offset_frac (float): how observed flows <= 0 are
            handled ('keep', 'offset' or 'skip'; see pm._paired_values). The
            pairing uses positive=False, so 'offset' does not shift it.

    Decisions (tutor's logic decisions, 2026-10-28):
        - Ties get average ranks (the paper does not say how to rank ties).
        - The pairing follows the nonpositive switch.
        - The observed and simulated means must be positive (ValueError).
        - A constant series (all ranks equal) raises ValueError.

    Returns:
        float: the score, ideal 1.
    """
    return combine_terms(_kge_np_terms(df, sim_col, obs_col, nonpositive, offset_frac), weights)


def _jsd(p, q):
    """
    Jensen-Shannon divergence, base 2, of two probability vectors of equal
    length that sum to 1: m = (p + q) / 2, JSD = 0.5 * sum(p * log2(p / m))
    + 0.5 * sum(q * log2(q / m)). A summand whose p_i (or q_i) is 0
    contributes 0 (Ficchi et al. 2026, Section 2.1.3).
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    m = (p + q) / 2.0
    total = 0.0
    for a in (p, q):
        nonzero = a > 0
        total += 0.5 * float(np.sum(a[nonzero] * np.log2(a[nonzero] / m[nonzero])))
    return total


def _bin_indices(x, edges):
    """
    Bin of each value of x for the bins defined by edges, with the paper's
    closure: the first bin is [edges[0], edges[1]], every later bin is
    (edges[i-1], edges[i]]. Values outside the edges (and a last edge that
    differs from the maximum by rounding) are clipped into the end bins.
    """
    indices = np.searchsorted(edges, x, side="left") - 1
    return np.clip(indices, 0, len(edges) - 2)


def _number_of_bins(x_obs, x_min, x_max, epsilon, ts_factor=1.0):
    """
    Number of equal-width bins (Ficchi et al. 2026, Section 2.1.3):
    Freedman-Diaconis width h_fd = 2 * IQR(x_obs) * n_obs ** (-1/3), or
    (x_max - x_min) / JSD_MIN_BINS when the IQR is 0; h = max(h_fd,
    min(100 * epsilon, 0.1)); n_raw = ceil(ts_factor ** (1/3) * (x_max - x_min)
    / h), the ratio rounded to 9 decimals before the ceiling; the result is
    n_raw limited to [JSD_MIN_BINS, JSD_MAX_BINS]. ts_factor is the time step in
    days (1 for daily data).
    """
    q25, q75 = np.percentile(x_obs, [25, 75])
    iqr = q75 - q25
    if iqr > 0:
        h_fd = 2.0 * iqr * len(x_obs) ** (-1.0 / 3.0)
    else:
        h_fd = (x_max - x_min) / JSD_MIN_BINS
    h = max(h_fd, min(100.0 * epsilon, 0.1))
    n_raw = math.ceil(round(ts_factor ** (1.0 / 3.0) * (x_max - x_min) / h, 9))
    return max(min(n_raw, JSD_MAX_BINS), JSD_MIN_BINS)


def _jsd_between(obs, sim, ts_factor):
    """
    JSD of the log-flow histograms of two non-negative finite arrays of equal
    length (Ficchi et al. 2026, Section 2.1.3): epsilon = min(1e-6, 0.1 *
    smallest positive value of obs and sim); zeros replaced by epsilon; natural
    log; bins on the common support of both series (_number_of_bins); densities
    count / (n * width) plus epsilon, renormalized to probabilities; _jsd.

    Raises ValueError if neither array has a positive value, or if every value
    of both series is identical (zero-width bins).
    """
    obs, sim = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    c = np.concatenate([obs[obs > 0], sim[sim > 0]])
    if c.size == 0:
        raise ValueError("JSD needs at least one positive flow value")
    epsilon = min(1e-6, 0.1 * c.min())
    obs = np.where(obs == 0, epsilon, obs)
    sim = np.where(sim == 0, epsilon, sim)
    x_obs, x_sim = np.log(obs), np.log(sim)
    x_min = min(x_obs.min(), x_sim.min())
    x_max = max(x_obs.max(), x_sim.max())
    if x_max == x_min:
        raise ValueError(f"every value of both series is identical (log value {x_min!r}); "
                         "the histogram bins would have zero width")
    n_bins = _number_of_bins(x_obs, x_min, x_max, epsilon, ts_factor)
    edges = x_min + (np.arange(n_bins + 1) / n_bins) * (x_max - x_min)
    count_obs = np.bincount(_bin_indices(x_obs, edges), minlength=n_bins)
    count_sim = np.bincount(_bin_indices(x_sim, edges), minlength=n_bins)
    widths = np.diff(edges)
    p_tilde = count_obs / (len(x_obs) * widths) + epsilon
    q_tilde = count_sim / (len(x_sim) * widths) + epsilon
    return _jsd(p_tilde / p_tilde.sum(), q_tilde / q_tilde.sum())


def jdkge(df, sim_col="Qsim", obs_col="Qobs", weights=None, ts_factor=1.0):
    """
    JDKGE (Ficchi et al. 2026, preprint egusphere-2026-43, Eq. 2): KGE' plus
    the squared Jensen-Shannon divergence of the log-flow histograms; 1 - sqrt
    of the weighted sum of four terms, higher is better, ideal 1. With all
    weights 1: 1 - sqrt((r-1)^2 + (beta-1)^2 + (alpha-1)^2 + JSD^2).

    Terms (names in JDKGE_TERMS):
        "r":     (1 - r) ** 2, r the Pearson correlation;
        "beta":  (1 - beta) ** 2, beta = mean_sim / mean_obs;
        "alpha": (1 - alpha) ** 2, alpha = cv_sim / cv_obs, cv = std / mean
                 (ddof 0 for both);
        "jsd":   JSD ** 2, the JSD (base 2) of the log-flow histograms
                 (Sections 2.1.2 and 2.1.3): zeros replaced by epsilon =
                 min(1e-6, 0.1 * smallest positive value), natural log, bins
                 on the union of both series, Freedman-Diaconis width from the
                 observed series, 25 to 100 bins, densities smoothed with
                 epsilon and renormalized.

    Pairs: rows where either value is missing, not finite or negative are
    dropped; the same cleaned sample (zeros included) is used for all four
    terms. jdkge has no nonpositive switch: this is its own pipeline.

    Parameters:
        df (pd.DataFrame): table containing `sim_col` and `obs_col`.
        sim_col, obs_col (str): simulated and observed discharge columns.
        weights (dict or None): weights by term name (see normalize_weights);
            an unknown name raises ValueError.
        ts_factor (float): the paper's ts_f, the time step in days (1 for
            daily data); scales the bin count and the record-length check.
            Checked first: a real number (TypeError otherwise, also for bool
            and strings), finite and > 0 (ValueError otherwise).

    Decisions (tutor's logic decisions, 2026-10-28):
        - Natural log; IQR with numpy's default percentile method; ddof 0
          for cv; ceiling after rounding to 9 decimals.
        - One cleaned sample for all four terms (the paper does not say which
          sample the KGE' terms use); zeros become epsilon only inside the
          JSD pipeline.
        - Fewer than JSD_MIN_YEARS years of pairs gives a UserWarning, not an
          error (the paper only recommends 4 years of daily data).
        - The fixed-reference option is not part of this step.
        - ts_factor is checked before anything else.

    Raises ValueError for an invalid ts_factor, fewer than 2 pairs, a zero
    observed or simulated std, or an observed or simulated mean <= 0.

    Returns:
        float: the score, ideal 1.
    """
    _check_real("ts_factor", ts_factor)
    if not math.isfinite(ts_factor) or ts_factor <= 0:
        raise ValueError(f"ts_factor must be finite and > 0, got {ts_factor!r}")
    values = df[[sim_col, obs_col]].to_numpy(dtype=float)
    keep = np.all(np.isfinite(values), axis=1) & np.all(values >= 0, axis=1)
    sim, obs = values[keep, 0], values[keep, 1]
    n = len(obs)
    if n < 2:
        raise ValueError(f"JDKGE needs at least 2 valid pairs, got {n}")
    years = n * ts_factor / 365.25
    if years < JSD_MIN_YEARS:
        warnings.warn(f"jdkge: {n} pairs cover about {years:.2f} years; Ficchi et al. (2026) "
                      f"recommend at least {JSD_MIN_YEARS} years of daily observed data",
                      UserWarning, stacklevel=2)

    obs_std, sim_std, obs_mean, sim_mean = obs.std(), sim.std(), obs.mean(), sim.mean()
    if obs_std == 0:
        raise ValueError("observed series has zero variance; JDKGE is undefined")
    if sim_std == 0:
        raise ValueError("simulated series has zero variance; JDKGE is undefined")
    if obs_mean <= 0:
        raise ValueError(f"observed series has mean {obs_mean!r} <= 0; JDKGE is undefined")
    if sim_mean <= 0:
        raise ValueError(f"simulated series has mean {sim_mean!r} <= 0; JDKGE is undefined")
    r = np.corrcoef(sim, obs)[0, 1]
    beta = sim_mean / obs_mean
    alpha = (sim_std / sim_mean) / (obs_std / obs_mean)
    jsd = _jsd_between(obs, sim, ts_factor)
    terms = {"r": (1.0 - r) ** 2, "beta": (1.0 - beta) ** 2, "alpha": (1.0 - alpha) ** 2,
             "jsd": jsd ** 2}
    return combine_terms(terms, weights)
