"""
Selection metrics for the gap-filling tests (OUTLINE_gap_metrics.md, Part 1).

This step holds only the shared helpers; the metrics are added in later steps.

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
"""

import math

RATIO_OPTIONS = ("linear", "log")


def _check_real(name, value):
    """Raise TypeError unless `value` is a real number (int or float, not bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
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
