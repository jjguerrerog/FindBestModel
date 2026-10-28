"""
Read one hold-out run of find_best_model_holdout.py and pick winners.

A run folder (performance_holdout/<gauge>/<run_label>/) holds run_info.json,
summary.parquet (indexed by model: name, n_ok, n_failed, best_<metric> and
file) and one model_<NN>_<name>.parquet per model (one row per realization:
the parameters, the metric columns, metrics_error and error).

Conventions:

- Every metric is "higher is better".
- A realization is ok when its "error" value is null; failed rows never win
  and are left out of every share and correlation.
- NaN metric values are ignored when picking a winner. A metric with no
  value at all has no winner (None), which is not an error.
- Ties go to the first row in table order (best_row) or to the lowest model
  number (best_model_per_metric).

Across gauges, winner_agreement_counts counts how often two metrics pick the
same winning model.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_run(run_dir):
    """
    Reads a run folder.

    Returns a dict with "run_info" (the dict from run_info.json), "summary"
    (the DataFrame from summary.parquet) and "tables" (dict from model number,
    an int, to the DataFrame of the file named in the summary's "file"
    column). The run is not required to be finished: check
    run_info["status"] ("finished" or "running").

    Raises FileNotFoundError if run_info.json or summary.parquet is missing.
    """
    run_dir = Path(run_dir)
    info_path, summary_path = run_dir / "run_info.json", run_dir / "summary.parquet"
    for path in (info_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(f"{path.name} not found in {run_dir}")
    run_info = json.loads(info_path.read_text())
    summary = pd.read_parquet(summary_path)
    tables = {int(model): pd.read_parquet(run_dir / file) for model, file in summary["file"].items()}
    return {"run_info": run_info, "summary": summary, "tables": tables}


def best_row(table, metric):
    """
    The ok row (null "error") with the highest non-NaN `metric` value, as a
    Series; ties go to the first such row in table order. None if no row
    qualifies. Raises KeyError if the metric column is missing.
    """
    if metric not in table.columns:
        raise KeyError(f"metric column {metric!r} not in the table")
    values = table[metric].where(table["error"].isna())
    valid = values.notna().to_numpy()
    if not valid.any():
        return None
    scores = np.where(valid, values.to_numpy(dtype=float), -np.inf)
    return table.iloc[int(np.argmax(scores))]


def best_model_per_metric(summary, metrics):
    """
    Winner and runner-up of each metric from the summary's best_<metric>
    columns: {metric: {"model", "value", "second_model", "second_value"}}.
    Highest value wins, ties go to the lowest model number, NaN is ignored;
    missing places are None. Raises KeyError naming a missing best_ column.
    """
    result = {}
    for metric in metrics:
        column = f"best_{metric}"
        if column not in summary.columns:
            raise KeyError(f"column {column!r} not in the summary")
        ranked = (pd.DataFrame({"model": summary.index, "value": summary[column].to_numpy()})
                  .dropna(subset=["value"])
                  .sort_values(["value", "model"], ascending=[False, True]))
        places = [(int(row.model), float(row.value)) for row in ranked.head(2).itertuples()]
        places += [(None, None)] * (2 - len(places))
        result[metric] = {"model": places[0][0], "value": places[0][1],
                          "second_model": places[1][0], "second_value": places[1][1]}
    return result


def shared_winner_groups(winners):
    """
    {model: [metrics it wins]} from {metric: winning model}, in the order of
    the input; metrics whose winner is None are left out.
    """
    groups = {}
    for metric, model in winners.items():
        if model is not None:
            groups.setdefault(model, []).append(metric)
    return groups


def winner_agreement_counts(winners_by_gauge, metrics):
    """
    (agree, both): two DataFrames indexed and columned by `metrics`, from
    {gauge: the dict of best_model_per_metric}. both[a, b] counts the gauges
    where a and b both have a winning model; agree[a, b] those where it is
    also the same model. The diagonal counts the gauges where the metric has
    a winner. Raises KeyError naming the gauge and the metric if a gauge's
    dict lacks a metric.
    """
    metrics = list(metrics)
    agree = pd.DataFrame(0, index=metrics, columns=metrics)
    both = pd.DataFrame(0, index=metrics, columns=metrics)
    for gauge, winners in winners_by_gauge.items():
        models = {}
        for metric in metrics:
            if metric not in winners:
                raise KeyError(f"gauge {gauge!r} has no entry for metric {metric!r}")
            models[metric] = winners[metric]["model"]
        for a in metrics:
            for b in metrics:
                if models[a] is not None and models[b] is not None:
                    both.loc[a, b] += 1
                    agree.loc[a, b] += int(models[a] == models[b])
    return agree, both


def _ok_rows(tables, columns):
    """The `columns` of the ok rows of all tables, stacked."""
    parts = [table.loc[table["error"].isna(), list(columns)] for table in tables.values()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(columns))


def nan_share(tables, metric):
    """Share of the ok rows of all tables whose `metric` is NaN; NaN if there are no ok rows."""
    values = _ok_rows(tables, [metric])[metric]
    if len(values) == 0:
        return float("nan")
    return float(values.isna().mean())


def metric_rank_correlation(tables, metrics):
    """
    Spearman rank correlation between the metric columns over the ok rows of
    all tables (DataFrame.corr(method="spearman"), pairwise complete). A
    metric that is NaN in every row gives NaN in its row and column.
    """
    rows = _ok_rows(tables, metrics).astype(float)
    return rows.corr(method="spearman")
