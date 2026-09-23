"""Scoring, paired comparison, and the change-control gate."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(a, p) -> float:
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(p, float))))


def mape(a, p) -> float:
    a, p = np.asarray(a, float), np.asarray(p, float)
    ok = np.abs(a) > 1e-9
    return float(np.mean(np.abs((a[ok] - p[ok]) / a[ok])) * 100)


def rmse(a, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(p, float)) ** 2)))


def direction_pct(a, p, p0) -> float:
    """Share of cases where the sign of the change from the origin is called right."""
    da = np.sign(np.asarray(a, float) - np.asarray(p0, float))
    dp = np.sign(np.asarray(p, float) - np.asarray(p0, float))
    ok = da != 0
    return float(np.mean(da[ok] == dp[ok]) * 100) if ok.any() else float("nan")


def score_long(df: pd.DataFrame, model_cols: list[str], by: list[str]) -> pd.DataFrame:
    """Long metrics table. df carries actual, origin_price, h and one column per model."""
    out = []
    for keys, g in df.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for m in model_cols:
            sub = g[g[m].notna()]
            if not len(sub):
                continue
            row = dict(zip(by, keys))
            row.update(model=m, n=len(sub),
                       MAE=mae(sub["actual"], sub[m]),
                       MAPE=mape(sub["actual"], sub[m]),
                       RMSE=rmse(sub["actual"], sub[m]),
                       direction_pct=direction_pct(sub["actual"], sub[m], sub["origin_price"]))
            out.append(row)
    return pd.DataFrame(out)


def paired_table(df: pd.DataFrame, challenger: str,
                 references: tuple[str, ...] = ("panel_fe", "naive")) -> pd.DataFrame:
    """One row per horizon. Positive pct_better means the challenger is better."""
    rows = []
    for h, g in df.groupby("h"):
        g = g[g[challenger].notna()]
        r = dict(h=int(h), n=len(g), challenger=challenger,
                 challenger_MAE=mae(g["actual"], g[challenger]),
                 challenger_MAPE=mape(g["actual"], g[challenger]),
                 challenger_direction_pct=direction_pct(g["actual"], g[challenger], g["origin_price"]))
        for ref in references:
            rm = mae(g["actual"], g[ref])
            r[f"{ref}_MAE"] = rm
            r[f"vs_{ref}_pct"] = (rm - r["challenger_MAE"]) / rm * 100
        rows.append(r)
    return pd.DataFrame(rows).sort_values("h").reset_index(drop=True)


def _dm_series_stats(d: np.ndarray, lag: int) -> tuple[float, float, int]:
    """Newey-West mean/variance of one chronologically-ordered loss-differential
    series. `d` must already be sorted in time order by the caller -- the lag
    terms assume consecutive entries are consecutive origins of the same series,
    which is the entire premise of the overlap correction below."""
    n = len(d)
    dbar = d.mean()
    g0 = np.mean((d - dbar) ** 2)
    var = g0
    for L in range(1, min(lag, n - 1) + 1):
        g = np.mean((d[L:] - dbar) * (d[:-L] - dbar))
        var += 2 * (1 - L / (lag + 1)) * g
    return dbar, max(var, 1e-12), n


def diebold_mariano(a, p1, p2, h: int, groups=None) -> dict:
    """Newey-West DM test on absolute-error differentials. Negative stat favours p1.

    The overlap correction (h-step forecasts from nearby origins share target
    weeks, hence correlated errors) is only valid within one chronologically
    ordered series. Pooling several markets into one table and running the
    correction across the concatenated rows mixes unrelated series at every
    market boundary, understating (or otherwise distorting) significance --
    found on the FEWSNET side of this project (D-50, main branch) and ported
    here since `run_afex.py`'s `groupby(["origin", "market", "h"])` has the
    identical date-first sort order and therefore the identical bug.

    Pass `groups` (e.g. one market/series label per row, in the same order as
    a/p1/p2) to compute the correction separately within each group's own time
    order and combine them assuming independence across groups, instead of one
    pooled, boundary-contaminated series. `a`/`p1`/`p2` must already be sorted
    by (group, time) when `groups` is given -- this function trusts that order
    within each group, it does not re-sort by any date column it isn't given.
    When `groups` is omitted, behaviour is unchanged from before (single
    series, whatever order the caller provides)."""
    a, p1, p2 = (np.asarray(x, float) for x in (a, p1, p2))
    d_all = np.abs(a - p1) - np.abs(a - p2)
    n_total = len(d_all)
    if n_total < 10:
        return {"dm_stat": float("nan"), "dm_p": float("nan"), "n": n_total}
    lag = max(0, h - 1)

    if groups is None:
        dbar, var, n = _dm_series_stats(d_all, lag)
        stat = dbar / np.sqrt(var / n)
    else:
        groups = np.asarray(groups)
        dbar_all = d_all.mean()
        var_of_mean_numerator = 0.0
        for g_val in np.unique(groups):
            d_g = d_all[groups == g_val]
            _, var_g, n_g = _dm_series_stats(d_g, lag)
            var_of_mean_numerator += n_g * var_g
        var_of_overall_mean = var_of_mean_numerator / (n_total ** 2)
        stat = dbar_all / np.sqrt(var_of_overall_mean)

    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(stat) / sqrt(2))))
    return {"dm_stat": float(stat), "dm_p": float(p), "n": n_total}


def change_control(paired: pd.DataFrame, gate_horizons=(4, 13), threshold=5.0) -> dict:
    """Section 5.3: challenger must beat the incumbent by >5% MAE at BOTH gate
    horizons simultaneously. This function adjudicates ONE review only; the rule
    also requires the result to hold across two consecutive quarterly reviews."""
    detail = {}
    for h in gate_horizons:
        row = paired[paired["h"] == h]
        if row.empty:
            detail[int(h)] = {"pct_better": None, "pass": False, "note": "horizon absent"}
            continue
        v = float(row["vs_panel_fe_pct"].iloc[0])
        detail[int(h)] = {"pct_better": v, "pass": bool(v > threshold)}
    passed = all(d["pass"] for d in detail.values())
    return {
        "gate_horizons": list(gate_horizons),
        "threshold_pct": threshold,
        "per_horizon": detail,
        "single_review_pass": passed,
        "verdict": ("PASS this review; requires a second consecutive quarterly "
                    "review to change production" if passed else
                    "FAIL; production model unchanged"),
    }
