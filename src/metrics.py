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
    found directly on this project's own FEWSNET panel, see docs/DECISIONS.md.

    Pass `groups` (e.g. one market label per row, in the same order as a/p1/p2)
    to compute the correction separately within each group's own time order and
    combine them assuming independence across groups, instead of one pooled,
    boundary-contaminated series. `a`/`p1`/`p2` must already be sorted by
    (group, time) when `groups` is given -- this function trusts that order
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


# ---------------------------------------------------------------------------
# bias (D-73): reported alongside error for every run
# ---------------------------------------------------------------------------
def _cluster_cov(X: np.ndarray, resid: np.ndarray, groups) -> np.ndarray:
    """OLS sandwich covariance, clustered by group (CR0); HC0 when fewer than 2 groups."""
    XtX_inv = np.linalg.inv(X.T @ X)
    g = np.asarray(groups) if groups is not None else np.arange(len(resid))
    if len(np.unique(g)) < 2:
        g = np.arange(len(resid))
    meat = np.zeros((X.shape[1], X.shape[1]))
    for k in np.unique(g):
        s = (X[g == k] * resid[g == k, None]).sum(axis=0)
        meat += np.outer(s, s)
    return XtX_inv @ meat @ XtX_inv


def bias_metrics(df: pd.DataFrame, pred: str, actual: str = "actual", naive: str = "naive",
                 h: str = "h", market: str = "market", origin: str = "origin") -> pd.DataFrame:
    """Per horizon:
    - share of forecasts above / below the actual, with a sign test against 50/50;
    - mean error (NGN/kg) and mean percentage error (positive = forecasts too high);
    - tracking signal, the planning convention: per market, sum of the last 12 errors / their
      mean absolute error (range -12..+12); reported as the share of (market, forecast) points
      where it is outside +/-4, the usual alarm level, and the mean of each market's latest value;
    - Mincer-Zarnowitz on changes from the origin price: log(actual/naive) = a + b log(pred/naive),
      unbiased if a = 0 and b = 1 (b < 1: forecasts move too much; b > 1: too little); standard
      errors clustered by market; joint Wald test, chi-square with 2 df."""
    import math
    rows = []
    for hh, d in df.groupby(h):
        d = d.dropna(subset=[actual, pred, naive]).sort_values([origin, market])
        a, p, n0 = d[actual].to_numpy(float), d[pred].to_numpy(float), d[naive].to_numpy(float)
        e = p - a
        n = len(e)
        over = int((e > 0).sum())
        z = (over - n / 2) / math.sqrt(n / 4) if n else float("nan")
        alarms, latest = [], []
        for _, q in d.groupby(market):
            eq = pd.Series((q[pred] - q[actual]).to_numpy(float))
            ts = eq.rolling(12, min_periods=12).sum() / eq.abs().rolling(12, min_periods=12).mean()
            ts = ts.dropna()
            if len(ts):
                alarms.extend((ts.abs() > 4).tolist())
                latest.append(float(ts.iloc[-1]))
        y = np.log(a / n0)
        x = np.log(p / n0)
        X = np.column_stack([np.ones(n), x])
        # numpy 2.0 + Apple Accelerate raises spurious "divide by zero in matmul" warnings here;
        # the results are checked for finiteness instead.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            V = _cluster_cov(X, y - (X * beta).sum(axis=1), d[market].to_numpy())
        assert np.isfinite(beta).all() and np.isfinite(V).all()
        dvec = beta - np.array([0.0, 1.0])
        try:
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                W = float(dvec @ np.linalg.inv(V) @ dvec)
        except np.linalg.LinAlgError:
            W = float("nan")
        se = np.sqrt(np.diag(V))
        rows.append(dict(
            h=hh, n=n, pct_over=round(100 * over / n, 1), pct_under=round(100 * (e < 0).sum() / n, 1),
            sign_test_p=round(math.erfc(abs(z) / math.sqrt(2)), 4),
            mean_error=round(float(e.mean()), 2), mean_pct_error=round(float(100 * (e / a).mean()), 1),
            ts12_pct_outside_4=round(100 * float(np.mean(alarms)), 1) if alarms else float("nan"),
            ts12_latest_mean=round(float(np.mean(latest)), 2) if latest else float("nan"),
            mz_intercept=round(float(beta[0]), 4), mz_intercept_se=round(float(se[0]), 4),
            mz_slope=round(float(beta[1]), 3), mz_slope_se=round(float(se[1]), 3),
            mz_joint_p=round(math.exp(-W / 2), 4) if np.isfinite(W) else float("nan")))
    return pd.DataFrame(rows)
