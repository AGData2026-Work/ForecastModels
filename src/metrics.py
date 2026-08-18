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


def diebold_mariano(a, p1, p2, h: int) -> dict:
    """Newey-West DM test on absolute-error differentials. Negative stat favours p1."""
    a, p1, p2 = (np.asarray(x, float) for x in (a, p1, p2))
    d = np.abs(a - p1) - np.abs(a - p2)
    n = len(d)
    if n < 10:
        return {"dm_stat": float("nan"), "dm_p": float("nan"), "n": n}
    dbar = d.mean()
    lag = max(0, h - 1)
    g0 = np.mean((d - dbar) ** 2)
    var = g0
    for L in range(1, lag + 1):
        g = np.mean((d[L:] - dbar) * (d[:-L] - dbar))
        var += 2 * (1 - L / (lag + 1)) * g
    var = max(var, 1e-12)
    stat = dbar / np.sqrt(var / n)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(stat) / sqrt(2))))
    return {"dm_stat": float(stat), "dm_p": float(p), "n": n}


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
