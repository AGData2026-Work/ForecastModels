"""
Joint forecast: GRU (production) combined with the analyst's Inflation Adjusted model (2026-10-26). A candidate, not
production.

    python src/joint_forecast.py evaluate --analyst outputs/joint_forecast/analyst_forecasts.csv \
        --out outputs/joint_forecast/backtest
    python src/joint_forecast.py live --analyst outputs/joint_forecast/analyst_forecasts.csv \
        --gru outputs/production_monthly_parity_serving/predictions/2026-10-12.csv \
        --out outputs/joint_forecast/predictions/2026-10-12.csv

The combination averages the two models' predicted changes in log terms and applies the result to OUR origin price:
joint = p0_gru x sqrt((gru / p0_gru) x (analyst / p0_analyst)). Where the two origin prices agree (every FEWS NET
observed month) this is the plain geometric mean of the two forecasts, the version scored in the 21 Oct head to head.
Where they differ (filled months: his fills come from the data team file, ours from panel v3) each model keeps its own
predicted change and the joint forecast starts from our published price; the gap is reported per row.

evaluate: production GRU forecasts (outputs/d75_monthly_parity_pit/h{1,3,6}/predictions_paired.csv) and the analyst's,
on shared (market, origin, horizon), scored on FEWS NET observed actuals against naive on the same pairs; periods all
years, 2019-24, 2025-26 and Oct 2024 to Aug 2026; Diebold-Mariano p for joint vs GRU; the two-period keep rule.

live: also writes 80% and 90% ranges, the production method applied to the joint forecast (quantiles of every past
miss whose target month is at or before the origin, pooled over markets, per horizon; GRU alone before Jan 2017).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import diebold_mariano, mae, mape  # noqa: E402

EVAL = "outputs/d75_monthly_parity_pit/h{h}/predictions_paired.csv"
PERIODS = [("All years", "2000-01-01", "2100-01-01"), ("2019-24", "2019-01-01", "2024-12-31"),
           ("2025-26", "2025-01-01", "2100-01-01"), ("Oct 2024 to Aug 2026", "2024-10-01", "2100-01-01")]


def combine(gru: pd.Series, p0_gru: pd.Series, analyst: pd.Series, p0_analyst: pd.Series) -> pd.Series:
    """Geometric mean of the two predicted changes, applied to our origin price."""
    return p0_gru * np.sqrt((gru / p0_gru) * (analyst / p0_analyst))


def load_analyst(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["origin", "target"])


def evaluate(analyst: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows, pairs = [], []
    for h in (1, 3, 6):
        g = pd.read_csv(EVAL.format(h=h), parse_dates=["origin"]).query("h == @h")
        g["target"] = g.origin + pd.DateOffset(months=h)
        x = g.merge(analyst[analyst.h == h][["market", "origin", "p0", "recommended"]], on=["market", "origin"])
        x["joint"] = combine(x.GRU, x.origin_price, x.recommended, x.p0)
        x["p0_gap_pct"] = 100 * (x.p0 / x.origin_price - 1)
        pairs.append(x)
        for per, a, z in PERIODS:
            s = x[(x.target >= a) & (x.target <= z)].sort_values(["market", "origin"])
            n = mae(s.actual, s.naive)
            r = dict(h=h, period=per, n=len(s), naive_mape=mape(s.actual, s.naive))
            for col, lab in [("GRU", "gru"), ("recommended", "analyst"), ("joint", "joint")]:
                r[f"{lab}_mape"] = mape(s.actual, s[col])
                r[f"{lab}_vs_naive_pct"] = 100 * (n - mae(s.actual, s[col])) / n
            r["p_joint_vs_gru"] = diebold_mariano(s.actual.values, s.joint.values, s.GRU.values, h, groups=s.market.values)["dm_p"]
            rows.append(r)
    t = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(out / "summary.csv", index=False)
    pd.concat(pairs).to_csv(out / "pairs.csv", index=False)
    k = t[t.period.isin(["2019-24", "2025-26"])].assign(gain=lambda d: d.joint_vs_naive_pct - d.gru_vs_naive_pct)
    k = k.pivot(index="h", columns="period", values="gain")
    k["keep_rule"] = (k["2019-24"] > 0) & (k["2025-26"] > 0)
    k.round(2).to_csv(out / "keep_rule.csv")
    return t, k


def past_misses(analyst: pd.DataFrame, h: int, until: pd.Timestamp) -> np.ndarray:
    """log(actual / joint) for every production forecast whose target month is at or before `until` (known by the
    forecast date); before the analyst's first forecast (Jan 2017) the joint falls back to the GRU alone."""
    g = pd.read_csv(EVAL.format(h=h), parse_dates=["origin"]).query("h == @h")
    g["target"] = g.origin + pd.DateOffset(months=h)
    x = g.merge(analyst[analyst.h == h][["market", "origin", "p0", "recommended"]], on=["market", "origin"], how="left")
    x["joint"] = combine(x.GRU, x.origin_price, x.recommended, x.p0).fillna(x.GRU)
    x = x[x.target <= until]
    return np.log(x.actual / x.joint).to_numpy()


def live(analyst: pd.DataFrame, gru_path: str, out: Path) -> pd.DataFrame:
    g = pd.read_csv(gru_path, parse_dates=["origin"])
    a = analyst[["market", "origin", "h", "p0", "recommended"]]
    x = g.merge(a, on=["market", "origin", "h"], how="left")
    x["joint"] = combine(x.forecast_price, x.origin_price, x.recommended, x.p0)
    x = x.rename(columns={"origin_price": "origin_price_gru", "p0": "origin_price_analyst",
                          "forecast_price": "gru", "recommended": "inflation_adjusted"})
    x["origin_price_gap_pct"] = 100 * (x.origin_price_analyst / x.origin_price_gru - 1)
    # 80% and 90% ranges: the production method (quantiles of all past misses known by the forecast date, pooled
    # over markets, per horizon), applied to the joint forecast
    for h in sorted(x.h.unique()):
        r = past_misses(analyst, int(h), x.origin.max())
        m = x.h == h
        for lv in (80, 90):
            a = (100 - lv) / 200
            x.loc[m, f"lo{lv}"] = x.loc[m, "joint"] * np.exp(np.quantile(r, a))
            x.loc[m, f"hi{lv}"] = x.loc[m, "joint"] * np.exp(np.quantile(r, 1 - a))
        x.loc[m, "n_past_misses"] = len(r)
    x["note"] = np.where(x.inflation_adjusted.isna(), "no analyst forecast for this market and origin; joint not issued",
                         np.where(x.origin_price_gap_pct.abs() > 1, "origin prices differ by more than 1%", ""))
    cols = ["market", "origin", "h", "origin_price_gru", "origin_price_analyst", "origin_price_gap_pct", "gru",
            "inflation_adjusted", "joint", "lo80", "hi80", "lo90", "hi90", "n_past_misses", "note"]
    out.parent.mkdir(parents=True, exist_ok=True)
    x[cols].round(2).to_csv(out, index=False)
    return x[cols]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["evaluate", "live"])
    ap.add_argument("--analyst", required=True)
    ap.add_argument("--gru", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    an = load_analyst(a.analyst)
    pd.set_option("display.width", 250)
    if a.mode == "evaluate":
        t, k = evaluate(an, Path(a.out))
        print(t.round(2).to_string(index=False)); print("\nkeep rule (joint minus GRU, points vs naive):\n", k.round(2).to_string())
    else:
        print(live(an, a.gru, Path(a.out)).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
