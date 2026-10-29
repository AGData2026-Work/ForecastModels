"""
Production checks for the joint forecast (GRU + the analyst's Inflation Adjusted model), 2026-10-29: the same boxes
the GRU passed for D-77, on the shared forecasts (origins Jan 2017 to Jul 2026, his first forecast is Jan 2017).

    python src/joint_production_checks.py --analyst outputs/joint_forecast/analyst_forecasts.csv \
        --out outputs/joint_forecast/production_checks

1. D-45 bar per horizon (src/quality_gate.py's rule): Diebold-Mariano p < .05 against naive AND margin over naive
   larger than the seed-to-seed spread. Seeds: the joint forecast is built once per GRU seed (his model is
   deterministic), so the spread is the GRU's seed spread passed through the combination; the scored forecast uses the
   seed median, exactly as production does.
2. Accuracy against naive and against the GRU alone (MAE, MAPE, DM), all years and Oct 2024 to Aug 2026.
3. Direction right, falls called, rises called, bias (median forecast / actual - 1; share of forecasts too high).
4. Ranges: the production method (quantiles of every past miss, log(actual / forecast), whose outcome was known by
   the forecast date, pooled over markets, per horizon) applied to the joint forecast; 50 / 80 / 90%; coverage, misses
   below and above, width; next to the same method on the GRU alone, on the same forecasts. The pool of past misses
   starts with the production forecasts of 2015, the joint falling back to the GRU before the analyst's first forecast.
Scored on FEWS NET observed actuals (production predictions_paired.csv). Writes CSVs to --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from joint_forecast import combine  # noqa: E402
from metrics import diebold_mariano, mae, mape  # noqa: E402

RUN = "outputs/d75_monthly_parity_pit/h{h}"
PERIODS = [("All years", "2000-01-01"), ("Oct 2024 to Aug 2026", "2024-10-01")]
LEVELS = (0.5, 0.8, 0.9)


def load(analyst: pd.DataFrame, h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = pd.read_csv(Path(RUN.format(h=h)) / "predictions_paired.csv", parse_dates=["origin"]).query("h == @h")
    seeds = pd.read_csv(Path(RUN.format(h=h)) / "forecasts.csv", parse_dates=["origin"]).query("h == @h")
    a = analyst[analyst.h == h][["market", "origin", "p0", "recommended"]]
    x = paired.merge(a, on=["market", "origin"])
    x["target"] = x.origin + pd.DateOffset(months=h)
    x["joint"] = combine(x.GRU, x.origin_price, x.recommended, x.p0)
    s = seeds.merge(a, on=["market", "origin"])
    s["joint"] = combine(s.pred, s.origin_price, s.recommended, s.p0)
    s["target"] = s.origin + pd.DateOffset(months=h)
    return x.sort_values(["market", "origin"]), s


def d45(x: pd.DataFrame, s: pd.DataFrame, h: int) -> list[dict]:
    rows = []
    for per, lo in PERIODS:
        xx, ss = x[x.target >= lo], s[s.target >= lo]
        n_mae = mae(xx.actual, xx.naive)
        seed_mae = ss.groupby("seed")[["actual", "joint"]].apply(lambda g: mae(g.actual, g.joint))
        margin = n_mae - mae(xx.actual, xx.joint)
        dm = diebold_mariano(xx.actual.values, xx.joint.values, xx.naive.values, h, groups=xx.market.values)
        gs = ss.groupby("seed")[["actual", "pred"]].apply(lambda g: mae(g.actual, g.pred))
        rows.append(dict(h=h, period=per, n=len(xx), naive_mae=n_mae, joint_mae=mae(xx.actual, xx.joint),
                         gru_mae=mae(xx.actual, xx.GRU), joint_vs_naive_pct=100 * margin / n_mae,
                         gru_vs_naive_pct=100 * (n_mae - mae(xx.actual, xx.GRU)) / n_mae,
                         dm_p_vs_naive=dm["dm_p"], dm_stat=dm["dm_stat"], seed_mae_sd=float(seed_mae.std()),
                         margin_over_seed_sd=margin / float(seed_mae.std()), gru_seed_mae_sd=float(gs.std()),
                         joint_mape=mape(xx.actual, xx.joint), gru_mape=mape(xx.actual, xx.GRU), naive_mape=mape(xx.actual, xx.naive),
                         dm_p_joint_vs_gru=diebold_mariano(xx.actual.values, xx.joint.values, xx.GRU.values, h, groups=xx.market.values)["dm_p"],
                         d45_pass=bool(dm["dm_p"] < 0.05 and dm["dm_stat"] < 0 and margin / float(seed_mae.std()) > 1)))
    return rows


def behaviour(x: pd.DataFrame, h: int) -> list[dict]:
    rows = []
    for per, lo in PERIODS:
        xx = x[x.target >= lo]
        up, down = xx.actual > xx.origin_price, xx.actual < xx.origin_price
        for c in ("GRU", "joint"):
            f = xx[c]
            rows.append(dict(h=h, period=per, model=c, n=len(xx),
                             direction_right_pct=100 * np.mean(np.sign(f - xx.origin_price) == np.sign(xx.actual - xx.origin_price)),
                             falls_called_pct=100 * np.mean((f < xx.origin_price)[down]), rises_called_pct=100 * np.mean((f > xx.origin_price)[up]),
                             median_error_pct=100 * np.median(f / xx.actual - 1), share_too_high_pct=100 * np.mean(f > xx.actual)))
    return rows


def ranges(x: pd.DataFrame, full: pd.DataFrame, h: int) -> tuple[list[dict], pd.DataFrame]:
    """Past misses come from every production forecast since 2015 (`full`); where the analyst's model has no forecast
    (before Jan 2017) the joint forecast falls back to the GRU, so the joint pool uses the GRU's misses there."""
    rows, band = [], []
    for c in ("GRU", "joint"):
        r_all = np.log(full.actual / full[c])
        for org in sorted(x.origin.unique()):
            past = r_all[full.target <= org]
            if len(past) < 60:
                continue
            cur = x[x.origin == org]
            for lv in LEVELS:
                ql, qh = np.quantile(past, (1 - lv) / 2), np.quantile(past, 1 - (1 - lv) / 2)
                band.append(pd.DataFrame(dict(model=c, h=h, level=lv, market=cur.market, origin=org, target=cur.target, actual=cur.actual,
                                              forecast=cur[c], lo=cur[c] * np.exp(ql), hi=cur[c] * np.exp(qh))))
    b = pd.concat(band)
    b["inside"] = (b.actual >= b.lo) & (b.actual <= b.hi)
    for (c, lv), g in b.groupby(["model", "level"]):
        for per, lo in PERIODS:
            gg = g[g.target >= lo]
            rows.append(dict(h=h, model=c, level=lv, period=per, n=len(gg), coverage_pct=100 * gg.inside.mean(),
                             below_pct=100 * (gg.actual < gg.lo).mean(), above_pct=100 * (gg.actual > gg.hi).mean(),
                             width_pct_median=100 * ((gg.hi - gg.lo) / gg.forecast).median()))
    return rows, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyst", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    an = pd.read_csv(a.analyst, parse_dates=["origin", "target"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D, B, R, bands = [], [], [], []
    for h in (1, 3, 6):
        x, s = load(an, h)
        D += d45(x, s, h); B += behaviour(x, h)
        full = pd.read_csv(Path(RUN.format(h=h)) / "predictions_paired.csv", parse_dates=["origin"]).query("h == @h")
        full["target"] = full.origin + pd.DateOffset(months=h)
        full = full.merge(x[["market", "origin", "joint"]], on=["market", "origin"], how="left")
        full["joint"] = full.joint.fillna(full.GRU)                    # fallback: GRU alone where he has no forecast
        rr, bb = ranges(x, full, h); R += rr; bands.append(bb)
    pd.set_option("display.width", 250)
    for name, rows in [("1_d45_and_accuracy", D), ("2_direction_bias", B), ("3_ranges", R)]:
        t = pd.DataFrame(rows); t.round(4).to_csv(out / f"{name}.csv", index=False)
        print(f"\n== {name}\n{t.round(2).to_string(index=False)}")
    pd.concat(bands).to_csv(out / "3_ranges_by_forecast.csv.gz", index=False)


if __name__ == "__main__":
    main()
