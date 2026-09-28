"""Regime-aware soft blend: the GRU forecast weighted against a per-market
seasonal-naive baseline by how volatile each market currently looks.

Adopted by D-59 as the documented companion to the production GRU (not the
headline forecast; on MAE it trails the raw GRU at h=4 and h=13). Brought over
from the regime-blend-exploration branch (its D-54 to D-56), with two
look-ahead leaks removed:

1. The seasonal index is point-in-time. The exploration version fitted each
   market's index on every complete year in the panel, including years after
   the origin being forecast, which is a statistic derived from test windows
   reaching the forecast. Here the index for an origin in year Y is built only
   from complete calendar years before Y (a year is only complete once it has
   ended). With fewer than MIN_COMPLETE_YEARS such years the blend falls back
   to the pure model (weight 1) and the count is logged.
2. z_scale is point-in-time. It was the std of z across every market and every
   date; here it is the std of z readings dated strictly before each origin.

Everything uses only the maize price panel and the food inflation series in
data/external/; no new data source.

    python src/regime_blend.py --run-dir outputs/build3_h4h13_extended/GRU_unconditional --out outputs/regime_blend_soft/GRU

Point --run-dir at outputs/build3_h4h13_extended (main's D-56 extension), not
outputs/build3_underfit_corrected; on the 1,590 historical pairs the two are
identical, the extension adds 57 pairs at h=4/h=13.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from metrics import diebold_mariano, mae, mape

SCORED_MARKETS = [
    "Biu", "Damaturu", "Dandume", "Giwa", "Gombe", "Gujungu", "Gwandu, Dodoru",
    "Ibadan, Bodija", "Kano, Dawanau", "Kaura Namoda", "Lagos, Mile 12",
    "Maiduguri", "Mubi", "Potiskum", "Saminaka",
]
TRAILING_WINDOW = 13  # weeks
MIN_COMPLETE_YEARS = 3
MIN_Z_READINGS = 52  # across all markets, before the origin, to estimate z_scale


def build_food_cpi(inflation_path: Path) -> dict:
    """Monthly food CPI, compounded from month-on-month food inflation.
    Base value is arbitrary (100 at the first observed month); only ratios
    within and across years are ever used, so the base cancels out."""
    infl = pd.read_excel(inflation_path, sheet_name="Monthly Inflation")
    infl = infl.rename(columns={"Unnamed: 0": "date"}).sort_values("date").reset_index(drop=True)
    infl["date"] = pd.to_datetime(infl["date"])
    mom = infl["Food Inflation (%)-MoM"] / 100.0
    infl["cpi_food"] = 100.0 * (1 + mom).cumprod()
    infl["year"] = infl["date"].dt.year
    infl["month"] = infl["date"].dt.month
    return infl.set_index(["year", "month"])["cpi_food"].to_dict()


def build_seasonal_index(panel: pd.DataFrame, market: str, cpi_lookup: dict,
                         before_year: int | None = None) -> dict | None:
    """Per-market deflated seasonal index, D-52's method: monthly nominal
    price deflated by food CPI, ratio to that year's own real average,
    geometric mean across complete (12-month) years, normalised to average
    100 across the twelve months. With before_year set, only rows dated
    before 1 January of that year are used. Returns None if fewer than
    MIN_COMPLETE_YEARS complete years are available."""
    df = panel[panel["market"] == market].copy()
    df["date"] = pd.to_datetime(df["date"])
    if before_year is not None:
        df = df[df["date"] < pd.Timestamp(before_year, 1, 1)]
    df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month
    monthly = df.groupby(["year", "month"], as_index=False)["price"].mean().rename(columns={"price": "N"})
    if monthly.empty:
        return None
    monthly["CPI"] = monthly.apply(lambda r: cpi_lookup.get((r["year"], r["month"]), np.nan), axis=1)
    monthly["R"] = monthly["N"] / monthly["CPI"]
    complete_years = monthly.groupby("year")["month"].nunique()
    complete_years = complete_years[complete_years == 12].index.tolist()
    if len(complete_years) < MIN_COMPLETE_YEARS:
        return None
    m = monthly[monthly["year"].isin(complete_years)].copy()
    m["ratio"] = m["R"] / m.groupby("year")["R"].transform("mean")
    gm = m.groupby("month")["ratio"].apply(lambda x: np.exp(np.log(x).mean()))
    K = gm.mean()
    return (100 * gm / K).to_dict()


def seasonal_naive_forecast(origin_price: float, origin_month: int, target_month: int, si: dict) -> float:
    return origin_price * (si[target_month] / si[origin_month])


def compute_regime_z(panel: pd.DataFrame, market: str, window: int = TRAILING_WINDOW) -> pd.Series:
    """log(trailing `window`-week volatility / expanding median of that same
    market's own past volatility readings). Positive = more volatile than
    this market's own history to date; negative = calmer. Uses only price
    data at or before each date -- the expanding median at date i is built
    from readings strictly before i, so nothing here can see its own future,
    let alone the target the eventual forecast is being scored against."""
    s = panel[panel["market"] == market].sort_values("date").reset_index(drop=True)
    prices = s["price"].values
    pct_change = np.diff(prices) / prices[:-1]
    trailing_vol = np.full(len(prices), np.nan)
    for i in range(len(prices)):
        window_changes = pct_change[max(0, i - window):i]
        if len(window_changes) >= 4:
            trailing_vol[i] = np.std(window_changes)
    expanding_median = np.full(len(prices), np.nan)
    seen: list[float] = []
    for i in range(len(prices)):
        if len(seen) >= 8:
            expanding_median[i] = np.median(seen)
        if not np.isnan(trailing_vol[i]):
            seen.append(trailing_vol[i])
    valid = (trailing_vol > 0) & (expanding_median > 0)
    z = np.full(len(prices), np.nan)
    z[valid] = np.log(trailing_vol[valid] / expanding_median[valid])
    return pd.Series(z, index=pd.to_datetime(s["date"].values))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="an existing run's output dir (predictions_paired.csv)")
    ap.add_argument("--kind", default="GRU", choices=["RNN", "GRU"],
                    help="GRU is the production model (D-59); RNN only for re-scoring archived runs")
    ap.add_argument("--panel", default="data/panel_weekly.parquet")
    ap.add_argument("--inflation", default="data/external/inflation.xlsx")
    ap.add_argument("--window", type=int, default=TRAILING_WINDOW)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    run_dir, out = Path(a.run_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(a.panel)
    panel = panel[panel["market"].isin(SCORED_MARKETS)].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    cpi_lookup = build_food_cpi(Path(a.inflation))
    z_by_market = {mkt: compute_regime_z(panel, mkt, a.window) for mkt in SCORED_MARKETS}

    pred = pd.read_csv(run_dir / "predictions_paired.csv")
    pred["origin"] = pd.to_datetime(pred["origin"])
    pred["target_date"] = pred["origin"] + pd.to_timedelta(pred["h"] * 7, unit="D")
    pred["origin_month"], pred["target_month"] = pred["origin"].dt.month, pred["target_date"].dt.month

    # point-in-time seasonal index, one per (market, origin year)
    si_cache: dict[tuple[str, int], dict | None] = {}
    sn = []
    for mkt, o, om, tm, p0 in zip(pred["market"], pred["origin"], pred["origin_month"],
                                   pred["target_month"], pred["origin_price"]):
        key = (mkt, o.year)
        if key not in si_cache:
            si_cache[key] = build_seasonal_index(panel, mkt, cpi_lookup, before_year=o.year)
        si = si_cache[key]
        sn.append(np.nan if si is None else seasonal_naive_forecast(p0, om, tm, si))
    pred["seasonal_naive"] = sn

    # point-in-time z_scale: std of every market's z readings dated before the origin
    all_z = pd.concat([z.dropna() for z in z_by_market.values()]).sort_index()
    z_dates, z_vals = all_z.index.values, all_z.values
    scale_by_origin = {}
    for o in pred["origin"].unique():
        n = int(np.searchsorted(z_dates, np.datetime64(o), side="left"))
        scale_by_origin[o] = float(np.std(z_vals[:n])) if n >= MIN_Z_READINGS else np.nan
    pred["z_scale"] = pred["origin"].map(scale_by_origin)
    pred["z"] = [z_by_market[m].get(o, np.nan) for m, o in zip(pred["market"], pred["origin"])]

    pred["weight"] = sigmoid(pred["z"] / pred["z_scale"])
    no_si = pred["seasonal_naive"].isna()
    no_regime = pred["weight"].isna() & ~no_si
    pred.loc[no_si | no_regime, "weight"] = 1.0  # no index or unknown regime -> pure model
    pred["blended"] = np.where(no_si, pred[a.kind],
                               pred["weight"] * pred[a.kind] + (1 - pred["weight"]) * pred["seasonal_naive"])
    pred.to_csv(out / "predictions_with_blend.csv", index=False)

    rows = []
    for h in sorted(pred["h"].unique()):
        s = pred[pred.h == h].sort_values(["market", "origin"])
        has_sn = s["seasonal_naive"].notna()
        row = dict(h=int(h), n=len(s), n_seasonal_index=int(has_sn.sum()),
                   n_fallback_no_index=int((~has_sn).sum()),
                   n_fallback_no_regime=int(no_regime.loc[s.index].sum()),
                   model_MAE=mae(s["actual"], s[a.kind]), model_MAPE=mape(s["actual"], s[a.kind]),
                   naive_MAE=mae(s["actual"], s["naive"]), naive_MAPE=mape(s["actual"], s["naive"]),
                   seasonal_naive_MAE=mae(s.loc[has_sn, "actual"], s.loc[has_sn, "seasonal_naive"]),
                   seasonal_naive_MAPE=mape(s.loc[has_sn, "actual"], s.loc[has_sn, "seasonal_naive"]),
                   blend_MAE=mae(s["actual"], s["blended"]), blend_MAPE=mape(s["actual"], s["blended"]))
        g = s["market"].values
        for label, other in [("model", a.kind), ("naive", "naive")]:
            d = diebold_mariano(s["actual"], s["blended"], s[other], h, groups=g)
            row[f"dm_stat_vs_{label}"], row[f"dm_p_vs_{label}"] = d["dm_stat"], d["dm_p"]
        d = diebold_mariano(s.loc[has_sn, "actual"], s.loc[has_sn, "blended"], s.loc[has_sn, "seasonal_naive"],
                            h, groups=g[has_sn.values])
        row["dm_stat_vs_seasonal_naive"], row["dm_p_vs_seasonal_naive"] = d["dm_stat"], d["dm_p"]
        rows.append(row)
        print(f"h={h}: blend {row['blend_MAE']:.2f}/{row['blend_MAPE']:.2f}%  "
              f"model {row['model_MAE']:.2f}/{row['model_MAPE']:.2f}%  naive {row['naive_MAE']:.2f}/{row['naive_MAPE']:.2f}%  "
              f"dm_p vs model={row['dm_p_vs_model']:.4f} vs naive={row['dm_p_vs_naive']:.4f}  "
              f"fallback no index={row['n_fallback_no_index']} no regime={row['n_fallback_no_regime']}")

    pd.DataFrame(rows).to_csv(out / "blend_metrics.csv", index=False)
    (out / "run_metadata.json").write_text(json.dumps({
        "run_dir": str(run_dir), "kind": a.kind, "mode": "soft", "window": a.window,
        "point_in_time": True, "min_complete_years": MIN_COMPLETE_YEARS, "min_z_readings": MIN_Z_READINGS,
        "seasonal_index_unavailable": sorted(f"{m} {y}" for (m, y), si in si_cache.items() if si is None),
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
