"""
Monthly panel and monthly evaluation grid for the monthly-cadence experiment (monthly-cadence experiment).

    python src/build_monthly_panel.py                    # month price = mean of its weeks
    python src/build_monthly_panel.py --price-agg last   # month price = its LAST observed week

Inputs: data/panel_weekly.parquet (the flagship's panel, 2012-05 to 2024-09-18) and
data/07_panel_fe_forecasts.parquet (the frozen weekly grid, used only for each market's
first and last origin month, so the monthly grid covers the same span per market).

Outputs:
  data/panel_monthly.parquet   one row per (market, month), month-start dates
  data/monthly_grid.parquet    the monthly evaluation grid, same columns as the weekly
                               baseline file (market, horizon, origin, target, actual,
                               naive, panel_fe), horizons in MONTHS (1, 3, 6), panel_fe NaN
  data/panel_monthly.log.json  every count behind the two files
With --price-agg last the three files carry a _last suffix (panel_monthly_last.parquet,
monthly_grid_last.parquet, panel_monthly_last.log.json); nothing else differs.

Conventions (why each exists):
- A month's price is the mean of that month's non-missing weekly prices, proxy-filled
  weeks included, because the flagship trains on the same weekly values. Owner decision
  2026-10-02: a single observed week is enough (MIN_WEEKS = 1). Nine months rest on one
  week, all in 2012-14 and outside the grid; `n_price_weeks` and `n_proxy_weeks` are kept
  per row so they can be found.
- --price-agg last (owner request 2026-10-02): a month's price is instead its latest
  non-missing weekly price, proxy-filled weeks included, so the origin price, the naive
  forecast and every target are a point at the end of the month rather than an average.
  If the month's final week is missing, the latest earlier week in that month is used;
  `price_week` holds the date of the week used. Only the maize price changes: diesel,
  upstream price, rainfall and NDVI stay monthly means.
- Diesel, upstream price, rainfall and NDVI are monthly means of the weekly values, so
  units match the weekly panel exactly; the share of interpolated weeks is kept.
- Fourier terms are recomputed on month of year (period 12), the monthly analogue of
  the weekly panel's week-of-year terms.
- An ORIGIN is the end of a completed calendar month o. The forecaster knows every week
  of month o, and "naive" is month o's average. Targets are the averages of months
  o+1, o+3, o+6. An origin is kept only if its own price and all three targets exist,
  the same completeness rule as the weekly grid (load_grid's default).
- Each market's origins run from the month of its first weekly grid origin to the month
  of its last, so both evaluations cover the same period per market.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_WEEKS = 1
HORIZONS = [1, 3, 6]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-agg", choices=["mean", "last"], default="mean")
    a = ap.parse_args()
    sfx = "" if a.price_agg == "mean" else "_last"
    w = pd.read_parquet("data/panel_weekly.parquet")
    w["date"] = pd.to_datetime(w["date"])
    w["month"] = w["date"].dt.to_period("M")
    log: dict = {"weekly_rows": int(len(w)), "min_weeks_per_month": MIN_WEEKS, "price_agg": a.price_agg}

    agg = w.groupby(["market", "month"]).agg(
        price=("price", a.price_agg),   # "last" skips NaN: the latest observed week
        n_price_weeks=("price", lambda s: int(s.notna().sum())),
        n_weeks=("date", "size"),
        n_proxy_weeks=("is_proxy_price", "sum"),
        diesel=("diesel", "mean"),
        share_interpolated_diesel=("is_interpolated_diesel", "mean"),
        upstream_price=("upstream_price", "mean"),
        upstream_market=("upstream_market", "first"),
        upstream_lag_weeks=("upstream_lag_weeks", "first"),
        market_kind=("market_kind", "first"),
        state=("state", "first"),
        rainfall_simple_average=("rainfall_simple_average", "mean"),
        rainfall_inverse_distance=("rainfall_inverse_distance", "mean"),
        ndvi_simple_average=("ndvi_simple_average", "mean"),
        ndvi_inverse_distance=("ndvi_inverse_distance", "mean"),
        share_interpolated_rainfall=("is_interpolated_rainfall", "mean"),
        share_interpolated_ndvi=("is_interpolated_ndvi", "mean"),
        market_id=("market_id", "first"),
    ).reset_index()
    if a.price_agg == "last":
        pw = (w[w["price"].notna()].groupby(["market", "month"])["date"].max()
              .rename("price_week").reset_index())
        agg = agg.merge(pw, on=["market", "month"], how="left")
        dom = agg["price_week"].dt.day
        last_week_of_month = (agg["price_week"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
                              - agg["price_week"]).dt.days <= 6
        log["last_value"] = {"months_with_price": int(agg["price_week"].notna().sum()),
                             "price_from_final_week_of_month": int(last_week_of_month.sum()),
                             "price_from_an_earlier_week": int((agg["price_week"].notna() & ~last_week_of_month).sum()),
                             "price_week_day_of_month_quartiles": [float(x) for x in dom.quantile([.25, .5, .75])]}
    thin = agg["price"].notna() & (agg["n_price_weeks"] < MIN_WEEKS)
    log["months_with_price_from_weekly"] = int(agg["price"].notna().sum())
    log["months_set_nan_below_min_weeks"] = int(thin.sum())
    agg.loc[thin, "price"] = np.nan
    log["months_with_no_price"] = int(agg["price"].isna().sum())

    agg["date"] = agg["month"].dt.to_timestamp()
    moy = agg["date"].dt.month
    for k in range(1, 5):
        agg[f"sin{k}"] = np.sin(2 * np.pi * k * moy / 12)
        agg[f"cos{k}"] = np.cos(2 * np.pi * k * moy / 12)
    months = pd.period_range(agg["month"].min(), agg["month"].max(), freq="M")
    full = (pd.MultiIndex.from_product([sorted(agg["market"].unique()), months], names=["market", "month"])
              .to_frame(index=False))
    missing_rows = len(full) - len(agg)
    agg = full.merge(agg, on=["market", "month"], how="left")
    agg["date"] = agg["month"].dt.to_timestamp()
    log["months_added_with_no_weekly_rows"] = int(missing_rows)
    agg["time_idx"] = (agg["month"] - agg["month"].min()).apply(lambda d: d.n)
    out = agg.drop(columns=["month"])
    out.to_parquet(f"data/panel_monthly{sfx}.parquet", index=False)
    log["panel_rows"] = int(len(out))
    log["panel_first_month"] = str(months[0])
    log["panel_last_month"] = str(months[-1])
    log["last_month_weeks"] = (agg[agg["month"] == months[-1]]
                               .set_index("market")["n_price_weeks"].dropna().astype(int).to_dict())

    # ---------------- grid ----------------
    wg = pd.read_parquet("data/07_panel_fe_forecasts.parquet")
    wg["origin"] = pd.to_datetime(wg["origin"])
    span = wg.groupby("market")["origin"].agg(["min", "max"])
    P = agg.pivot(index="month", columns="market", values="price")
    rows, dropped = [], {"no_origin_price": 0, "missing_target": 0}
    for m, (lo, hi) in span.iterrows():
        for o in pd.period_range(lo.to_period("M"), hi.to_period("M"), freq="M"):
            p0 = P.at[o, m] if o in P.index else np.nan
            tg = [P.at[o + h, m] if (o + h) in P.index else np.nan for h in HORIZONS]
            if not np.isfinite(p0):
                dropped["no_origin_price"] += 1
                continue
            if not np.isfinite(tg).all():
                dropped["missing_target"] += 1
                continue
            for h, a in zip(HORIZONS, tg):
                rows.append(dict(market=m, horizon=h, origin=o.to_timestamp(),
                                 target=(o + h).to_timestamp(), actual=float(a),
                                 naive=float(p0), panel_fe=np.nan))
    grid = pd.DataFrame(rows)
    grid.to_parquet(f"data/monthly_grid{sfx}.parquet", index=False)
    log["grid"] = {
        "horizons_months": HORIZONS,
        "pairs_per_horizon": int(len(grid) / len(HORIZONS)),
        "origins_dropped": dropped,
        "origins_per_market": grid[grid.horizon == 1].groupby("market").size().to_dict(),
        "first_origin": str(grid["origin"].min().date()),
        "last_origin": str(grid["origin"].max().date()),
        "last_target": str(grid["target"].max().date()),
    }
    Path(f"data/panel_monthly{sfx}.log.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
