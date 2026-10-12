"""
Monthly panel and grid on the gap-filled FEWS NET maize panel (owner request 2026-10-12: run the
monthly daily-FX variant on it).

    python src/build_monthly_gapfilled.py --filled <20261012_FEWSNET_Maize_Monthly_GapFilled_Panel_v2.xlsx>

Writes data/panel_monthly_gapfilled.parquet, data/monthly_grid_gapfilled.parquet,
data/panel_monthly_gapfilled.log.json.

Construction (May 2012 to Aug 2026, month-start dates, the monthly panel's layout):
- price: the gap-filled panel's value (FEWS NET API monthly where observed; AFEX / NAERLS old crop /
  Kaduna-Kano other source where filled; linear interpolation for FEWS NET gaps of 2 months or less).
  The config sets price_interp_limit_weeks: 0 so load_panel adds no fills of its own.
- target_price: the FEWS NET observed value only (NaN on every filled month), so no filled value is
  ever a training target; target_known_date = the next month, i.e. known by the end of the month
  after (FEWS NET's API values appear around the 15th of the next month).
- upstream_price: the upstream market's gap-filled price in the same month.
- diesel, rainfall, NDVI: monthly means of data/panel_weekly_oos2026_fewsdiesel.parquet, missing
  months carried forward (counted in the log). Not as-published (same as every monthly config).
- Fourier terms on month of year, as src/build_monthly_panel.py.
Grid: 15 scored markets, origins Sep 2015 to Jul 2026, horizons 1, 3, 6. actual = FEWS NET observed
value of the target month only (filled months are never scored); naive = the panel price at the
origin. A row is dropped when the origin has no price, and when any interpolated value inside the
6-month input window takes its right-hand anchor from after the origin (it would not exist yet).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

HORIZONS = [1, 3, 6]
LOOKBACK = 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filled", required=True)
    a = ap.parse_args()
    log: dict = {"source": a.filled}
    L = pd.read_excel(a.filled, "Long_panel")
    L["month"] = pd.PeriodIndex(L["month"], freq="M")
    months = pd.period_range("2012-05", "2026-08", freq="M")
    L = L[L["month"].isin(months)]
    price = L.set_index(["market", "month"])["price_NGN_per_kg"]
    flag = L.set_index(["market", "month"])["flag"]
    fill_kind = L.set_index(["market", "month"])["fill_kind"].fillna("")
    markets = sorted(L["market"].unique())

    w = pd.read_parquet("data/panel_weekly_oos2026_fewsdiesel.parquet")
    w["date"] = pd.to_datetime(w["date"])
    w["month"] = w["date"].dt.to_period("M")
    drv = w.groupby(["market", "month"]).agg(
        diesel=("diesel", "mean"), rainfall_simple_average=("rainfall_simple_average", "mean"),
        rainfall_inverse_distance=("rainfall_inverse_distance", "mean"),
        ndvi_simple_average=("ndvi_simple_average", "mean"), ndvi_inverse_distance=("ndvi_inverse_distance", "mean"),
        upstream_market=("upstream_market", "first"), upstream_lag_weeks=("upstream_lag_weeks", "first"),
        market_kind=("market_kind", "first"), state=("state", "first"), market_id=("market_id", "first"))
    P = drv.reindex(pd.MultiIndex.from_product([markets, months], names=["market", "month"])).reset_index()
    for col in ["upstream_market", "upstream_lag_weeks", "market_kind", "state", "market_id"]:
        P[col] = P.groupby("market")[col].transform(lambda s: s.ffill().bfill())
    carried = {}
    for col in ["diesel", "rainfall_simple_average", "rainfall_inverse_distance", "ndvi_simple_average", "ndvi_inverse_distance"]:
        before = P[col].isna()
        P[col] = P.groupby("market")[col].transform(lambda s: s.ffill())
        carried[col] = int((before & P[col].notna()).sum())
    log["driver_months_carried_forward"] = carried
    key = list(zip(P["market"], P["month"]))
    P["price"] = [price.get(k, np.nan) for k in key]
    P["price_flag"] = [flag.get(k, "missing") for k in key]
    P["is_proxy_price"] = P["price_flag"].str.startswith("filled")
    P["target_price"] = np.where(P["price_flag"] == "observed", P["price"], np.nan)
    P["target_from_price"] = False
    P["target_known_date"] = (P["month"] + 1).dt.to_timestamp()
    P["upstream_price"] = [price.get((u, mo), np.nan) if isinstance(u, str) else np.nan
                           for u, mo in zip(P["upstream_market"], P["month"])]
    for col, v in [("n_price_weeks", 0), ("n_weeks", 0), ("n_proxy_weeks", 0), ("share_interpolated_diesel", 1.0),
                   ("share_interpolated_rainfall", 1.0), ("share_interpolated_ndvi", 1.0)]:
        P[col] = v
    P["date"] = P["month"].dt.to_timestamp()
    moy = P["date"].dt.month
    for k in range(1, 5):
        P[f"sin{k}"] = np.sin(2 * np.pi * k * moy / 12)
        P[f"cos{k}"] = np.cos(2 * np.pi * k * moy / 12)
    P["time_idx"] = (P["month"] - months[0]).apply(lambda d: d.n)
    P["market_id"] = P["market_id"].astype("int8")
    log["price_cells_by_flag"] = P["price_flag"].value_counts().to_dict()
    log["target_cells_observed"] = int(P["target_price"].notna().sum())
    P.drop(columns=["month"]).to_parquet("data/panel_monthly_gapfilled.parquet", index=False)

    # right-hand anchor month of every interpolated cell, from its fill_kind text
    anchor = {}
    for (m, mo), fk in fill_kind.items():
        if flag.get((m, mo)) == "filled_interpolated":
            anchor[(m, mo)] = pd.Period(fk.split(" and ")[1][:7], freq="M")
    rows, drops = [], {"no_origin_price": 0, "interpolated_input_from_after_origin": 0}
    for m in [x for x in markets if x != "Aba"]:
        for o in pd.period_range("2015-09", "2026-07", freq="M"):
            p0 = price.get((m, o), np.nan)
            if not np.isfinite(p0):
                drops["no_origin_price"] += 1
                continue
            window = [o - k for k in range(LOOKBACK)]
            if any(anchor.get((m, x), o) > o for x in window):
                drops["interpolated_input_from_after_origin"] += 1
                continue
            for h in HORIZONS:
                t = o + h
                act = price.get((m, t), np.nan) if flag.get((m, t)) == "observed" else np.nan
                if np.isfinite(act):
                    rows.append(dict(market=m, horizon=h, origin=o.to_timestamp(), target=t.to_timestamp(),
                                     actual=act, naive=p0, panel_fe=np.nan, origin_flag=flag.get((m, o))))
    G = pd.DataFrame(rows)
    G.to_parquet("data/monthly_grid_gapfilled.parquet", index=False)
    log["grid"] = {"rows_by_horizon": G.groupby("horizon").size().to_dict(), "origins_dropped": drops,
                   "rows_with_filled_origin_by_horizon": G[G.origin_flag != "observed"].groupby("horizon").size().to_dict(),
                   "first_origin": str(G["origin"].min().date()), "last_origin": str(G["origin"].max().date())}
    json.dump(log, open("data/panel_monthly_gapfilled.log.json", "w"), indent=2, default=str)
    print(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main()
