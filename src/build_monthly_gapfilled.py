"""
Monthly panel and grid on the gap-filled FEWS NET maize panel (owner request 2026-10-12: run the
monthly daily-FX variant on it).

    python src/build_monthly_gapfilled.py --filled <20261012_FEWSNET_Maize_Monthly_GapFilled_Panel_v2.xlsx>

Writes data/panel_monthly_gapfilled.parquet, data/monthly_grid_gapfilled.parquet,
data/panel_monthly_gapfilled.log.json (with --published: the same names ending _pit; see that flag).

Construction (May 2012 to Aug 2026, month-start dates, the monthly panel's layout):
- price: the gap-filled panel's value (FEWS NET API monthly where observed; AFEX / NAERLS old crop /
  Kaduna-Kano other source where filled; linear interpolation for FEWS NET gaps of 2 months or less).
  The config sets price_interp_limit_weeks: 0 so load_panel adds no fills of its own.
- target_price: the FEWS NET observed value only (NaN on every filled month), so no filled value is
  ever a training target; target_known_date = the next month, i.e. known by the end of the month
  after (FEWS NET's API values appear around the 15th of the next month).
- upstream_price: the upstream market's gap-filled price in the SAME month, no lag (production; both prices are
  released together). Checks only: --upstream-lag N (N months earlier), --no-upstream (blank).
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


DRIVERS = ["diesel", "rainfall_simple_average", "rainfall_inverse_distance", "ndvi_simple_average", "ndvi_inverse_distance"]


def build_panel(L: pd.DataFrame, w: pd.DataFrame, published: bool, observed_only: bool = False,
                driver_lag: int | None = None, diesel: pd.DataFrame | None = None, upstream_lag: int = 0,
                no_upstream: bool = False, last_month: str = "2026-08"):
    """Monthly model panel from the gap-filled long panel `L` and the weekly driver panel `w`.
    Returns (panel, log, price, flag, fill_kind, markets)."""
    log: dict = {"published": published, "observed_only": observed_only, "upstream_lag_months": upstream_lag,
                 "no_upstream": no_upstream}
    L = L.copy()
    L["month"] = pd.PeriodIndex(L["month"], freq="M")
    months = pd.period_range("2012-05", last_month, freq="M")
    L = L[L["month"].isin(months)]
    if observed_only:  # D-75 sensitivity: FEWS NET observed values only, every fill removed
        L.loc[L["flag"] != "observed", ["price_NGN_per_kg"]] = np.nan
        L.loc[L["flag"] != "observed", "flag"] = "missing"
    price = L.set_index(["market", "month"])["price_NGN_per_kg"]
    flag = L.set_index(["market", "month"])["flag"]
    fill_kind = L.set_index(["market", "month"])["fill_kind"].fillna("")
    markets = sorted(L["market"].unique())
    w = w.copy()
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
    if diesel is not None:
        # FEWS NET-only diesel (owner request 2026-10-13): one source for the whole history instead of
        # NBS state diesel back-extended before Jun 2015 and spliced to FEWS NET after Oct 2024. `diesel` is
        # month x market, NGN/litre. Months before a market's first FEWS NET value stay missing.
        dz = diesel.copy(); dz.index = pd.PeriodIndex(dz.index, freq="M")
        P["diesel"] = [dz[m].get(mo, np.nan) if m in dz.columns else np.nan for m, mo in zip(P["market"], P["month"])]
        log["diesel_source"] = "FEWS NET API retail diesel"
        log["diesel_months_before_first_value"] = int(P.groupby("market")["diesel"].transform(lambda s: s.ffill().isna()).sum())
    carried = {}
    for col in DRIVERS:
        before = P[col].isna()
        P[col] = P.groupby("market")[col].transform(lambda s: s.ffill())
        carried[col] = int((before & P[col].notna()).sum())
    log["driver_months_carried_forward"] = carried
    if published:
        # Diesel for month M appears on the 20th of M+1 (D-67), after the 15th-of-next-month forecast
        # date; a month's mean rainfall/NDVI needs dekads to its end plus the release lag (D-67 uses 4
        # weeks). At the forecast for month o, both are known through month o-1.
        lag = 1 if driver_lag is None else driver_lag
        for col in DRIVERS:
            P[col] = P.groupby("market")[col].shift(lag)
        log["drivers_lagged_months"] = lag
    key = list(zip(P["market"], P["month"]))
    P["price"] = [price.get(k, np.nan) for k in key]
    P["price_flag"] = [flag.get(k, "missing") for k in key]
    P["is_proxy_price"] = P["price_flag"].str.startswith("filled")
    P["target_price"] = np.where(P["price_flag"] == "observed", P["price"], np.nan)
    P["target_from_price"] = False
    # month-end design: month t is known by the end of t+1. As-published design: the origin index of
    # month o already stands for the 15th of o+1, so month t is known from its own index.
    P["target_known_date"] = (P["month"] + (0 if published else 1)).dt.to_timestamp()
    # UPSTREAM TIMING (production, owner-confirmed 2026-10-14): each market gets its upstream market's price for
    # the SAME month (upstream_lag = 0). Both are FEWS NET monthly values released together around the 15th of
    # the next month, so the same-month value is public on the forecast date. The weekly panel's
    # upstream_lag_weeks (1 to 4 weeks) is carried but not read here. A 1-month lag was worse at every horizon
    # (docs/DECISIONS.md, D-77 follow-up of 2026-10-14); --upstream-lag N exists only for that check.
    P["upstream_price"] = [price.get((u, mo - upstream_lag), np.nan) if isinstance(u, str) else np.nan
                           for u, mo in zip(P["upstream_market"], P["month"])]
    if no_upstream:  # D-77 check: no upstream information at all (has_upstream stays, every value is NaN)
        P["upstream_price"] = np.nan
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
    return P.drop(columns=["month"]), log, price, flag, fill_kind, markets


def build_grid(price, flag, fill_kind, markets, horizons=None):
    horizons = horizons or HORIZONS
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
            for h in horizons:
                t = o + h
                act = price.get((m, t), np.nan) if flag.get((m, t)) == "observed" else np.nan
                if np.isfinite(act):
                    rows.append(dict(market=m, horizon=h, origin=o.to_timestamp(), target=t.to_timestamp(),
                                     actual=act, naive=p0, panel_fe=np.nan, origin_flag=flag.get((m, o))))
    return pd.DataFrame(rows), drops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filled", required=True)
    ap.add_argument("--published", action="store_true",
                    help="as-published variant: the forecast for month o is made on the 15th of o+1 "
                         "(after FEWS NET's release of month o); diesel, rainfall and NDVI lag one month; "
                         "a month's price is a usable target from its own origin index")
    ap.add_argument("--observed-only", action="store_true",
                    help="sensitivity: drop every fill, FEWS NET observed prices only")
    ap.add_argument("--driver-lag", type=int, default=None, help="months diesel/rainfall/NDVI lag (published default 1)")
    ap.add_argument("--horizons", type=int, nargs="+", default=None, help="grid horizons (default 1 3 6)")
    ap.add_argument("--suffix", default=None, help="override the output file suffix")
    ap.add_argument("--upstream-lag", type=int, default=0, help="months the upstream market's price lags (default 0)")
    ap.add_argument("--no-upstream", action="store_true",
                    help="check only: blank every upstream price, so the upstream channel and its mask are zero")
    ap.add_argument("--last-month", default="2026-08",
                    help="last panel month (default 2026-08, production); later months need a gap-filled panel that has them")
    ap.add_argument("--fewsnet-diesel", default=None,
                    help="FEWS NET API export (.xlsx, sheet diesel_retail_litre): use it as the only diesel source")
    a = ap.parse_args()
    sfx = a.suffix if a.suffix is not None else ("_pit" if a.published else "") + ("_obsonly" if a.observed_only else "")
    P, log, price, flag, fill_kind, markets = build_panel(
        pd.read_excel(a.filled, "Long_panel"), pd.read_parquet("data/panel_weekly_oos2026_fewsdiesel.parquet"),
        a.published, a.observed_only, a.driver_lag,
        pd.read_excel(a.fewsnet_diesel, "diesel_retail_litre").set_index("month") if a.fewsnet_diesel else None,
        a.upstream_lag, a.no_upstream, a.last_month)
    log["source"] = a.filled
    P.to_parquet(f"data/panel_monthly_gapfilled{sfx}.parquet", index=False)
    G, drops = build_grid(price, flag, fill_kind, markets, a.horizons)
    G.to_parquet(f"data/monthly_grid_gapfilled{sfx}.parquet", index=False)
    log["grid"] = {"rows_by_horizon": G.groupby("horizon").size().to_dict(), "origins_dropped": drops,
                   "rows_with_filled_origin_by_horizon": G[G.origin_flag != "observed"].groupby("horizon").size().to_dict(),
                   "first_origin": str(G["origin"].min().date()), "last_origin": str(G["origin"].max().date())}
    json.dump(log, open(f"data/panel_monthly_gapfilled{sfx}.log.json", "w"), indent=2, default=str)
    print(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main()
