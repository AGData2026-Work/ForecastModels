"""
Rescore the Sep/Oct 2026 proxy check (src/proxy_sepoct_check.py) against two other "actuals" (owner request, 2026-11-04).

    python src/proxy_rescore.py --out outputs/proxy_20261104

1. Panel v9 (20261104_FEWSNET_Maize_Monthly_GapFilled_Panel_v9_HarvestRatio.xlsx; src/build_gapfilled_panel.py
   --harvest-ratio): AFEX fills in Sep and Oct levelled by the Sep-Oct-only prior overlap ratio. The GRU forecast from the
   Sep origin is rerun on v9 (outputs/proxy_20261104/gru_serving_origin_202609_v9harvest), since its starting prices moved.
   Every other forecast was made before Sep and does not change.
2. FEWS NET weekly (data/external/20261104_FEWSNET_API_Nigeria_WeeklyMaizeWhite_v1.csv, src/fetch_fewsnet_weekly.py on
   4 Nov 2026): the monthly mean of the weekly white quotes for Sep 2026, out for 8 markets before the monthly release.
   From 2015 to Jun 2026 the weekly mean equals the published monthly value (1,791 market-months, median gap 0.0%); in
   Jul and Aug 2026 it differed by up to 2.6% (Dandume Aug 6.4%), so it is a close but provisional stand-in for the
   monthly figure. FEWS NET's own numbers, not a proxy. No weekly quote for Oct yet, and none for Kano in Sep.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

V9 = "/Users/augmentumadvisory/Downloads/20261104_FEWSNET_Maize_Monthly_GapFilled_Panel_v9_HarvestRatio.xlsx"
WEEKLY = "data/external/20261104_FEWSNET_API_Nigeria_WeeklyMaizeWhite_v1.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/proxy_20261104")
    a = ap.parse_args()
    out = Path(a.out)
    from proxy_sepoct_check import GRADE, V8, forecasts, ranges
    d = forecasts()   # every market with a forecast, so markets FEWS NET reports but no proxy covers (Ibadan, Maiduguri) count
    d["grade"] = d.market.map(GRADE)
    d["headline"] = np.where(d.joint.notna(), d.joint, np.where(d.gru.notna(), d.gru, d.analyst))
    v8 = pd.read_excel(V8, "Long_panel")
    v8 = v8[v8.month.isin(["2026-09", "2026-10"])][["market", "month", "price_NGN_per_kg", "source", "source_market"]]
    v8["target"] = pd.to_datetime(v8.month + "-01")
    d = d.merge(v8.drop(columns="month").rename(columns={"price_NGN_per_kg": "actual_v8", "source": "source_v8",
                                                        "source_market": "source_market_v8"}), on=["market", "target"], how="left")
    # set B on v9: new starting prices, new GRU forecasts, ranges by the same production method
    b = pd.read_csv(out / "gru_serving_origin_202609_v9harvest/predictions/2026-11-04.csv", parse_dates=["origin"]).query("h == 1")
    b["lo80"], b["hi80"] = ranges("gru", 1, pd.Timestamp("2026-09-01"), b.forecast_price)
    nb = "proxy origin Sep -> Oct (1 month, GRU only)"
    bv9 = pd.DataFrame({"set": nb + " [v9 rerun]", "market": b.market, "origin": b.origin, "h": 1,
                        "target": pd.Timestamp("2026-10-01"), "origin_price": b.origin_price, "gru": b.forecast_price,
                        "analyst": np.nan, "joint": np.nan, "lo80": b.lo80, "hi80": b.hi80, "headline": b.forecast_price})
    bv9["grade"] = bv9.market.map(GRADE)
    d = pd.concat([d, bv9], ignore_index=True)
    v9 = pd.read_excel(V9, "Long_panel")
    v9 = v9[v9.month.isin(["2026-09", "2026-10"])][["market", "month", "price_NGN_per_kg", "source", "source_market"]]
    v9["target"] = pd.to_datetime(v9.month + "-01")
    d = d.merge(v9.drop(columns="month").rename(columns={"price_NGN_per_kg": "actual_v9", "source": "source_v9",
                                                        "source_market": "source_market_v9"}), on=["market", "target"], how="left")
    w = pd.read_csv(WEEKLY, parse_dates=["period_date"])
    w["target"] = w.period_date.dt.to_period("M").dt.to_timestamp()
    wm = w.groupby(["market", "target"]).price_ngn_per_kg.agg(actual_fews_weekly="mean", fews_weeks="size").reset_index()
    d = d.merge(wm, on=["market", "target"], how="left")
    for k in ("v8", "v9", "fews_weekly"):
        y = d[f"actual_{k}"]
        d[f"err_pct_{k}"] = 100 * (d.headline / y - 1)
        d[f"err_naive_pct_{k}"] = 100 * (d.origin_price / y - 1)
        d[f"inside80_{k}"] = np.where(y.notna(), (y >= d.lo80) & (y <= d.hi80), np.nan)
    d.round(2).to_csv(out / "rescore_rows.csv", index=False)
    rows = []
    for (s, k), g in [((s, k), g) for s, g in d.groupby("set", sort=False) for k in ("v8", "v9", "fews_weekly")]:
        g = g[g[f"actual_{k}"].notna() & g.headline.notna()]
        if s == nb and k == "v9":   # set B's v8-panel forecast scored on v9 actuals mixes two panels; report the v9 rerun instead
            continue
        if not len(g):
            continue
        rows.append({"set": s, "actual": k, "n": len(g), "markets": ", ".join(g.market), "our_mape": g[f"err_pct_{k}"].abs().mean(),
                     "naive_mape": g[f"err_naive_pct_{k}"].abs().mean(), "bias_pct": g[f"err_pct_{k}"].mean(),
                     "inside80": f"{int(g[f'inside80_{k}'].sum())} of {len(g)}"})
    S = pd.DataFrame(rows)
    S.round(1).to_csv(out / "rescore_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print(S.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
