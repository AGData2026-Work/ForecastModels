"""
Task 9, first cut: how much of the foreknowledge advantage is buying real
driver forecastability, versus how much is just perfect foresight.

    python src/driver_forecasts.py

Two things this script does, both read-only against the panel (no model
training):

1. Validates the diesel method choice (random walk with drift versus plain
   carry-forward) against realised diesel, on every (market, origin, horizon)
   in the evaluation grid, using only information available at the origin.
   Whichever wins is what src/data.py's `diesel_forecast` defaults to
   (DIESEL_FORECAST_METHOD, see docs/DECISIONS.md D-25).
2. Reports each driver forecast's own accuracy against its realised target-
   week value (not the multi-week window average used inside build_flat's
   flat features): climatology for rainfall/NDVI, the winning diesel method,
   seasonal-naive for upstream. All three are compared against plain
   carry-forward, since a driver forecast no better than carry-forward
   explains a null result in the trained-model comparison.

The four-row headline table (operational / forecast_drivers / conditional_exog
/ foreknowledge) needs trained runs under all four conventions and is
produced separately once those runs exist (see docs/DECISIONS.md D-25 for
the table once available; this script only measures the driver forecasts
themselves).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import (climatology_forecast, diesel_forecast, load_grid, load_panel,  # noqa: E402
                  upstream_forecast_seasonal_naive)

HORIZONS = [4, 13, 26]


def mae(a, p) -> float:
    a, p = np.asarray(a, float), np.asarray(p, float)
    ok = np.isfinite(a) & np.isfinite(p)
    return float(np.mean(np.abs(a[ok] - p[ok]))) if ok.any() else float("nan")


def main() -> None:
    panel = load_panel("data/panel_weekly.parquet")
    grid, glog = load_grid("data/07_panel_fe_forecasts.parquet", HORIZONS,
                           drop_targets=["2019-04-24"])

    rows = []
    for _, r in grid.iterrows():
        j = panel.midx.get(r["market"])
        i = panel.pos.get(pd.Timestamp(r["origin"]))
        if j is None or i is None:
            continue
        for h in HORIZONS:
            t = i + h
            if t >= panel.price.shape[0]:
                continue
            row = dict(market=r["market"], origin=r["origin"], h=h)

            row["diesel_realised"] = panel.diesel[t, j]
            row["diesel_rw_drift"] = diesel_forecast(panel.diesel[:, j], i, t, method="rw_drift")
            row["diesel_carry_forward"] = diesel_forecast(panel.diesel[:, j], i, t, method="carry_forward")

            if panel.has_upstream[j] and np.isfinite(panel.upstream[i, j]) and panel.upstream[i, j] > 0:
                row["upstream_realised"] = panel.upstream[t, j]
                row["upstream_seasonal_naive"] = upstream_forecast_seasonal_naive(panel.upstream[:, j], t)
                row["upstream_carry_forward"] = panel.upstream[i, j]

            row["rain_realised"] = panel.rainfall[t, j]
            row["rain_climatology"] = climatology_forecast(panel.rainfall[:, j], i, t)
            row["rain_carry_forward"] = panel.rainfall[i, j]

            row["ndvi_realised"] = panel.ndvi[t, j]
            row["ndvi_climatology"] = climatology_forecast(panel.ndvi[:, j], i, t)
            row["ndvi_carry_forward"] = panel.ndvi[i, j]
            rows.append(row)

    df = pd.DataFrame(rows)

    print("DIESEL: random walk with drift vs plain carry-forward, by horizon")
    print("(the source task asks this to be validated once, globally; whichever wins")
    print(" here is what src/data.py's DIESEL_FORECAST_METHOD is set to)")
    diesel_summary = []
    for h, g in df.groupby("h"):
        rw = mae(g["diesel_realised"], g["diesel_rw_drift"])
        cf = mae(g["diesel_realised"], g["diesel_carry_forward"])
        diesel_summary.append(dict(h=h, n=g["diesel_realised"].notna().sum(),
                                   rw_drift_MAE=rw, carry_forward_MAE=cf,
                                   rw_drift_wins=rw < cf))
    diesel_df = pd.DataFrame(diesel_summary)
    print(diesel_df.round(3).to_string(index=False))
    overall_rw = mae(df["diesel_realised"], df["diesel_rw_drift"])
    overall_cf = mae(df["diesel_realised"], df["diesel_carry_forward"])
    winner = "rw_drift" if overall_rw < overall_cf else "carry_forward"
    print(f"\nOverall (pooled across horizons): rw_drift MAE={overall_rw:.3f}, "
          f"carry_forward MAE={overall_cf:.3f} -> winner: {winner}")
    print("Confirm this matches DIESEL_FORECAST_METHOD in src/data.py; if not, that "
          "constant needs updating and every forecast_drivers run needs a rerun.")

    print("\nUPSTREAM: seasonal-naive vs carry-forward, by horizon (upstream markets only)")
    up = df.dropna(subset=["upstream_realised"])
    up_summary = []
    for h, g in up.groupby("h"):
        sn = mae(g["upstream_realised"], g["upstream_seasonal_naive"])
        cf = mae(g["upstream_realised"], g["upstream_carry_forward"])
        up_summary.append(dict(h=h, n=len(g), seasonal_naive_MAE=sn, carry_forward_MAE=cf,
                               seasonal_naive_wins=sn < cf))
    print(pd.DataFrame(up_summary).round(3).to_string(index=False))

    print("\nRAINFALL: climatology vs carry-forward, by horizon")
    rain_summary = []
    for h, g in df.groupby("h"):
        cl = mae(g["rain_realised"], g["rain_climatology"])
        cf = mae(g["rain_realised"], g["rain_carry_forward"])
        rain_summary.append(dict(h=h, n=len(g), climatology_MAE=cl, carry_forward_MAE=cf,
                                 climatology_wins=cl < cf))
    print(pd.DataFrame(rain_summary).round(3).to_string(index=False))

    print("\nNDVI: climatology vs carry-forward, by horizon")
    ndvi_summary = []
    for h, g in df.groupby("h"):
        cl = mae(g["ndvi_realised"], g["ndvi_climatology"])
        cf = mae(g["ndvi_realised"], g["ndvi_carry_forward"])
        ndvi_summary.append(dict(h=h, n=len(g), climatology_MAE=cl, carry_forward_MAE=cf,
                                 climatology_wins=cl < cf))
    print(pd.DataFrame(ndvi_summary).round(3).to_string(index=False))

    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/driver_forecast_accuracy_detail.csv", index=False)
    print("\nwrote outputs/driver_forecast_accuracy_detail.csv")


if __name__ == "__main__":
    main()
