"""
Monthly refresh of the GRU's weekly drivers: rainfall, NDVI and FEWS NET diesel (2026-11-06; until then rainfall and NDVI
had been carried forward from Jun and May 2026).

Step 1, sheets: put HDX's state-level dekadal values into the cleaned source sheets climate_assignment.py reads.

    python src/climate_refresh.py sheets --kind rainfall --hdx <date,PCODE,value csv> --sheet <rainfall_dekadal xlsx> \
        --replace-from 2026-01-01 --out <new xlsx>
    python src/climate_refresh.py sheets --kind ndvi --hdx <csv> --sheet <ndvi_dekadal xlsx> --replace-from 2026-05-21 --out <xlsx>

  Source: WFP on HDX, nga-rainfall-subnational (column rfh, 10-day rainfall, mm) and nga-ndvi-subnational (column vim),
  rows with adm_level 1; PCODE is the sheet's Location Code. HDX answers scripted downloads with HTTP 202 and no body, so
  the files are read in a browser. Rainfall: replace from the first month still held as CHIRPS preliminary (HDX column
  version), use final dekads only. NDVI: keep history (its smoothing revises recent dekads a little), append new dekads,
  including the first dekads after the panel end that the weekly interpolation needs.

Step 2, run climate_assignment.py (tools/climate/, from the original Maize-Forecasting project) on the new sheets:
    python tools/climate/src/climate_assignment.py --end <last weekly date of the panel>

Step 3, drivers: write the weekly drivers file the monthly panel reads (build_monthly_gapfilled.py --weekly-drivers).

    python src/climate_refresh.py drivers --weekly <current weekly drivers parquet> --climate <climate_weekly_by_market.parquet> \
        --climate-from 2026-01-01 --fewsnet-monthly <FEWS NET monthly rows (parquet or csv)> --diesel-month 2026-09 \
        --out <new weekly drivers parquet>

  Rainfall and NDVI columns are replaced from --climate-from on (earlier weeks must already agree; checked). Diesel for
  --diesel-month = FEWS NET monthly retail diesel x the market's splice ratio, read from the month before (weekly
  diesel / FEWS NET diesel, constant per market by construction in build_oos_2024_26.py); markets without FEWS NET
  diesel keep their carried-forward value. Writes <out> and <out>.log.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLIM = ["rainfall_simple_average", "rainfall_inverse_distance", "ndvi_simple_average", "ndvi_inverse_distance"]
VALUE = {"rainfall": "Rainfall (mm)", "ndvi": "NDVI"}


def sheets(a) -> None:
    old = pd.read_excel(a.sheet, "Cleaned Data")
    n = pd.read_csv(a.hdx, parse_dates=["date"])
    n = n[n.date >= a.replace_from]
    t = old.iloc[0]
    add = pd.DataFrame({"Data Sub Type": t["Data Sub Type"], "Date": n.date, "Year": n.date.dt.year, "Month": n.date.dt.month,
                        "Week": n.date.dt.isocalendar().week.astype(int), "Day": n.date.dt.day, "Location Code": n.PCODE,
                        "Country": "Nigeria", VALUE[a.kind]: n.value, "Unit": t["Unit"]})
    if set(add["Location Code"]) != set(old["Location Code"]):
        raise SystemExit("HDX state codes differ from the sheet's")
    out = pd.concat([old[old.Date < a.replace_from], add], ignore_index=True).sort_values(["Location Code", "Date"])
    with pd.ExcelWriter(a.out) as w:
        out.to_excel(w, sheet_name="Cleaned Data", index=False)
    print(f"{a.kind}: kept {int((old.Date < a.replace_from).sum())}, added {len(add)}, to {out.Date.max().date()} -> {a.out}")


def drivers(a) -> None:
    w = pd.read_parquet(a.weekly)
    w["date"] = pd.to_datetime(w.date)
    c = pd.read_parquet(a.climate)
    c["date"] = pd.to_datetime(c.date)
    m = w.merge(c[["market", "date"] + CLIM], on=["market", "date"], how="left", suffixes=("", "_new"))
    early = m.date < a.climate_from
    for col in CLIM:
        if not np.allclose(m.loc[early, col], m.loc[early, col + "_new"], equal_nan=True, atol=1e-6):
            raise SystemExit(f"{col}: the climate run disagrees with the weekly file before {a.climate_from}")
        m.loc[~early, col] = m.loc[~early, col + "_new"]
    m = m.drop(columns=[x + "_new" for x in CLIM])
    r = pd.read_parquet(a.fewsnet_monthly) if a.fewsnet_monthly.endswith(".parquet") else pd.read_csv(a.fewsnet_monthly)
    r = r[r["product"].str.contains("Diesel", na=False) & (r.collection_schedule == "Monthly")] if "collection_schedule" in r \
        else r[r["product"].str.contains("Diesel", na=False)]
    r["mo"] = r.period_date.astype(str).str[:7]
    api = r.groupby(["market", "mo"]).value.mean()
    prev = str(pd.Period(a.diesel_month, freq="M") - 1)
    m["mo"] = m.date.dt.strftime("%Y-%m")
    log = {}
    for mk in m.market.unique():
        a_prev, a_new = api.get((mk, prev)), api.get((mk, a.diesel_month))
        wk = m.loc[(m.market == mk) & (m.mo == prev) & ~m.is_interpolated_diesel.astype(bool), "diesel"]
        if a_prev is None or a_new is None or wk.empty:
            continue
        ratio = float(wk.iloc[0] / a_prev)
        sel = (m.market == mk) & (m.mo == a.diesel_month)
        m.loc[sel, "diesel"], m.loc[sel, "is_interpolated_diesel"] = a_new * ratio, False
        log[mk] = dict(fewsnet=float(a_new), ratio=round(ratio, 4), weekly=round(a_new * ratio, 1), weeks=int(sel.sum()))
    m = m.drop(columns="mo")
    if list(m.columns) != list(w.columns) or len(m) != len(w):
        raise SystemExit("output layout differs from the input weekly file")
    m.to_parquet(a.out, index=False)
    json.dump({"weekly": a.weekly, "climate": a.climate, "climate_from": a.climate_from, "diesel_month": a.diesel_month,
               "diesel": log}, open(str(a.out) + ".log.json", "w"), indent=1)
    print(f"climate replaced from {a.climate_from}; diesel {a.diesel_month} for {len(log)} markets -> {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheets")
    s.add_argument("--kind", choices=["rainfall", "ndvi"], required=True)
    s.add_argument("--hdx", required=True)
    s.add_argument("--sheet", required=True)
    s.add_argument("--replace-from", required=True)
    s.add_argument("--out", required=True)
    d = sub.add_parser("drivers")
    d.add_argument("--weekly", required=True)
    d.add_argument("--climate", required=True)
    d.add_argument("--climate-from", required=True)
    d.add_argument("--fewsnet-monthly", required=True)
    d.add_argument("--diesel-month", required=True)
    d.add_argument("--out", required=True)
    a = ap.parse_args()
    sheets(a) if a.cmd == "sheets" else drivers(a)


if __name__ == "__main__":
    main()
