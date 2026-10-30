"""
The analyst's model inputs as plain CSV files, so the Inflation Adjusted model runs without his Excel workbook
(2026-10-30).

    python src/analyst_inputs.py export --workbook data/external/analyst_v4/20261021_Analyst_Maize_Forecast_v4_White.xlsx \
        --out data/analyst_inputs
    python src/analyst_inputs.py check --workbook <same workbook> --inputs data/analyst_inputs

His code (vendor/analyst_v4/v2_models.py, unchanged) reads one bundle of tables, D_OVERRIDE, which his
refit_feature_model_v4.py builds from seven workbook tabs. `export` writes that bundle to the CSVs below, once;
`load_inputs` rebuilds the identical bundle from them; `check` proves the two give identical forecasts. After the
export the workbook is never needed again: each month src/analyst_inputs_update.py appends the new month to these
files from the sources.

Files (all months are month-start dates, YYYY-MM-01; markets are column headers, quoted where they contain commas):
  white_price_ngn_per_kg.csv      month x 15 markets, white maize wholesale (his In_Price)
  white_observed.csv              month x 15 markets, 1 = FEWS NET observed, 0 = filled, blank = no price (In_Price_Observed)
  yellow_price_ngn_per_kg.csv     month x 11 markets, yellow maize wholesale (In_Price_Yellow; training only)
  sorghum_price_ngn_per_100kg.csv month x 15 markets, white sorghum wholesale, NGN per 100 kg (In_Sorghum)
  macro_monthly.csv               month, headline_mom, food_mom, cpi, fx_parallel, fx_official, world_maize, diesel,
                                  petrol, logistics, duty (In_Macro columns B-K; food_mom, diesel, petrol unused)
  weekly_quotes.csv               type (White / Yellow), market, week_end, price_ngn_per_kg, source (In_Weekly_Prices)
  zones.csv, zone_bands.csv       market -> zone; zone -> allowed error band (In_Zone_Bands; scoring only)
  kasuwa_weekly.csv               market, week_end, price_ngn_per_kg, source: Kasuwa white maize quotes, used only for a
                                  market-month with no FEWS NET white quote (his rule); new quotes are appended here
  manual_monthly.csv              month, headline_mom, food_mom, cpi (NBS CPI report), fx_parallel (street BDC quotes):
                                  the only values a person enters each month; src/analyst_inputs_update.py copies them in
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "analyst_v4"
MACRO = ["headline_mom", "food_mom", "cpi", "fx_parallel", "fx_official", "world_maize", "diesel", "petrol", "logistics", "duty"]
WIDE = {"white": "white_price_ngn_per_kg.csv", "obs": "white_observed.csv", "yellow": "yellow_price_ngn_per_kg.csv",
        "sorghum": "sorghum_price_ngn_per_100kg.csv"}


def inputs_from_workbook(workbook: str) -> dict:
    """His own input-reading code (refit_feature_model_v4.py up to D_OVERRIDE), run on the workbook."""
    src = (VENDOR / "refit_feature_model_v4.py").read_text()
    src = src.split("src = open(os.path.join(here, 'v2_models.py'))")[0]
    src = src.replace("path = sys.argv[1]", f"path = {str(workbook)!r}")
    src = src.replace("here = os.path.dirname(os.path.abspath(__file__))", f"here = {str(VENDOR)!r}")
    g = {"__name__": "analyst_inputs"}
    exec(compile(src, "refit_feature_model_v4.py", "exec"), g)
    g["wb"].close()
    D = g["D_OVERRIDE"]
    D["_obs"] = g["obs"]                                   # the observed flags as read, before he joins them with yellow
    D["_weekly_source"] = None
    return D


def export(D: dict, out: Path, workbook: str | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    wide = {"white": D["white"], "obs": D["_obs"], "yellow": D["yellow"], "sorghum": D["sorghum"]}
    for k, df in wide.items():
        df.rename_axis("month").to_csv(out / WIDE[k], date_format="%Y-%m-%d")
    D["macro"].rename_axis("month").to_csv(out / "macro_monthly.csv", date_format="%Y-%m-%d")
    w = D["weekly"].rename(columns={"end": "week_end", "price_kg": "price_ngn_per_kg"})
    if workbook:                                           # keep the source label (FEWS NET / Kasuwa ...) for traceability
        import openpyxl
        wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
        src = [r[5] for r in wb["In_Weekly_Prices"].iter_rows(min_row=5, values_only=True) if r[0] is not None]
        wb.close()
        w["source"] = src
    w.to_csv(out / "weekly_quotes.csv", index=False, date_format="%Y-%m-%d")
    if "source" in w:                                      # Kasuwa quotes he used where FEWS NET had none: kept as their own input
        k = w[w.source.astype(str).str.startswith("Kasuwa")]
        k[["market", "week_end", "price_ngn_per_kg", "source"]].to_csv(out / "kasuwa_weekly.csv", index=False, date_format="%Y-%m-%d")
    # the four monthly values a person enters (NBS CPI report; street BDC quotes), seeded from his history
    D["macro"][["headline_mom", "food_mom", "cpi", "fx_parallel"]].rename_axis("month").to_csv(out / "manual_monthly.csv", date_format="%Y-%m-%d")
    pd.Series(D["ZONE"], name="zone").rename_axis("market").to_csv(out / "zones.csv")
    pd.Series(D["bands"], name="allowed_band").rename_axis("zone").to_csv(out / "zone_bands.csv")


def load_inputs(inputs: str | Path) -> dict:
    """The same D_OVERRIDE bundle his refit script builds, from the CSV files."""
    d = Path(inputs)
    rd = lambda f: pd.read_csv(d / f, index_col="month", parse_dates=["month"]).astype(float)
    white, obs, yellow, sorg = (rd(WIDE[k]) for k in ("white", "obs", "yellow", "sorghum"))
    for df in (white, obs, yellow, sorg):
        df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp()
        df.index.name = None
    MK, YMK = list(white.columns), list(yellow.columns)
    SER = [("White", m) for m in MK] + [("Yellow", m) for m in YMK]
    flags = pd.concat([obs.reindex(columns=MK), yellow.notna().astype(float).where(yellow.notna())], axis=1)
    flags.columns = pd.MultiIndex.from_tuples(SER)
    mac = rd("macro_monthly.csv")[MACRO]
    mac.index = pd.DatetimeIndex(mac.index).to_period("M").to_timestamp(); mac.index.name = None
    w = pd.read_csv(d / "weekly_quotes.csv", parse_dates=["week_end"])
    weekly = pd.DataFrame({"type": w.type, "market": w.market, "end": w.week_end, "price_kg": w.price_ngn_per_kg.astype(float)})
    ZONE = pd.read_csv(d / "zones.csv").set_index("market").zone.to_dict()
    bands = pd.read_csv(d / "zone_bands.csv").set_index("zone").allowed_band.to_dict()
    return dict(SER=SER, MK=MK, TYPES=["White", "Yellow"], white=white, yellow=yellow, MONTHS=white.index, macro=mac,
                weekly=weekly, sorghum=sorg.reindex(white.index), bands=bands, ZONE=ZONE, flags=flags)


def run_model(D: dict, as_published: bool = True) -> dict:
    """Run his v2_models.py (unchanged) on a D_OVERRIDE bundle; return its globals. as_published lags his headline
    inflation one month (NBS publishes month o around the 15th-16th of o+1), exactly as src/analyst_forecast.py."""
    D = {k: v for k, v in D.items() if not k.startswith("_")}
    if as_published:
        mac = D["macro"].copy(); mac["headline_mom"] = mac["headline_mom"].shift(1); D["macro"] = mac
    src = (VENDOR / "v2_models.py").read_text().split("pd.to_pickle(dict(panel=pn")[0]
    src = src.replace("sys.path.insert(0, '/home/claude'); os.chdir('/home/claude'); ", "")
    g = {"D_OVERRIDE": D, "__name__": "analyst_inputs"}
    cwd = os.getcwd()
    try:
        exec(compile(src, str(VENDOR / "v2_models.py"), "exec"), g)
    finally:
        os.chdir(cwd)
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "check"])
    ap.add_argument("--workbook", required=True)
    ap.add_argument("--out", default="data/analyst_inputs")
    ap.add_argument("--inputs", default="data/analyst_inputs")
    a = ap.parse_args()
    if a.mode == "export":
        export(inputs_from_workbook(a.workbook), Path(a.out), a.workbook)
        print("wrote", sorted(p.name for p in Path(a.out).iterdir()))
        return
    worst = 0.0
    for pub in (True, False):
        A = run_model(inputs_from_workbook(a.workbook), pub)["pn"]
        B = run_model(load_inputs(a.inputs), pub)["pn"]
        for h in (1, 3, 6):
            for c in (f"FM_{h}", f"BL_{h}", f"RS_{h}", f"IW_{h}"):
                x, y = A[c].to_numpy(float), B[c].to_numpy(float)
                assert np.array_equal(np.isnan(x), np.isnan(y)), f"{c}: missing values differ"
                m = ~np.isnan(x)
                worst = max(worst, float(np.max(np.abs(x[m] / y[m] - 1))) if m.any() else 0.0)
    print(f"forecasts from CSV inputs vs workbook, as published and as in workbook: max relative difference {worst:.1e}")


if __name__ == "__main__":
    main()
