"""
Monthly update of the analyst model's CSV inputs (data/analyst_inputs/, src/analyst_inputs.py) straight from the
sources, with no Excel workbook in the loop (2026-10-30).

    python src/analyst_inputs_update.py --months 2026-09 \
        --consolidated <data team consolidated price file .xlsx> \
        --weekly-white <src/fetch_fewsnet_weekly.py output> --weekly-yellow <same, --product yellow> \
        --fewsnet-monthly <FEWS NET monthly API pull, API columns: market, product, price_type, period_date, value, unit> \
        --cbn <src/fetch_cbn_fx.py output> --parity <src/build_import_parity.py output> \
        [--kasuwa <kasuwa_weekly.csv>] [--check [--report <csv>]]

Each month's rows, by file, and the rule (matched against the analyst's own history, Jan 2015 to Aug 2026):
  white_price_ngn_per_kg   consolidated file, sheet Data, period_type month, value            (2,006 of 2,006 equal)
  white_observed           1 - is_estimate from the same rows                                  (2,006 of 2,006)
  yellow_price_ngn_per_kg  FEWS NET monthly, Maize Grain (Yellow), Wholesale, NGN per kg       (1,306 of 1,349; his
                           history mixes this with the mean of weekly quotes, kept as he had it)
  sorghum_price_ngn_per_100kg  FEWS NET monthly, Sorghum (White), Wholesale, NGN per 100 kg    (1,681 of 1,703)
  weekly_quotes            FEWS NET weekly, white and yellow (API), week_end = period_date; Kasuwa quotes from
                           kasuwa_weekly.csv (or --kasuwa) only for a market-month with no FEWS NET white quote (his rule)
  macro_monthly            from manual_monthly.csv in the inputs folder (headline_mom, food_mom, cpi from the NBS CPI
                           report; fx_parallel = monthly mean of street BDC quotes), fx_official = CBN central rate
                           monthly mean, world_maize / logistics / duty = import parity CSV fob_usd_t / freight_usd_t /
                           duty (140 of 140 equal); diesel and petrol are not used by the model and stay blank
--check builds the rows and compares them with what the files already hold, writing nothing. Without it the months'
rows are replaced (or added) in place; every other month is untouched. Missing sources leave cells blank and are
reported, never filled.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analyst_inputs import MACRO, WIDE


def kg(unit: str) -> float:
    if unit == "kg":
        return 1.0
    m = re.fullmatch(r"(\d+)_kg", str(unit))
    if not m:
        raise ValueError(f"unexpected unit {unit!r}")
    return float(m.group(1))


def read_any(path: str, sheet: str | None = None) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet or 0) if str(path).endswith(".xlsx") else pd.read_csv(path)


def build(month: pd.Timestamp, a, cols: dict) -> dict:
    out = {}
    c = read_any(a.consolidated, "Data")
    c = c[(c.period_type == "month") & (pd.to_datetime(c.period_start) == month)].set_index("market_name")
    out["white"] = pd.Series({m: c.value.get(m, np.nan) for m in cols["white"]}, dtype=float)
    out["obs"] = pd.Series({m: (1.0 - float(c.is_estimate[m])) if m in c.index else np.nan for m in cols["white"]}, dtype=float)
    f = read_any(a.fewsnet_monthly, "all_rows_as_pulled")
    f = f[(pd.to_datetime(f.period_date).dt.to_period("M").dt.to_timestamp() == month) & (f.price_type == "Wholesale") & f.value.notna()]
    def fews(product, cols_, per):
        g = f[f["product"] == product]
        v = (g.value / g.unit.map(kg) * per).groupby(g.market).mean()
        return pd.Series({m: v.get(m, np.nan) for m in cols_}, dtype=float)
    out["yellow"] = fews("Maize Grain (Yellow)", cols["yellow"], 1.0)
    out["sorghum"] = fews("Sorghum (White)", cols["sorghum"], 100.0)
    wk = []
    for typ, path in (("White", a.weekly_white), ("Yellow", a.weekly_yellow)):
        w = pd.read_csv(path, parse_dates=["period_date"])
        w = w[w.period_date.dt.to_period("M").dt.to_timestamp() == month]
        wk.append(pd.DataFrame({"type": typ, "market": w.market, "week_end": w.period_date, "price_ngn_per_kg": w.price_ngn_per_kg,
                                "source": "FEWS NET"}))
    wk = pd.concat(wk, ignore_index=True)
    if a.kasuwa:
        k = pd.read_csv(a.kasuwa, parse_dates=["week_end"])
        k = k[k.week_end.dt.to_period("M").dt.to_timestamp() == month]
        have = set(wk[wk.type == "White"].market)
        k = k[~k.market.isin(have)]
        wk = pd.concat([wk, pd.DataFrame({"type": "White", "market": k.market, "week_end": k.week_end,
                                          "price_ngn_per_kg": k.price_ngn_per_kg, "source": k.source})], ignore_index=True)
    out["weekly"] = wk.sort_values(["type", "market", "week_end"]).reset_index(drop=True)
    man = pd.read_csv(Path(a.inputs) / "manual_monthly.csv", parse_dates=["month"]).set_index("month")
    cbn = pd.read_csv(a.cbn, parse_dates=["date"])
    cbn = cbn[cbn.date.dt.to_period("M").dt.to_timestamp() == month].centralrate.astype(float)
    par = pd.read_csv(a.parity); par["month"] = pd.to_datetime(par.month); par = par.set_index("month")
    row = {k: np.nan for k in MACRO}
    if month in man.index:
        for k in ("headline_mom", "food_mom", "cpi", "fx_parallel"):
            row[k] = float(man.at[month, k])
    row["fx_official"] = float(cbn.mean()) if len(cbn) else np.nan
    if month in par.index:
        row["world_maize"], row["logistics"], row["duty"] = (float(par.at[month, k]) for k in ("fob_usd_t", "freight_usd_t", "duty"))
    out["macro"] = pd.Series(row, dtype=float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", required=True)
    ap.add_argument("--inputs", default="data/analyst_inputs")
    for k in ("consolidated", "weekly_white", "weekly_yellow", "fewsnet_monthly", "cbn", "parity"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    ap.add_argument("--kasuwa", default=None, help="Kasuwa weekly quotes CSV (default: kasuwa_weekly.csv in the inputs folder)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", default=None, help="with --check: also write the comparison table to this CSV")
    a = ap.parse_args()
    d = Path(a.inputs)
    if a.kasuwa is None and (d / "kasuwa_weekly.csv").exists():
        a.kasuwa = str(d / "kasuwa_weekly.csv")
    tabs = {k: pd.read_csv(d / f, index_col="month", parse_dates=["month"]) for k, f in WIDE.items()}
    macro = pd.read_csv(d / "macro_monthly.csv", index_col="month", parse_dates=["month"])
    weekly = pd.read_csv(d / "weekly_quotes.csv", parse_dates=["week_end"])
    cols = {k: list(v.columns) for k, v in tabs.items()}
    cols["obs"] = cols["white"]
    report = []
    for ym in a.months:
        m = pd.Timestamp(ym + "-01")
        new = build(m, a, cols)
        for k in ("white", "obs", "yellow", "sorghum"):
            if a.check and m in tabs[k].index:
                old = tabs[k].loc[m].astype(float); n = new[k]
                both = old.notna() & n.notna()
                report.append(dict(month=ym, file=WIDE[k], cells_old=int(old.notna().sum()), cells_new=int(n.notna().sum()),
                                   equal=int(np.isclose(old[both], n[both], rtol=5e-3).sum())))
            tabs[k].loc[m] = new[k].reindex(tabs[k].columns).values
        if a.check and m in macro.index:
            old = macro.loc[m, MACRO].astype(float); n = new["macro"][MACRO]
            used = ["headline_mom", "cpi", "fx_parallel", "world_maize", "logistics", "duty", "fx_official"]
            report.append(dict(month=ym, file="macro_monthly.csv (columns the model uses)", cells_old=int(old[used].notna().sum()),
                               cells_new=int(n[used].notna().sum()),
                               equal=int(np.isclose(old[used], n[used], rtol=5e-3, equal_nan=False).sum())))
        macro.loc[m] = new["macro"].reindex(macro.columns).values
        in_m = weekly.week_end.dt.to_period("M").dt.to_timestamp() == m
        if a.check:
            o = weekly[in_m & (weekly.source == "FEWS NET")]; n = new["weekly"][new["weekly"].source == "FEWS NET"]
            j = o.merge(n, on=["type", "market", "week_end"], how="outer", suffixes=("_old", "_new"), indicator=True)
            both = j._merge == "both"
            report.append(dict(month=ym, file="weekly_quotes.csv (FEWS NET rows)", cells_old=int((j._merge != "right_only").sum()),
                               cells_new=int((j._merge != "left_only").sum()),
                               equal=int(np.isclose(j.price_ngn_per_kg_old[both], j.price_ngn_per_kg_new[both], rtol=5e-3).sum())))
        weekly = pd.concat([weekly[~in_m], new["weekly"]], ignore_index=True)
    if a.check:
        pd.set_option("display.width", 200)
        t = pd.DataFrame(report)
        print(t.to_string(index=False))
        print("\nTOTAL", t.groupby("file")[["cells_old", "cells_new", "equal"]].sum().to_string())
        if a.report:
            t.to_csv(a.report, index=False)
        return
    for k, f in WIDE.items():
        tabs[k].sort_index().rename_axis("month").to_csv(d / f, date_format="%Y-%m-%d")
    macro.sort_index().rename_axis("month").to_csv(d / "macro_monthly.csv", date_format="%Y-%m-%d")
    order = {"White": 0, "Yellow": 1}
    weekly.assign(_t=weekly.type.map(order)).sort_values(["_t", "market", "week_end"]).drop(columns="_t") \
        .to_csv(d / "weekly_quotes.csv", index=False, date_format="%Y-%m-%d")
    print(f"updated {', '.join(a.months)} in {d}")


if __name__ == "__main__":
    main()
