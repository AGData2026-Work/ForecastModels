"""
Extract the weekly price series used for the relative-price features (D-60)
from the raw FEWS NET staple food price file.

    python src/build_relprice.py --raw ~/Downloads/FEWS_NET_Staple_Food_Price_Data.xlsx

Writes data/external/fews_relprice_weekly.parquet: one row per (date, market,
series) with the price per kg (per litre for fuel). Only the series named in
SERIES are kept. Units in the raw file are per 50 kg or 100 kg bag for most
wholesale series and per kg / per litre for retail; they are divided out here
so every value is per single unit. Rows with no value or a value <= 0 are
dropped and counted.

The white maize wholesale column in the same file matches data/panel_weekly
.parquet's `price` exactly on every overlapping (market, week), checked on
2026-09-30, so the two files share a date index and market names.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SERIES = {
    "rice": ("Rice (5% Broken)", "Wholesale"),
    "gasoline": ("Gasoline", "Retail"),
}
UNIT_DIVISOR = {"kg": 1, "L": 1, "50_kg": 50, "100_kg": 100}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data/external/fews_relprice_weekly.parquet")
    a = ap.parse_args()

    d = pd.read_excel(Path(a.raw).expanduser(), sheet_name="data", parse_dates=["period_date"])
    log = {"raw_path": str(a.raw), "raw_rows": int(len(d)), "series": {}}
    frames = []
    for name, (product, price_type) in SERIES.items():
        g = d[(d["product"] == product) & (d["price_type"] == price_type)].copy()
        unknown = sorted(set(g["unit"]) - set(UNIT_DIVISOR))
        if unknown:
            raise SystemExit(f"{product} {price_type}: unknown unit(s) {unknown}")
        n_rows = len(g)
        g = g[g["value"].notna() & (g["value"] > 0)]
        g["price"] = g["value"] / g["unit"].map(UNIT_DIVISOR)
        g = (g.groupby(["period_date", "market"], as_index=False)["price"].mean()
               .rename(columns={"period_date": "date"}))
        g["series"] = name
        frames.append(g)
        log["series"][name] = {"product": product, "price_type": price_type,
                               "rows_in_raw": int(n_rows), "rows_kept": int(len(g)),
                               "dropped_missing_or_nonpositive": int(n_rows - len(g)),
                               "markets": int(g["market"].nunique()),
                               "first": str(g["date"].min().date()), "last": str(g["date"].max().date())}
    out = pd.concat(frames, ignore_index=True)[["date", "market", "series", "price"]]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    Path(a.out).with_suffix(".log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))
    print(f"wrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
