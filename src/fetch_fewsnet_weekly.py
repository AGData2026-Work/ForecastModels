"""
FEWS NET weekly wholesale white (15 markets) or yellow (11 markets) maize quotes, from the public FEWS NET Data Warehouse API.

    python src/fetch_fewsnet_weekly.py --out data/external/<YYYYMMDD>_FEWSNET_API_Nigeria_WeeklyMaizeWhite_v1.csv
    python src/fetch_fewsnet_weekly.py --product yellow --out data/external/<YYYYMMDD>_FEWSNET_API_Nigeria_WeeklyMaizeYellow_v1.csv

The API returns weekly rows only per dataseries and only with schedule=Weekly
(marketpricefacts/?schedule=Weekly&dataseries=<id>); a country-wide query, with or without collection_schedule=Weekly,
returns the monthly aggregates only. Dataseries IDs from the analyst's v4 workbook (Data_Pipeline tab); each response
is checked against the expected market and product. Kept: every row with a value, converted to NGN/kg, with its
status and the API's created / status_changed timestamps (when the quote entered the warehouse), so publication lags can
be measured. Rows without a value are counted in the printout, not written.
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request

import pandas as pd

API = "https://fdw.fews.net/api/marketpricefacts/?format=json&schedule=Weekly&dataseries={}"
SERIES = {"Maiduguri": 15129, "Gombe": 15081, "Gujungu": 15093, "Giwa": 15069, "Saminaka": 15141, "Kano, Dawanau": 15045,
          "Dandume": 15033, "Gwandu, Dodoru": 15057, "Lagos, Mile 12": 15117, "Ibadan, Bodija": 15021,
          "Kaura Namoda": 15105, "Biu": 25467, "Damaturu": 25453, "Mubi": 25481, "Potiskum": 25439}
# yellow maize wholesale, the analyst's Data_Pipeline tab (11 markets; Biu, Damaturu, Mubi and Potiskum are white only)
SERIES_YELLOW = {"Maiduguri": 15130, "Gombe": 15082, "Gujungu": 15094, "Giwa": 15070, "Saminaka": 15142,
                 "Kano, Dawanau": 15046, "Dandume": 15034, "Gwandu, Dodoru": 15058, "Lagos, Mile 12": 15118,
                 "Ibadan, Bodija": 15022, "Kaura Namoda": 15106}
PRODUCT = {"white": ("Maize Grain (White)", SERIES), "yellow": ("Maize Grain (Yellow)", SERIES_YELLOW)}
KEEP = ["period_date", "start_date", "value", "unit", "status", "collection_status", "source_organization",
        "source_document", "created", "status_changed", "collection_status_changed"]


def kg_per_unit(unit: str) -> float:
    if unit == "kg":
        return 1.0
    hit = re.fullmatch(r"(\d+)_kg", unit or "")
    if not hit:
        raise ValueError(f"unexpected unit {unit!r}")
    return float(hit.group(1))


def fetch(market: str, sid: int, product: str = "Maize Grain (White)") -> tuple[pd.DataFrame, int]:
    req = urllib.request.Request(API.format(sid), headers={"User-Agent": "fewsnet-weekly"})
    with urllib.request.urlopen(req, timeout=120) as r:
        rows = json.load(r)
    rows = rows if isinstance(rows, list) else rows.get("results", [])
    bad = {(x.get("market"), x.get("product"), x.get("price_type"), x.get("collection_schedule")) for x in rows} - \
          {(market, product, "Wholesale", "Weekly")}
    if bad:
        raise ValueError(f"dataseries {sid} is not {market} {product} wholesale weekly: {bad}")
    d = pd.DataFrame(rows)[KEEP]
    n_empty = int(d.value.isna().sum())
    d = d[d.value.notna()].copy()
    d.insert(0, "market", market)
    d.insert(1, "dataseries", sid)
    d["price_ngn_per_kg"] = d.value / d.unit.map(kg_per_unit)
    return d, n_empty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--product", choices=["white", "yellow"], default="white")
    a = ap.parse_args()
    product, series = PRODUCT[a.product]
    out = []
    for market, sid in series.items():
        d, n_empty = fetch(market, sid, product)
        out.append(d)
        print(f"{market:16s} {sid}: {len(d):4d} weeks with a value ({n_empty} without), "
              f"{d.period_date.min()} to {d.period_date.max()}")
    pd.concat(out, ignore_index=True).sort_values(["market", "period_date"]).to_csv(a.out, index=False)
    print("->", a.out)


if __name__ == "__main__":
    main()
