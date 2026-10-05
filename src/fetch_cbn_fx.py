"""
Rebuild the CBN daily US dollar rate file the production config reads (D-66, D-68).

    python src/fetch_cbn_fx.py --out data/external/YYYYMMDD_CBN_API_Nigeria_USDNGN_Daily_<first>_<last>_v1.csv
    python src/fetch_cbn_fx.py --json saved_GetAllExchangeRates.json --out ...   # from a saved download

Source: https://www.cbn.gov.ng/api/GetAllExchangeRates, the public endpoint behind the CBN's
exchange-rate page (every currency, every business day since Dec 2001). Kept: currency
"US DOLLAR" (the API also has one "US DOLLAR " with a trailing space), one row per date (the
first, by API order, when a date repeats), columns date, buyingrate, centralrate, sellingrate.
Values are left exactly as published, typos included; the feature code smooths them
(src/data.py _fx_daily_features).
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import pandas as pd

URL = "https://www.cbn.gov.ng/api/GetAllExchangeRates"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None, help="use a saved API response instead of downloading")
    a = ap.parse_args()
    if a.json:
        rows = json.load(open(a.json))
    else:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        rows = json.loads(urllib.request.urlopen(req, timeout=120).read())
    d = pd.DataFrame(rows)
    u = d[d["currency"].str.strip() == "US DOLLAR"].copy()
    u["date"] = pd.to_datetime(u["ratedate"])
    u = u.sort_values("date", kind="stable").drop_duplicates("date")
    out = u[["date", "buyingrate", "centralrate", "sellingrate"]]
    out.to_csv(a.out, index=False, date_format="%Y-%m-%d")
    print(f"{len(out)} business days, {out.date.min().date()} to {out.date.max().date()} -> {a.out}")


if __name__ == "__main__":
    main()
