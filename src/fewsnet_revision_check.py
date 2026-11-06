"""
Has FEWS NET revised what we hold? Re-pull Nigeria's full monthly history, one month per request, and diff it against
the export the production panel reads (owner request after the 4 Nov 2026 finding that Jul and Aug 2026 had changed).

    .venv/bin/python src/fewsnet_revision_check.py --first 2004-10 --last 2026-09 \
        --ours ~/Downloads/20261012_FEWSNET_API_Nigeria_MarketPrices_2003-10_2026-08_v2.xlsx \
        --raw data/external/20261105_FEWSNET_API_Nigeria_AllRows_Monthly_2004-10_2026-09_v1.parquet \
        --out outputs/fewsnet_revisions_20261105

Why one month per request: on 4 Nov 2026 a Jun-Sep 2026 request returned Jun-Aug only (891 rows), although Sep had
been published on 4 Oct. On 5 Nov the same request returned all four months (1,188 rows), so the cause is not
confirmed (no row cap, no client cache; the API sends Cache-Control max-age=900, private). Small requests plus a
cache-busting parameter, and a row count logged per month, make a short or empty answer visible.

Writes to --out:
  request_log.csv       month, rows returned, white maize wholesale rows, seconds
  white_maize_diff.csv  every market-month where ours and the API differ, incl. added and removed values, NGN/kg
  diesel_diff.csv       the same for diesel retail, NGN per litre as published (the export's diesel sheet)
  summary.json          counts by kind of change and by year
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://fdw.fews.net/api/marketpricefacts.json"
TOL = 0.011  # NGN per kg or litre; values are per 100 kg in whole naira, so 0.01 is rounding, not a revision


def pull_month(ym: str, cache: Path) -> tuple[str, list, float]:
    """One month's rows; saved to cache/<ym>.json as they arrive so a dropped connection resumes, not restarts."""
    f = cache / f"{ym}.json"
    if f.exists():
        return ym, json.loads(f.read_text()), 0.0
    p = pd.Period(ym, "M")
    q = {"country_code": "NG", "start_date": p.start_time.date().isoformat(), "end_date": p.end_time.date().isoformat(),
         "nocache": str(time.time_ns())}
    t = time.time()
    for attempt in range(6):
        try:
            req = urllib.request.Request(API + "?" + urllib.parse.urlencode(q), headers={"User-Agent": "fewsnet-revision-check"})
            with urllib.request.urlopen(req, timeout=180) as r:
                got = json.load(r)
            got = got if isinstance(got, list) else got.get("results", [])
            f.write_text(json.dumps(got))
            return ym, got, time.time() - t
        except Exception:
            if attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))


def per_kg(d: pd.DataFrame) -> pd.Series:
    k = d.unit.map(lambda u: 1.0 if u == "kg" else float(u.split("_")[0]) if isinstance(u, str) and u.endswith("_kg")
                   and u.split("_")[0].isdigit() else {"mt": 1000.0, "tonne": 1000.0}.get(u, np.nan))
    return d.value / k


def diff(ours: pd.DataFrame, api: pd.DataFrame, what: str) -> pd.DataFrame:
    cols = sorted(set(ours.columns) | set(api.columns))
    idx = sorted(set(ours.index) | set(api.index))
    a, b = ours.reindex(index=idx, columns=cols), api.reindex(index=idx, columns=cols)
    s = pd.concat([a.stack(dropna=False).rename("ours"), b.stack(dropna=False).rename("api")], axis=1)
    s.index.names = ["month", "market"]
    s = s[s.ours.notna() | s.api.notna()].reset_index()
    s["change"] = np.select([s.ours.isna(), s.api.isna(), (s.api - s.ours).abs() > TOL],
                            ["added by FEWS NET", "removed by FEWS NET", "value revised"], "same")
    s["pct"] = 100 * (s.api / s.ours - 1)
    s["series"] = what
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default="2004-10")
    ap.add_argument("--last", required=True)
    ap.add_argument("--ours", required=True, help="the FEWS NET export the panel reads (xlsx with the two wide sheets)")
    ap.add_argument("--raw", required=True, help="where to save every API row (parquet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    months = [str(p) for p in pd.period_range(a.first, a.last, freq="M")]
    cache = out / "months"  # delete this folder to force a fresh pull
    cache.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(lambda ym: pull_month(ym, cache), months))
    rows, log = [], []
    for ym, r, sec in res:
        for x in r:
            x["requested_month"] = ym
        rows += r
        wm = sum(1 for x in r if x.get("product") == "Maize Grain (White)" and x.get("price_type") == "Wholesale")
        log.append(dict(month=ym, rows=len(r), white_maize_wholesale_rows=wm, seconds=round(sec, 1)))
    log = pd.DataFrame(log)
    log.to_csv(out / "request_log.csv", index=False)
    raw = pd.DataFrame(rows)
    raw.to_parquet(a.raw, index=False)
    raw["month"] = raw.period_date.str[:7]
    off = raw[raw.month != raw.requested_month]
    print(f"{len(months)} requests, {len(raw)} rows; rows outside their requested month: {len(off)}; "
          f"empty months: {(log.rows == 0).sum()}")

    m = raw[(raw["product"] == "Maize Grain (White)") & (raw.price_type == "Wholesale")
            & (raw.collection_schedule == "Monthly")].copy()
    m["price"] = per_kg(m)
    dup = m.groupby(["month", "market"]).size()
    api_w = m.groupby(["month", "market"]).price.mean().unstack()
    ours_w = (pd.read_excel(a.ours, "white_maize_wholesale_kg").set_index("month").drop(columns="markets_reporting")
              .dropna(how="all"))
    dw = diff(ours_w, api_w, "white maize wholesale NGN/kg")
    pub = m.groupby(["month", "market"]).agg(created=("created", "max"), status_changed=("status_changed", "max"),
                                             status=("collection_status", "first")).reset_index()
    dw = dw.merge(pub, on=["month", "market"], how="left")

    d = raw[(raw["product"].str.contains("Diesel", na=False)) & (raw.collection_schedule == "Monthly")]
    ours_d = pd.read_excel(a.ours, "diesel_retail_litre").set_index("month")
    ours_d = ours_d.drop(columns=[c for c in ours_d.columns if c == "markets_reporting"]).dropna(how="all")
    api_d = d.groupby(["month", "market"]).value.mean().unstack()
    dd = diff(ours_d, api_d, "diesel retail NGN/litre")

    for name, x in [("white_maize_diff.csv", dw), ("diesel_diff.csv", dd)]:
        x[x.change != "same"].round(3).to_csv(out / name, index=False)
    summ = {"requests": len(months), "rows": len(raw), "empty_months": log.loc[log.rows == 0, "month"].tolist(),
            "white_maize_market_months_with_more_than_one_row": int((dup > 1).sum())}
    for x in (dw, dd):
        k = x.series.iloc[0]
        summ[k] = {"by_change": x.change.value_counts().to_dict(),
                   "changed_by_year": x[x.change != "same"].month.str[:4].value_counts().sort_index().to_dict()}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
