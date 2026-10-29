"""
Early-month price signal for the monthly GRU (idea 1, 2026-10-19): how far a market's own weekly source price has
moved from month o to the first two weeks of month o+1. The monthly forecast from month o is made on the 15th of
o+1, so reports dated the 1st to the 14th of o+1 are already public.

    python src/build_nowcast.py --afex <20261019_AFEX_Maize_Observed_AllMarkets_v2.xlsx> \
        --kasuwa "<Price Data From January Till Date.xlsx>" \
        --out data/external/20261019_Nowcast_EarlyMonth_v1.csv

Per FEWS NET market and origin month o: nowcast = log(mean early-o+1 price / mean month-o price), both from the SAME
source series, so no level adjustment is needed. AFEX weekly rows are dated by their Wednesday and cover that
Wednesday to the next Tuesday: a week counts as early o+1 only if it starts on the 1st to the 8th (so it ends by the
14th), and as month o if it starts in month o. Kasuwa quotes are dated: early o+1 = the 1st to the 14th.
Sources, same market or same town only: Dandume, Saminaka, Giwa and Dawanau (Kano) from AFEX; Potiskum, Mubi, Lashe
Money (Biu) and Giwa (fallback) from the 2026 price file. Every other market-month has no value; the model sees 0
with a mask of 0 (src/data.py attach_nowcast).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_gapfilled_panel import load_price_data_2026  # noqa: E402  (same cleaning of the 2026 file)

AFEX_MAP = {"Dandume": "Dandume", "Saminaka": "Saminaka", "Giwa": "Giwa", "Kano, Dawanau": "Dawanau"}
KASUWA_MAP = {"Potiskum": "Potiskum", "Mubi": "Mubi", "Biu": "Lashe Money", "Giwa": "Giwa"}


def afex_rows(path: str) -> pd.DataFrame:
    w = pd.read_excel(path, "Weekly_Panel")
    w = w[(w["commodity"] == "Maize") & w["price_NGN_per_kg"].notna()].copy()
    w["date"] = pd.to_datetime(w["date"])
    out = []
    for fews, src in AFEX_MAP.items():
        s = w[w.market == src]
        if s.empty:
            continue
        month_mean = s.groupby(s.date.dt.to_period("M")).price_NGN_per_kg.mean()
        early = s[s.date.dt.day <= 8]
        early_mean = early.groupby(early.date.dt.to_period("M")).price_NGN_per_kg.agg(["mean", "size"])
        for o, base in month_mean.items():
            e = early_mean.reindex([o + 1])
            if e["mean"].notna().iloc[0]:
                out.append(dict(market=fews, origin_month=str(o), nowcast=float(np.log(e["mean"].iloc[0] / base)),
                                source=f"AFEX {src}", n_early=int(e["size"].iloc[0])))
    return pd.DataFrame(out)


def kasuwa_rows(path: str) -> pd.DataFrame:
    d = pd.read_excel(path).iloc[:, :8]
    d = d[d["Commodity"].astype(str).str.strip().str.lower() == "white maize"].copy()
    d["Market"] = d["Market"].astype(str).str.strip().str.title()
    calc = d["Price per Bag"] / d["Weight of Bag (kg)"]
    d = d[(d["Weight of Bag (kg)"] <= 200) & (d["price per kg"] <= 2 * calc)]   # as load_price_data_2026
    d["date"] = pd.to_datetime(d["Date"])
    out = []
    for fews, src in KASUWA_MAP.items():
        s = d[d.Market == src]
        if s.empty:
            continue
        month_mean = s.groupby(s.date.dt.to_period("M"))["price per kg"].mean()
        early = s[s.date.dt.day <= 14]
        early_mean = early.groupby(early.date.dt.to_period("M"))["price per kg"].agg(["mean", "size"])
        for o, base in month_mean.items():
            e = early_mean.reindex([o + 1])
            if e["mean"].notna().iloc[0]:
                out.append(dict(market=fews, origin_month=str(o), nowcast=float(np.log(e["mean"].iloc[0] / base)),
                                source=f"2026 price file {src}", n_early=int(e["size"].iloc[0])))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--afex", required=True)
    ap.add_argument("--kasuwa", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A, K = afex_rows(a.afex), kasuwa_rows(a.kasuwa)
    K = K[~K.set_index(["market", "origin_month"]).index.isin(A.set_index(["market", "origin_month"]).index)]  # AFEX first
    N = pd.concat([A, K], ignore_index=True).sort_values(["market", "origin_month"])
    N.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(N)} market-months")
    print(N.groupby("market").agg(months=("nowcast", "size"), first=("origin_month", "min"), last=("origin_month", "max"),
                                  median_abs_pct=("nowcast", lambda s: round(100 * s.abs().median(), 1))).to_string())


if __name__ == "__main__":
    main()
