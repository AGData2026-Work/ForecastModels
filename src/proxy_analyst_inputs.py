"""
The analyst's model inputs extended to Sep and Oct 2026 for the proxy check only (owner request, 2026-11-04). Writes a
COPY of data/analyst_inputs; the production inputs are not touched.

    python src/proxy_analyst_inputs.py --out outputs/proxy_20261104/analyst/inputs_v8
    python src/analyst_forecast.py --inputs outputs/proxy_20261104/analyst/inputs_v8 --out outputs/proxy_20261104/analyst/analyst_forecasts_v8.csv

Rows added, and where each comes from (his Data_Pipeline tab names the source of every input):
  white price      gap-filled panel v8 (or --panel, e.g. v10 with FEWS NET weekly Sep) Sep and Oct (the same starting prices the GRU used); no value where v8 has none
                   (Ibadan, Maiduguri, Gombe, Gujungu, Gwandu, Lagos; Mubi in Oct). white_observed = 0 on every cell:
                   all are proxies (FEWS NET's monthly release for Sep and Oct is not out)
  yellow price     Sep: mean of FEWS NET weekly yellow quotes (his Data_Pipeline rule for yellow); Oct: none
  sorghum          none (FEWS NET monthly not out); his feature model reads a missing feature as no signal
  weekly quotes    FEWS NET weekly white and yellow for Sep (fetched 4 Nov 2026); Kasuwa quotes from kasuwa_weekly.csv
                   for market-months with no FEWS NET white quote (his rule)
  headline_mom     Sep 0.06, food_mom Sep -1.55 (NBS CPI report for Sep 2026, published mid-Oct). Oct not out; the joint
                   forecast lags headline inflation one month, so the Oct origin uses Sep's value
  cpi              Sep chained from Aug by headline_mom (NBS rebased to 2024 = 100 with this report, so the new index
                   level is not on his Jan-2015 = 100 scale); Oct blank. Only complete calendar years enter his
                   seasonal index, so 2026 CPI is not read by any 2026 forecast
  fx_parallel      ESTIMATES from press street (BDC) quotes, his post-Jan-2025 method: Sep 1395, Oct 1377 (see FX_NOTE)
  fx_official      CBN central rate, monthly mean (data/external/20261104_CBN_..._2026-11_v1.csv)
  world maize, logistics, duty   import parity CSV: Sep as published by NADIH; Oct = Sep carried forward (no Oct row yet,
                   as in the GRU's Oct-origin run). --parity <csv> swaps the file; the owner's
                   2026-11-04 rerun uses ..._SepFobFRED_OctCarried_v1.csv (Sep world maize from IMF via FRED)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

V8 = "/Users/augmentumadvisory/Downloads/20261104_FEWSNET_Maize_Monthly_GapFilled_Panel_v8.xlsx"
WEEKLY = {"White": "data/external/20261104_FEWSNET_API_Nigeria_WeeklyMaizeWhite_v1.csv",
          "Yellow": "data/external/20261104_FEWSNET_API_Nigeria_WeeklyMaizeYellow_v1.csv"}
CBN = "data/external/20261104_CBN_API_Nigeria_USDNGN_Daily_2001-12_2026-11_v1.csv"
PARITY = "data/external/20261104_NADIH_ImportParity_Maize_Monthly_2006-01_2026-10_OctIsSepCarriedForward_v1.csv"
NBS = {"2026-09": {"headline_mom": 0.06, "food_mom": -1.55}}
FX = {"2026-09": 1395.0, "2026-10": 1377.0}
FX_NOTE = ("Sep: sell quotes about 1410 (2 and 5 Sep), 1390 (15 Sep), 1385 (29-30 Sep). Oct: about 1385 (1 Oct), 1380 (12 Oct), "
           "1365-1380 (20-22 Oct), 1385-1390 (29-30 Oct). Press trackers (thecityceleb, investorsking, arbiterz, vanguard, "
           "withinnigeria), user-submitted and indicative; sources differ by 20-30 NGN on a day.")
MONTHS = ["2026-09", "2026-10"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/analyst_inputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel", default=V8, help="gap-filled panel xlsx (Long_panel) for the Sep/Oct white prices")
    ap.add_argument("--parity", default=PARITY, help="import parity CSV (fob_usd_t, freight_usd_t, duty by month)")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(a.src, out, ignore=shutil.ignore_patterns("_backup"))
    rd = lambda f: pd.read_csv(out / f, index_col="month", parse_dates=["month"])
    white, obs, yellow, sorg = (rd(f) for f in ("white_price_ngn_per_kg.csv", "white_observed.csv",
                                                 "yellow_price_ngn_per_kg.csv", "sorghum_price_ngn_per_100kg.csv"))
    v8 = pd.read_excel(a.panel, "Long_panel")
    wk = {t: pd.read_csv(p, parse_dates=["period_date"]) for t, p in WEEKLY.items()}
    log = []
    for ym in MONTHS:
        m = pd.Timestamp(ym + "-01")
        p = v8[(v8.month == ym)].set_index("market").price_NGN_per_kg
        white.loc[m] = [p.get(c, np.nan) for c in white.columns]
        src_ = v8[(v8.month == ym)].set_index("market").source   # FEWS NET's own weekly mean counts as observed
        obs.loc[m] = [(1.0 if src_.get(c) == "FEWS NET weekly" else 0.0) if pd.notna(p.get(c, np.nan)) else np.nan for c in white.columns]
        y = wk["Yellow"][wk["Yellow"].period_date.dt.strftime("%Y-%m") == ym].groupby("market").price_ngn_per_kg.mean()
        yellow.loc[m] = [y.get(c, np.nan) for c in yellow.columns]
        sorg.loc[m] = np.nan
        log.append(dict(month=ym, white_cells=int(white.loc[m].notna().sum()), yellow_cells=int(yellow.loc[m].notna().sum())))
    for df, f in ((white, "white_price_ngn_per_kg.csv"), (obs, "white_observed.csv"), (yellow, "yellow_price_ngn_per_kg.csv"),
                  (sorg, "sorghum_price_ngn_per_100kg.csv")):
        df.sort_index().rename_axis("month").to_csv(out / f, date_format="%Y-%m-%d")
    # weekly quotes: FEWS NET Sep weeks, then Kasuwa where FEWS NET has no white quote in that market-month
    w = pd.read_csv(out / "weekly_quotes.csv", parse_dates=["week_end"])
    w = w[~w.week_end.dt.strftime("%Y-%m").isin(MONTHS)]
    new = []
    for t, d in wk.items():
        d = d[d.period_date.dt.strftime("%Y-%m").isin(MONTHS)]
        new.append(pd.DataFrame({"type": t, "market": d.market, "week_end": d.period_date, "price_ngn_per_kg": d.price_ngn_per_kg,
                                 "source": "FEWS NET"}))
    new = pd.concat(new, ignore_index=True)
    k = pd.read_csv(out / "kasuwa_weekly.csv", parse_dates=["week_end"])
    k = k[k.week_end.dt.strftime("%Y-%m").isin(MONTHS)]
    have = set(zip(new[new.type == "White"].market, new[new.type == "White"].week_end.dt.strftime("%Y-%m")))
    k = k[[(mk, d.strftime("%Y-%m")) not in have for mk, d in zip(k.market, k.week_end)]]
    new = pd.concat([new, pd.DataFrame({"type": "White", "market": k.market, "week_end": k.week_end,
                                        "price_ngn_per_kg": k.price_ngn_per_kg, "source": k.source})], ignore_index=True)
    w = pd.concat([w, new], ignore_index=True)
    w.to_csv(out / "weekly_quotes.csv", index=False, date_format="%Y-%m-%d")
    # macro
    mac = rd("macro_monthly.csv"); man = rd("manual_monthly.csv")
    cbn = pd.read_csv(CBN, parse_dates=["date"]); cbn["ym"] = cbn.date.dt.strftime("%Y-%m")
    par = pd.read_csv(a.parity).set_index("month")
    for ym in MONTHS:
        m = pd.Timestamp(ym + "-01"); prev = m - pd.DateOffset(months=1)
        r = {c: np.nan for c in mac.columns}
        r.update(NBS.get(ym, {}))
        if pd.notna(r.get("headline_mom")) and prev in mac.index:
            r["cpi"] = mac.at[prev, "cpi"] * (1 + r["headline_mom"] / 100)
        r["fx_parallel"] = FX[ym]
        r["fx_official"] = float(pd.to_numeric(cbn[cbn.ym == ym].centralrate, errors="coerce").mean())
        r["world_maize"], r["logistics"], r["duty"] = (float(par.at[ym, c]) for c in ("fob_usd_t", "freight_usd_t", "duty"))
        mac.loc[m] = [r[c] for c in mac.columns]
        man.loc[m] = [r[c] for c in man.columns]
    mac.sort_index().rename_axis("month").to_csv(out / "macro_monthly.csv", date_format="%Y-%m-%d")
    man.sort_index().rename_axis("month").to_csv(out / "manual_monthly.csv", date_format="%Y-%m-%d")
    (out / "README_proxy_check.txt").write_text(__doc__ + "\nfx_parallel quotes: " + FX_NOTE + "\n")
    print(pd.DataFrame(log).to_string(index=False)); print(mac.tail(4).round(2).to_string())
    print(w[w.week_end >= "2026-09-01"].groupby(["type", "source"]).size())


if __name__ == "__main__":
    main()
