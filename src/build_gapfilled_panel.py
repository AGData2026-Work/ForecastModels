"""
Gap-filled monthly FEWS NET white maize wholesale panel (D-75), as a function so it can be rebuilt
with information truncated at any date (src/truncation_test_monthly.py).

    python src/build_gapfilled_panel.py --dl ~/Downloads --out <...GapFilled_Panel_v3.xlsx>

Inputs (in --dl, not in the repo):
  20261012_FEWSNET_API_Nigeria_MarketPrices_2003-10_2026-08_v2.xlsx  sheet white_maize_wholesale_kg
  20261012_FEWSNET Price Data_v1.xlsx        value_one_year_ago / value_one_month_ago columns
  20260923_AFEX_MultiCommodity_Weekly_Panel.xlsx                     sheet Panel_Long
  20260811_NAERLS Wholesale Maize Prices_v0 (2).xlsx
  20260930_OtherSource_Kaduna_Kano_Consolidated_v1_1.xlsx             sheet Weekly_Panel
Optional (--extra-2026, off by default so v3 stays reproducible):
  Price Data From January Till Date.xlsx     white maize, Dec 2025 to Sep 2026, 8 markets, collector
                                             unconfirmed (its Giwa and Pambegua values track the Kaduna/Kano
                                             other source closely). Used for Potiskum, Mubi and Giwa, and its
                                             Lashe Money market (in Biu town) for Biu.
--afex-chat <_chat.txt> (off by default): in months where an AFEX reporter sent a new-crop maize quote next to
an old or untagged one, replace the AFEX panel's monthly value with the old and untagged quotes only
(src/afex_chat.py). The panel's narrow new-crop rule left new crop in some harvest months, e.g. Dandume Sep
2026: 279 in the panel, 345 old crop only.
--afex-maize <xlsx> (off by default): take the AFEX source from the corrected all-markets AFEX maize panel
(sheet Weekly_Panel; e.g. 20261019_AFEX_Maize_Observed_AllMarkets_v2.xlsx: narrow new-crop rule, harvest check,
Sep 2026 extract re-tagged) instead of the 23 Sep observed panel. Monthly value = mean of observed weeks.
--fews-retail <xlsx> (off by default): FEWS NET's own RETAIL white maize price for the same market (Long_panel of
20261019_FEWSNET_Maize_Monthly_Retail_Panel_v1.xlsx) as the first fill source for every market, level-adjusted
to wholesale by the usual prior-overlap ratio. Retail tracks wholesale closely at these markets (ratio 0.96-1.01,
monthly changes correlate 0.81-0.96). With --extra-2026 it also adds Damaturu <- Potiskum from the 2026 file.
--through YYYY-MM adds empty FEWS NET months after the export's last month, so a source can fill a month
FEWS NET has not released yet (e.g. Sep 2026). Such months can never be interpolated (no later anchor).

Rules, fixed before filling:
  - Observed FEWS NET values are never changed. Months FEWS NET never published as a value but reports
    in a later row's value_one_year_ago / value_one_month_ago are treated as observed (v3).
  - Candidates per market in a fixed order: same-market sources first, then same-state stand-ins ranked
    by their alignment with FEWS NET (PLAN). Kano, Dawanau is not source-filled.
  - Level ratio = median(FEWS / source) over months where both exist BEFORE the gap month, at least 6,
    otherwise the source value is used unadjusted and flagged.
  - Remaining FEWS NET gaps of 2 months or less are interpolated linearly; longer gaps stay empty.
`cutoff` (a month, 'YYYY-MM') keeps only what was known at the forecast for that month, made on the
15th of the next month: FEWS NET values and source values for months <= cutoff, and recovered values
only from rows dated <= cutoff (a value reported as value_one_year_ago of month m+12 is known from
m+12's own release).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PLAN = {  # (source, source market, kind)
    "Dandume": [("AFEX", "Dandume", "direct"), ("NAERLS (old crop)", "Dandume", "direct"),
                ("AFEX", "Bakori", "stand-in, Katsina"), ("AFEX", "Danja", "stand-in, Katsina")],
    "Saminaka": [("AFEX", "Saminaka", "direct"), ("AFEX", "Pambegua", "stand-in, Kaduna"),
                 ("Other source Kaduna/Kano", "Pambegua", "stand-in, Kaduna")],
    "Giwa": [("Other source Kaduna/Kano", "Giwa", "direct"), ("NAERLS (old crop)", "Giwa", "direct"),
             ("AFEX", "Tundun Saibu", "stand-in, Kaduna"), ("AFEX", "Pambegua", "stand-in, Kaduna"),
             ("AFEX", "Anchau", "stand-in, Kaduna")],
    "Kaura Namoda": [("NAERLS (old crop)", "Tsafe", "stand-in, Zamfara")],
}
# Added only when --extra-2026 is given (owner request 2026-10-15); same-market, after existing sources.
PLAN_2026 = {"Potiskum": [("Price Data 2026", "Potiskum", "direct")],
             "Mubi": [("Price Data 2026", "Mubi", "direct")],
             "Giwa": [("Price Data 2026", "Giwa", "direct")],
             # Lashe Money is a market in Biu town (owner, 2026-10-15): FEWS NET Biu / Lashe Money 0.92 to 1.05
             # over Jan-Aug 2026, monthly changes correlate 0.65 (against Maiduguri: 1.10 to 1.36, 0.09)
             "Biu": [("Price Data 2026", "Lashe Money", "stand-in, Biu town")]}
# Added only with --fews-retail (owner request 2026-10-19): Damaturu from the 2026 file's Potiskum (same state, 9
# overlap months in 2026, FEWS NET Damaturu / file Potiskum 0.87, monthly changes correlate 0.63).
PLAN_V7 = {"Damaturu": [("Price Data 2026", "Potiskum", "stand-in, Yobe")]}
MIN_OVERLAP = 6
MAX_INTERP = 2


def load_price_data_2026(path: str | Path) -> pd.DataFrame:
    """Monthly mean of the file's own 'price per kg' per market. Dropped as entry errors: bag weights over
    200 kg (235 kg Pambegua bags, a 1,100 kg Mubi bag) and per-kg values more than twice bag price / weight."""
    d = pd.read_excel(path).iloc[:, :8]
    d = d[d["Commodity"].astype(str).str.strip().str.lower() == "white maize"].copy()
    d["Market"] = d["Market"].astype(str).str.strip().str.title()
    calc = d["Price per Bag"] / d["Weight of Bag (kg)"]
    d = d[(d["Weight of Bag (kg)"] <= 200) & (d["price per kg"] <= 2 * calc)]
    d["month"] = pd.to_datetime(d["Date"]).dt.strftime("%Y-%m")
    return d.groupby(["month", "Market"])["price per kg"].mean().unstack()


def load_inputs(dl: str | Path, extra_2026: str | Path | None = None, through: str | None = None,
                afex_chat: str | Path | None = None, afex_maize: str | Path | None = None,
                fews_retail: str | Path | None = None) -> dict:
    dl = Path(dl)
    F = (pd.read_excel(dl / "20261012_FEWSNET_API_Nigeria_MarketPrices_2003-10_2026-08_v2.xlsx",
                       "white_maize_wholesale_kg").set_index("month").drop(columns="markets_reporting"))
    X = pd.read_excel(dl / "20261012_FEWSNET Price Data_v1.xlsx")
    X = X[(X["product"] == "Maize Grain (White)") & (X.price_type == "Wholesale")].copy()
    X["mp"] = pd.to_datetime(X.period_date).dt.to_period("M")
    L = pd.read_excel(dl / "20260923_AFEX_MultiCommodity_Weekly_Panel.xlsx", "Panel_Long")
    L = L[(L.commodity == "Maize") & (L.row_status == "observed")].copy()
    L["month"] = pd.to_datetime(L.s).dt.strftime("%Y-%m")
    n = pd.read_excel(dl / "20260811_NAERLS Wholesale Maize Prices_v0 (2).xlsx")
    n = n[(n["Maize Type"] == "White") & (n["Maize Age"] != "New")].copy()
    n["month"] = pd.to_datetime(n.Date).dt.strftime("%Y-%m")
    o = pd.read_excel(dl / "20260930_OtherSource_Kaduna_Kano_Consolidated_v1_1.xlsx", "Weekly_Panel")
    o = o[(o.commodity == "Maize") & o.price_NGN_per_kg.notna()].copy()
    o["month"] = pd.to_datetime(o.date).dt.strftime("%Y-%m")
    if through and through > F.index.max():  # months FEWS NET has not released yet stay empty
        F = F.reindex(list(F.index) + [str(p) for p in pd.period_range(F.index.max(), through, freq="M")[1:]])
    src = {"AFEX": L.groupby(["market", "month"]).price_NGN_per_kg.mean().unstack(0),
           "NAERLS (old crop)": (n.groupby(["Market", "month"]).Value.mean() / 100).unstack(0),
           "Other source Kaduna/Kano": o.groupby(["market", "month"]).price_NGN_per_kg.mean().unstack(0)}
    if afex_maize:
        a = pd.read_excel(afex_maize, "Weekly_Panel")
        a = a[(a["commodity"] == "Maize") & a["price_NGN_per_kg"].notna()].copy()
        a["month"] = pd.to_datetime(a["date"]).dt.strftime("%Y-%m")
        src["AFEX"] = a.groupby(["market", "month"]).price_NGN_per_kg.mean().unstack(0)
    if extra_2026:
        src["Price Data 2026"] = load_price_data_2026(extra_2026)
    if fews_retail:
        r = pd.read_excel(fews_retail, "Long_panel")
        r = r[(r["flag"] == "observed") & r["price_NGN_per_kg"].notna()]
        src["FEWS NET retail"] = r.pivot_table(index="month", columns="market", values="price_NGN_per_kg")
    overridden = set()
    if afex_chat:
        from afex_chat import old_crop_overrides
        ov = old_crop_overrides(afex_chat)
        A = src["AFEX"]
        A = A.reindex(sorted(set(A.index) | set(ov.index)))
        for mo in ov.index:
            for mk in ov.columns:
                if mk in A.columns and pd.notna(ov.loc[mo, mk]):
                    A.loc[mo, mk] = ov.loc[mo, mk]
                    overridden.add((mk, mo))
        src["AFEX"] = A
    return {"F": F, "X": X, "src": src, "afex_old_crop": overridden}


def build_long(inp: dict, cutoff: str | None = None) -> pd.DataFrame:
    F = inp["F"].copy()
    months = list(F.index)
    if cutoff:
        F.loc[[m for m in months if m > cutoff]] = np.nan
    X = inp["X"]
    if cutoff:
        X = X[X.mp <= pd.Period(cutoff, freq="M")]
    recov = {}
    for col, lag in [("value_one_year_ago", 12), ("value_one_month_ago", 1)]:
        for mk, mp, v in zip(X.market, X.mp, X[col]):
            k = (mk, str(mp - lag))
            if pd.notna(v) and k[1] in F.index and mk in F.columns and pd.isna(F.loc[k[1], mk]):
                recov[k] = v / 100
    for (mk, mo), v in recov.items():
        F.loc[mo, mk] = v
    src = {k: (s.loc[[m for m in s.index if m <= cutoff]] if cutoff else s) for k, s in inp["src"].items()}
    rows = []
    for mkt in F.columns:
        col = F[mkt]
        for i, m in enumerate(months):
            v = col.iloc[i]
            r = dict(market=mkt, month=m, price_NGN_per_kg=np.nan, flag="missing", source="", source_market="",
                     fill_kind="", source_value=np.nan, ratio_applied=np.nan, ratio_overlap_months=np.nan)
            if pd.notna(v):
                r.update(price_NGN_per_kg=round(v, 2), flag="observed", source_market=mkt,
                         source="FEWS NET API, value_one_year_ago" if (mkt, m) in recov else "FEWS NET API")
            else:
                retail = [("FEWS NET retail", mkt, "direct, retail")] if "FEWS NET retail" in src else []
                v7 = PLAN_V7.get(mkt, []) if "FEWS NET retail" in src else []
                cands = retail + PLAN.get(mkt, []) + PLAN_2026.get(mkt, []) + v7
                cands = [c for c in cands if c[2].startswith("direct")] + [c for c in cands if not c[2].startswith("direct")]  # same market first
                for s_name, sm, kind in cands:
                    if s_name not in src:
                        continue
                    S = src[s_name]
                    if sm not in S or m not in S.index or pd.isna(S.loc[m, sm]):
                        continue
                    s = S[sm]
                    prior = [x for x in months if x < m and pd.notna(F.loc[x, mkt]) and x in s.index and pd.notna(s[x])]
                    if len(prior) >= MIN_OVERLAP:
                        ratio = float(np.median([F.loc[x, mkt] / s[x] for x in prior])); adj = "adjusted"
                    else:
                        ratio = 1.0; adj = "unadjusted (fewer than 6 prior overlap months)"
                    oc = "; old crop only (AFEX chat)" if s_name == "AFEX" and (sm, m) in inp.get("afex_old_crop", ()) else ""
                    r.update(price_NGN_per_kg=round(s[m] * ratio, 2),
                             flag="filled_direct" if kind.startswith("direct") else "filled_standin",
                             source=s_name, source_market=sm, fill_kind=f"{kind}; {adj}{oc}",
                             source_value=round(s[m], 2), ratio_applied=round(ratio, 4), ratio_overlap_months=len(prior))
                    break
            if r["flag"] == "missing":
                a = next((j for j in range(i - 1, -1, -1) if pd.notna(col.iloc[j])), None)
                b = next((j for j in range(i + 1, len(months)) if pd.notna(col.iloc[j])), None)
                if a is not None and b is not None and b - a - 1 <= MAX_INTERP:
                    k = b - a - 1
                    val = col.iloc[a] + (col.iloc[b] - col.iloc[a]) * (i - a) / (k + 1)
                    harvest = any(int(months[j][5:]) in (9, 10, 11, 12) for j in range(a + 1, b))
                    r.update(price_NGN_per_kg=round(val, 2), flag="filled_interpolated", source="FEWS NET, linear",
                             source_market=mkt,
                             fill_kind=f"{k}-month gap between {months[a]} and {months[b]}" + ("; harvest months, caution" if harvest and k == 2 else ""))
            r["is_proxy_price"] = r["flag"].startswith("filled")
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", default=str(Path.home() / "Downloads"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra-2026", default=None, help="path to 'Price Data From January Till Date.xlsx' (Potiskum, Mubi, Giwa)")
    ap.add_argument("--through", default=None, help="extend the months to YYYY-MM with empty FEWS NET values")
    ap.add_argument("--fews-retail", default=None, help="FEWS NET retail panel xlsx (Long_panel): same-market retail as the first fill source")
    ap.add_argument("--afex-maize", default=None, help="corrected AFEX maize panel (Weekly_Panel sheet) to use as the AFEX source")
    ap.add_argument("--afex-chat", default=None, help="AFEX WhatsApp _chat.txt: old-crop AFEX values where new crop was mixed in")
    a = ap.parse_args()
    inp = load_inputs(a.dl, a.extra_2026, a.through, a.afex_chat, a.afex_maize, a.fews_retail)
    if a.afex_chat:
        print(f"AFEX cells replaced with old crop only: {len(inp['afex_old_crop'])}")
    P = build_long(inp)
    P.to_excel(a.out, sheet_name="Long_panel", index=False)
    print(P.flag.value_counts().to_dict(), a.out)


if __name__ == "__main__":
    main()
