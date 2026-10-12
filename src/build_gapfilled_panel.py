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
MIN_OVERLAP = 6
MAX_INTERP = 2


def load_inputs(dl: str | Path) -> dict:
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
    src = {"AFEX": L.groupby(["market", "month"]).price_NGN_per_kg.mean().unstack(0),
           "NAERLS (old crop)": (n.groupby(["Market", "month"]).Value.mean() / 100).unstack(0),
           "Other source Kaduna/Kano": o.groupby(["market", "month"]).price_NGN_per_kg.mean().unstack(0)}
    return {"F": F, "X": X, "src": src}


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
                for s_name, sm, kind in PLAN.get(mkt, []):
                    S = src[s_name]
                    if sm not in S or m not in S.index or pd.isna(S.loc[m, sm]):
                        continue
                    s = S[sm]
                    prior = [x for x in months if x < m and pd.notna(F.loc[x, mkt]) and x in s.index and pd.notna(s[x])]
                    if len(prior) >= MIN_OVERLAP:
                        ratio = float(np.median([F.loc[x, mkt] / s[x] for x in prior])); adj = "adjusted"
                    else:
                        ratio = 1.0; adj = "unadjusted (fewer than 6 prior overlap months)"
                    r.update(price_NGN_per_kg=round(s[m] * ratio, 2),
                             flag="filled_direct" if kind == "direct" else "filled_standin",
                             source=s_name, source_market=sm, fill_kind=f"{kind}; {adj}",
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
    a = ap.parse_args()
    P = build_long(load_inputs(a.dl))
    P.to_excel(a.out, sheet_name="Long_panel", index=False)
    print(P.flag.value_counts().to_dict(), a.out)


if __name__ == "__main__":
    main()
