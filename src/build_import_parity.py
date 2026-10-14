"""
Monthly import parity for maize at Apapa (D-75), from the NADIH volatility workbook.

    python src/build_import_parity.py --workbook <20261012_NADIH_FEWSNET_Maize-Volatility_v3.xlsx> \
        --out data/external/20261012_NADIH_ImportParity_Maize_Monthly_2006-01_2026-09_v1.csv

Reads sheet 'Parity price', columns B (Int'l Price fob, US$/MT), C (Exchange rate, NGN per US$),
E (Logistics, US$/MT) and G (Customs duty, a fraction). Columns D, F and H are spreadsheet formulas and
are recomputed here rather than read:
    parity_ngn_per_kg = (fob + freight) x naira_per_usd / 1000 x (1 + duty)
Rows with any of the four inputs missing are dropped. src/data.py attach_parity reads parity_ngn_per_kg.
"""
from __future__ import annotations

import argparse

import pandas as pd


def build(workbook: str) -> pd.DataFrame:
    s = pd.read_excel(workbook, sheet_name="Parity price", header=0)
    q = pd.DataFrame({
        "month": pd.to_datetime(s.iloc[:, 0], errors="coerce").dt.strftime("%Y-%m"),
        "fob_usd_t": pd.to_numeric(s.iloc[:, 1], errors="coerce"),
        "naira_per_usd": pd.to_numeric(s.iloc[:, 2], errors="coerce"),
        "freight_usd_t": pd.to_numeric(s.iloc[:, 4], errors="coerce"),
        "duty": pd.to_numeric(s.iloc[:, 6], errors="coerce"),
    }).dropna()
    q["parity_ngn_per_kg"] = (q.fob_usd_t + q.freight_usd_t) * q.naira_per_usd / 1000 * (1 + q.duty)
    return q.round(4).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    q = build(a.workbook)
    q.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(q)} months, {q.month.iloc[0]} to {q.month.iloc[-1]}")


if __name__ == "__main__":
    main()
