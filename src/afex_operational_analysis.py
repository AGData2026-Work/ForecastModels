"""
Consolidated deliverable: price direction analysis and per-market error
analysis for the AFEX operational runs (maize only, RNN and GRU).

    python src/afex_operational_analysis.py

Reads outputs/afex_directional_overall.csv, outputs/afex_directional_by_market.csv
(from afex_directional.py) and outputs/afex_operational/{RNN,GRU}/metrics_by_market.csv
directly. No retraining.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

FONT = "Arial"
THIN = Side(style="thin", color="000000")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def write_sheet(wb, name, df, notes=None, pct_cols=(), num_cols=()):
    ws = wb.create_sheet(name[:31])
    r = 1
    if notes:
        for n in notes:
            ws.cell(row=r, column=1, value=n).font = Font(name=FONT, size=10, italic=True)
            r += 1
        r += 1
    header_row = r
    for j, c in enumerate(df.columns, start=1):
        cell = ws.cell(row=header_row, column=j, value=c)
        cell.font = Font(name=FONT, size=10, bold=True)
        cell.border = BORDER
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.fill = PatternFill("solid", fgColor="D9D9D9")
    for i, (_, row) in enumerate(df.iterrows(), start=header_row + 1):
        for j, c in enumerate(df.columns, start=1):
            v = row[c]
            if pd.isna(v):
                v = None
            elif hasattr(v, "item"):
                v = v.item()
            cell = ws.cell(row=i, column=j, value=v)
            cell.font = Font(name=FONT, size=10)
            cell.border = BORDER
            if c in pct_cols:
                cell.number_format = "0.00"
            elif c in num_cols:
                cell.number_format = "#,##0.00"
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
    for j, c in enumerate(df.columns, start=1):
        width = max(len(str(c)) + 2, 11)
        if c in ("market", "run"):
            width = max(width, 22)
        ws.column_dimensions[get_column_letter(j)].width = min(width, 30)
    ws.auto_filter.ref = f"A{header_row}:{get_column_letter(len(df.columns))}{header_row + len(df)}"


def write_readme(wb, counts):
    ws = wb.create_sheet("README", 0)
    lines = [
        ("AFEX operational runs: price direction and per-market error, maize only", True),
        ("", False),
        (f"Generated {datetime.now():%Y-%m-%d %H:%M}", False),
        ("Scope: operational setting only (D-28: safeguard-removed discontinued for this panel, "
         "found worse at every horizon in D-27). RNN and GRU, maize series only, using the "
         "already-saved forecasts -- nothing retrained for this analysis.", False),
        ("", False),
        ("WHY THERE IS NO \"DO-NOTHING DIRECTION\" COLUMN", True),
        ("The do-nothing (carry-forward) forecast predicts no change at every origin, so it "
         "never asserts an up or down call. Scored the same way the model is, it would show 0% "
         "directional accuracy on every window where price actually moved -- not a meaningful "
         "comparison, just a restatement that a flat forecast has no opinion. The real benchmark "
         "is the panel's own base rate: the share of windows where price actually rose (or "
         "fell). A model with no real skill should land near that share by chance; a model that "
         "cannot even reach the base rate is not adding directional information. This mirrors "
         "the method already used for the main FEWSNET panel (DECISIONS D-19).", False),
        ("", False),
        ("HEADLINE DIRECTIONAL RESULT", True),
        ("5 of 6 run/horizon combinations FAIL to beat the stronger of always-up/always-down. "
         "Only RNN at 13 weeks clears it (61.4% vs a 58.7% always-up share). GRU at 4 weeks "
         "scores 35.2%, below a 50% coin flip and 24.5 points below the 59.7% always-up share.", False),
        ("RNN is the stronger architecture for direction at every horizon here. This does not "
         "match the error-rate picture (D-27), where GRU edges ahead on MAE at 26 weeks: the two "
         "metrics do not point at the same architecture.", False),
        ("", False),
        ("HEADLINE PER-MARKET ERROR RESULT", True),
        ("GRU has lower MAE than RNN at every one of the 14 maize markets at 26 weeks. At 4 and "
         "13 weeks the picture is mixed, RNN ahead in most markets but not all.", False),
        ("Ikara | Maize is the weakest market at every horizon (roughly double the next-weakest "
         "market's MAE at 26 weeks) and also one of the thinnest, 24 scored windows. Pambegua | "
         "Maize looks strongest at every horizon but rests on only 5 scored windows -- too few "
         "to trust as a real result rather than a favourable sample. Treat per-market rankings "
         "as indicative, not decisive, especially for markets under about 15 scored windows.", False),
        ("", False),
        ("SHEETS", True),
        ("Directional_Overall     model directional accuracy vs the always-up/always-down base rate, "
         "maize pooled across markets, per run and horizon.", False),
        ("Directional_By_Market   the same, split by individual maize market.", False),
        ("Error_By_Market         MAE and MAPE per market, RNN and GRU side by side, per horizon.", False),
    ]
    for i, (text, bold) in enumerate(lines, start=1):
        c = ws.cell(row=i, column=1, value=text)
        c.font = Font(name=FONT, size=11, bold=bold)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 118
    r = len(lines) + 2
    ws.cell(row=r, column=1, value="ROW COUNTS").font = Font(name=FONT, size=11, bold=True)
    for k, v in counts.items():
        r += 1
        ws.cell(row=r, column=1, value=f"{k}: {v}").font = Font(name=FONT, size=11)


def main():
    overall = pd.read_csv("outputs/afex_directional_overall.csv")
    by_market_dir = pd.read_csv("outputs/afex_directional_by_market.csv")
    err = pd.read_csv("outputs/afex_operational_per_market_error.csv")

    wb = Workbook()
    wb.remove(wb.active)

    write_sheet(wb, "Directional_Overall",
                overall[["run", "h", "n", "n_directional", "model_direction_pct",
                        "always_up_share_pct", "always_down_share_pct",
                        "margin_vs_coinflip_pp", "beats_always_up_and_down"]],
                notes=["Maize pooled across markets. model_direction_pct excludes windows where price did not move at all."],
                num_cols=(), pct_cols=("model_direction_pct", "always_up_share_pct",
                                       "always_down_share_pct", "margin_vs_coinflip_pp"))

    write_sheet(wb, "Directional_By_Market",
                by_market_dir[["run", "h", "market", "n", "model_direction_pct",
                              "always_up_share_pct", "always_down_share_pct",
                              "beats_always_up_and_down"]],
                notes=["Cell counts are small for several markets; check n before drawing conclusions."],
                pct_cols=("model_direction_pct", "always_up_share_pct", "always_down_share_pct"))

    write_sheet(wb, "Error_By_Market",
                err[["run", "h", "market", "n", "challenger_MAE", "challenger_MAPE",
                    "naive_MAE", "naive_MAPE"]],
                notes=["MAE in NGN/kg, MAPE in percent. naive is plain carry-forward."],
                num_cols=("challenger_MAE", "naive_MAE"), pct_cols=("challenger_MAPE", "naive_MAPE"))

    write_readme(wb, {"directional overall rows": len(overall),
                      "directional by-market rows": len(by_market_dir),
                      "error by-market rows": len(err)})

    stamp = datetime.now().strftime("%Y%m%d")
    out = f"outputs/{stamp}_AFEX_Operational_DirectionAndPerMarketError_v1.xlsx"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
