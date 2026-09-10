"""
Consolidated predicted-vs-actual report for the AFEX multi-commodity
farmgate panel workstream: operational vs safeguard-removed, RNN vs GRU.

    python src/afex_report.py

Reads the four run directories under outputs/afex_operational/ and
outputs/afex_safeguard_removed/ and writes one xlsx with headline metrics,
predicted-vs-actual detail per period, and per-market detail.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

FONT = "Arial"
RUNS = [
    ("RNN operational", "outputs/afex_operational/RNN"),
    ("GRU operational", "outputs/afex_operational/GRU"),
    ("RNN safeguard-removed", "outputs/afex_safeguard_removed/RNN"),
    ("GRU safeguard-removed", "outputs/afex_safeguard_removed/GRU"),
]

THIN = Side(style="thin", color="000000")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def load_all():
    horizon, period, market, detail = [], [], [], []
    for label, path in RUNS:
        p = Path(path)
        h = pd.read_csv(p / "metrics_by_horizon.csv")
        h["run"] = label
        horizon.append(h)
        pe = pd.read_csv(p / "metrics_by_period.csv")
        pe["run"] = label
        period.append(pe)
        m = pd.read_csv(p / "metrics_by_market.csv")
        m["run"] = label
        market.append(m)
        pp = pd.read_csv(p / "predictions_paired.csv", parse_dates=["origin"])
        kind = label.split(" ")[0]
        pp = pp.rename(columns={kind: "pred"})
        pp["run"] = label
        pp["error_NGN_per_kg"] = pp["pred"] - pp["actual"]
        pp["abs_pct_error"] = (pp["error_NGN_per_kg"] / pp["actual"]).abs() * 100
        pp["target"] = pp["origin"] + pd.to_timedelta(pp["h"], unit="W")
        pp["period"] = pp["target"].dt.to_period("Q").astype(str)
        detail.append(pp)
    return (pd.concat(horizon, ignore_index=True), pd.concat(period, ignore_index=True),
            pd.concat(market, ignore_index=True), pd.concat(detail, ignore_index=True))


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
            if isinstance(v, (pd.Timestamp, datetime)):
                v = v.date()
            elif pd.isna(v):
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
        if c in ("market", "run", "period"):
            width = max(width, 22)
        ws.column_dimensions[get_column_letter(j)].width = min(width, 30)
    ws.auto_filter.ref = f"A{header_row}:{get_column_letter(len(df.columns))}{header_row + len(df)}"


def write_readme(wb, counts):
    ws = wb.create_sheet("README", 0)
    lines = [
        ("AFEX multi-commodity farmgate panel: predicted vs actual, all runs", True),
        ("", False),
        (f"Generated {datetime.now():%Y-%m-%d %H:%M}", False),
        ("Source panel: 20260824_AFEX_MultiCommodity_Weekly_Panel_v1.xlsx, 51 series across 18 "
         "markets and 7 commodities, price only (no diesel/rainfall/NDVI/upstream data exists "
         "for this panel), 2021-04-07 to 2026-07-22 (277 weeks).", False),
        ("No incumbent forecast file exists for this panel (unlike the main FEWSNET evaluation), "
         "so there is no panel-FE comparison here. The benchmark is a plain carry-forward "
         "\"naive\" prediction: predicted price = origin-week price, unchanged.", False),
        ("All 51 series (7 commodities) train the pooled embedding; only the 16 maize series "
         "(one, Dandume | Maize, dropped -- see below) are scored, since maize is the stated "
         "priority. This mirrors DECISIONS D-03's precedent of a series training without being "
         "scored.", False),
        ("Dandume | Maize clears the source panel's own 112-week entry bar but has zero usable "
         "78-week windows after new-crop removal (flagged in the source file's own "
         "Build_Decisions sheet). Dropped from both training and scoring.", False),
        ("", False),
        ("FOUR RUNS", True),
        ("RNN / GRU operational       same hyperparameters as configs/build2.yaml's unconditional arm.", False),
        ("RNN / GRU safeguard-removed same hyperparameters as configs/build3.yaml (capacity doubled, "
         "dropout/weight decay/recency-weighting/most of the purge window removed).", False),
        ("On the MAIN panel, safeguard-removed improved MAE at every horizon (DECISIONS D-23). "
         "On THIS panel it is worse at every horizon for both architectures -- the opposite "
         "result. Plausible reading: the main panel's under-learning diagnosis does not carry "
         "over to a panel this much shorter (277 weeks vs 12 years) and thinner per series; here "
         "the safeguards built against overfitting may be doing real work.", False),
        ("", False),
        ("A NOTE ON THE VALIDATION SPLIT", True),
        ("In several \"operational\" cuts the validation set is LARGER than the training set "
         "(e.g. RNN operational cut 11: train=212, val=397). Mechanism: the purge step only "
         "removes TRAINING windows near the validation cutoff; validation itself is always a "
         "fixed 15% of the pre-purge total. With purge_weeks=78 on a panel this short, and many "
         "series' histories concentrated unevenly across the five-year span, purge can remove a "
         "very large share of the training pool while validation stays fixed. This did not "
         "happen on the main 12-year panel, where the training pool at any cut was large enough "
         "that a 78-week purge was a small fraction of it. Early-stopping decisions in these "
         "cuts should be read cautiously; the safeguard-removed runs (purge=26) do not show this "
         "issue.", False),
        ("", False),
        ("SHEETS", True),
        ("Summary_By_Horizon    MAE/MAPE per run and horizon, challenger vs the naive benchmark.", False),
        ("By_Period             the same, split by calendar quarter of the TARGET week.", False),
        ("By_Market             the same, split by maize series (market).", False),
        ("Predicted_vs_Actual   every scored forecast: origin, target, predicted price, actual "
         "price, error, for all four runs. Sort or filter by run, market or period.", False),
        ("", False),
        ("COLUMN NOTES", True),
        ("error_NGN_per_kg     predicted minus actual. Positive means the model forecast too high.", False),
        ("abs_pct_error        absolute error as a percentage of the actual price (this is MAPE's "
         "per-row ingredient).", False),
        ("vs_naive_pct         positive means the challenger beat plain carry-forward.", False),
    ]
    for i, (text, bold) in enumerate(lines, start=1):
        c = ws.cell(row=i, column=1, value=text)
        c.font = Font(name=FONT, size=11, bold=bold)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 120
    r = len(lines) + 2
    ws.cell(row=r, column=1, value="ROW COUNTS").font = Font(name=FONT, size=11, bold=True)
    for k, v in counts.items():
        r += 1
        ws.cell(row=r, column=1, value=f"{k}: {v}").font = Font(name=FONT, size=11)


def main():
    horizon, period, market, detail = load_all()

    wb = Workbook()
    wb.remove(wb.active)

    write_sheet(wb, "Summary_By_Horizon", horizon[["run", "h", "n", "challenger_MAE",
                "challenger_MAPE", "naive_MAE", "naive_MAPE", "vs_naive_pct"]],
                notes=["MAE in NGN/kg, MAPE in percent. vs_naive_pct positive means the challenger beat carry-forward."],
                num_cols=("challenger_MAE", "naive_MAE"), pct_cols=("challenger_MAPE", "naive_MAPE", "vs_naive_pct"))

    write_sheet(wb, "By_Period", period[["run", "h", "period", "n", "challenger_MAE",
                "challenger_MAPE", "naive_MAE", "naive_MAPE"]],
                notes=["Split by calendar quarter of the TARGET week. Cell counts are small in some quarters; check n."],
                num_cols=("challenger_MAE", "naive_MAE"), pct_cols=("challenger_MAPE", "naive_MAPE"))

    write_sheet(wb, "By_Market", market[["run", "h", "market", "n", "challenger_MAE",
                "challenger_MAPE", "naive_MAE", "naive_MAPE"]],
                notes=["Split by maize series. Dandume | Maize is absent (see README)."],
                num_cols=("challenger_MAE", "naive_MAE"), pct_cols=("challenger_MAPE", "naive_MAPE"))

    pv = detail[["run", "market", "h", "origin", "target", "origin_price", "actual", "pred",
                "naive", "error_NGN_per_kg", "abs_pct_error", "period", "pred_sd"]].sort_values(
                ["run", "market", "h", "origin"])
    write_sheet(wb, "Predicted_vs_Actual", pv,
                notes=["Every scored forecast, all four runs. pred_sd is the spread across the seven seeds."],
                num_cols=("origin_price", "actual", "pred", "naive", "error_NGN_per_kg", "pred_sd"),
                pct_cols=("abs_pct_error",))

    write_readme(wb, {"runs": len(RUNS), "horizon rows": len(horizon),
                      "period rows": len(period), "market rows": len(market),
                      "predicted-vs-actual rows": len(pv)})

    stamp = datetime.now().strftime("%Y%m%d")
    out = f"outputs/{stamp}_AFEX_PredictedVsActual_OperationalVsSafeguardRemoved_v1.xlsx"
    wb.save(out)
    print(f"wrote {out}")
    print(f"rows: horizon={len(horizon)} period={len(period)} market={len(market)} detail={len(pv)}")


if __name__ == "__main__":
    main()
