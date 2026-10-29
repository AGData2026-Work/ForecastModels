"""
Export the analyst's Inflation Adjusted model forecasts (his v4 workbook model) for the joint forecast (2026-10-26).

    python src/analyst_forecast.py --workbook <Maize_Forecast_v4_White.xlsx> \
        --out outputs/joint_forecast/analyst_forecasts.csv

Runs his own code, unchanged, from vendor/analyst_v4/ (refit_feature_model_v4.py and v2_models.py, received 2026-10-26):
the input-reading part of refit_feature_model_v4.py and all of v2_models.py, which re-learns his weights for every
forecast month from past data only and builds every model's forecasts. It stops before his script writes weights back
into the workbook, so the workbook is only read. Checked on 2026-10-26 against his workbook: weights equal to 4e-6 and
forecasts to 6e-8 (relative) on every shared forecast (outputs/d77_checks/v4_refit_check/).

Output, one row per white-maize market, forecast month (origin) and horizon: p0 (his price at the origin, from his
In_Price tab), his feature model (fm), blend (bl), import switch (rs), his import-window model (iw, only while his window
is on) and the window flag, his recommended model (fm at 1 month, rs at 3 and 6), and the actual price and observed flag
where known. Yellow maize is used only for his training, as in his
workbook. Each month: refresh his workbook's input tabs (his Data_Pipeline tab lists them), then rerun this.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "analyst_v4"
RECOMMENDED = {1: "fm", 3: "rs", 6: "rs"}


AS_PUBLISHED_NOTE = ("headline inflation lagged one month: NBS publishes month o's CPI around the 15th-16th of o+1 "
                     "(docs/PUBLICATION_LAGS.md), so on our forecast date only month o-1's figure is reliably out")


def run_his_model(workbook: str, as_published: bool = True) -> dict:
    """Globals of his v2_models.py after it runs (pn = panel of forecasts, plus every feature table).
    as_published (default, the joint forecast's setting): his headline month-on-month inflation, used only by his
    original V5/V7 models and therefore by his blend and import switch at 3 and 6 months, is lagged one month. His
    code is unchanged; only the input table it reads is shifted. as_published=False reproduces his workbook exactly."""
    src = (VENDOR / "refit_feature_model_v4.py").read_text()
    src = src.split("# column layout of each weights tab")[0]       # stop before it rewrites the workbook
    if as_published:
        anchor = "wk = [r for r in wb['In_Weekly_Prices']"
        assert src.count(anchor) == 1
        src = src.replace(anchor, "mac['headline_mom'] = mac['headline_mom'].shift(1)\n" + anchor)
    src = src.replace("path = sys.argv[1]", f"path = {str(workbook)!r}")
    src = src.replace("here = os.path.dirname(os.path.abspath(__file__))", f"here = {str(VENDOR)!r}")
    g = {"__name__": "analyst_forecast"}
    cwd = os.getcwd()
    try:
        exec(compile(src, str(VENDOR / "refit_feature_model_v4.py"), "exec"), g)
    finally:
        os.chdir(cwd)
    return g["g"]


def export(pn: pd.DataFrame) -> pd.DataFrame:
    w = pn[pn.type == "White"]
    rows = []
    for h in (1, 3, 6):
        d = pd.DataFrame({"market": w.market, "origin": w.origin, "h": h, "target": w[f"tgt{h}"], "p0": w.p0,
                          "fm": w[f"FM_{h}"], "bl": w[f"BL_{h}"], "rs": w[f"RS_{h}"], "iw": w[f"IW_{h}"],
                          "import_window": w.regime, "actual": w[f"A{h}"], "actual_observed": w[f"obs{h}"]})
        d["recommended"] = d[RECOMMENDED[h]]
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    return out[out.recommended.notna()].sort_values(["h", "origin", "market"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", required=True)
    ap.add_argument("--out", default="outputs/joint_forecast/analyst_forecasts.csv")
    ap.add_argument("--as-in-workbook", action="store_true", help="reproduce his workbook exactly (inflation not lagged)")
    a = ap.parse_args()
    out = export(run_his_model(os.path.abspath(a.workbook), as_published=not a.as_in_workbook)["pn"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    last = out.origin.max()
    print(f"{len(out)} forecasts, origins {out.origin.min():%b %Y} to {last:%b %Y}; latest origin: "
          f"{out[out.origin == last].market.nunique()} markets -> {a.out}")


if __name__ == "__main__":
    main()
