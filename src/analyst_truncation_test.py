"""
Look-ahead check of the analyst's model code (2026-10-29), the counterpart of src/truncation_test_monthly.py for GRU.

    python src/analyst_truncation_test.py --inputs data/analyst_inputs \
        --origins 2018-03 2020-07 2022-11 2024-06 2024-12 2025-05 2025-09 2026-02 --out outputs/joint_forecast/production_checks

For each test origin o, every input his code reads (white and yellow prices, observed flags, macro, weekly quotes,
sorghum) is cut off after month o (weekly quotes after the last day of o), his model is rerun from scratch on that
truncated data, and its forecasts made at o are compared with the full-data run. Identical forecasts mean his code
uses nothing dated after the forecast month. It cannot test publication timing inside month o+1 (when each of month
o's inputs became public); that is in the timing audit (docs/PUBLICATION_LAGS.md).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "analyst_v4"


def his_inputs(workbook: str) -> dict:
    """Run only the input-reading part of his refit script; return its D_OVERRIDE dict."""
    src = (VENDOR / "refit_feature_model_v4.py").read_text()
    src = src.split("src = open(os.path.join(here, 'v2_models.py'))")[0]
    src = src.replace("path = sys.argv[1]", f"path = {workbook!r}")
    src = src.replace("here = os.path.dirname(os.path.abspath(__file__))", f"here = {str(VENDOR)!r}")
    g = {"__name__": "analyst_truncation"}
    exec(compile(src, "refit_feature_model_v4.py", "exec"), g)
    g["wb"].close()
    return g["D_OVERRIDE"]


def run_model(D: dict) -> pd.DataFrame:
    src = (VENDOR / "v2_models.py").read_text().split("pd.to_pickle(dict(panel=pn")[0]
    src = src.replace("sys.path.insert(0, '/home/claude'); os.chdir('/home/claude'); ", "")
    g = {"D_OVERRIDE": D, "__name__": "analyst_truncation"}
    cwd = os.getcwd()
    try:
        exec(compile(src, "v2_models.py", "exec"), g)
    finally:
        os.chdir(cwd)
    return g["pn"]


def truncate(D: dict, o: pd.Timestamp) -> dict:
    T = dict(D)
    keep = D["MONTHS"][D["MONTHS"] <= o]
    for k in ("white", "yellow", "flags", "sorghum"):
        T[k] = D[k].loc[D[k].index <= o]
    T["macro"] = D["macro"].loc[D["macro"].index <= o]
    T["weekly"] = D["weekly"][D["weekly"].end <= o + pd.offsets.MonthEnd(0)]
    T["MONTHS"] = keep
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--inputs", help="folder of his CSV inputs (src/analyst_inputs.py)")
    src.add_argument("--workbook", help="legacy: his Excel workbook")
    ap.add_argument("--origins", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.inputs:
        from analyst_inputs import load_inputs
        D = load_inputs(a.inputs)
    else:
        D = his_inputs(str(Path(a.workbook).resolve()))
    full = run_model(D)
    rows = []
    for o in a.origins:
        o = pd.Timestamp(o + "-01")
        tr = run_model(truncate(D, o))
        f = full[(full.origin == o) & (full.type == "White")].set_index("market")
        t = tr[(tr.origin == o) & (tr.type == "White")].set_index("market")
        for h in (1, 3, 6):
            for code in ("FM", "BL", "RS"):
                x = pd.concat([f[f"{code}_{h}"], t[f"{code}_{h}"]], axis=1, keys=["full", "trunc"]).dropna(how="all")
                diff = (x.trunc / x.full - 1).abs()
                rows.append(dict(origin=o.strftime("%Y-%m"), h=h, model=code, n_markets=int(x.full.notna().sum()),
                                 n_missing_in_truncated=int((x.full.notna() & x.trunc.isna()).sum()),
                                 max_rel_diff=float(np.nanmax(diff.values)) if diff.notna().any() else np.nan))
        print(o.strftime("%Y-%m"), "done")
    t = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t.to_csv(Path(a.out) / "4_analyst_truncation_test.csv", index=False)
    print(t.groupby(["origin"]).agg(max_rel_diff=("max_rel_diff", "max"), missing=("n_missing_in_truncated", "sum")).to_string())
    print("overall max relative difference:", t.max_rel_diff.max())


if __name__ == "__main__":
    main()
