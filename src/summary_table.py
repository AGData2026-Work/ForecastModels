"""
Consolidated MAPE and MAE table: every completed run against the incumbent, at
every horizon, on identical pairs.

    python src/summary_table.py
    python src/summary_table.py --metric MAE
    python src/summary_table.py --csv outputs/20260818_MAPE_summary.csv

Reference rows (panel_fe, naive) are computed straight from the baseline forecast
file on the same 1,592-pair grid the builds use, so the table is complete and
correct even before any challenger has finished. Challenger rows appear as they
complete; missing runs show as blank rather than being silently omitted, so it is
obvious what has not been run.

Any row whose run was tagged `smoke: true` is labelled and excluded from the
change-control column. A smoke run covers a fraction of the grid and is not
comparable to a full one; treating it as a result is the grid artefact that cost
the prior session a reversed headline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from data import load_grid
from metrics import direction_pct, mae, mape

HORIZONS = [4, 13, 26]
REF = ("panel_fe", "naive")


def reference_rows(grid_path: str, drop_targets: list[str]) -> pd.DataFrame:
    g, log = load_grid(grid_path, HORIZONS, drop_targets=drop_targets)
    rows = []
    for name in REF:
        for h in HORIZONS:
            a = g[f"actual_h{h}"]
            p = g[f"{name}_h{h}"]
            rows.append(dict(build="reference", model=name, convention="n/a",
                             smoke=False, h=h, n=len(g),
                             MAPE=mape(a, p), MAE=mae(a, p),
                             direction_pct=direction_pct(a, p, g["origin_price"])))
    return pd.DataFrame(rows), log


def run_rows(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.glob("*/*/predictions_paired.csv")):
        meta_path = p.parent / "run_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        df = pd.read_csv(p, parse_dates=["origin"])
        kind = meta.get("kind") or next(c for c in df.columns if c in ("RNN", "GRU"))
        for h in HORIZONS:
            s = df[df.h == h]
            if not len(s):
                continue
            rows.append(dict(build=p.parent.parent.name, model=kind,
                             convention=meta.get("convention", "?"),
                             smoke=bool(meta.get("smoke", False)), h=h, n=len(s),
                             MAPE=mape(s.actual, s[kind]), MAE=mae(s.actual, s[kind]),
                             direction_pct=direction_pct(s.actual, s[kind], s.origin_price),
                             panel_fe_MAE_same_pairs=mae(s.actual, s.panel_fe),
                             panel_fe_MAPE_same_pairs=mape(s.actual, s.panel_fe)))
    return pd.DataFrame(rows)


def expected_rows() -> pd.DataFrame:
    out = []
    for b in ("build1_replication", "build2_best_practice"):
        for k in ("RNN", "GRU"):
            for c in ("unconditional", "conditional"):
                for h in HORIZONS:
                    out.append(dict(build=b, model=k, convention=c, h=h))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--grid", default="data/07_panel_fe_forecasts.parquet")
    ap.add_argument("--drop-targets", nargs="*", default=["2019-04-24"])
    ap.add_argument("--metric", default="MAPE", choices=["MAPE", "MAE", "direction_pct"])
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    ref, glog = reference_rows(a.grid, a.drop_targets)
    runs = run_rows(Path(a.outdir))
    exp = expected_rows()
    if len(runs):
        full = exp.merge(runs, on=["build", "model", "convention", "h"], how="left")
    else:
        full = exp.assign(smoke=False, n=pd.NA, MAPE=np.nan, MAE=np.nan,
                          direction_pct=np.nan,
                          panel_fe_MAE_same_pairs=np.nan,
                          panel_fe_MAPE_same_pairs=np.nan)
    full["smoke"] = full["smoke"].fillna(False).astype(bool)
    both = pd.concat([ref, full], ignore_index=True)

    # pivot_table would cross the index levels and invent rows like
    # "build1 / panel_fe / conditional". Build the row key explicitly instead.
    both["row"] = both.apply(
        lambda r: f"{r['model']}" if r["build"] == "reference"
        else f"{r['build'].replace('_', ' ')} | {r['model']} | {r['convention']}"
             + (" | SMOKE" if r["smoke"] else ""), axis=1)
    order = both.drop_duplicates("row")["row"].tolist()
    piv = both.pivot(index="row", columns="h", values=a.metric).reindex(order)
    piv.columns = [f"h={c}" for c in piv.columns]
    npiv = both.pivot(index="row", columns="h", values="n").reindex(order)

    print(f"\n{a.metric} by horizon. Reference rows use n={glog['n_market_origin_pairs']} "
          f"pairs per horizon.")
    print("Blank challenger rows have not been run. Rows marked SMOKE cover only a "
          "fraction of the grid and are NOT comparable to the reference.\n")
    out = piv.round(2).copy()
    out["n"] = pd.to_numeric(npiv[npiv.columns[0]], errors="coerce").astype("Int64")
    print(out.to_string(na_rep="   --  "))

    done = runs[~runs.smoke] if len(runs) else runs
    if len(done):
        print("\nChallenger vs incumbent on identical pairs "
              "(positive = challenger better):")
        d = done.assign(
            vs_panel_fe_MAE_pct=lambda x: (x.panel_fe_MAE_same_pairs - x.MAE)
            / x.panel_fe_MAE_same_pairs * 100,
            vs_panel_fe_MAPE_pct=lambda x: (x.panel_fe_MAPE_same_pairs - x.MAPE)
            / x.panel_fe_MAPE_same_pairs * 100)
        print(d[["build", "model", "convention", "h", "n", "MAPE", "MAE",
                 "vs_panel_fe_MAPE_pct", "vs_panel_fe_MAE_pct"]]
              .round(2).to_string(index=False))
        print("\nThe Section 5.3 gate is on MAE at h=4 and h=13, not MAPE. "
              "MAPE is a Section 5.4 secondary metric.")
    else:
        print("\nNo completed full runs yet. Run `bash run_all.sh` then re-run this.")

    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        both.to_csv(a.csv, index=False)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
