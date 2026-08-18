"""
Read finished runs and print comparison tables. Retrains nothing.

    python src/report.py                          # every run found, headline table
    python src/report.py --h 13                   # per-market at one horizon
    python src/report.py --market Gombe           # forecast vs actual detail
    python src/report.py --export                 # -> outputs/comparison.csv
    python src/report.py --gate                   # change-control summary only

Every table states its own n. Two runs with different n are not comparable; the
prior session's largest error came from comparing MAEs computed on different
origin sets, so n is printed everywhere rather than tucked into a footnote.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from metrics import mae, mape, direction_pct


def discover(root: Path) -> list[dict]:
    runs = []
    for p in sorted(root.glob("*/*/predictions_paired.csv")):
        meta_path = p.parent / "run_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        df = pd.read_csv(p, parse_dates=["origin"])
        kind = meta.get("kind") or next(c for c in df.columns if c in ("RNN", "GRU"))
        runs.append(dict(path=p.parent, build=p.parent.parent.name, kind=kind,
                         convention=meta.get("convention", "?"),
                         smoke=meta.get("smoke", False),
                         n_params=meta.get("n_params"), df=df))
    return runs


def headline(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        d, k = r["df"], r["kind"]
        for h in sorted(d.h.unique()):
            g = d[d.h == h]
            rows.append(dict(
                build=r["build"], model=k, convention=r["convention"],
                smoke=r["smoke"], h=int(h), n=len(g),
                MAE=mae(g.actual, g[k]), MAPE=mape(g.actual, g[k]),
                direction_pct=direction_pct(g.actual, g[k], g.origin_price),
                panel_fe_MAE=mae(g.actual, g.panel_fe),
                naive_MAE=mae(g.actual, g.naive),
                vs_panel_fe_pct=(mae(g.actual, g.panel_fe) - mae(g.actual, g[k]))
                / mae(g.actual, g.panel_fe) * 100,
                vs_naive_pct=(mae(g.actual, g.naive) - mae(g.actual, g[k]))
                / mae(g.actual, g.naive) * 100,
                seed_sd_mean=float(g.pred_sd.mean()) if "pred_sd" in g else np.nan))
    return pd.DataFrame(rows).sort_values(["build", "convention", "model", "h"])


def per_market(runs, h: int) -> pd.DataFrame:
    rows = []
    for r in runs:
        d, k = r["df"], r["kind"]
        g = d[d.h == h]
        for m, s in g.groupby("market"):
            rows.append(dict(build=r["build"], model=k, convention=r["convention"],
                             market=m, n=len(s), MAE=mae(s.actual, s[k]),
                             panel_fe_MAE=mae(s.actual, s.panel_fe),
                             beats_panel_fe=mae(s.actual, s[k]) < mae(s.actual, s.panel_fe)))
    return pd.DataFrame(rows).sort_values(["build", "convention", "model", "MAE"])


def detail(runs, market: str) -> pd.DataFrame:
    out = []
    for r in runs:
        d, k = r["df"], r["kind"]
        s = d[d.market == market].copy()
        s["error"] = s[k] - s.actual
        s["abs_pct_error"] = (s.error / s.actual).abs() * 100
        s["build"], s["model"], s["convention"] = r["build"], k, r["convention"]
        out.append(s.rename(columns={k: "pred"})[
            ["build", "model", "convention", "market", "origin", "h",
             "origin_price", "actual", "pred", "panel_fe", "naive",
             "error", "abs_pct_error"]])
    return pd.concat(out).sort_values(["build", "convention", "model", "h", "origin"])


def gate(runs) -> pd.DataFrame:
    rows = []
    for r in runs:
        f = r["path"] / "change_control.json"
        if not f.exists():
            continue
        cc = json.loads(f.read_text())
        row = dict(build=r["build"], model=r["kind"], convention=r["convention"],
                   smoke=r["smoke"], verdict=cc["verdict"])
        for h, d in cc["per_horizon"].items():
            row[f"h{h}_pct_better"] = d["pct_better"]
            row[f"h{h}_pass"] = d["pass"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--market", default=None)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()

    root = Path(a.outdir)
    runs = discover(root)
    if not runs:
        raise SystemExit(f"no completed runs under {root}")

    smoky = [r for r in runs if r["smoke"]]
    if smoky:
        print("WARNING: smoke runs present; these are pipeline checks, not results:")
        for r in smoky:
            print(f"  {r['build']}/{r['kind']}_{r['convention']}")
        print()

    if a.market:
        t = detail(runs, a.market)
        print(t.to_string(index=False))
    elif a.h:
        t = per_market(runs, a.h)
        print(t.round(2).to_string(index=False))
    elif a.gate:
        t = gate(runs)
        print(t.to_string(index=False))
    else:
        t = headline(runs)
        print(t.round(2).to_string(index=False))
        print("\nvs_panel_fe_pct and vs_naive_pct: positive means the challenger is better.")
        print("Change-control gate (Section 5.3): >5% better than panel_fe at BOTH h=4 and h=13.")
        print("\n" + gate(runs).to_string(index=False))

    if a.export:
        p = root / "comparison.csv"
        t.to_csv(p, index=False)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
