"""
Directional accuracy against the always-up / always-down benchmark, not 50%.

Prices in this panel trend upward, strongly from 2023, so a model that always
predicted "up" would score well above 50% while knowing nothing. Comparing
reported directional accuracy against a coin flip is the wrong reference; this
compares it against the share of windows that actually rose (and, for
completeness, the share that fell).

    python src/directional_benchmark.py
    python src/directional_benchmark.py --outdir outputs --out outputs/directional_benchmark.csv

Reads every completed run's predictions_paired.csv directly (repo mode, same
as error_analysis.py). Periods bucket by TARGET date, since the question is
"what did the price actually do during this period", not when the forecast
was issued.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PERIODS = [
    ("2015-19", "2015-01-01", "2019-12-31"),
    ("2020-22", "2020-01-01", "2022-12-31"),
    ("2023-24", "2023-01-01", "2024-12-31"),
]


def load_runs(root: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(root.glob("*/*/predictions_paired.csv")):
        meta_path = p.parent / "run_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("smoke"):
            continue
        df = pd.read_csv(p, parse_dates=["origin"])
        kind = meta.get("kind") or next(c for c in df.columns if c in ("RNN", "GRU"))
        conv = meta.get("convention", "unknown")
        build = meta.get("build", p.parent.parent.name)
        df = df.rename(columns={kind: "pred"})
        df["model"] = kind
        df["convention"] = conv
        df["build"] = build
        df["run"] = f"{build}|{kind}|{conv}"
        df["target"] = df["origin"] + pd.to_timedelta(df["h"], unit="W")
        frames.append(df)
    if not frames:
        raise SystemExit(f"no completed runs under {root}")
    return pd.concat(frames, ignore_index=True)


def period_of(ts: pd.Timestamp) -> str:
    for label, lo, hi in PERIODS:
        if pd.Timestamp(lo) <= ts <= pd.Timestamp(hi):
            return label
    return "other"


def _direction_pct(actual, pred, origin_price) -> tuple[float, int]:
    """Model directional accuracy, and n windows where direction is defined
    (excludes windows where the actual price did not move at all)."""
    da = np.sign(np.asarray(actual, float) - np.asarray(origin_price, float))
    dp = np.sign(np.asarray(pred, float) - np.asarray(origin_price, float))
    ok = da != 0
    if not ok.any():
        return float("nan"), 0
    return float(np.mean(da[ok] == dp[ok]) * 100), int(ok.sum())


def summarise(g: pd.DataFrame) -> dict:
    rose = g["actual"] > g["origin_price"]
    fell = g["actual"] < g["origin_price"]
    flat = ~rose & ~fell
    model_pct, n_directional = _direction_pct(g["actual"], g["pred"], g["origin_price"])
    up_share = float(rose.mean() * 100)
    down_share = float(fell.mean() * 100)
    return dict(
        n=len(g), n_directional=n_directional, n_flat=int(flat.sum()),
        always_up_share_pct=up_share, always_down_share_pct=down_share,
        model_direction_pct=model_pct,
        margin_vs_always_up_pp=model_pct - up_share,
        margin_vs_always_down_pp=model_pct - down_share,
        directional_claim_valid=bool(model_pct > max(up_share, down_share)),
    )


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, build, model, conv, h), g in df.groupby(
            ["run", "build", "model", "convention", "h"]):
        base = dict(build=build, model=model, convention=conv, h=int(h), scope="overall", scope_value="")
        base.update(summarise(g))
        rows.append(base)

        g2 = g.assign(period=g["target"].apply(period_of))
        for period, gp in g2.groupby("period"):
            base = dict(build=build, model=model, convention=conv, h=int(h),
                       scope="period", scope_value=period)
            base.update(summarise(gp))
            rows.append(base)

        for market, gm in g.groupby("market"):
            base = dict(build=build, model=model, convention=conv, h=int(h),
                       scope="market", scope_value=market)
            base.update(summarise(gm))
            rows.append(base)
    return pd.DataFrame(rows).sort_values(
        ["build", "convention", "model", "h", "scope", "scope_value"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--out", default="outputs/directional_benchmark.csv")
    a = ap.parse_args()

    df = load_runs(Path(a.outdir))
    table = build_table(df)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)

    overall = table[table["scope"] == "overall"]
    print(overall[["build", "model", "convention", "h", "n", "always_up_share_pct",
                  "model_direction_pct", "margin_vs_always_up_pp",
                  "directional_claim_valid"]].round(2).to_string(index=False))

    void = overall[~overall["directional_claim_valid"]]
    if len(void):
        print(f"\nVOID at overall scope ({len(void)} run/horizon combinations): "
              "model directional accuracy does not exceed the always-up (or "
              "always-down) share.")
        print(void[["build", "model", "convention", "h", "always_up_share_pct",
                    "model_direction_pct"]].round(2).to_string(index=False))
    else:
        print("\nNo overall-scope combination is void: every run beats the "
              "stronger of always-up/always-down at every horizon.")

    void_scoped = table[~table["directional_claim_valid"]]
    print(f"\n{len(void_scoped)} of {len(table)} rows across all scopes "
          "(overall + period + market) are void.")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
