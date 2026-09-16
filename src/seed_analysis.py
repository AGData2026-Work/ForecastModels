"""
Run-to-run (seed) variation of the HEADLINE METRIC, not of individual predictions.

predictions_paired.csv's pred_sd column is the spread of the model's individual
predictions across seeds. That is a different and smaller number than the
spread of MAE across seeds, by roughly the square root of the sample size.
Decisions need the latter: a margin over the incumbent that is smaller than
the run's own seed-to-seed MAE spread is not a result.

    python src/seed_analysis.py
    python src/seed_analysis.py --outdir outputs --out outputs/seed_analysis.csv

Reads every completed (non-smoke) run's forecasts.csv, the per-seed,
per-origin, per-horizon prediction file, directly (repo mode).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def mae(a, p) -> float:
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(p, float))))


def load_runs(root: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(root.glob("*/*/forecasts.csv")):
        meta_path = p.parent / "run_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("smoke"):
            continue
        df = pd.read_csv(p, parse_dates=["origin"])
        build = df["build"].iloc[0]
        model = df["model"].iloc[0]
        conv = df["convention"].iloc[0] if "convention" in df.columns else "n/a"
        df["convention"] = conv
        df["run"] = f"{build}|{model}|{conv}"
        frames.append(df)
    if not frames:
        raise SystemExit(f"no completed runs under {root}")
    return pd.concat(frames, ignore_index=True)


def per_seed_mae(g: pd.DataFrame) -> pd.Series:
    """One MAE per seed, pooled over all markets and origins in this (run, h)."""
    return g.groupby("seed")[["actual", "pred"]].apply(lambda s: mae(s["actual"], s["pred"]))


def median_ensemble_mae(g: pd.DataFrame, seeds: list) -> float:
    piv = g.pivot_table(index=["market", "origin"], columns="seed", values="pred")
    actual = g.drop_duplicates(["market", "origin"]).set_index(["market", "origin"])["actual"]
    med = piv[seeds].median(axis=1)
    return mae(actual.loc[med.index], med)


def seed_count_curve(g: pd.DataFrame, seeds: list) -> pd.DataFrame:
    """For k = 1..len(seeds), every k-subset of seeds (exact enumeration --
    at most C(7,3)=35 -- median-ensembled, then mean/sd of that MAE across
    subsets of the same size."""
    piv = g.pivot_table(index=["market", "origin"], columns="seed", values="pred")
    actual = g.drop_duplicates(["market", "origin"]).set_index(["market", "origin"])["actual"]
    rows = []
    for k in range(1, len(seeds) + 1):
        maes = []
        for combo in itertools.combinations(seeds, k):
            med = piv[list(combo)].median(axis=1)
            maes.append(mae(actual.loc[med.index], med))
        rows.append(dict(k=k, n_subsets=len(maes),
                         mean_mae=float(np.mean(maes)), sd_mae=float(np.std(maes))))
    return pd.DataFrame(rows)


def per_market_stability(g: pd.DataFrame, seeds: list) -> pd.DataFrame:
    """Per market: does the beats-naive verdict change depending on which
    single seed's fit is used?"""
    rows = []
    for market, gm in g.groupby("market"):
        naive_row = gm.drop_duplicates(["origin"]).set_index("origin")["naive"]
        actual_row = gm.drop_duplicates(["origin"]).set_index("origin")["actual"]
        naive_mae_m = mae(actual_row, naive_row)
        verdicts = {}
        for seed in seeds:
            gs = gm[gm["seed"] == seed]
            if not len(gs):
                continue
            verdicts[seed] = mae(gs["actual"], gs["pred"]) < naive_mae_m
        rows.append(dict(market=market, n_seeds_checked=len(verdicts),
                         verdict_flips=len(set(verdicts.values())) > 1,
                         n_seeds_beating_naive=sum(verdicts.values())))
    return pd.DataFrame(rows)


def analyse_run_horizon(g: pd.DataFrame) -> dict:
    seeds = sorted(g["seed"].unique())
    ps_mae = per_seed_mae(g)
    naive_row = g.drop_duplicates(["market", "origin"])
    naive_mae = mae(naive_row["actual"], naive_row["naive"])
    median_mae = median_ensemble_mae(g, seeds)

    stability = per_market_stability(g, seeds)
    n_flip = int(stability["verdict_flips"].sum())

    margin = naive_mae - median_mae
    seed_sd = float(ps_mae.std())
    ratio = margin / seed_sd if seed_sd > 0 else float("nan")

    return dict(
        n_seeds=len(seeds),
        naive_MAE=naive_mae,
        seed_MAE_mean=float(ps_mae.mean()),
        seed_MAE_sd=seed_sd,
        seed_MAE_min=float(ps_mae.min()),
        seed_MAE_max=float(ps_mae.max()),
        seed_MAE_range=float(ps_mae.max() - ps_mae.min()),
        median_ensemble_MAE=median_mae,
        ensemble_gain_vs_mean_of_seed_MAE=float(ps_mae.mean() - median_mae),
        margin_over_naive_at_median=margin,
        margin_over_naive_over_seed_sd=ratio,
        margin_below_1_seed_sd=bool(abs(ratio) < 1) if np.isfinite(ratio) else None,
        n_markets_with_flipping_verdict=n_flip,
        n_markets_total=len(stability),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--out", default="outputs/seed_analysis.csv")
    ap.add_argument("--curve-out", default="outputs/seed_count_curve.csv")
    a = ap.parse_args()

    df = load_runs(Path(a.outdir))

    rows, curves = [], []
    for (run, build, model, conv, h), g in df.groupby(
            ["run", "build", "model", "convention", "h"]):
        base = dict(build=build, model=model, convention=conv, h=int(h))
        base.update(analyse_run_horizon(g))
        rows.append(base)

        seeds = sorted(g["seed"].unique())
        curve = seed_count_curve(g, seeds)
        curve["build"], curve["model"], curve["convention"], curve["h"] = build, model, conv, int(h)
        curves.append(curve)

    table = pd.DataFrame(rows).sort_values(["build", "convention", "model", "h"]).reset_index(drop=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)

    curve_table = pd.concat(curves, ignore_index=True).sort_values(
        ["build", "convention", "model", "h", "k"]).reset_index(drop=True)
    curve_table.to_csv(a.curve_out, index=False)

    print(table[["build", "model", "convention", "h", "n_seeds", "seed_MAE_mean",
                "seed_MAE_sd", "seed_MAE_range", "median_ensemble_MAE",
                "ensemble_gain_vs_mean_of_seed_MAE", "margin_over_naive_over_seed_sd",
                "margin_below_1_seed_sd", "n_markets_with_flipping_verdict",
                "n_markets_total"]].round(3).to_string(index=False))

    below1 = table[table["margin_below_1_seed_sd"] == True]  # noqa: E712
    if len(below1):
        print(f"\n{len(below1)} run/horizon combinations have a margin over naive "
              "below 1 seed-MAE standard deviation -- indistinguishable from seed "
              "noise at this seed count:")
        print(below1[["build", "model", "convention", "h",
                      "margin_over_naive_over_seed_sd"]].round(2).to_string(index=False))

    # At k = n_seeds there is exactly one subset (all seeds), so sd_mae is
    # mechanically zero there regardless of whether the curve has actually
    # flattened. Flattening is read off mean_mae's convergence instead: how
    # much the ensemble estimate still moves from k-1 to k.
    max_k = curve_table["k"].max()
    at_max = curve_table[curve_table["k"] == max_k]
    at_prev = curve_table[curve_table["k"] == max_k - 1]
    cmp = at_max.merge(at_prev, on=["build", "model", "convention", "h"], suffixes=("", "_prev"))
    cmp["mean_mae_pct_change"] = (
        (cmp["mean_mae"] - cmp["mean_mae_prev"]).abs() / cmp["mean_mae_prev"] * 100)
    still_moving = cmp[cmp["mean_mae_pct_change"] > 0.5]
    if len(still_moving):
        print(f"\nEnsemble MAE still moved more than 0.5% from k={max_k-1} to k={max_k} "
              f"for {len(still_moving)} run/horizon combinations -- the seed-count curve "
              f"has not demonstrably flattened by {max_k} seeds for these "
              "(k=7 sd_mae is not informative here: only one 7-of-7 subset exists):")
        print(still_moving[["build", "model", "convention", "h",
                            "mean_mae_prev", "mean_mae", "mean_mae_pct_change"]]
              .round(3).to_string(index=False))
    else:
        print(f"\nEnsemble MAE moved less than 0.5% from k={max_k-1} to k={max_k} "
              "everywhere; no evidence more seeds are needed.")

    print(f"\nwrote {a.out}")
    print(f"wrote {a.curve_out}")


if __name__ == "__main__":
    main()
