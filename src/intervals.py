"""
Split conformal prediction intervals: distribution-free, no retraining, a
coverage guarantee rather than an average.

    python src/intervals.py
    python src/intervals.py --alphas 0.20 0.10 --calib-frac 0.6

Method, per run and per horizon
--------------------------------
1. Split ORIGIN DATES chronologically, never shuffled: the first
   --calib-frac (default 0.6, not specified in the source task -- see
   docs/DECISIONS.md D-24 for why 0.6 was chosen) become calibration, the
   rest test.
2. Residuals are normalised by the origin price before quantiles are taken:
   s = |pred - actual| / origin_price. Skipping this makes intervals far too
   narrow in 2023-24 and far too wide in 2015-19, because absolute errors
   scale with the price level (28 to 1,221 NGN/kg across the panel).
3. qhat = quantile(s_calib, min(ceil((n+1)*(1-alpha))/n, 1.0)), the
   finite-sample-correct split conformal quantile.
4. Interval = pred +/- qhat * origin_price.
5. Realised coverage and mean width (NGN/kg) are reported on the TEST
   portion against the nominal 1-alpha, for alpha in {0.20, 0.10}.

Three uncertainty components, reported separately rather than collapsed:
  Model uncertainty    seed_MAE_sd from outputs/seed_analysis.csv (Task 4).
  Data uncertainty      conformal interval width, this script.
  Regime uncertainty    realised coverage computed separately for 2015-19,
                        2020-22, 2023-24 (this script, outputs/intervals_by_regime.csv).
                        Rows inside the calibration window are flagged
                        in-sample; only out-of-sample coverage validates the
                        method.

Per-market caution: ~106 forecasts per market is thin for a market-specific
quantile, so this uses the POOLED (run, horizon) qhat with no per-market
refit -- per-market coverage in outputs/intervals_by_market.csv is a
diagnostic on the pooled interval, not a separately calibrated one.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REGIME_PERIODS = [
    ("2015-19", "2015-01-01", "2019-12-31"),
    ("2020-22", "2020-01-01", "2022-12-31"),
    ("2023-24", "2023-01-01", "2024-12-31"),
]


def period_of(ts: pd.Timestamp) -> str:
    for label, lo, hi in REGIME_PERIODS:
        if pd.Timestamp(lo) <= ts <= pd.Timestamp(hi):
            return label
    return "other"


def load_runs(root: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(root.glob("*/*/predictions_paired.csv")):
        meta_path = p.parent / "run_metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("smoke"):
            continue
        df = pd.read_csv(p, parse_dates=["origin"])
        kind = meta.get("kind") or next(c for c in df.columns if c in ("RNN", "GRU"))
        build = meta.get("build", p.parent.parent.name)
        conv = meta.get("convention", "unknown")
        df = df.rename(columns={kind: "pred"})
        df["model"] = kind
        df["build"] = build
        df["convention"] = conv
        df["run"] = f"{build}|{kind}|{conv}"
        df["target"] = df["origin"] + pd.to_timedelta(df["h"], unit="W")
        frames.append(df)
    if not frames:
        raise SystemExit(f"no completed runs under {root}")
    return pd.concat(frames, ignore_index=True)


def qhat(s: np.ndarray, alpha: float) -> float:
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(s, min(k, 1.0)))


def split_calib_test(g: pd.DataFrame, calib_frac: float):
    origins = np.sort(g["origin"].unique())
    cut_idx = max(1, int(len(origins) * calib_frac))
    cutoff = pd.Timestamp(origins[cut_idx - 1])
    return g[g["origin"] <= cutoff], g[g["origin"] > cutoff], cutoff


def analyse_run_horizon(g: pd.DataFrame, alphas: list[float], calib_frac: float):
    calib, test, cutoff = split_calib_test(g, calib_frac)
    if not len(calib) or not len(test):
        return [], [], []
    calib_s = ((calib["pred"] - calib["actual"]).abs() / calib["origin_price"]).values

    headline_rows, regime_rows, market_rows = [], [], []
    for alpha in alphas:
        q = qhat(calib_s, alpha)
        allrows = g.copy()
        allrows["halfwidth"] = q * allrows["origin_price"]
        allrows["lo"] = allrows["pred"] - allrows["halfwidth"]
        allrows["hi"] = allrows["pred"] + allrows["halfwidth"]
        allrows["covered"] = (allrows["actual"] >= allrows["lo"]) & (allrows["actual"] <= allrows["hi"])
        allrows["in_calibration"] = allrows["origin"] <= cutoff

        test_rows = allrows[~allrows["in_calibration"]]
        headline_rows.append(dict(
            alpha=alpha, nominal_coverage_pct=(1 - alpha) * 100, qhat=q,
            calib_cutoff=str(cutoff.date()), n_calib=len(calib), n_test=len(test_rows),
            realised_coverage_test_pct=float(test_rows["covered"].mean() * 100),
            mean_width_NGN_per_kg_test=float((2 * test_rows["halfwidth"]).mean()),
        ))

        allrows["regime"] = allrows["target"].apply(period_of)
        for regime, gr in allrows.groupby("regime"):
            regime_rows.append(dict(
                alpha=alpha, regime=regime, n=len(gr),
                pct_in_calibration=float(gr["in_calibration"].mean() * 100),
                realised_coverage_pct=float(gr["covered"].mean() * 100),
                mean_width_NGN_per_kg=float((2 * gr["halfwidth"]).mean()),
            ))

        for market, gm in test_rows.groupby("market"):
            market_rows.append(dict(
                alpha=alpha, market=market, n=len(gm),
                realised_coverage_pct=float(gm["covered"].mean() * 100),
                mean_width_NGN_per_kg=float((2 * gm["halfwidth"]).mean()),
            ))
    return headline_rows, regime_rows, market_rows


def attach_model_uncertainty(headline: pd.DataFrame, seed_analysis_path: Path) -> pd.DataFrame:
    if not seed_analysis_path.exists():
        headline["seed_MAE_sd_model_uncertainty"] = np.nan
        return headline
    sa = pd.read_csv(seed_analysis_path)
    sa = sa.rename(columns={"h": "h"})[["build", "model", "convention", "h", "seed_MAE_sd"]]
    sa = sa.rename(columns={"seed_MAE_sd": "seed_MAE_sd_model_uncertainty"})
    return headline.merge(sa, on=["build", "model", "convention", "h"], how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.20, 0.10])
    ap.add_argument("--calib-frac", type=float, default=0.6,
                    help="share of chronological origins used for calibration")
    ap.add_argument("--out", default="outputs/intervals_summary.csv")
    ap.add_argument("--regime-out", default="outputs/intervals_by_regime.csv")
    ap.add_argument("--market-out", default="outputs/intervals_by_market.csv")
    a = ap.parse_args()

    df = load_runs(Path(a.outdir))
    headline, regime, market = [], [], []
    for (run, build, model, conv, h), g in df.groupby(
            ["run", "build", "model", "convention", "h"]):
        hl, rg, mk = analyse_run_horizon(g, a.alphas, a.calib_frac)
        for r in hl:
            r.update(build=build, model=model, convention=conv, h=int(h))
            headline.append(r)
        for r in rg:
            r.update(build=build, model=model, convention=conv, h=int(h))
            regime.append(r)
        for r in mk:
            r.update(build=build, model=model, convention=conv, h=int(h))
            market.append(r)

    headline_df = pd.DataFrame(headline).sort_values(
        ["build", "convention", "model", "h", "alpha"]).reset_index(drop=True)
    headline_df = attach_model_uncertainty(headline_df, Path(a.outdir) / "seed_analysis.csv")
    regime_df = pd.DataFrame(regime).sort_values(
        ["build", "convention", "model", "h", "alpha", "regime"]).reset_index(drop=True)
    market_df = pd.DataFrame(market).sort_values(
        ["build", "convention", "model", "h", "alpha", "market"]).reset_index(drop=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    headline_df.to_csv(a.out, index=False)
    regime_df.to_csv(a.regime_out, index=False)
    market_df.to_csv(a.market_out, index=False)

    print("HEADLINE (test portion, out-of-sample)")
    print(headline_df[["build", "model", "convention", "h", "alpha",
                       "nominal_coverage_pct", "realised_coverage_test_pct",
                       "mean_width_NGN_per_kg_test",
                       "seed_MAE_sd_model_uncertainty"]].round(2).to_string(index=False))

    print("\nREGIME COVERAGE (2023-24 expected to be poor; report, do not widen to fix)")
    show = regime_df[regime_df["alpha"] == a.alphas[0]]
    print(show[["build", "model", "convention", "h", "regime", "n",
               "pct_in_calibration", "realised_coverage_pct",
               "mean_width_NGN_per_kg"]].round(2).to_string(index=False))

    poor = regime_df[(regime_df["regime"] == "2023-24") &
                     (regime_df["pct_in_calibration"] < 50) &
                     (regime_df["realised_coverage_pct"] <
                      (100 - regime_df["alpha"] * 100) - 10)]
    if len(poor):
        print(f"\n{len(poor)} out-of-sample 2023-24 rows undercover by more than 10 points "
              "versus nominal -- a true statement about a regime break, not something to widen away.")

    print(f"\nwrote {a.out}\nwrote {a.regime_out}\nwrote {a.market_out}")


if __name__ == "__main__":
    main()
