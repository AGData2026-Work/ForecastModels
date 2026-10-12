"""
Bias report for any finished run (D-73). Retrains nothing.

    python src/bias_report.py --run-dir outputs/production_fxdaily_pit_extended/GRU_unconditional
    python src/bias_report.py --run-dir outputs/regime_blend_soft_production/GRU --col blended --file predictions_with_blend.csv

Writes bias_metrics.csv into the run directory and prints it, with the target period split
(2015-19, 2020-22, 2023-24 and later) because a balanced total can hide opposite biases.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import bias_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--file", default="predictions_paired.csv")
    ap.add_argument("--col", default=None, help="forecast column (default: GRU or RNN)")
    a = ap.parse_args()
    run = Path(a.run_dir)
    d = pd.read_csv(run / a.file, parse_dates=["origin"])
    col = a.col or next(c for c in ("GRU", "RNN") if c in d.columns)
    b = bias_metrics(d, col)
    b.to_csv(run / "bias_metrics.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"{run}  column {col}\n" + b.to_string(index=False))
    d["target"] = d["origin"] + pd.to_timedelta(d["h"] * 7, unit="D")
    d["period"] = pd.cut(d["target"].dt.year, [2014, 2019, 2022, 2100], labels=["2015-19", "2020-22", "2023 on"])
    print("\nby target period")
    for p_, q in d.groupby("period", observed=True):
        print(f"-- {p_}\n" + bias_metrics(q, col)[["h", "n", "pct_over", "pct_under", "mean_pct_error",
                                                        "ts12_pct_outside_4", "ts12_latest_mean",
                                                        "mz_intercept", "mz_slope", "mz_joint_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
