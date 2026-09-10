"""
Price direction analysis for the AFEX operational runs: does the model call
the direction of the price move better than doing nothing, and better than
the panel's own base rate of rises and falls.

    python src/afex_directional.py

Reads outputs/afex_operational/RNN and outputs/afex_operational/GRU
directly (existing saved forecasts, no retraining).

Why "vs the do-nothing variant" needs care
-------------------------------------------
The do-nothing (carry-forward) forecast predicts NO CHANGE at every origin,
by construction (see run_afex.py: naive_h{h} = origin_price, always). It
therefore never asserts a direction at all -- sign(naive - origin) is
always exactly 0. Scored the same way the model is scored, its directional
accuracy is 0% on every window where the price actually moved, which is
almost every window. That is not a meaningful comparison; it is a
restatement of the fact that a flat forecast has no directional opinion.

The meaningful comparison, and the one this project already uses for the
main panel (src/directional_benchmark.py, DECISIONS D-19), is against the
panel's own base rate: the share of windows where price actually rose (or
fell). A model with no skill at all, right only by chance, should land near
50%; a model that merely tracks the panel's dominant trend can look good
against 50% while adding nothing. This script reports both, plus per-market
breakdowns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RUNS = [
    ("RNN operational", "outputs/afex_operational/RNN"),
    ("GRU operational", "outputs/afex_operational/GRU"),
]


def direction_pct(actual, pred, origin_price) -> tuple[float, int]:
    da = np.sign(np.asarray(actual, float) - np.asarray(origin_price, float))
    dp = np.sign(np.asarray(pred, float) - np.asarray(origin_price, float))
    ok = da != 0
    if not ok.any():
        return float("nan"), 0
    return float(np.mean(da[ok] == dp[ok]) * 100), int(ok.sum())


def summarise(g: pd.DataFrame, kind: str) -> dict:
    rose = g["actual"] > g["origin_price"]
    fell = g["actual"] < g["origin_price"]
    model_pct, n_dir = direction_pct(g["actual"], g[kind], g["origin_price"])
    naive_pct, _ = direction_pct(g["actual"], g["naive"], g["origin_price"])
    up_share = float(rose.mean() * 100)
    down_share = float(fell.mean() * 100)
    return dict(
        n=len(g), n_directional=n_dir,
        model_direction_pct=model_pct,
        naive_direction_pct=naive_pct,
        always_up_share_pct=up_share, always_down_share_pct=down_share,
        margin_vs_coinflip_pp=model_pct - 50.0,
        margin_vs_always_up_pp=model_pct - up_share,
        margin_vs_always_down_pp=model_pct - down_share,
        beats_always_up_and_down=bool(model_pct > max(up_share, down_share)),
    )


def main() -> None:
    rows_overall, rows_market = [], []
    for label, path in RUNS:
        kind = label.split(" ")[0]
        g = pd.read_csv(f"{path}/predictions_paired.csv", parse_dates=["origin"])
        for h, gh in g.groupby("h"):
            base = dict(run=label, h=int(h))
            base.update(summarise(gh, kind))
            rows_overall.append(base)
            for market, gm in gh.groupby("market"):
                mbase = dict(run=label, h=int(h), market=market)
                mbase.update(summarise(gm, kind))
                rows_market.append(mbase)

    overall = pd.DataFrame(rows_overall).sort_values(["run", "h"]).reset_index(drop=True)
    market = pd.DataFrame(rows_market).sort_values(["run", "h", "market"]).reset_index(drop=True)

    Path("outputs").mkdir(exist_ok=True)
    overall.to_csv("outputs/afex_directional_overall.csv", index=False)
    market.to_csv("outputs/afex_directional_by_market.csv", index=False)

    print("NAIVE (CARRY-FORWARD) DIRECTIONAL ACCURACY, FOR THE RECORD")
    print("naive_direction_pct is 0.0 or NaN throughout by construction: carry-forward never")
    print("asserts a direction (predicted change is always exactly zero), so it cannot be scored")
    print("as calling a direction right. The real benchmark is the panel's own move share, below.\n")

    print("OVERALL (maize, all markets pooled), operational runs")
    print(overall[["run", "h", "n", "n_directional", "model_direction_pct",
                  "always_up_share_pct", "always_down_share_pct",
                  "margin_vs_coinflip_pp", "beats_always_up_and_down"]]
          .round(2).to_string(index=False))

    void = overall[~overall["beats_always_up_and_down"]]
    if len(void):
        print(f"\n{len(void)} of {len(overall)} run/horizon combinations do not beat the "
              "stronger of always-up/always-down:")
        print(void[["run", "h", "model_direction_pct", "always_up_share_pct",
                    "always_down_share_pct"]].round(2).to_string(index=False))
    else:
        print("\nEvery run/horizon combination beats the stronger of always-up/always-down.")

    print(f"\nwrote outputs/afex_directional_overall.csv")
    print(f"wrote outputs/afex_directional_by_market.csv")


if __name__ == "__main__":
    main()
