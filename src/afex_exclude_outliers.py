"""
Rescore existing AFEX forecasts excluding two markets flagged, independently
of each other, as unreliable before this script existed:

    Giwa | Maize   the source panel's own documented near-duplicate of
                   Anchau | Maize (0.947 correlation after an 81-week
                   repair -- Build_Decisions sheet, DECISIONS D-27/D-33).
    Ikara | Maize  the thinnest, noisiest maize series throughout this
                   workstream (24 scored windows, consistently the weakest
                   market at every horizon regardless of build -- D-30).

Excluded rather than down-weighted, for consistency with this project's own
precedent (Dandume | Maize was already dropped outright for a zero-window
data-quality reason, D-27) rather than inventing a new weighting scheme for
a problem the exclusion already handles cleanly.

    python src/afex_exclude_outliers.py

Reads saved forecasts only, no retraining. Covers operational,
safeguard-removed and agroclimatic (sorghum excluded per instruction --
D-32 was already a clean result and mixing this rescoring into that
comparison would confound it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXCLUDE = ["Giwa | Maize", "Ikara | Maize"]

RUNS = [
    ("RNN operational", "outputs/afex_operational/RNN"),
    ("GRU operational", "outputs/afex_operational/GRU"),
    ("RNN safeguard-removed", "outputs/afex_safeguard_removed/RNN"),
    ("GRU safeguard-removed", "outputs/afex_safeguard_removed/GRU"),
    ("RNN agroclimatic", "outputs/afex_operational_agroclimatic/RNN"),
    ("GRU agroclimatic", "outputs/afex_operational_agroclimatic/GRU"),
]


def mae(a, p) -> float:
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(p, float))))


def mape(a, p) -> float:
    a, p = np.asarray(a, float), np.asarray(p, float)
    ok = np.abs(a) > 1e-9
    return float(np.mean(np.abs((a[ok] - p[ok]) / a[ok])) * 100) if ok.any() else float("nan")


def direction_pct(actual, pred, origin_price) -> tuple[float, int]:
    da = np.sign(np.asarray(actual, float) - np.asarray(origin_price, float))
    dp = np.sign(np.asarray(pred, float) - np.asarray(origin_price, float))
    ok = da != 0
    return (float(np.mean(da[ok] == dp[ok]) * 100), int(ok.sum())) if ok.any() else (float("nan"), 0)


def main() -> None:
    rows = []
    for label, path in RUNS:
        kind = label.split(" ")[0]
        df = pd.read_csv(f"{path}/predictions_paired.csv", parse_dates=["origin"])
        for scope, sub in [("all markets (n=15)", df),
                           ("excluding Giwa+Ikara (n=13)", df[~df["market"].isin(EXCLUDE)])]:
            for h, gh in sub.groupby("h"):
                dir_pct, n_dir = direction_pct(gh["actual"], gh[kind], gh["origin_price"])
                rose = gh["actual"] > gh["origin_price"]
                up_share = float(rose.mean() * 100)
                rows.append(dict(
                    run=label, scope=scope, h=int(h), n=len(gh),
                    MAE=mae(gh["actual"], gh[kind]), MAPE=mape(gh["actual"], gh[kind]),
                    naive_MAE=mae(gh["actual"], gh["naive"]),
                    direction_pct=dir_pct, always_up_share_pct=up_share,
                ))

    out = pd.DataFrame(rows)
    out["vs_naive_pct"] = (out["naive_MAE"] - out["MAE"]) / out["naive_MAE"] * 100
    Path("outputs").mkdir(exist_ok=True)
    out.to_csv("outputs/afex_excl_outliers_comparison.csv", index=False)

    pd.set_option("display.width", 220)
    for h in [4, 13, 26]:
        print(f"=== h={h} ===")
        sub = out[out["h"] == h]
        piv = sub.pivot(index="run", columns="scope", values=["MAE", "direction_pct"])
        print(piv.round(2).to_string())
        print()

    print(f"wrote outputs/afex_excl_outliers_comparison.csv")


if __name__ == "__main__":
    main()
