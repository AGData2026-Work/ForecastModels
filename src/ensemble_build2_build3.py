"""
Free ensemble check: average build2 and build3's already-trained predictions
(same architecture, unconditional convention, same 1,590-pair grid) and see
whether combining two differently-regularised fits beats either alone.

    python src/ensemble_build2_build3.py

No retraining. Reads predictions_paired.csv from both builds directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RUNS = ["RNN", "GRU"]


def mae(a, p):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(p, float))))


def mape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    ok = np.abs(a) > 1e-9
    return float(np.mean(np.abs((a[ok] - p[ok]) / a[ok])) * 100)


def main() -> None:
    rows = []
    for kind in RUNS:
        b2 = pd.read_csv(f"outputs/build2_best_practice/{kind}_unconditional/predictions_paired.csv")
        b3 = pd.read_csv(f"outputs/build3_underfit_corrected/{kind}_unconditional/predictions_paired.csv")
        merged = b2.merge(b3, on=["origin", "market", "h"], suffixes=("_b2", "_b3"))
        assert len(merged) == len(b2) == len(b3), "grids don't match, cannot ensemble cleanly"

        merged["ens_mean"] = (merged[f"{kind}_b2"] + merged[f"{kind}_b3"]) / 2
        merged["ens_median"] = merged[["ens_mean"]]  # placeholder, 2 values -> mean == median

        for h, g in merged.groupby("h"):
            for label, col in [("build2 alone", f"{kind}_b2"), ("build3 alone", f"{kind}_b3"),
                               ("ensemble (mean)", "ens_mean")]:
                m = mae(g["actual_b2"], g[col])
                pf = g["panel_fe_b2"]
                rows.append(dict(
                    kind=kind, h=int(h), variant=label, n=len(g),
                    MAE=m, MAPE=mape(g["actual_b2"], g[col]),
                    vs_panel_fe_pct=(mae(g["actual_b2"], pf) - m) / mae(g["actual_b2"], pf) * 100,
                    vs_naive_pct=(mae(g["actual_b2"], g["naive_b2"]) - m) / mae(g["actual_b2"], g["naive_b2"]) * 100,
                ))

    out = pd.DataFrame(rows)
    out.to_csv("outputs/ensemble_build2_build3.csv", index=False)
    pd.set_option("display.width", 200)
    for kind in RUNS:
        print(f"=== {kind} ===")
        print(out[out["kind"] == kind].pivot(index="variant", columns="h",
              values="vs_panel_fe_pct").round(2).to_string())
        print()
    print("wrote outputs/ensemble_build2_build3.csv")


if __name__ == "__main__":
    main()
