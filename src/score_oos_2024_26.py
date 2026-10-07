"""
Score the frozen out-of-sample runs (Oct 2024 to Aug 2026) against FEWS NET's monthly prices.

    python src/score_oos_2024_26.py

Reads outputs/oos2026/{fewsdiesel,flatdiesel}_h{4,13,26}/ (one run per horizon, the same
model fitted once at 2024-09-25) and data/oos_grid_2024_26.parquet. The actual is the
FEWS NET value for the target week's month; naive is the last published value at the origin,
the same number the model starts from. Reported with and without origins whose latest
price is more than 13 weeks old.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import diebold_mariano, direction_pct, mae, mape  # noqa: E402

CONTINUED = ["Biu", "Damaturu", "Dandume", "Ibadan, Bodija", "Kano, Dawanau", "Kaura Namoda", "Maiduguri", "Mubi", "Potiskum"]


def load(variant: str) -> pd.DataFrame:
    parts = []
    for h in (4, 13, 26):
        p = pd.read_csv(f"outputs/oos2026/{variant}_h{h}/predictions_paired.csv", parse_dates=["origin"])
        parts.append(p[p["h"] == h])
    p = pd.concat(parts, ignore_index=True)
    g = pd.read_parquet("data/oos_grid_2024_26.parquet").rename(columns={"horizon": "h"})
    p = p.merge(g[["market", "origin", "h", "target", "target_month", "origin_data_age_weeks", "stale_over_13w"]],
                on=["market", "origin", "h"], how="left")
    assert p["target"].notna().all()
    return p


def block(s: pd.DataFrame, label: str) -> dict:
    s = s.sort_values(["market", "origin"])
    e, n = mae(s["actual"], s["GRU"]), mae(s["actual"], s["naive"])
    out = dict(set=label, n=len(s), markets=s["market"].nunique(), MAE=round(e, 2), naive_MAE=round(n, 2),
               margin_vs_naive_pct=round(100 * (n - e) / n, 1), MAPE=round(mape(s["actual"], s["GRU"]), 1),
               naive_MAPE=round(mape(s["actual"], s["naive"]), 1),
               direction_pct=round(direction_pct(s["actual"], s["GRU"], s["naive"]), 1),
               actual_rose_pct=round(100 * (s["actual"] > s["naive"]).mean(), 1),
               predicted_rise_pct=round(100 * (s["GRU"] > s["naive"]).mean(), 1),
               mean_error_pct=round(100 * ((s["GRU"] - s["actual"]) / s["actual"]).mean(), 1))
    if len(s) >= 10 and s["market"].nunique() >= 2:
        d = diebold_mariano(s["actual"], s["GRU"], s["naive"], int(s["h"].iloc[0]), groups=s["market"].values)
        out["DM_p_vs_naive"] = round(d["dm_p"], 4)
    return out


def main() -> None:
    rows, mk = [], []
    for variant in ("fewsdiesel", "flatdiesel"):
        p = load(variant)
        for h in (4, 13, 26):
            s = p[p["h"] == h]
            rows.append(dict(variant=variant, h=h, **block(s, "all scorable")))
            rows.append(dict(variant=variant, h=h, **block(s[~s["stale_over_13w"]], "latest price <= 13 weeks old")))
            rows.append(dict(variant=variant, h=h, **block(s[s["market"].isin(CONTINUED) & ~s["stale_over_13w"]], "9 continued markets, <= 13 weeks")))
            if variant == "fewsdiesel":
                for m, q in s.groupby("market"):
                    mk.append(dict(h=h, market=m, n=len(q), MAE=round(mae(q["actual"], q["GRU"]), 2),
                                   naive_MAE=round(mae(q["actual"], q["naive"]), 2),
                                   MAPE=round(mape(q["actual"], q["GRU"]), 1),
                                   mean_error_pct=round(100 * ((q["GRU"] - q["actual"]) / q["actual"]).mean(), 1)))
        p.to_csv(f"outputs/oos2026/{variant}_scored_pairs.csv", index=False)
    T = pd.DataFrame(rows)
    T.to_csv("outputs/oos2026/summary.csv", index=False)
    M = pd.DataFrame(mk)
    M.to_csv("outputs/oos2026/by_market_fewsdiesel.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.to_string(index=False))
    print()
    print(M.pivot_table(index="market", columns="h", values=["MAE", "naive_MAE", "mean_error_pct"]).round(1).to_string())


if __name__ == "__main__":
    main()
