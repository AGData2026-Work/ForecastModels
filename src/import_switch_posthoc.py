"""
The analyst's import switch applied to our forecasts after the fact (2026-10-26, no retraining).

    python src/import_switch_posthoc.py --analyst outputs/joint_forecast/analyst_forecasts.csv --out outputs/d77_checks/v4ideas/import_switch

His rule (v2_models.py, M10/M11): while his import window is ON (0% duty and national price above landed import cost at
the street rate; Aug 2024 to Sep 2025), mix the forecast 50/50 (geometric) with his import-window model, a straight
line of past window months' change beyond the usual seasonal move on the gap to import cost, fitted only on window
months whose outcome was known. Outside the window the forecast is unchanged. Applied here to the GRU and to the joint
forecast; his window model's expected change is applied to our origin price (identical where the origin prices agree).
Scored on the shared production pairs against FEWS NET observed actuals, with naive on the same pairs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from joint_forecast import EVAL, PERIODS, combine  # noqa: E402
from metrics import diebold_mariano, mae, mape  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyst", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    an = pd.read_csv(a.analyst, parse_dates=["origin", "target"])
    rows = []
    for h in (1, 3, 6):
        g = pd.read_csv(EVAL.format(h=h), parse_dates=["origin"]).query("h == @h")
        g["target"] = g.origin + pd.DateOffset(months=h)
        x = g.merge(an[an.h == h][["market", "origin", "p0", "recommended", "iw"]], on=["market", "origin"])
        x["joint"] = combine(x.GRU, x.origin_price, x.recommended, x.p0)
        iw_ours = x.origin_price * (x.iw / x.p0)                      # his window model's change, from our price
        on = x.iw.notna()
        x["gru_switch"] = np.where(on, np.sqrt(x.GRU * iw_ours), x.GRU)
        x["joint_switch"] = np.where(on, np.sqrt(x.joint * iw_ours), x.joint)
        for per, lo, hi in PERIODS + [("Window months only", "2000-01-01", "2100-01-01")]:
            s = x[(x.target >= lo) & (x.target <= hi)]
            if per == "Window months only":
                s = x[on]
            s = s.sort_values(["market", "origin"])
            n = mae(s.actual, s.naive)
            r = dict(h=h, period=per, n=len(s), n_window=int(s.iw.notna().sum()))
            for c in ["GRU", "gru_switch", "joint", "joint_switch"]:
                r[f"{c}_mape"] = mape(s.actual, s[c]); r[f"{c}_vs_naive"] = 100 * (n - mae(s.actual, s[c])) / n
            r["p_gru_switch_vs_gru"] = diebold_mariano(s.actual.values, s.gru_switch.values, s.GRU.values, h, groups=s.market.values)["dm_p"] if len(s) > 20 else np.nan
            rows.append(r)
    t = pd.DataFrame(rows)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t.round(4).to_csv(Path(a.out) / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
