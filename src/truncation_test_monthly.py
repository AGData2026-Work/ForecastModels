"""
Truncation test for the monthly as-published GRU (D-75), the monthly counterpart of D-67's FX test.

    python src/truncation_test_monthly.py --dl ~/Downloads --step 2 --out outputs/d75_truncation_test.csv

For each sampled origin month o (forecast on the 15th of o+1), rebuild everything from only what was
public by then: FEWS NET monthly values for months <= o, values recovered from a later row only once that
row exists (src/build_gapfilled_panel.py cutoff rule), fill sources for months <= o, the weekly driver panel
up to the 15th of o+1, and the CBN daily file up to the 14th. Then compare the model inputs at (market, o)
with those built from the full data. Any difference means the full-data pipeline used something that
was not yet known at the forecast date.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_gapfilled_panel import build_long, load_inputs  # noqa: E402
from build_monthly_gapfilled import build_grid, build_panel  # noqa: E402
from data import attach_fx_summary, build_flat, build_sequence, load_panel  # noqa: E402

FXD = "data/external/20261002_CBN_API_Nigeria_USDNGN_Daily_2001-12_2026-09_v1.csv"
KEEP = ["fx_dep3_rank", "fx_dep6_rank", "fx_dep12_rank", "months_since_deval"]
SEQ = ["price", "diesel", "upstream", "rainfall", "ndvi", "sin1", "cos1", "sin2", "cos2", "upstream_mask"]
FLAT = ["anchor_h1", "anchor_h3", "anchor_h6", "lastyr_h1", "lastyr_h3", "lastyr_h6"] + KEEP


def features(parquet: str, fx_path: str):
    p = load_panel(parquet, rainfall_scheme="inverse_distance", fourier_k=2, interp_limit=0, frequency="MS")
    attach_fx_summary(p, "data/external/fx_rate.xlsx", "data/external/inflation.xlsx", fx_lag_months=0,
                      inflation_lag_months=1, keep=KEEP, fx_daily_path=fx_path, origin_is_month_end=True,
                      monthly_cutoff_day=14)
    return p


def window(p, o: pd.Timestamp, m: str):
    i, j = p.pos.get(o), p.midx.get(m)
    if i is None or j is None:
        return None, None
    X, ok = build_sequence(p, i, j, 6)
    F, okf = build_flat(p, i, j, [1, 3, 6], True, False, use_fx_summary=True)
    return (X if ok else None), (F if okf else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", default=str(Path.home() / "Downloads"))
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--out", default="outputs/d75_truncation_test.csv")
    a = ap.parse_args()
    inp = load_inputs(a.dl)
    weekly = pd.read_parquet("data/panel_weekly_oos2026_fewsdiesel.parquet")
    weekly["date"] = pd.to_datetime(weekly["date"])
    fxd = pd.read_csv(FXD, parse_dates=["date"])
    tmp = Path(tempfile.mkdtemp())
    full = features("data/panel_monthly_gapfilled_pit.parquet", FXD)
    G = pd.read_parquet("data/monthly_grid_gapfilled_pit.parquet")
    scored = set(zip(G.market, pd.to_datetime(G.origin)))
    rows = []
    for o in pd.period_range("2015-09", "2026-07", freq="M")[:: a.step]:
        cut = str(o)
        forecast_day = (o + 1).to_timestamp() + pd.Timedelta(days=14)
        P, *_ = build_panel(build_long(inp, cutoff=cut), weekly[weekly.date <= forecast_day], published=True)
        P.to_parquet(tmp / "p.parquet", index=False)
        fxd[fxd.date <= forecast_day - pd.Timedelta(days=1)].to_csv(tmp / "fx.csv", index=False)
        tr = features(str(tmp / "p.parquet"), str(tmp / "fx.csv"))
        ot = o.to_timestamp()
        for m in [x for x in full.markets if x != "Aba"]:
            Xf, Ff = window(full, ot, m)
            Xt, Ft = window(tr, ot, m)
            r = dict(origin=cut, market=m, scored=(m, ot) in scored, full_ok=Xf is not None and Ff is not None,
                     truncated_ok=Xt is not None and Ft is not None)
            if r["full_ok"] and r["truncated_ok"]:
                dx = np.abs(Xf - Xt).max(axis=0)
                df = np.abs(Ff - Ft)
                for k, c in enumerate(SEQ):
                    r[f"d_{c}"] = float(dx[k])
                for k, c in enumerate(FLAT):
                    r[f"d_{c}"] = float(df[k])
                r["max_abs_diff"] = float(max(dx.max(), df.max()))
            rows.append(r)
        print(cut, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(a.out, index=False)
    both = R[R.full_ok & R.truncated_ok]
    print("windows compared", len(both), "identical (max diff < 1e-9):", int((both.max_abs_diff < 1e-9).sum()))
    print("full-data window exists but not on truncated data:", int((R.full_ok & ~R.truncated_ok).sum()),
          "of which scored:", int((R.full_ok & ~R.truncated_ok & R.scored).sum()))
    bad = both[both.max_abs_diff >= 1e-9]
    if len(bad):
        cols = [c for c in bad.columns if c.startswith("d_")]
        print("channels that differ (count of windows):", (bad[cols] > 1e-9).sum()[lambda s: s > 0].to_dict())
        print("scored windows that differ:", int(bad.scored.sum()))


if __name__ == "__main__":
    main()
