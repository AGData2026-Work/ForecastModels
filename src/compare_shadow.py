"""
Score the production model against the shadow model (D-77) on live forecasts, once FEWS NET has
published the months they target.

    python src/compare_shadow.py \
        --prod outputs/production_monthly_parity_serving \
        --shadow outputs/shadow_monthly_parity_fdiesel_l0_lb3_serving \
        --panel data/panel_monthly_gapfilled_pit.parquet

Reads every dated forecast file in <dir>/predictions/ (written by src/predict.py), matches production and
shadow on (market, origin, h), and scores them against the FEWS NET observed price of the target month
(panel rows with price_flag == "observed"; filled months are never scored). Rebuild the panel with the new
month first. Naive = the origin price. Prints MAE and the margin over naive per horizon and in total, and
how many pairs are still waiting for their target month.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load(d: Path) -> pd.DataFrame:
    files = sorted((d / "predictions").glob("*.csv"))
    if not files:
        raise SystemExit(f"no forecasts in {d}/predictions")
    f = pd.concat([pd.read_csv(x).assign(file=x.name) for x in files])
    f = f.sort_values("file").drop_duplicates(["market", "origin", "h"], keep="last")
    return f[["market", "origin", "h", "origin_price", "forecast_price"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", required=True)
    ap.add_argument("--shadow", required=True)
    ap.add_argument("--panel", default="data/panel_monthly_gapfilled_pit.parquet")
    a = ap.parse_args()
    m = load(Path(a.prod)).merge(load(Path(a.shadow)), on=["market", "origin", "h"], suffixes=("_prod", "_shadow"))
    m["target"] = [(pd.Period(o[:7], "M") + int(h)).to_timestamp() for o, h in zip(m.origin, m.h)]
    p = pd.read_parquet(a.panel)
    obs = p[p.price_flag == "observed"].assign(date=lambda x: pd.to_datetime(x.date)).set_index(["market", "date"]).price
    m["actual"] = [obs.get((mk, t)) for mk, t in zip(m.market, m.target)]
    done = m[m.actual.notna()]
    print(f"forecast pairs: {len(m)}; scored: {len(done)}; waiting for their target month: {len(m) - len(done)}")
    if not len(done):
        return
    rows = []
    for lab, s in [(f"{h} month(s)", done[done.h == h]) for h in sorted(done.h.unique())] + [("total", done)]:
        if not len(s):
            continue
        n = (s.actual - s.origin_price_prod).abs().mean()
        r = dict(horizon=lab, n=len(s), naive_MAE=round(n, 2))
        for k in ("prod", "shadow"):
            e = (s.actual - s[f"forecast_price_{k}"]).abs().mean()
            r[f"{k}_MAE"] = round(e, 2)
            r[f"{k}_vs_naive_pct"] = round(100 * (n - e) / n, 1)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
