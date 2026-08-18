"""
Reproduce every number in docs/DATA_AUDIT.md from the source files.

    python src/audit.py
    python src/audit.py --out outputs/audit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

UP = "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=f"{UP}/panel_weekly.parquet")
    ap.add_argument("--panel-alt", default=f"{UP}/panel_weekly_yellow.parquet")
    ap.add_argument("--grid", default=f"{UP}/07_panel_fe_forecasts.parquet")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = pd.read_parquet(a.panel)
    f = pd.read_parquet(a.grid)
    dates = pd.DatetimeIndex(sorted(d["date"].unique()))
    step = pd.Series(np.diff(dates).astype("timedelta64[D]").astype(int))

    print("=" * 74)
    print("PANEL")
    print("=" * 74)
    print(f"shape {d.shape} | {len(dates)} weeks {dates[0].date()} -> {dates[-1].date()}")
    print(f"date spacing (days): {step.value_counts().to_dict()} | weekday "
          f"{set(dates.day_name())}")
    print(f"markets {d.market.nunique()} | states {d.state.nunique()}")
    obs = d.price.notna()
    print(f"price observed {int(obs.sum())}/{len(d)} ({100*obs.mean():.1f}%) "
          f"| range {d.price.min()} - {d.price.max()} NGN/kg")

    # which panel did the incumbent use
    print("\nWHICH PANEL THE INCUMBENT USED")
    for name, path in [("white (panel_weekly)", a.panel), ("yellow", a.panel_alt)]:
        p = pd.read_parquet(path)[["market", "date", "price"]]
        m = f.merge(p.rename(columns={"date": "target", "price": "pt"}),
                    on=["market", "target"], how="left")
        m = m.merge(p.rename(columns={"date": "origin", "price": "po"}),
                    on=["market", "origin"], how="left")
        print(f"  {name:22} actual==price@target {np.isclose(m.actual, m.pt.fillna(-1)).mean():.4f}"
              f" | naive==price@origin {np.isclose(m.naive, m.po.fillna(-1)).mean():.4f}")

    # per-market record
    print("\nPER-MARKET PRICE RECORD")
    rows = []
    for m, g in d.sort_values("date").groupby("market"):
        s = g.set_index("date").price
        o = s.dropna()
        inner = s.loc[o.index.min():o.index.max()] if len(o) else s
        isn = inner.isna().values
        runs, c = [], 0
        for v in isn:
            if v:
                c += 1
            elif c:
                runs.append(c)
                c = 0
        if c:
            runs.append(c)
        rows.append(dict(market=m, state=g.state.iloc[0], kind=g.market_kind.iloc[0],
                         first_price=o.index.min().date() if len(o) else None,
                         observed=int(s.notna().sum()),
                         proxy=int(g.is_proxy_price.sum()),
                         longest_gap=max(runs) if runs else 0,
                         upstream=g.upstream_market.iloc[0],
                         upstream_lag=g.upstream_lag_weeks.iloc[0]))
    pm = pd.DataFrame(rows)
    print(pm.to_string(index=False))
    print(f"\nconsumption-zone markets: {(pm['kind']=='consumption').sum()} of {len(pm)}"
          "  (Section 8.8 decision rule: >4 of 16 favours the supply-shed scheme)")

    # outage weeks
    piv = d.pivot(index="date", columns="market", values="price").reindex(dates)
    outage = piv.index[piv.isna().all(axis=1)]
    print(f"\nFULL-PANEL OUTAGE WEEKS ({len(outage)}): "
          f"{[str(x.date()) for x in outage]}")

    # cross-sectional variation of the two climate schemes
    print("\nCLIMATE SCHEME VARIATION (distinct values per week across markets)")
    for c in ["rainfall_simple_average", "rainfall_inverse_distance",
              "ndvi_simple_average", "ndvi_inverse_distance"]:
        v = d.pivot(index="date", columns="market", values=c).nunique(axis=1).mean()
        print(f"  {c:30} {v:5.2f}")
    dz = d.pivot(index="date", columns="market", values="diesel").nunique(axis=1)
    print(f"  {'diesel':30} {dz.mean():5.2f}   (state-level, not national)")
    print(f"  diesel range {d.diesel.min():.2f} - {d.diesel.max():.2f} | "
          f"interpolated {100*d.is_interpolated_diesel.mean():.1f}%")

    # window feasibility at grid origins
    print("\n" + "=" * 74)
    print("EVALUATION GRID")
    print("=" * 74)
    print(f"rows {len(f)} | pairs per horizon {len(f)//f.horizon.nunique()} | "
          f"origins {f.origin.nunique()} {f.origin.min().date()} -> {f.origin.max().date()}")
    print(f"scored markets {f.market.nunique()} | absent from grid: "
          f"{sorted(set(d.market) - set(f.market))}")
    sub = f[f.horizon == f.horizon.min()]
    sp = []
    for m, g in sub.groupby("market"):
        o = np.sort(g.origin.unique())
        sp.append((m, len(o), pd.Series(np.diff(o).astype("timedelta64[D]").astype(int))
                   .value_counts().to_dict()))
    print("\nper-market origin spacing (confirms 4-weekly staggered, not weekly):")
    for m, n, s in sp:
        print(f"  {m:18} {n:>4} origins  spacing {s}")
    print(f"\nmarkets per origin: mean {sub.groupby('origin').market.nunique().mean():.2f} "
          f"min {sub.groupby('origin').market.nunique().min()} "
          f"max {sub.groupby('origin').market.nunique().max()}")
    print(f"distinct market-sets across origins: "
          f"{sub.groupby('origin').market.apply(lambda s: tuple(sorted(s))).nunique()}")
    print("target minus origin, days, by horizon: "
          f"{f.assign(x=(f.target-f.origin).dt.days).groupby('horizon').x.unique().to_dict()}")

    # lookback feasibility
    pos = {t: i for i, t in enumerate(dates)}
    grid = sub[["market", "origin"]].drop_duplicates()
    res = {c: [] for c in ["price", "diesel", "upstream_price",
                           "rainfall_inverse_distance", "ndvi_inverse_distance"]}
    arr = {c: d.pivot(index="date", columns="market", values=c).reindex(dates)
           for c in res}
    for m, g in grid.groupby("market"):
        for c in res:
            s = arr[c][m].values
            for o in g.origin:
                i = pos[pd.Timestamp(o)]
                res[c].append(int(np.sum(np.isfinite(s[i-51:i+1]))))
    print("\n52-WEEK LOOKBACK COMPLETENESS AT GRID ORIGINS (before interpolation)")
    for c, v in res.items():
        v = np.array(v)
        print(f"  {c:30} all 52 present {np.mean(v==52)*100:5.1f}%  min {v.min():>3}")

    # incumbent performance
    print("\nINCUMBENT ON THE FULL GRID")
    mae = lambda x, y: float(np.mean(np.abs(np.asarray(x) - np.asarray(y))))
    for h in sorted(f.horizon.unique()):
        g = f[f.horizon == h]
        pf, nv = mae(g.actual, g.panel_fe), mae(g.actual, g.naive)
        print(f"  h={h:>2}w  n={len(g)}  panel_fe {pf:6.2f}  naive {nv:6.2f}  "
              f"panel_fe vs naive {100*(nv-pf)/nv:+5.1f}%")

    # the 2019-04-24 problem
    p = d[["market", "date", "price"]].rename(columns={"date": "target", "price": "pt"})
    m = f.merge(p, on=["market", "target"], how="left")
    bad = m[~np.isclose(m.actual, m.pt.fillna(-1))]
    print(f"\nGRID ROWS WHOSE ACTUAL IS NOT IN THE PANEL: {len(bad)}")
    if len(bad):
        print(bad[["market", "horizon", "origin", "target", "actual"]].to_string(index=False))

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        pm.to_csv(out / "per_market_record.csv", index=False)
        pd.DataFrame({"outage_week": [x.date() for x in outage]}).to_csv(
            out / "full_panel_outage_weeks.csv", index=False)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
