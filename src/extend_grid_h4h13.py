"""
Extend the frozen grid with new origins past each market's last historical
origin, for h=4 and h=13 only -- these targets already have real actuals in
panel_weekly.parquet even though h=26's target hasn't happened yet. No new
data source, no proxying: same panel, same markets, just origins the frozen
grid never reached because it always required all three horizons together.

Each market's own 28-day cadence is continued exactly (not reinvented -- see
CLAUDE.md's D-01 warning about grid staggering). No h=26 row is added for any
new origin; run.py's --require-horizons 4 13 (paired with this file) is what
lets those origins through without one.

Run:  python src/extend_grid_h4h13.py
Out:  data/baseline_h4h13_extended.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H = [4, 13]


def main() -> None:
    old_grid = pd.read_parquet(ROOT / "data/07_panel_fe_forecasts.parquet")
    panel = pd.read_parquet(ROOT / "data/panel_weekly.parquet")
    price = panel.pivot(index="date", columns="market", values="price")
    max_date = price.index.max()

    last_origin = old_grid.groupby("market")["origin"].max()

    rows = []
    for market, last in last_origin.items():
        origin = last + pd.Timedelta(weeks=4)
        while True:
            targets_ok = all(origin + pd.Timedelta(weeks=h) <= max_date for h in H)
            if not targets_ok:
                break
            for h in H:
                target = origin + pd.Timedelta(weeks=h)
                actual = price.at[target, market] if target in price.index else None
                naive = price.at[origin, market] if origin in price.index else None
                if pd.isna(actual) or pd.isna(naive):
                    continue  # a genuine reporting gap at this date, not fabricated
                rows.append(dict(market=market, horizon=h, origin=origin,
                                 target=target, actual=actual,
                                 panel_fe=naive,  # inert placeholder -- D-36 dropped
                                 naive=naive))     # panel_fe as this project's benchmark
            origin += pd.Timedelta(weeks=4)

    new_grid = pd.DataFrame(rows)
    full = pd.concat([old_grid, new_grid], ignore_index=True)
    out = ROOT / "data/baseline_h4h13_extended.parquet"
    full.to_parquet(out, index=False)
    print(f"wrote {out.relative_to(ROOT)}  shape={full.shape}")

    print("\nnew origins per market:")
    for market in last_origin.index:
        g = new_grid[new_grid.market == market]
        n_origins = g["origin"].nunique()
        if n_origins:
            print(f"  {market:16s} {n_origins:2d} new origins "
                  f"({g.origin.min().date()} -> {g.origin.max().date()})")
        else:
            print(f"  {market:16s}  0 new origins")


if __name__ == "__main__":
    main()
