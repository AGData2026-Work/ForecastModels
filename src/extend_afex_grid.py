"""
Build the extension grid (new (market, origin, horizon) rows) for Dandume,
Kano/Dawanau and Saminaka past their last origin in the frozen
07_panel_fe_forecasts.parquet, continuing each market's own 28-day origin
cadence exactly (verified from the existing grid, not invented -- see
CLAUDE.md's D-01 warning about grid staggering).

No panel_fe: this workstream dropped the incumbent as a benchmark at D-36
and its generation script was never available anyway. panel_fe is filled
with the naive value as an inert placeholder so load_grid's pivot (which
still expects the column) does not break; it is never reported.

Run:  python src/extend_afex_grid.py
Out:  data/baseline_afex_extended.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H = [4, 13, 26]

LAST_ORIGIN = {
    "Dandume": pd.Timestamp("2024-03-20"),
    "Kano, Dawanau": pd.Timestamp("2024-02-28"),
    "Saminaka": pd.Timestamp("2024-03-06"),
}


def main() -> None:
    panel = pd.read_parquet(ROOT / "data/panel_weekly_afex_extended.parquet")
    price = panel.pivot(index="date", columns="market", values="price")

    old_grid = pd.read_parquet(ROOT / "data/07_panel_fe_forecasts.parquet")

    rows = []
    for market, last_origin in LAST_ORIGIN.items():
        origin = last_origin + pd.Timedelta(weeks=4)
        while origin <= price.index.max():
            for h in H:
                target = origin + pd.Timedelta(weeks=h)
                if target > price.index.max():
                    continue
                actual = price.at[target, market] if target in price.index else None
                naive = price.at[origin, market] if origin in price.index else None
                rows.append(dict(market=market, horizon=h, origin=origin,
                                 target=target, actual=actual,
                                 panel_fe=naive,  # inert placeholder, see docstring
                                 naive=naive))
            origin += pd.Timedelta(weeks=4)

    new_grid = pd.DataFrame(rows)
    full_grid = pd.concat([old_grid, new_grid], ignore_index=True)
    out = ROOT / "data/baseline_afex_extended.parquet"
    full_grid.to_parquet(out, index=False)
    print(f"wrote {out.relative_to(ROOT)}  shape={full_grid.shape}")

    print("\nnew origins per market, and how many have all 3 horizons' actuals:")
    for market in LAST_ORIGIN:
        g = new_grid[new_grid.market == market]
        n_origins = g["origin"].nunique()
        complete = g.groupby("origin")["actual"].apply(lambda s: s.notna().all())
        print(f"  {market:16s} {n_origins:3d} candidate origins "
              f"({g.origin.min().date()} -> {g.origin.max().date()}), "
              f"{complete.sum():3d} with all 3 horizons scoreable")


if __name__ == "__main__":
    main()
