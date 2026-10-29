"""
Monthly panel with yellow maize as extra TRAINING-ONLY series (2026-10-26, the analyst's "learn from white and yellow").

    python src/build_yellow_panel.py --fewsnet <20261012_FEWSNET_API_Nigeria_MarketPrices_2003-10_2026-08_v2.xlsx> \
        --out data/panel_monthly_gapfilled_pit_yellow.parquet

Each of our markets with FEWS NET yellow maize wholesale prices gets a second series "<market> (yellow)": every driver
column (diesel, rainfall, NDVI, upstream market price, season terms) is copied from the white series of the same market
and month; price and training target are FEWS NET's monthly yellow price (NGN per kg = value per 100 kg / 100), observed
only, known in the same month as the white price. No fills: a yellow month without a FEWS NET value has no price, so no
window is built over it. The evaluation grid is unchanged, so only white maize is ever scored; yellow series only add
training windows (train_markets: all_in_panel).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

PANEL = "data/panel_monthly_gapfilled_pit.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fewsnet", required=True)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = pd.read_parquet(a.panel)
    raw = pd.read_excel(a.fewsnet, sheet_name="all_rows_as_pulled")
    y = raw[(raw["product"] == "Maize Grain (Yellow)") & (raw.price_type == "Wholesale") & raw.value.notna()].copy()
    assert set(y.unit) == {"100_kg"}, y.unit.unique()
    y["date"] = pd.to_datetime(y.period_date).dt.to_period("M").dt.to_timestamp()
    y = y.groupby(["market", "date"]).value.mean().div(100).rename("yellow").reset_index()
    mk = sorted(set(y.market) & set(w.market))
    yw = w[w.market.isin(mk)].merge(y, on=["market", "date"], how="left")
    yw["price"] = yw.yellow
    yw["price_flag"] = np.where(yw.yellow.notna(), "observed", "missing")
    yw["is_proxy_price"] = False
    yw["target_price"] = yw.yellow
    yw["target_from_price"] = False
    yw["target_known_date"] = yw.date
    ids = {m: int(w.market_id.max()) + 1 + k for k, m in enumerate(mk)}
    yw["market_id"] = yw.market.map(ids).astype(w.market_id.dtype)
    yw["market"] = yw.market + " (yellow)"
    out = pd.concat([w, yw.drop(columns="yellow")], ignore_index=True).sort_values(["market", "date"]).reset_index(drop=True)
    out.to_parquet(a.out, index=False)
    print(f"white rows {len(w)}, yellow rows {len(yw)} ({int(yw.price.notna().sum())} with a price) for {len(mk)} markets -> {a.out}")


if __name__ == "__main__":
    main()
