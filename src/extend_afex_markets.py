"""
Extend Dandume, Kano/Dawanau and Saminaka past 2024-09-18 using AFEX price
data (proxy-filled) for the three markets that overlap between the two
workstreams, plus a re-run of the original climate_assignment.py rainfall/NDVI
pipeline extended to the same date. See docs/DECISIONS.md on this branch for
the full reasoning; this script only builds the data, it does not train or
score anything.

Every other one of the 16 FEWSNET markets gets rows on the extended weekly
grid too (this panel is pivoted wide by market, so the grid must stay
rectangular -- see CLAUDE.md "things that will bite" #1), but with a null
price and every derived channel null, since there is nothing to report for
them past 2024-09-18. They will not produce new scorable origins; that is
the correct behaviour, not a bug.

Run:  python src/extend_afex_markets.py
Out:  data/panel_weekly_afex_extended.parquet
      data/baseline_afex_extended.parquet   (the new grid rows only)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FOURIER_PERIOD_WEEKS = 52.18  # fit exactly from the existing panel's own sin1/cos1

TARGET_MARKETS = {
    "Dandume": "Dandume",          # FEWSNET name -> AFEX name
    "Kano, Dawanau": "Dawanau",
    "Saminaka": "Saminaka",
}


def load_afex_price(market_afex: str) -> pd.Series:
    """Weekly maize price_NGN_per_kg for one AFEX market, proxy-filled panel."""
    path = ROOT / "data/external/afex_multicommodity_panel.xlsx"
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Panel_Long"]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    recs = [dict(zip(header, r)) for r in rows
            if r[header.index("market")] == market_afex and r[header.index("commodity")] == "Maize"]
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["price_NGN_per_kg"].sort_index()


def load_afex_is_proxy(market_afex: str) -> pd.Series:
    path = ROOT / "data/external/afex_multicommodity_panel.xlsx"
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Panel_Long"]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    recs = [dict(zip(header, r)) for r in rows
            if r[header.index("market")] == market_afex and r[header.index("commodity")] == "Maize"]
    df = pd.DataFrame(recs)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["is_proxy_price"].sort_index()


def load_climate_extension() -> pd.DataFrame:
    """Re-run the original climate_assignment.py logic, already validated
    byte-identical against the existing panel on 647 overlapping weeks
    (see chat log / DECISIONS.md this branch). Reuses its own output rather
    than recomputing, since it was already generated this session."""
    path = Path("/Users/augmentumadvisory/Downloads/Maize-Forecasting/data/processed/climate_weekly_by_market.parquet")
    df = pd.read_parquet(path)
    return df.set_index(["date", "market"])


def fourier(time_idx: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for k in range(1, 5):
        ang = 2 * np.pi * k * time_idx / FOURIER_PERIOD_WEEKS
        out[f"sin{k}"] = np.sin(ang)
        out[f"cos{k}"] = np.cos(ang)
    return out


def main() -> None:
    panel = pd.read_parquet(ROOT / "data/panel_weekly.parquet")
    last_date = panel["date"].max()
    last_time_idx = int(panel["time_idx"].max())
    assert last_date == pd.Timestamp("2024-09-18")

    new_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), "2026-09-16", freq="W-WED")
    print(f"extending {len(new_dates)} weeks: {new_dates[0].date()} -> {new_dates[-1].date()}")

    market_meta = (panel[["market", "market_id", "state", "market_kind",
                          "upstream_market", "upstream_lag_weeks"]]
                   .drop_duplicates("market").set_index("market"))
    last_diesel = panel.sort_values("date").groupby("market")["diesel"].last()

    climate = load_climate_extension()

    afex_price, afex_is_proxy = {}, {}
    for fewsnet_name, afex_name in TARGET_MARKETS.items():
        afex_price[fewsnet_name] = load_afex_price(afex_name)
        afex_is_proxy[fewsnet_name] = load_afex_is_proxy(afex_name)

    rows = []
    for i, date in enumerate(new_dates):
        time_idx = last_time_idx + 1 + i
        four = fourier(np.array([time_idx]))
        for market in market_meta.index:
            meta = market_meta.loc[market]
            row = {
                "date": date, "market": market, "time_idx": time_idx,
                "market_id": meta["market_id"], "state": meta["state"],
                "market_kind": meta["market_kind"],
                "upstream_market": meta["upstream_market"],
                "upstream_lag_weeks": meta["upstream_lag_weeks"],
                **{k: v[0] for k, v in four.items()},
                "price": np.nan, "is_proxy_price": False,
                "diesel": np.nan, "is_interpolated_diesel": False,
                "upstream_price": np.nan,
                "rainfall_simple_average": np.nan, "rainfall_inverse_distance": np.nan,
                "ndvi_simple_average": np.nan, "ndvi_inverse_distance": np.nan,
                "is_interpolated_rainfall": False, "is_interpolated_ndvi": False,
            }
            if market in TARGET_MARKETS:
                p = afex_price[market].get(date, np.nan)
                row["price"] = p
                row["is_proxy_price"] = bool(afex_is_proxy[market].get(date, False))
                row["diesel"] = last_diesel[market]
                row["is_interpolated_diesel"] = True
                if (date, market) in climate.index:
                    c = climate.loc[(date, market)]
                    row["rainfall_simple_average"] = c["rainfall_simple_average"]
                    row["rainfall_inverse_distance"] = c["rainfall_inverse_distance"]
                    row["ndvi_simple_average"] = c["ndvi_simple_average"]
                    row["ndvi_inverse_distance"] = c["ndvi_inverse_distance"]
                    row["is_interpolated_rainfall"] = True
                    row["is_interpolated_ndvi"] = True
            rows.append(row)

    new_panel = pd.DataFrame(rows)

    # upstream_price for Dawanau/Saminaka: Dandume's own (AFEX) price, lagged.
    # Dandume's price series now spans old FEWSNET + new AFEX rows continuously.
    dandume_full = pd.concat([
        panel[panel.market == "Dandume"].set_index("date")["price"],
        new_panel[new_panel.market == "Dandume"].set_index("date")["price"],
    ]).sort_index()
    for market in ["Kano, Dawanau", "Saminaka"]:
        lag_weeks = int(market_meta.loc[market, "upstream_lag_weeks"])
        mask = new_panel.market == market
        lagged_dates = new_panel.loc[mask, "date"] - pd.Timedelta(weeks=lag_weeks)
        new_panel.loc[mask, "upstream_price"] = lagged_dates.map(dandume_full)

    full = pd.concat([panel, new_panel], ignore_index=True).sort_values(["market", "date"])
    out_path = ROOT / "data/panel_weekly_afex_extended.parquet"
    full.to_parquet(out_path, index=False)
    print(f"wrote {out_path.relative_to(ROOT)}  shape={full.shape}")

    print("\nnew rows, price coverage by target market:")
    for market in TARGET_MARKETS:
        g = new_panel[new_panel.market == market]
        print(f"  {market:16s} {g.price.notna().sum():3d}/{len(g)} weeks priced, "
              f"upstream {g.upstream_price.notna().sum() if market != 'Dandume' else 0}/{len(g)}, "
              f"rainfall {g.rainfall_inverse_distance.notna().sum()}/{len(g)}")


if __name__ == "__main__":
    main()
