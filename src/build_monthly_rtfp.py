"""
Monthly panel and grid with the World Bank RTFP maize ESTIMATE (c_maize_fao) as the price for
every market and month (owner request 2026-10-07: "use the c_fao value across the board").

    python src/build_monthly_rtfp.py --rtfp <NGA_RTFP_mkt_2007_2026-09-28.xlsx or a parquet copy>

Writes data/panel_monthly_rtfp.parquet, data/monthly_grid_rtfp.parquet, data/panel_monthly_rtfp.log.json.

What this is and is not (read before quoting any result):
- c_maize_fao is the World Bank's modelled "close" price. For 9 of our 15 markets RTFP holds no
  observed maize at all, so their whole series is model output; for the other 6 (Ibadan, Kaura
  Namoda, Lagos, Kano, Giwa, Gombe) observed values exist for some months and differ from the
  estimate (RTFP's Kano series, Jul 2026: observed 381, estimate 544; RTFP also has an estimate-only
  Dawanau series, which is the one mapped below, as is the estimate-only Maiduguri rather than Monday
  market). RTFP re-estimates history as new data
  arrive, so the series is not as-published: every result here carries that look-ahead.
- Scoring is against the same estimates, so results measure how well the model forecasts the
  World Bank series, not real prices. The grid also carries FEWS NET's real monthly wholesale
  value (fews_actual) where it exists, for a secondary score.

Construction (May 2012 to Sep 2026, month-start dates, the monthly panel's layout):
- price: c_maize_fao of the RTFP market with our market's name (Kano, Dawanau -> "Dawanau",
  Gwandu, Dodoru -> "Gwandu", Ibadan, Bodija -> "Ibadan", Lagos, Mile 12 -> "Lagos").
- upstream_price: c_maize_fao of the panel's upstream market, same month.
- diesel, rainfall, NDVI: monthly means of data/panel_weekly_oos2026_fewsdiesel.parquet (the
  weekly panel extended to 2026, D-70); months with no diesel (FEWS NET did not collect) and
  NDVI after its source ends (May 2026) carry the last month forward, counted in the log.
- Fourier terms on month of year, as src/build_monthly_panel.py.
Grid: origins from Sept 2015 (the monthly grid's first) to Mar 2026, horizons 1, 3, 6 months;
actual = c_maize_fao of the target month, naive = c_maize_fao of the origin month.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

RTFP_NAME = {"Aba": "Aba", "Biu": "Biu", "Damaturu": "Damaturu", "Dandume": "Dandume", "Giwa": "Giwa",
             "Gombe": "Gombe", "Gujungu": "Gujungu", "Gwandu, Dodoru": "Gwandu", "Ibadan, Bodija": "Ibadan",
             "Kano, Dawanau": "Dawanau", "Kaura Namoda": "Kaura Namoda", "Lagos, Mile 12": "Lagos",
             "Maiduguri": "Maiduguri", "Mubi": "Mubi", "Potiskum": "Potiskum", "Saminaka": "Saminaka"}
HORIZONS = [1, 3, 6]
API = "data/external/20261005_FEWSNET_API_Nigeria_MarketPrices_2024-09_2026-08_v2.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtfp", required=True)
    a = ap.parse_args()
    log: dict = {"source": a.rtfp, "price_column": "c_maize_fao"}
    r = pd.read_parquet(a.rtfp) if a.rtfp.endswith(".parquet") else pd.read_excel(a.rtfp)
    r["month"] = pd.to_datetime(r["price_date"]).dt.to_period("M")
    rk = {v: k for k, v in RTFP_NAME.items()}
    r = r[r["mkt_name"].isin(rk)].copy()
    r["market"] = r["mkt_name"].map(rk)
    est = r.set_index(["market", "month"])["c_maize_fao"]
    obs = r.set_index(["market", "month"])["maize_fao"]
    log["markets_with_any_observed_maize"] = sorted(r.loc[r["maize_fao"].notna(), "market"].unique().tolist())

    w = pd.read_parquet("data/panel_weekly_oos2026_fewsdiesel.parquet")
    w["date"] = pd.to_datetime(w["date"])
    w["month"] = w["date"].dt.to_period("M")
    months = pd.period_range("2012-05", "2026-09", freq="M")
    drv = w.groupby(["market", "month"]).agg(
        diesel=("diesel", "mean"), rainfall_simple_average=("rainfall_simple_average", "mean"),
        rainfall_inverse_distance=("rainfall_inverse_distance", "mean"),
        ndvi_simple_average=("ndvi_simple_average", "mean"), ndvi_inverse_distance=("ndvi_inverse_distance", "mean"),
        upstream_market=("upstream_market", "first"), upstream_lag_weeks=("upstream_lag_weeks", "first"),
        market_kind=("market_kind", "first"), state=("state", "first"), market_id=("market_id", "first"))
    full = pd.MultiIndex.from_product([sorted(RTFP_NAME), months], names=["market", "month"])
    P = drv.reindex(full).reset_index()
    for col in ["upstream_market", "upstream_lag_weeks", "market_kind", "state", "market_id"]:
        P[col] = P.groupby("market")[col].transform(lambda s: s.ffill().bfill())
    filled = {}
    for col in ["diesel", "rainfall_simple_average", "rainfall_inverse_distance", "ndvi_simple_average", "ndvi_inverse_distance"]:
        before = P[col].isna()
        P[col] = P.groupby("market")[col].transform(lambda s: s.ffill())
        filled[col] = int((before & P[col].notna()).sum())
    log["driver_months_carried_forward"] = filled
    P["price"] = [est.get((m, mo), np.nan) for m, mo in zip(P["market"], P["month"])]
    P["rtfp_observed_maize"] = [obs.get((m, mo), np.nan) for m, mo in zip(P["market"], P["month"])]
    P["upstream_price"] = [est.get((u, mo), np.nan) if isinstance(u, str) else np.nan for u, mo in zip(P["upstream_market"], P["month"])]
    for col, v in [("n_price_weeks", 0), ("n_weeks", 0), ("n_proxy_weeks", 0), ("share_interpolated_diesel", 1.0),
                   ("share_interpolated_rainfall", 1.0), ("share_interpolated_ndvi", 1.0)]:
        P[col] = v
    P["date"] = P["month"].dt.to_timestamp()
    moy = P["date"].dt.month
    for k in range(1, 5):
        P[f"sin{k}"] = np.sin(2 * np.pi * k * moy / 12)
        P[f"cos{k}"] = np.cos(2 * np.pi * k * moy / 12)
    P["time_idx"] = (P["month"] - months[0]).apply(lambda d: d.n)
    P["market_id"] = P["market_id"].astype("int8")
    log["price_missing"] = int(P["price"].isna().sum())
    log["price_months_with_rtfp_observation"] = int(P["rtfp_observed_maize"].notna().sum())
    P.drop(columns=["month"]).to_parquet("data/panel_monthly_rtfp.parquet", index=False)

    # real FEWS NET monthly wholesale, for the secondary score
    pw = pd.read_parquet("data/panel_weekly.parquet")
    pw["month"] = pd.to_datetime(pw["date"]).dt.to_period("M")
    fw = pw[pw["price"].notna() & ~pw["is_proxy_price"].astype(bool)].groupby(["market", "month"])["price"].mean()
    api = pd.read_csv(API, parse_dates=["period_date"])
    api = api[(api["product"] == "Maize Grain (White)") & (api["price_type"] == "Wholesale") & api["value"].notna()]
    fa = (api.assign(month=api["period_date"].dt.to_period("M"), p=api["value"] / 100.0)
          .groupby(["market", "month"])["p"].first())
    fews = pd.concat([fw, fa[fa.index.get_level_values(1) > pd.Period("2024-09")]])

    rows = []
    for m in [x for x in RTFP_NAME if x != "Aba"]:
        for o in pd.period_range("2015-09", "2026-03", freq="M"):
            p0 = est.get((m, o), np.nan)
            for h in HORIZONS:
                t = o + h
                rows.append(dict(market=m, horizon=h, origin=o.to_timestamp(), target=t.to_timestamp(),
                                 actual=est.get((m, t), np.nan), naive=p0, panel_fe=np.nan,
                                 fews_actual=fews.get((m, t), np.nan), rtfp_observed_actual=obs.get((m, t), np.nan)))
    G = pd.DataFrame(rows)
    G.to_parquet("data/monthly_grid_rtfp.parquet", index=False)
    log["grid"] = {"pairs_per_horizon": int(len(G) / 3), "first_origin": str(G["origin"].min().date()),
                   "last_origin": str(G["origin"].max().date()),
                   "with_fews_actual_by_h": G.dropna(subset=["fews_actual"]).groupby("horizon").size().to_dict()}
    json.dump(log, open("data/panel_monthly_rtfp.log.json", "w"), indent=2, default=str)
    print(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main()
