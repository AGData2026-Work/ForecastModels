"""
Out-of-sample test inputs, Oct 2024 to Aug 2026: the weekly panel extended with FEWS NET's
monthly prices, every value placed as it would have been published, plus a scoring grid.

    python src/build_oos_2024_26.py --climate <climate_weekly_by_market.parquet to 2026-09-30>

Writes
  data/panel_weekly_oos2026_fewsdiesel.parquet  diesel continued with FEWS NET market diesel
  data/panel_weekly_oos2026_flatdiesel.parquet  diesel left to carry forward (no new diesel)
  data/oos_grid_2024_26.parquet                 origins, monthly actuals, naive
  data/oos_2024_26.log.json                     every count behind them

How the monthly data is applied (each rule exists to keep inputs to what was knowable):
- Price after 2024-09-18 is FEWS NET's monthly white-maize wholesale value
  (data/external/20261005_FEWSNET_API_..._v2.csv, NGN per 100 kg / 100). A month's value is
  treated as public on the 15th of the next month (assumption; the API gives no release
  date) and is held flat on every Wednesday from then until the next month is public. Weeks
  before September's value is public carry the last weekly observation forward. Nothing is
  interpolated: a straight line to the next month would use a price not yet published.
- Missing months are not filled; the last published value is carried forward and its age
  is recorded (oos_data_age_weeks = weeks between the forecast date and the end of the
  month the value describes). The grid flags origins whose value is over 13 weeks old.
- Upstream price is the upstream market's series built the same way, lagged as in the panel.
- Diesel: panel state diesel ends Sep 2024. Variant "fewsdiesel" continues it with FEWS NET's
  monthly market diesel (API, retail, per litre) times a per-market ratio, median of
  state/FEWS NET over 2023-01..2024-09 from the raw FEWS NET file; months FEWS NET did not
  collect stay unpublished. Variant "flatdiesel" adds no diesel after Sep 2024. Both then go
  through make_exog_point_in_time (published on the 20th of the next month, carried forward).
- Rainfall and NDVI come from the original climate_assignment.py run to 2026-09-30 (validated
  identical to the panel on every overlapping week); NDVI after its source ends (2026-05-20)
  is set to NaN rather than extrapolated. The production config lags both 4 weeks.
- Fourier terms continue the panel's own formula (period 52.18 weeks from 2012-05-02).

Grid: 15 scored markets; origins 2024-09-25 and then the first Wednesday on or after the 15th
of each month, Oct 2024 to Jul 2026 (the first forecast date after each month's value is
public). Target week = origin + 4, 13, 26 weeks; actual = FEWS NET's value for the target
week's month, only where FEWS NET has one (never filled). Naive = the price the model sees at
the origin (the last published value), so model and naive use the same information.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

PANEL = "data/panel_weekly.parquet"
API = "data/external/20261005_FEWSNET_API_Nigeria_MarketPrices_2024-09_2026-08_v2.csv"
END = pd.Timestamp("2026-09-30")
AVAIL_DAY = 15
NDVI_SOURCE_END = pd.Timestamp("2026-05-20")
STALE_WEEKS = 13
HORIZONS = [4, 13, 26]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--climate", required=True)
    ap.add_argument("--raw-diesel", required=True, help="raw FEWS NET diesel rows (parquet), for the splice ratio")
    a = ap.parse_args()
    log: dict = {}

    pw = pd.read_parquet(PANEL)
    pw["date"] = pd.to_datetime(pw["date"])
    pend = pw["date"].max()
    new_dates = pd.date_range(pend + pd.Timedelta(days=7), END, freq="7D")
    t0 = pw["date"].min()
    log["panel_end"] = str(pend.date())
    log["new_weeks"] = len(new_dates)

    api = pd.read_csv(API, parse_dates=["period_date"])
    w = api[(api["product"] == "Maize Grain (White)") & (api["price_type"] == "Wholesale") & api["value"].notna()].copy()
    assert set(w["unit"]) == {"100_kg"}
    w["p"] = w["value"] / 100.0
    w["month"] = w["period_date"].dt.to_period("M")
    w["avail"] = (w["month"] + 1).dt.to_timestamp() + pd.Timedelta(days=AVAIL_DAY - 1)
    w["month_end"] = w["month"].dt.to_timestamp(how="end").dt.normalize()

    markets = sorted(pw["market"].unique())
    meta = pw.sort_values("date").groupby("market").last()

    # ---------------- price, as published
    price_rows = []
    for m in markets:
        hist = pw[(pw["market"] == m) & pw["price"].notna()].sort_values("date")
        d0, v0 = hist["date"].iloc[-1], float(hist["price"].iloc[-1])
        mm = w[w["market"] == m].sort_values("month")
        for d in new_dates:
            pub = mm[mm["avail"] <= d]
            if len(pub):
                r = pub.iloc[-1]
                price_rows.append((m, d, float(r["p"]), "fews_monthly_as_published", (d - r["month_end"]).days / 7, str(r["month"])))
            else:
                price_rows.append((m, d, v0, "carry_forward_last_weekly", (d - d0).days / 7, str(d0.date())))
    P = pd.DataFrame(price_rows, columns=["market", "date", "price", "oos_price_source", "oos_data_age_weeks", "oos_value_refers_to"])
    log["price_sources"] = P["oos_price_source"].value_counts().to_dict()
    log["months_published_per_market"] = w.groupby("market")["month"].nunique().to_dict()

    # ---------------- upstream from the extended price series
    allp = pd.concat([pw[["market", "date", "price"]], P[["market", "date", "price"]]]).set_index(["market", "date"])["price"]
    up = []
    for m in markets:
        um, lag = meta.loc[m, "upstream_market"], meta.loc[m, "upstream_lag_weeks"]
        for d in new_dates:
            if pd.isna(um):
                up.append(np.nan)
            else:
                up.append(allp.get((um, d - pd.Timedelta(weeks=int(lag))), np.nan))
    P["upstream_price"] = up

    # ---------------- climate
    c = pd.read_parquet(a.climate)
    c["date"] = pd.to_datetime(c["date"])
    c.loc[c["date"] > NDVI_SOURCE_END, ["ndvi_simple_average", "ndvi_inverse_distance"]] = np.nan
    ccols = ["rainfall_simple_average", "rainfall_inverse_distance", "ndvi_simple_average", "ndvi_inverse_distance",
             "is_interpolated_rainfall", "is_interpolated_ndvi"]
    P = P.merge(c[["market", "date"] + ccols], on=["market", "date"], how="left")
    log["climate_missing_cells"] = {k: int(P[k].isna().sum()) for k in ccols[:4]}

    # ---------------- static and calendar columns
    for col in ["state", "upstream_market", "upstream_lag_weeks", "market_kind", "market_id"]:
        P[col] = P["market"].map(meta[col])
    P["time_idx"] = ((P["date"] - t0).dt.days // 7).astype(int)
    for k in range(1, 5):
        P[f"sin{k}"] = np.sin(2 * np.pi * k * P["time_idx"] / 52.18)
        P[f"cos{k}"] = np.cos(2 * np.pi * k * P["time_idx"] / 52.18)
    chk = pw.drop_duplicates("date")
    assert np.allclose(np.sin(2 * np.pi * chk["time_idx"] / 52.18), chk["sin1"], atol=1e-9)
    P["is_proxy_price"] = False

    # ---------------- diesel variants
    sep = pw[pw["date"].dt.to_period("M") == pd.Period("2024-09")].groupby("market")["diesel"].first()
    rd = pd.read_parquet(a.raw_diesel)
    rd["period_date"] = pd.to_datetime(rd["period_date"])
    rd["month"] = rd["period_date"].dt.to_period("M")
    fm = rd.groupby(["market", "month"])["value"].mean()
    st = pw[~pw["is_interpolated_diesel"].astype(bool)].assign(month=lambda d: d["date"].dt.to_period("M")).groupby(["market", "month"])["diesel"].first()
    j = pd.concat([st, fm], axis=1, keys=["state", "fews"]).dropna().reset_index()
    j = j[j["month"] >= pd.Period("2023-01")]
    ratio = j.assign(r=np.log(j["state"] / j["fews"])).groupby("market")["r"].median().pipe(np.exp)
    log["diesel_splice_ratio_state_over_fews"] = ratio.round(4).to_dict()
    ad = api[(api["product"] == "Diesel") & api["value"].notna()].copy()
    ad["month"] = ad["period_date"].dt.to_period("M")
    adm = ad.groupby(["market", "month"])["value"].mean()
    P["month"] = P["date"].dt.to_period("M")
    actual_m = w.set_index(["market", "month"])["p"]
    outs = {}
    for variant in ("fewsdiesel", "flatdiesel"):
        Q = P.copy()
        dv, interp = [], []
        for m, mo in zip(Q["market"], Q["month"]):
            if mo == pd.Period("2024-09"):
                dv.append(float(sep.get(m, np.nan))); interp.append(False)
            elif variant == "fewsdiesel" and (m, mo) in adm.index and m in ratio.index:
                dv.append(float(adm[(m, mo)] * ratio[m])); interp.append(False)
            else:
                dv.append(np.nan); interp.append(True)
        Q["diesel"], Q["is_interpolated_diesel"] = dv, interp
        log[f"{variant}_new_diesel_months"] = int(Q.loc[~Q["is_interpolated_diesel"] & (Q["month"] > pd.Period("2024-09")), ["market", "month"]].drop_duplicates().shape[0])
        cols = list(pw.columns) + ["oos_price_source", "oos_data_age_weeks", "oos_value_refers_to"]
        # D-71 training targets: the real value of the week's own month, known from the 15th of
        # the next month; NaN where FEWS NET has no value. History rows: NaN (= use price).
        Q["target_price"] = [actual_m.get((m, mo), np.nan) for m, mo in zip(Q["market"], Q["month"])]
        Q["target_known_date"] = [(mo + 1).to_timestamp() + pd.Timedelta(days=AVAIL_DAY - 1) for mo in Q["month"]]
        cols2 = cols + ["target_price", "target_known_date"]
        Q["target_from_price"] = 0.0
        hist = pw.assign(oos_price_source="panel_weekly", oos_data_age_weeks=0.0, oos_value_refers_to="",
                         target_price=np.nan, target_known_date=pw["date"], target_from_price=1.0)
        ext = pd.concat([hist, Q[cols2 + ["target_from_price"]]], ignore_index=True)
        ext = ext.sort_values(["market", "date"]).reset_index(drop=True)
        ext["is_proxy_price"] = ext["is_proxy_price"].astype(bool)
        ext.to_parquet(f"data/panel_weekly_oos2026_{variant}.parquet", index=False)
        outs[variant] = ext
        log[f"{variant}_rows"] = len(ext)

    # ---------------- grid
    ext = outs["fewsdiesel"].set_index(["market", "date"])
    origins = [pd.Timestamp("2024-09-25")]
    for mo in pd.period_range("2024-10", "2026-07", freq="M"):
        d = mo.to_timestamp() + pd.Timedelta(days=AVAIL_DAY - 1)
        origins.append(d + pd.Timedelta(days=(2 - d.dayofweek) % 7))
    actual = w.set_index(["market", "month"])["p"]
    scored = [m for m in markets if m != "Aba"]
    rows = []
    for m in scored:
        for o in origins:
            naive = float(ext.loc[(m, o), "price"])
            age = float(ext.loc[(m, o), "oos_data_age_weeks"])
            for h in HORIZONS:
                t = o + pd.Timedelta(weeks=h)
                rows.append(dict(market=m, horizon=h, origin=o, target=t, target_month=str(t.to_period("M")),
                                 actual=actual.get((m, t.to_period("M")), np.nan), naive=naive, panel_fe=np.nan,
                                 origin_data_age_weeks=age, stale_over_13w=age > STALE_WEEKS))
    G = pd.DataFrame(rows)
    G.to_parquet("data/oos_grid_2024_26.parquet", index=False)
    log["grid"] = {"origins": [str(o.date()) for o in origins], "rows": len(G),
                   "scorable_by_h": G.dropna(subset=["actual"]).groupby("horizon").size().to_dict(),
                   "scorable_not_stale_by_h": G[G["actual"].notna() & ~G["stale_over_13w"]].groupby("horizon").size().to_dict(),
                   "scorable_by_market": G.dropna(subset=["actual"]).groupby("market").size().to_dict()}
    json.dump(log, open("data/oos_2024_26.log.json", "w"), indent=2, default=str)
    print(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main()
