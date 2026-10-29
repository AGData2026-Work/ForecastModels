"""
Per-market flat features for the 2026-10-21 checks, taken from the analyst's v3 regression model
(Maize_Model_v3_Explained.pdf). Read by src/data.py attach_checks (features.extra_flat_path).

    python src/build_extra_signals.py weekly --out data/external/20261021_FEWSNET_WeeklyExit_Monthly_v1.csv
    python src/build_extra_signals.py weekly --weekly-csv data/external/20261021_FEWSNET_API_Nigeria_WeeklyMaizeWhite_v1.csv \
        [--as-published] --out data/external/20261021_FEWSNET_WeeklyExit_Monthly_v2.csv
    python src/build_extra_signals.py producers --out data/external/20261021_ProducerIndex_Monthly_v1.csv

weekly: where the month ended against its own average, from FEWS NET's weekly quotes (data/panel_weekly.parquet,
proxy weeks dropped). wk_exit = log(last observed week of month o / mean of month o's observed weeks), needing at least
2 observed weeks; wk_exit_national = mean wk_exit over the markets that have one; wk_exit_rel = wk_exit minus the
national mean; wk_mask = 1 where wk_exit exists. The monthly FEWS NET price is the mean of these same weeks, so the
quotes are public with it (by the 15th of o+1). The weekly panel ends 2024-09-18: every later month has mask 0.
--weekly-csv reads the FEWS NET API pull instead (src/fetch_fewsnet_weekly.py, same values, weeks to Aug 2026).
--as-published keeps only weeks whose API status_changed is on or before the 15th of o+1 (2025-26: 73% of month-end
weeks; older timestamps include later bulk edits, so this understates what was public then).

producers: one price for the producing areas (Dandume, Saminaka, Giwa, Kaura Namoda, Gwandu Dodoru), chain-linked so
a market dropping out does not move it: the index's monthly change is the mean log change of the producers priced in
both months. prod_chg1 / prod_chg3 = the index's log change over 1 / 3 months to o; prod_gap = log(market price at o)
minus the log index (its level is arbitrary, so a market's typical gap sits in the market's own constant). Prices are
the monthly panel's (data/panel_monthly_gapfilled_pit.parquet, the gap-filled v3 prices the model already sees), same
month, as the upstream channel. prod_mask = 1 where all three exist.

importwin: the analyst's import window, ON in month o when import duty is 0 and the national average price (mean log
price over markets priced that month) is above import parity (data/external/..._NADIH_ImportParity_...csv, official
rate). iw_gap = window x (log national price - log parity); iw_flag = window; national, the same for every market. Used
ALONGSIDE the production parity gap, not in place of it.

crossmkt: the analyst's market-vs-rest signals. gap = log market price - national mean log price; out_of_line = gap
minus its mean over the previous 12 months (at least 6 priced); own_jump = this market's 1-month log change minus the
national mean change; xm_mask = 1 where both exist.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

PRODUCERS = ["Dandume", "Saminaka", "Giwa", "Kaura Namoda", "Gwandu, Dodoru"]


def weekly(path: str, api_csv: str | None = None, as_published: bool = False) -> pd.DataFrame:
    if api_csv:
        w = pd.read_csv(api_csv, parse_dates=["period_date", "status_changed"])
        w = w.rename(columns={"period_date": "date", "price_ngn_per_kg": "price"})
        if as_published:
            due = (w.date.dt.to_period("M") + 1).dt.to_timestamp() + pd.Timedelta(days=14)
            w = w[w.status_changed <= due]
    else:
        w = pd.read_parquet(path)
        w = w[w.price.notna() & ~w.is_proxy_price.astype(bool)]
    w = w.sort_values(["market", "date"])
    w["origin_month"] = w.date.dt.to_period("M").astype(str)
    g = w.groupby(["market", "origin_month"]).price
    d = pd.DataFrame({"n": g.size(), "last": g.last(), "mean": g.mean()}).reset_index()
    d = d[d.n >= 2]
    d["wk_exit"] = np.log(d["last"] / d["mean"])
    d["wk_exit_national"] = d.groupby("origin_month").wk_exit.transform("mean")
    d["wk_exit_rel"] = d.wk_exit - d.wk_exit_national
    d["wk_mask"] = 1.0
    return d[["market", "origin_month", "wk_exit", "wk_exit_national", "wk_exit_rel", "wk_mask"]]


def producers(path: str) -> pd.DataFrame:
    m = pd.read_parquet(path)
    lp = np.log(m.pivot(index="date", columns="market", values="price").sort_index())
    chg = lp[PRODUCERS].diff().mean(axis=1, skipna=True)        # NaN only when no producer is priced in both months
    idx = chg.fillna(0).cumsum() + lp[PRODUCERS].iloc[0].mean()
    idx[lp[PRODUCERS].notna().sum(axis=1) == 0] = np.nan       # no producer priced at all: no index that month
    out = []
    for mk in lp.columns:
        out.append(pd.DataFrame({"market": mk, "origin_month": lp.index.to_period("M").astype(str),
                                 "prod_chg1": idx.diff(1).values, "prod_chg3": idx.diff(3).values,
                                 "prod_gap": (lp[mk] - idx).values}))
    d = pd.concat(out, ignore_index=True)
    d["prod_mask"] = d[["prod_chg1", "prod_chg3", "prod_gap"]].notna().all(axis=1).astype(float)
    return d


def _monthly_logs(path: str) -> pd.DataFrame:
    m = pd.read_parquet(path)
    return np.log(m.pivot(index="date", columns="market", values="price").sort_index())


def _long(lp: pd.DataFrame, cols: dict) -> pd.DataFrame:
    out = []
    for mk in lp.columns:
        out.append(pd.DataFrame({"market": mk, "origin_month": lp.index.to_period("M").astype(str),
                                 **{k: (v[mk].values if isinstance(v, pd.DataFrame) else v.values) for k, v in cols.items()}}))
    return pd.concat(out, ignore_index=True)


def importwin(path: str, parity_path: str) -> pd.DataFrame:
    lp = _monthly_logs(path)
    q = pd.read_csv(parity_path).set_index("month")
    months = lp.index.to_period("M").astype(str)
    nat = lp.mean(axis=1, skipna=True)
    gap = nat - np.log(q.parity_ngn_per_kg.reindex(months).values)
    flag = ((q.duty.reindex(months).values == 0) & (gap > 0)).astype(float)
    d = _long(lp, {"iw_gap": pd.Series(np.where(flag > 0, gap, 0.0), index=lp.index),
                   "iw_flag": pd.Series(flag, index=lp.index)})
    on = sorted(set(d.loc[d.iw_flag > 0, "origin_month"]))
    print(f"import window ON {len(on)} months: {on[0] if on else '-'} to {on[-1] if on else '-'}")
    return d


def crossmkt(path: str) -> pd.DataFrame:
    lp = _monthly_logs(path)
    gap = lp.sub(lp.mean(axis=1, skipna=True), axis=0)
    usual = gap.shift(1).rolling(12, min_periods=6).mean()
    dl = lp.diff()
    jump = dl.sub(dl.mean(axis=1, skipna=True), axis=0)
    d = _long(lp, {"out_of_line": gap - usual, "own_jump": jump})
    d["xm_mask"] = d[["out_of_line", "own_jump"]].notna().all(axis=1).astype(float)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["weekly", "producers", "importwin", "crossmkt"])
    ap.add_argument("--parity", default="data/external/20261012_NADIH_ImportParity_Maize_Monthly_2006-01_2026-09_v1.csv")
    ap.add_argument("--weekly-panel", default="data/panel_weekly.parquet")
    ap.add_argument("--weekly-csv", default=None, help="FEWS NET API weekly pull (src/fetch_fewsnet_weekly.py)")
    ap.add_argument("--as-published", action="store_true")
    ap.add_argument("--monthly-panel", default="data/panel_monthly_gapfilled_pit.parquet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = {"weekly": lambda: weekly(a.weekly_panel, a.weekly_csv, a.as_published),
         "producers": lambda: producers(a.monthly_panel),
         "importwin": lambda: importwin(a.monthly_panel, a.parity),
         "crossmkt": lambda: crossmkt(a.monthly_panel)}[a.kind]()
    d.to_csv(a.out, index=False)
    mask = d.filter(regex="mask|flag").iloc[:, 0]
    print(f"{a.kind}: {len(d)} rows, {int(mask.sum())} with a value, months {d.origin_month.min()} to "
          f"{d.loc[mask > 0, 'origin_month'].max()} -> {a.out}")


if __name__ == "__main__":
    main()
