"""
Per-market ranges (owner request 2026-11-06): "calmer markets should have calmer ranges; the composite ranges are still
too wide". Tests whether the production range (quantiles of ALL markets' past joint-forecast misses, pooled) is too wide
for calm markets and too narrow for volatile ones, and whether a per-market range does better.

    .venv/bin/python src/range_per_market.py --out outputs/range_per_market_20261106

Same past misses as production (src/range_experiments.table: joint forecast = average of the GRU's and the analyst's
log changes; miss = log(actual / forecast)), with FEWS NET's revised Jul and Aug 2026 actuals from the production grid
(data/monthly_grid_gapfilled_pit.parquet, rebuilt 2026-11-06). Walk-forward: a forecast with origin o only uses misses
whose target month is at or before o (the production rule). Scored on targets from Jan 2019, as the range experiments.

Methods (80% and 90% ranges around the same joint point forecast):
  pooled         production: quantiles of all markets' past misses
  own            quantiles of this market's own past misses (pooled when it has fewer than N_MIN)
  scaled_miss    pooled quantiles of misses divided by their market's miss scale, times this market's scale;
                 scale = root mean square of the market's past misses, shrunk to the pooled value with weight K
                 (scale = (n * s_market + K * s_pooled) / (n + K)), so a market with little history stays near pooled
  scaled_price   pooled quantiles times (this market's price volatility / all markets' price volatility); volatility =
                 root mean square of h-month log price changes in the production panel, observed months up to the origin

Scores: coverage (overall and per market), median width as % of the forecast, and the interval score (Gneiting and
Raftery 2007): width + (2 / alpha) x the distance by which the actual falls outside, in % of the actual; lower is
better, and a narrower range only wins if it still holds the price. Differences against pooled: mean per target month,
Newey-West standard error with h - 1 lags (markets in the same month are not independent; h-month targets overlap).

Writes scores.csv (method x horizon x period), by_market.csv (method x horizon x market), calm_vs_volatile.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_experiments import table  # noqa: E402

N_MIN, K = 24, 24
TEST_FROM = pd.Timestamp("2019-01-01")
PERIODS = [("2019 to Aug 2026", "2019-01-01", "2100-01-01"), ("2019-24", "2019-01-01", "2024-12-31"),
           ("Oct 2024 to Aug 2026", "2024-10-01", "2100-01-01")]
LEVELS = {80: (0.10, 0.90), 90: (0.05, 0.95)}
METHODS = ["pooled", "own", "scaled_miss", "scaled_price"]


def load(h: int) -> pd.DataFrame:
    x = table(h)
    g = pd.read_parquet("data/monthly_grid_gapfilled_pit.parquet").query("horizon == @h")
    g = g.rename(columns={"actual": "actual_rev"})[["market", "origin", "actual_rev"]]
    g["origin"] = pd.to_datetime(g.origin)
    x = x.merge(g, on=["market", "origin"], how="left")
    x["actual_changed"] = x.actual_rev.notna() & ~np.isclose(x.actual_rev, x.actual)
    x["actual"] = x.actual_rev.fillna(x.actual)
    x["y"] = np.log(x.actual / x.origin_price)
    x["e"] = x.y - x.r_joint
    x["point"] = x.origin_price * np.exp(x.r_joint)
    return x.dropna(subset=["e", "point"]).reset_index(drop=True)


def price_vol(h: int) -> pd.DataFrame:
    """Per market and month: h-month log price changes between observed FEWS NET months (production panel)."""
    p = pd.read_parquet("data/panel_monthly_gapfilled_pit.parquet")[["market", "date", "price", "price_flag"]]
    p["date"] = pd.to_datetime(p.date)
    w = p[p.price_flag == "observed"].pivot(index="date", columns="market", values="price")
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="MS"))
    return np.log(w / w.shift(h))


def ranges(x: pd.DataFrame, h: int) -> pd.DataFrame:
    dv = price_vol(h)
    test = x[x.target >= TEST_FROM]
    out = []
    for o, f in test.groupby("origin"):
        past = x[x.target <= o]
        e, m = past.e.to_numpy(), past.market.to_numpy()
        s_pool = np.sqrt(np.mean(e ** 2))
        n_by = past.groupby("market").e.agg(["size", lambda v: np.sqrt(np.mean(v ** 2))])
        n_by.columns = ["n", "s"]
        shr = (n_by.n * n_by.s + K * s_pool) / (n_by.n + K)
        z = e / pd.Series(m).map(shr).to_numpy()
        vh = dv[dv.index <= o]
        v_pool = np.sqrt(np.nanmean(vh.to_numpy() ** 2))
        v_mkt = np.sqrt((vh ** 2).mean())
        for r in f.itertuples():
            row = dict(market=r.market, origin=o, target=r.target, h=h, point=r.point, actual=r.actual,
                       actual_changed=r.actual_changed, n_pool=len(e), n_own=int(n_by.n.get(r.market, 0)),
                       scale_ratio_miss=float(shr.get(r.market, s_pool) / s_pool),
                       scale_ratio_price=float(v_mkt.get(r.market, np.nan) / v_pool) if pd.notna(v_mkt.get(r.market, np.nan)) else 1.0)
            own = e[m == r.market]
            for lv, (ql, qh) in LEVELS.items():
                q = {"pooled": np.quantile(e, [ql, qh]),
                     "own": np.quantile(own, [ql, qh]) if len(own) >= N_MIN else np.quantile(e, [ql, qh]),
                     "scaled_miss": np.quantile(z, [ql, qh]) * shr.get(r.market, s_pool),
                     "scaled_price": np.quantile(e, [ql, qh]) * row["scale_ratio_price"]}
                for k, (a, b) in q.items():
                    row[f"lo{lv}_{k}"], row[f"hi{lv}_{k}"] = r.point * np.exp(a), r.point * np.exp(b)
            out.append(row)
    return pd.DataFrame(out)


def interval_score(lo, hi, y, alpha):
    return 100 * ((hi - lo) + (2 / alpha) * np.clip(lo - y, 0, None) + (2 / alpha) * np.clip(y - hi, 0, None)) / y


def nw_se(d: np.ndarray, lags: int) -> float:
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    return float(np.sqrt(v / n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/range_per_market_20261106")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    R = []
    for h in (1, 3, 6):
        x = load(h)
        print(f"h={h}: {len(x)} misses, {int(x.actual_changed.sum())} actuals revised (Jul/Aug 2026)")
        R.append(ranges(x, h))
    d = pd.concat(R, ignore_index=True)
    for k in METHODS:
        for lv, (ql, qh) in LEVELS.items():
            lo, hi = d[f"lo{lv}_{k}"], d[f"hi{lv}_{k}"]
            d[f"in{lv}_{k}"] = (d.actual >= lo) & (d.actual <= hi)
            d[f"w{lv}_{k}"] = 100 * (hi - lo) / d.point
            d[f"is{lv}_{k}"] = interval_score(lo, hi, d.actual, 2 * ql)
    d.round(4).to_csv(out / "rows.csv.gz", index=False)

    rows = []
    for h in (1, 3, 6):
        for per, s0, s1 in PERIODS:
            q = d[(d.h == h) & (d.target >= s0) & (d.target <= s1)]
            for k in METHODS:
                r = dict(h=h, period=per, method=k, n=len(q))
                for lv in LEVELS:
                    r[f"inside{lv}_pct"] = 100 * q[f"in{lv}_{k}"].mean()
                    r[f"width{lv}_median_pct"] = q[f"w{lv}_{k}"].median()
                    r[f"interval_score{lv}"] = q[f"is{lv}_{k}"].mean()
                    if k != "pooled":
                        diff = (q[f"is{lv}_{k}"] - q[f"is{lv}_pooled"]).groupby(q.target).mean().sort_index().to_numpy()
                        se = nw_se(diff, h - 1)
                        r[f"is{lv}_vs_pooled_pct"] = 100 * diff.mean() / q[f"is{lv}_pooled"].mean()
                        r[f"is{lv}_vs_pooled_t"] = diff.mean() / se if se > 0 else np.nan
                pm = q.groupby("market")[f"in80_{k}"].mean() * 100
                r["markets"] = len(pm)
                r["inside80_market_min"], r["inside80_market_max"] = pm.min(), pm.max()
                r["inside80_market_mean_abs_gap_from_80"] = (pm - 80).abs().mean()
                rows.append(r)
    S = pd.DataFrame(rows)
    S.round(2).to_csv(out / "scores.csv", index=False)

    q = d[d.target >= TEST_FROM]
    B = []
    for (h, mk), g in q.groupby(["h", "market"]):
        r = dict(h=h, market=mk, n=len(g), miss_rms_pct=100 * np.sqrt(np.mean(np.log(g.actual / g.point) ** 2)),
                 scale_ratio_miss=g.scale_ratio_miss.iloc[-1], scale_ratio_price=g.scale_ratio_price.iloc[-1])
        for k in METHODS:
            r[f"inside80_{k}"] = 100 * g[f"in80_{k}"].mean()
            r[f"width80_{k}"] = g[f"w80_{k}"].median()
            r[f"is80_{k}"] = g[f"is80_{k}"].mean()
        B.append(r)
    B = pd.DataFrame(B)
    B.round(2).to_csv(out / "by_market.csv", index=False)

    # calm vs volatile: markets split at the median of their own test-period miss size, per horizon
    C = []
    for h in (1, 3, 6):
        b = B[B.h == h]
        cut = b.miss_rms_pct.median()
        for grp, mk in [("calm half", b[b.miss_rms_pct <= cut].market), ("volatile half", b[b.miss_rms_pct > cut].market)]:
            g = q[(q.h == h) & q.market.isin(mk)]
            r = dict(h=h, group=grp, markets="; ".join(sorted(mk)), n=len(g))
            for k in METHODS:
                r[f"inside80_{k}"] = 100 * g[f"in80_{k}"].mean()
                r[f"width80_{k}"] = g[f"w80_{k}"].median()
                r[f"is80_{k}"] = g[f"is80_{k}"].mean()
            C.append(r)
    C = pd.DataFrame(C)
    C.round(2).to_csv(out / "calm_vs_volatile.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["h", "period", "method", "n", "inside80_pct", "width80_median_pct", "interval_score80", "is80_vs_pooled_pct", "is80_vs_pooled_t",
            "inside90_pct", "interval_score90", "is90_vs_pooled_pct", "inside80_market_min", "inside80_market_max", "inside80_market_mean_abs_gap_from_80"]
    print(S[cols].round(1).to_string(index=False))
    print(C.drop(columns="markets").round(1).to_string(index=False))
    print(C[["h", "group", "markets"]].to_string(index=False))
    print(B[["h", "market", "n", "miss_rms_pct", "scale_ratio_miss", "scale_ratio_price", "inside80_pooled", "width80_pooled",
             "inside80_scaled_miss", "width80_scaled_miss", "inside80_own", "width80_own"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
