"""
Calm-period vs shock-period ranges (owner hypothesis 2026-11-06): one range for calm periods and one for shock periods,
on its own and layered on the per-market ranges of src/range_per_market.py.

    .venv/bin/python src/range_regimes.py --out outputs/range_regimes_20261106

Regime of a forecast = regime of its origin month, from information available at the origin. Two definitions, both
fixed before the first run:
  price  national price swing: root mean square, over all markets, of 1-month log changes in FEWS NET observed prices
         over the 3 months ending at the origin (production panel)
  fx     official naira move: absolute 3-month log change of the CBN central rate (monthly mean) ending at the origin
A month is a SHOCK month when its value is above the 75th percentile of all months up to and including the origin
(expanding, point-in-time); otherwise CALM. Past misses are labelled with the same threshold as the forecast they serve.

Methods (80% and 90% ranges around the same joint point forecast; walk-forward, misses with target at or before origin):
  pooled          production
  own             per-market (src/range_per_market.py)
  regime          quantiles of past misses whose origin was in the same regime (pooled when fewer than N_REGIME)
  regime_own      own-market quantiles stretched by the regime: own quantile x (regime pooled quantile / all pooled
                  quantile), lower and upper ends separately (log scale)
  regime_scaled   same-regime quantiles of misses divided by their market's shrunk miss scale, times this market's scale
Scores and test as src/range_per_market.py (interval score, Newey-West t on target-month means vs pooled).
Writes scores.csv, by_regime.csv, regime_months.csv, rows.csv.gz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_per_market import K, LEVELS, N_MIN, PERIODS, TEST_FROM, interval_score, load, nw_se  # noqa: E402

N_REGIME = 60
Q_SHOCK = 0.75
FX = "data/external/20261002_CBN_API_Nigeria_USDNGN_Daily_2001-12_2026-09_v1.csv"
METHODS = ["pooled", "own", "regime", "regime_own", "regime_scaled"]


def indicators() -> pd.DataFrame:
    p = pd.read_parquet("data/panel_monthly_gapfilled_pit.parquet")[["market", "date", "price", "price_flag"]]
    p["date"] = pd.to_datetime(p.date)
    w = p[p.price_flag == "observed"].pivot(index="date", columns="market", values="price")
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="MS"))
    r = np.log(w / w.shift(1)) ** 2
    price = np.sqrt(r.rolling(3, min_periods=2).sum().sum(1) / r.rolling(3, min_periods=2).count().sum(1))
    f = pd.read_csv(FX, parse_dates=["date"])
    fm = f.set_index("date").centralrate.resample("MS").mean()
    fx = np.log(fm / fm.shift(3)).abs()
    return pd.DataFrame({"price": price, "fx": fx}).loc["2012-01-01":]


def shock_flags(ind: pd.Series, o: pd.Timestamp) -> pd.Series:
    hist = ind[ind.index <= o].dropna()
    return hist > hist.quantile(Q_SHOCK)


def ranges(x: pd.DataFrame, h: int, ind: pd.DataFrame, kind: str) -> pd.DataFrame:
    test = x[x.target >= TEST_FROM]
    out = []
    for o, f in test.groupby("origin"):
        flags = shock_flags(ind[kind], o)
        if o not in flags.index:
            continue
        reg_o = bool(flags[o])
        past = x[x.target <= o]
        e, m = past.e.to_numpy(), past.market.to_numpy()
        reg_p = past.origin.map(flags).fillna(False).to_numpy().astype(bool)
        same = e[reg_p == reg_o]
        s_pool = np.sqrt(np.mean(e ** 2))
        n_by = past.groupby("market").e.agg(["size", lambda v: np.sqrt(np.mean(v ** 2))])
        n_by.columns = ["n", "s"]
        shr = (n_by.n * n_by.s + K * s_pool) / (n_by.n + K)
        z = e / pd.Series(m).map(shr).to_numpy()
        z_same = z[reg_p == reg_o]
        for r in f.itertuples():
            row = dict(market=r.market, origin=o, target=r.target, h=h, regime_kind=kind, shock=reg_o, point=r.point,
                       actual=r.actual, n_same_regime=len(same))
            own = e[m == r.market]
            for lv, (ql, qh) in LEVELS.items():
                qp = np.quantile(e, [ql, qh])
                qr = np.quantile(same, [ql, qh]) if len(same) >= N_REGIME else qp
                qo = np.quantile(own, [ql, qh]) if len(own) >= N_MIN else qp
                q = {"pooled": qp, "own": qo, "regime": qr,
                     "regime_own": qo * (qr / qp),
                     "regime_scaled": (np.quantile(z_same, [ql, qh]) if len(z_same) >= N_REGIME else np.quantile(z, [ql, qh]))
                     * shr.get(r.market, s_pool)}
                for k, (a, b) in q.items():
                    row[f"lo{lv}_{k}"], row[f"hi{lv}_{k}"] = r.point * np.exp(a), r.point * np.exp(b)
            out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/range_regimes_20261106")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ind = indicators()
    reg = pd.DataFrame({k: shock_flags(ind[k], ind.index.max()) for k in ("price", "fx")})
    pd.concat([ind, reg.add_suffix("_shock_with_full_history")], axis=1).round(4).to_csv(out / "regime_months.csv")
    D = []
    for h in (1, 3, 6):
        x = load(h)
        for kind in ("price", "fx"):
            D.append(ranges(x, h, ind, kind))
    d = pd.concat(D, ignore_index=True)
    for k in METHODS:
        for lv, (ql, qh) in LEVELS.items():
            lo, hi = d[f"lo{lv}_{k}"], d[f"hi{lv}_{k}"]
            d[f"in{lv}_{k}"] = (d.actual >= lo) & (d.actual <= hi)
            d[f"w{lv}_{k}"] = 100 * (hi - lo) / d.point
            d[f"is{lv}_{k}"] = interval_score(lo, hi, d.actual, 2 * ql)
    d.round(4).to_csv(out / "rows.csv.gz", index=False)

    rows, regs = [], []
    for kind in ("price", "fx"):
        for h in (1, 3, 6):
            for per, s0, s1 in PERIODS:
                q = d[(d.regime_kind == kind) & (d.h == h) & (d.target >= s0) & (d.target <= s1)]
                for k in METHODS:
                    if kind == "fx" and k in ("pooled", "own"):
                        continue  # identical to the price block
                    r = dict(regime_kind=kind, h=h, period=per, method=k, n=len(q), shock_share_pct=100 * q.shock.mean())
                    for lv in LEVELS:
                        r[f"inside{lv}_pct"] = 100 * q[f"in{lv}_{k}"].mean()
                        r[f"width{lv}_median_pct"] = q[f"w{lv}_{k}"].median()
                        r[f"interval_score{lv}"] = q[f"is{lv}_{k}"].mean()
                        if k != "pooled":
                            diff = (q[f"is{lv}_{k}"] - q[f"is{lv}_pooled"]).groupby(q.target).mean().sort_index().to_numpy()
                            se = nw_se(diff, h - 1)
                            r[f"is{lv}_vs_pooled_pct"] = 100 * diff.mean() / q[f"is{lv}_pooled"].mean()
                            r[f"is{lv}_vs_pooled_t"] = diff.mean() / se if se > 0 else np.nan
                    rows.append(r)
        q = d[(d.regime_kind == kind) & (d.target >= TEST_FROM)]
        for (h, s), g in q.groupby(["h", "shock"]):
            r = dict(regime_kind=kind, h=h, regime="shock" if s else "calm", n=len(g))
            for k in METHODS:
                r[f"inside80_{k}"] = 100 * g[f"in80_{k}"].mean()
                r[f"below80_{k}"] = 100 * (g.actual < g[f"lo80_{k}"]).mean()
                r[f"width80_{k}"] = g[f"w80_{k}"].median()
                r[f"is80_{k}"] = g[f"is80_{k}"].mean()
            regs.append(r)
    S, R = pd.DataFrame(rows), pd.DataFrame(regs)
    S.round(2).to_csv(out / "scores.csv", index=False)
    R.round(2).to_csv(out / "by_regime.csv", index=False)
    pd.set_option("display.width", 260)
    c = ["regime_kind", "h", "period", "method", "n", "shock_share_pct", "inside80_pct", "width80_median_pct", "interval_score80",
         "is80_vs_pooled_pct", "is80_vs_pooled_t", "inside90_pct", "is90_vs_pooled_pct"]
    print(S[c].round(1).to_string(index=False))
    print(R.round(1).to_string(index=False))
    sh = reg[reg.index >= "2018-10-01"]
    for k in ("price", "fx"):
        mm = sh.index[sh[k].fillna(False).astype(bool)].strftime("%Y-%m").tolist()
        print(k, "shock months (full-history threshold) since Oct 2018:", len(mm), mm)


if __name__ == "__main__":
    main()
