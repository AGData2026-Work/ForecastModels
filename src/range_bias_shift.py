"""
Bias-shifted ranges (owner go-ahead 2026-11-06). The range experiments found the 2025-26 range shortfall is bias, not
width: at 3 and 6 months every miss fell below the range. Test: keep each range's width, move its centre to where the
recent misses have been, point forecast unchanged.

    .venv/bin/python src/range_bias_shift.py --out outputs/range_bias_shift_20261106

Past misses as src/range_per_market.py (joint forecast, FEWS NET revised actuals, miss e = log(actual / joint)).
Walk-forward: a forecast with origin o uses only misses whose target month is at or before o.

Methods, fixed before the first run (80% and 90% ranges around the same joint point forecast):
  production     the D-80 rule: own-market quantiles at 1 month (pooled if fewer than 24), pooled at 3 and 6
  shift12        production quantiles - median(all past misses used) + median(pooled misses with target in the last
                 12 months up to o); i.e. the long-run centre replaced by the recent one
  shift12_half   half of that shift (shrunk toward production)
  shift6         as shift12 with a 6-month window
  shift12_own    as shift12, recent median from this market's own misses (at least 6, else the pooled recent median)
Recent windows need at least 6 misses; otherwise no shift.

Scores as src/range_per_market.py: coverage, share below / above, median width, interval score (lower is better),
difference vs production as mean over target months with Newey-West t (h - 1 lags). Also split by what the price did
(actual vs origin price: fell, rose) and how often the point forecast falls outside its own shifted range.
Writes scores.csv, by_direction.csv, shifts.csv, rows.csv.gz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from range_per_market import LEVELS, N_MIN, TEST_FROM, interval_score, load, nw_se  # noqa: E402

PERIODS = [("2019 to latest", "2019-01-01", "2100-01-01"), ("2019-22", "2019-01-01", "2022-12-31"),
           ("2023-24", "2023-01-01", "2024-12-31"), ("2025 to latest", "2025-01-01", "2100-01-01")]
METHODS = ["production", "shift12", "shift12_half", "shift6", "shift12_own"]
N_RECENT = 6


def ranges(x: pd.DataFrame, h: int) -> pd.DataFrame:
    test = x[x.target >= TEST_FROM]
    out = []
    for o, f in test.groupby("origin"):
        past = x[x.target <= o]
        e, m, t = past.e.to_numpy(), past.market.to_numpy(), past.target.to_numpy()
        r12 = t > np.datetime64(o - pd.DateOffset(months=12))
        r6 = t > np.datetime64(o - pd.DateOffset(months=6))
        med12 = np.median(e[r12]) if r12.sum() >= N_RECENT else None
        med6 = np.median(e[r6]) if r6.sum() >= N_RECENT else None
        for r in f.itertuples():
            own = e[m == r.market]
            use_own = h == 1 and len(own) >= N_MIN
            base = own if use_own else e
            c0 = np.median(base)
            own12 = e[(m == r.market) & r12]
            med12_own = np.median(own12) if len(own12) >= N_RECENT else med12
            s = {"production": 0.0,
                 "shift12": 0.0 if med12 is None else med12 - c0,
                 "shift12_half": 0.0 if med12 is None else 0.5 * (med12 - c0),
                 "shift6": 0.0 if med6 is None else med6 - c0,
                 "shift12_own": 0.0 if med12_own is None else med12_own - c0}
            row = dict(market=r.market, origin=o, target=r.target, h=h, point=r.point, actual=r.actual,
                       origin_price=r.origin_price, n_recent12=int(r12.sum()), n_recent12_own=len(own12),
                       **{f"shift_{k}": v for k, v in s.items()})
            for lv, (ql, qh) in LEVELS.items():
                qb = np.quantile(base, [ql, qh])
                for k, v in s.items():
                    row[f"lo{lv}_{k}"], row[f"hi{lv}_{k}"] = r.point * np.exp(qb[0] + v), r.point * np.exp(qb[1] + v)
            out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/range_bias_shift_20261106")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.concat([ranges(load(h), h) for h in (1, 3, 6)], ignore_index=True)
    d["direction"] = np.where(d.actual < d.origin_price, "fell", "rose or flat")
    for k in METHODS:
        for lv, (ql, qh) in LEVELS.items():
            lo, hi = d[f"lo{lv}_{k}"], d[f"hi{lv}_{k}"]
            d[f"in{lv}_{k}"] = (d.actual >= lo) & (d.actual <= hi)
            d[f"below{lv}_{k}"] = d.actual < lo
            d[f"above{lv}_{k}"] = d.actual > hi
            d[f"w{lv}_{k}"] = 100 * (hi - lo) / d.point
            d[f"is{lv}_{k}"] = interval_score(lo, hi, d.actual, 2 * ql)
            d[f"point_out{lv}_{k}"] = (d.point < lo) | (d.point > hi)
    d.round(4).to_csv(out / "rows.csv.gz", index=False)

    rows = []
    for h in (1, 3, 6):
        for per, s0, s1 in PERIODS:
            q = d[(d.h == h) & (d.target >= s0) & (d.target <= s1)]
            for k in METHODS:
                r = dict(h=h, period=per, method=k, n=len(q), median_shift_pct=100 * q[f"shift_{k}"].median())
                for lv in LEVELS:
                    r[f"inside{lv}_pct"] = 100 * q[f"in{lv}_{k}"].mean()
                    r[f"below{lv}_pct"] = 100 * q[f"below{lv}_{k}"].mean()
                    r[f"above{lv}_pct"] = 100 * q[f"above{lv}_{k}"].mean()
                    r[f"width{lv}_median_pct"] = q[f"w{lv}_{k}"].median()
                    r[f"point_outside{lv}_pct"] = 100 * q[f"point_out{lv}_{k}"].mean()
                    r[f"interval_score{lv}"] = q[f"is{lv}_{k}"].mean()
                    if k != "production":
                        diff = (q[f"is{lv}_{k}"] - q[f"is{lv}_production"]).groupby(q.target).mean().sort_index().to_numpy()
                        se = nw_se(diff, h - 1)
                        r[f"is{lv}_vs_production_pct"] = 100 * diff.mean() / q[f"is{lv}_production"].mean()
                        r[f"is{lv}_vs_production_t"] = diff.mean() / se if se > 0 else np.nan
                rows.append(r)
    S = pd.DataFrame(rows)
    S.round(2).to_csv(out / "scores.csv", index=False)

    B = []
    q = d[d.target >= TEST_FROM]
    for (h, dr), g in q.groupby(["h", "direction"]):
        r = dict(h=h, direction=dr, n=len(g))
        for k in METHODS:
            r[f"inside80_{k}"] = 100 * g[f"in80_{k}"].mean()
            r[f"below80_{k}"] = 100 * g[f"below80_{k}"].mean()
            r[f"above80_{k}"] = 100 * g[f"above80_{k}"].mean()
            r[f"is80_{k}"] = g[f"is80_{k}"].mean()
        B.append(r)
    B = pd.DataFrame(B)
    B.round(2).to_csv(out / "by_direction.csv", index=False)
    sh = d.groupby(["h", "origin"])[[f"shift_{k}" for k in METHODS[1:]]].median().mul(100).round(2)
    sh.to_csv(out / "shifts.csv")

    pd.set_option("display.width", 260)
    c = ["h", "period", "method", "n", "median_shift_pct", "inside80_pct", "below80_pct", "above80_pct", "width80_median_pct",
         "point_outside80_pct", "interval_score80", "is80_vs_production_pct", "is80_vs_production_t", "inside90_pct",
         "is90_vs_production_pct", "is90_vs_production_t"]
    print(S[c].round(1).to_string(index=False))
    print(B.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
