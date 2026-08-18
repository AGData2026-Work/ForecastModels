"""
Infer whether the incumbent panel fixed-effects forecasts were produced
CONDITIONALLY (driver values over the forecast window plugged in as if known) or
UNCONDITIONALLY (only information available at the origin).

Why this needs inferring at all: the script that produced
07_panel_fe_forecasts.parquet was not available, and the answer decides which of
the two challenger arms is a like-for-like comparison. A challenger forced to
work from origin-time information cannot fairly be measured against an incumbent
that was handed realised rainfall and diesel.

Method: rebuild panel FE both ways with the documented specification (own-price
lags 1, 2, 4, 13, 52; market fixed effects; no time effects; diesel, upstream
price, inverse-distance rainfall and NDVI; Fourier K=2), direct h-step, expanding
walk-forward on the real origin grid, and ask which version's PREDICTIONS sit
closer to the published ones.

This is a replication, not the original code. It is evidence, not proof.

    python src/diagnose_convention.py --out outputs/convention_diagnostic.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LAGS = [1, 2, 4, 13, 52]
HORIZONS = [4, 13, 26]


def load(panel_path: str, grid_path: str):
    d = pd.read_parquet(panel_path)
    f = pd.read_parquet(grid_path)
    dates = pd.DatetimeIndex(sorted(d["date"].unique()))
    markets = sorted(d["market"].unique())
    pos = {t: i for i, t in enumerate(dates)}
    midx = {m: j for j, m in enumerate(markets)}

    def g(col):
        return (d.pivot(index="date", columns="market", values=col)
                 .reindex(dates)[markets].to_numpy(float))

    arrs = {k: g(k) for k in ["price", "diesel", "upstream_price",
                              "rainfall_inverse_distance", "ndvi_inverse_distance"]}
    fou = (d.drop_duplicates("date").set_index("date")[["sin1", "cos1", "sin2", "cos2"]]
             .reindex(dates).to_numpy(float))
    return d, f, dates, markets, pos, midx, arrs, fou


def design(arrs, fou, ts, ms, h, realised_drivers):
    di = ts + h if realised_drivers else ts
    cols = [arrs["price"][ts - (L - 1), ms] for L in LAGS]
    cols += [arrs["diesel"][di, ms], arrs["upstream_price"][di, ms],
             arrs["rainfall_inverse_distance"][di, ms],
             arrs["ndvi_inverse_distance"][di, ms]]
    cols += [fou[di, k] for k in range(fou.shape[1])]
    return np.column_stack(cols)


def run(realised_drivers, panel_path, grid_path):
    d, f, dates, markets, pos, midx, arrs, fou = load(panel_path, grid_path)
    T, M = arrs["price"].shape
    out = []
    for h in HORIZONS:
        g = f[f["horizon"] == h]
        ts, ms = np.meshgrid(np.arange(51, T - h), np.arange(M), indexing="ij")
        ts, ms = ts.ravel(), ms.ravel()
        X = design(arrs, fou, ts, ms, h, realised_drivers)
        y = arrs["price"][ts + h, ms]
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        Xa, ya, ta, ma = X[ok], y[ok], ts[ok], ms[ok]
        D = np.zeros((len(Xa), M))
        D[np.arange(len(Xa)), ma] = 1.0
        A = np.hstack([Xa, D])
        preds = []
        for o, sub in g.groupby("origin"):
            oi = pos[pd.Timestamp(o)]
            tr = ta + h <= oi                     # target realised strictly by the origin
            if tr.sum() < 200:
                continue
            beta, *_ = np.linalg.lstsq(A[tr], ya[tr], rcond=None)
            for _, r in sub.iterrows():
                mj = midx[r["market"]]
                x = design(arrs, fou, np.array([oi]), np.array([mj]), h, realised_drivers)[0]
                if not np.isfinite(x).all():
                    continue
                dm = np.zeros(M)
                dm[mj] = 1.0
                preds.append((r["actual"], float(np.concatenate([x, dm]) @ beta),
                              r["panel_fe"], r["naive"]))
        p = pd.DataFrame(preds, columns=["actual", "replica", "panel_fe", "naive"])
        out.append(dict(
            convention="conditional" if realised_drivers else "unconditional",
            h=h, n=len(p),
            replica_MAE=np.mean(np.abs(p.actual - p.replica)),
            published_panel_fe_MAE=np.mean(np.abs(p.actual - p.panel_fe)),
            naive_MAE=np.mean(np.abs(p.actual - p.naive)),
            replica_vs_published_MAD=np.mean(np.abs(p.replica - p.panel_fe)),
            corr_replica_published=np.corrcoef(p.replica, p.panel_fe)[0, 1],
        ))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panel_weekly.parquet")
    ap.add_argument("--grid", default="data/07_panel_fe_forecasts.parquet")
    ap.add_argument("--out", default="outputs/convention_diagnostic.csv")
    a = ap.parse_args()

    res = pd.concat([run(False, a.panel, a.grid), run(True, a.panel, a.grid)],
                    ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    print(res.round(2).to_string(index=False))
    print("\nRead the replica_vs_published_MAD column: the convention whose replica "
          "predictions sit closer to the published ones is the likelier convention.")


if __name__ == "__main__":
    main()
