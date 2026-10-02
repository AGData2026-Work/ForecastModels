"""
FX pass-through overlay on an existing GRU run (FX restructuring, option 1). Retrains nothing.

    python src/fx_overlay.py --run-dir outputs/build3_underfit_corrected/GRU_unconditional --frequency W
    python src/fx_overlay.py --run-dir outputs/build3_monthly/GRU_unconditional --frequency MS

Rules, fixed before any result was seen:
- d = log(FX_known(t) / FX_known(t - L months)), the naira's depreciation over the last L
  months (primary L = 6; 3 and 12 reported as sensitivity). FX_known is point-in-time: the
  latest monthly official rate published by the forecast date, i.e. the previous month for
  a weekly origin, the origin month itself for a month-end origin.
- adjusted = GRU x exp(beta_h x d). beta_h is re-estimated at every origin t by least
  squares without intercept, log(actual / GRU) = beta_h d, on past forecasts of the same
  horizon whose target has already happened (target <= t). No intercept, so beta measures
  FX alone, not general bias.
- beta_h = 0 until at least MIN_INFORMATIVE realised forecasts have |d| >= 2%.
- "capped" variant: d is clipped to the largest |d| among the realised forecasts beta was
  fitted on, so the overlay never extrapolates beyond what it has seen.
Applied to the seed-median forecast and to each seed separately (for the seed-paired t).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import diebold_mariano, mae, mape  # noqa: E402

MIN_INFORMATIVE = 30
INFORMATIVE = 0.02


def fx_monthly() -> pd.Series:
    x = pd.read_excel("data/external/fx_rate.xlsx")
    x["Months"] = pd.to_datetime(x["Months"])
    s = x.set_index("Months")["Official Exchange Rate"].sort_index()
    s.index = s.index.to_period("M")
    return s


def depreciation(origins: pd.Series, freq: str, L: int, fx: pd.Series) -> np.ndarray:
    known = origins.dt.to_period("M") - (1 if freq == "W" else 0)
    a = fx.reindex(known).to_numpy(dtype=float)
    b = fx.reindex(known - L).to_numpy(dtype=float)
    return np.log(a / b)


def overlay(df: pd.DataFrame, pred_col: str, capped: bool) -> np.ndarray:
    """df: one horizon, columns origin, target, actual, d, pred_col. Returns adjusted forecasts."""
    out = df[pred_col].to_numpy(dtype=float).copy()
    r = np.log(df["actual"].to_numpy() / df[pred_col].to_numpy())
    d = df["d"].to_numpy(); tgt = df["target"].to_numpy(); org = df["origin"].to_numpy()
    order = np.argsort(tgt)
    tgt_s, d_s, r_s = tgt[order], d[order], r[order]
    cxr, cxx = np.cumsum(d_s * r_s), np.cumsum(d_s * d_s)
    cinf = np.cumsum(np.abs(d_s) >= INFORMATIVE)
    runmax = np.maximum.accumulate(np.abs(d_s))
    betas = {}
    for o in np.unique(org):
        n = int(np.searchsorted(tgt_s, o, side="right"))
        if n == 0 or cinf[n - 1] < MIN_INFORMATIVE or cxx[n - 1] <= 0:
            betas[o] = (0.0, np.inf)
        else:
            betas[o] = (cxr[n - 1] / cxx[n - 1], runmax[n - 1])
    for i in range(len(out)):
        beta, cap = betas[org[i]]
        di = np.clip(d[i], -cap, cap) if capped else d[i]
        out[i] = out[i] * np.exp(beta * di)
    return out, betas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--frequency", choices=["W", "MS"], required=True)
    a = ap.parse_args()
    run = Path(a.run_dir); freq = a.frequency
    fx = fx_monthly()
    pp = pd.read_csv(run / "predictions_paired.csv", parse_dates=["origin"])
    fc = pd.read_csv(run / "forecasts.csv", parse_dates=["origin"])
    for d_ in (pp, fc):
        if freq == "W":
            d_["target"] = d_["origin"] + pd.to_timedelta(d_["h"] * 7, unit="D")
        else:  # a monthly target is known once its month has ended
            d_["target"] = (d_["origin"].dt.to_period("M") + d_["h"]).dt.to_timestamp(how="end").dt.normalize()
            d_["origin"] = d_["origin"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
    kind = "GRU"
    H = sorted(pp.h.unique())
    print(f"run {run}  frequency {freq}  pairs per horizon {len(pp) // len(H)}")
    for L in [6, 3, 12]:
        pp["d"] = depreciation(pp["origin"], freq, L, fx); fc["d"] = depreciation(fc["origin"], freq, L, fx)
        for capped in ([False, True] if L == 6 else [False]):
            tag = f"L={L}m{' capped' if capped else ''}"
            print(f"\n== overlay {tag}" + ("   [PRIMARY]" if (L == 6 and not capped) else ""))
            for h in H:
                s = pp[pp.h == h].copy()
                s["adj"], betas = overlay(s, kind, capped)
                s = s.sort_values(["market", "origin"])
                dm = diebold_mariano(s.actual, s.adj, s[kind], int(h), groups=s.market.values)
                ds = []
                for sd in sorted(fc.seed.unique()):
                    f = fc[(fc.seed == sd) & (fc.h == h)].copy()
                    f["adj"], _ = overlay(f, "pred", capped)
                    ds.append(mae(f.actual, f.adj) - mae(f.actual, f.pred))
                ds = np.array(ds); t = ds.mean() / (ds.std(ddof=1) / np.sqrt(len(ds))) if ds.std() > 0 else 0.0
                nm = mae(s.actual, s.naive)
                b = np.array([v[0] for v in betas.values()])
                s["period"] = pd.cut(s.target.dt.year, [2014, 2019, 2022, 2025], labels=["2015-19", "2020-22", "2023-24"])
                per = "  ".join(f"{p}: {mae(q.actual, q[kind]):.2f}->{mae(q.actual, q.adj):.2f}" for p, q in s.groupby("period", observed=True))
                print(f"  h={h:2d}  GRU {mae(s.actual, s[kind]):.2f}/{mape(s.actual, s[kind]):.1f}% ({100*(nm-mae(s.actual, s[kind]))/nm:+.1f}%)  "
                      f"+FX {mae(s.actual, s.adj):.2f}/{mape(s.actual, s.adj):.1f}% ({100*(nm-mae(s.actual, s.adj))/nm:+.1f}%)  "
                      f"DM {dm['dm_stat']:+.2f} p={dm['dm_p']:.3f}  seed t {t:+.2f} ({(ds < 0).sum()}/{len(ds)} better)  "
                      f"beta range {b.min():+.2f}..{b.max():+.2f} (last {b[-1]:+.2f})")
                print(f"         by period MAE: {per}")


if __name__ == "__main__":
    main()
