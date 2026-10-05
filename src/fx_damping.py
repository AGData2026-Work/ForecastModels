"""
FX damping in devaluation periods. Retrains nothing.

    python src/fx_damping.py

Inputs: the daily-FX GRU (outputs/build3_fxdaily_fxonly), the no-FX flagship GRU
(outputs/build3_underfit_corrected), both on the frozen weekly grid, and the soft-blend
weights and seasonal-naive forecasts (model-independent; read from
outputs/regime_blend_soft_fxdaily/GRU/predictions_with_blend.csv).

Rules, fixed before any result was seen:
- An origin t is IN A DEVALUATION PERIOD if the daily-rate feature months_since_deval
  (src/data.py _fx_daily_features, read the day before t) is below R months: a >= 10%
  one-month fall of the smoothed CBN rate happened within the last R months. Primary
  R = 12; 6 and 24 are sensitivity.
- In a devaluation period: damped = w x FX-GRU + (1 - w) x no-FX GRU. Outside: FX-GRU.
- Primary (fitted, point-in-time): at each origin t and horizon h, w is the value in
  {0, .25, .5, .75, 1} with the lowest MAE on past in-period pairs of the same horizon
  whose target has happened (target <= t). w = 1 until MIN_PAST such pairs exist. Ties go
  to the larger w (less damping).
- Sensitivity: w = 0.5 fixed.
Applied to the seed-median forecasts, and to each seed separately (seed s of both runs,
w fitted on that seed's own history) for the seed-paired t. The soft blend is then applied
with the production weights: blended = b x damped + (1 - b) x seasonal_naive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data import _fx_daily_features  # noqa: E402
from metrics import diebold_mariano, direction_pct, mae, mape  # noqa: E402

FX = Path("outputs/build3_fxdaily_fxonly/GRU_unconditional")
BASE = Path("outputs/build3_underfit_corrected/GRU_unconditional")
BLEND = Path("outputs/regime_blend_soft_fxdaily/GRU/predictions_with_blend.csv")
PROD_BLEND = Path("outputs/regime_blend_soft/GRU/predictions_with_blend.csv")
DAILY = "data/external/20261002_CBN_API_Nigeria_USDNGN_Daily_2001-12_2026-09_v1.csv"
WGRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
MIN_PAST = 30
K = ["market", "origin", "h"]


def months_since_deval(origins: pd.Series) -> pd.Series:
    u = pd.DatetimeIndex(sorted(origins.unique()))
    X, _ = _fx_daily_features(DAILY, u - pd.Timedelta(days=1))
    return origins.map(dict(zip(u, X[:, 3] * 24.0)))


def damp(df: pd.DataFrame, fx: str, base: str, R: int, fixed: float | None):
    """df: one horizon. Returns (damped forecasts, w per row)."""
    inr = df["msd"].to_numpy() < R
    f, b, a = df[fx].to_numpy(), df[base].to_numpy(), df["actual"].to_numpy()
    w = np.ones(len(df))
    if fixed is not None:
        w[inr] = fixed
    else:
        tgt, org = df["target"].to_numpy(), df["origin"].to_numpy()
        past_idx = np.where(inr)[0]
        past_idx = past_idx[np.argsort(tgt[past_idx])]
        tsorted = tgt[past_idx]
        # cumulative abs error of each candidate w over in-period pairs, ordered by target date
        err = np.abs(a[past_idx, None] - (WGRID * f[past_idx, None] + (1 - WGRID) * b[past_idx, None]))
        cerr = np.cumsum(err, axis=0)
        for o in np.unique(org[inr]):
            n = int(np.searchsorted(tsorted, o, side="right"))
            if n >= MIN_PAST:
                c = cerr[n - 1]
                best = np.flatnonzero(c <= c.min() + 1e-9).max()   # ties -> larger w
                w[(org == o) & inr] = WGRID[best]
    return w * f + (1 - w) * b, w


def main() -> None:
    pf = pd.read_csv(FX / "predictions_paired.csv", parse_dates=["origin"])
    pb = pd.read_csv(BASE / "predictions_paired.csv", parse_dates=["origin"])
    bl = pd.read_csv(BLEND, parse_dates=["origin"])[K + ["weight", "seasonal_naive"]]
    prod = pd.read_csv(PROD_BLEND, parse_dates=["origin"])[K + ["blended"]].rename(columns={"blended": "prod_blend"})
    m = (pf[K + ["actual", "naive", "GRU"]].rename(columns={"GRU": "fx"})
         .merge(pb[K + ["actual", "GRU"]].rename(columns={"GRU": "nofx", "actual": "a2"}), on=K)
         .merge(bl, on=K).merge(prod, on=K))
    assert np.allclose(m.actual, m.a2) and len(m) == len(pf)
    m["target"] = m.origin + pd.to_timedelta(m.h * 7, unit="D")
    m["msd"] = months_since_deval(m.origin)
    m["period"] = pd.cut(m.target.dt.year, [2014, 2019, 2022, 2025], labels=["2015-19", "2020-22", "2023-24"])
    ff = pd.read_csv(FX / "forecasts.csv", parse_dates=["origin"])
    fb = pd.read_csv(BASE / "forecasts.csv", parse_dates=["origin"])
    seeds = sorted(ff.seed.unique())
    S = (ff[["seed"] + K + ["pred", "actual"]].rename(columns={"pred": "fx"})
         .merge(fb[["seed"] + K + ["pred"]].rename(columns={"pred": "nofx"}), on=["seed"] + K)
         .merge(m[K + ["target", "msd", "weight", "seasonal_naive"]], on=K))

    def blend(x, wb, sn):
        return np.where(np.isnan(sn), x, wb * x + (1 - wb) * sn)

    print(f"pairs {len(m)}; origins in a devaluation period (R=12): "
          f"{int((m[m.h == 4].msd < 12).sum())} of {int((m.h == 4).sum())} per horizon")
    for R, fixed, tag in [(12, None, "PRIMARY R=12m, fitted w"), (12, 0.5, "R=12m, w=0.5 fixed"),
                          (6, None, "R=6m, fitted w"), (24, None, "R=24m, fitted w")]:
        print(f"\n== {tag}")
        for h in [4, 13, 26]:
            s = m[m.h == h].sort_values(["market", "origin"]).copy()
            s["damped"], s["w"] = damp(s, "fx", "nofx", R, fixed)
            s["damped_blend"] = blend(s.damped.to_numpy(), s.weight.to_numpy(), s.seasonal_naive.to_numpy())
            a, g = s.actual, s.market.values
            inr = s.msd < R
            res = {}
            for lab, col, ref, scol in [("raw", "damped", "nofx", None), ("blend", "damped_blend", "prod_blend", None)]:
                d = diebold_mariano(a, s[col], s[ref], h, groups=g)
                ds = []
                for sd in seeds:
                    q = S[(S.seed == sd) & (S.h == h)].sort_values(["market", "origin"]).copy()
                    q["d"], _ = damp(q, "fx", "nofx", R, fixed)
                    if lab == "blend":
                        new = blend(q.d.to_numpy(), q.weight.to_numpy(), q.seasonal_naive.to_numpy())
                        old = blend(q.nofx.to_numpy(), q.weight.to_numpy(), q.seasonal_naive.to_numpy())
                    else:
                        new, old = q.d.to_numpy(), q.nofx.to_numpy()
                    ds.append(mae(q.actual, new) - mae(q.actual, old))
                ds = np.array(ds); t = ds.mean() / (ds.std(ddof=1) / np.sqrt(len(ds)))
                res[lab] = (mae(a, s[ref]), mae(a, s[col]), d["dm_p"], t, int((ds < 0).sum()))
            r, b = res["raw"], res["blend"]
            print(f"  h={h:2d}  raw: flagship {r[0]:.2f}  FX {mae(a, s.fx):.2f}  damped {r[1]:.2f}  DM p={r[2]:.4f} t {r[3]:+.2f} ({r[4]}/7)"
                  f"  |  blend: production {b[0]:.2f}  damped {b[1]:.2f}  DM p={b[2]:.4f} t {b[3]:+.2f} ({b[4]}/7)"
                  f"  |  dir {direction_pct(a, s.damped_blend, s.naive):.1f}%")
            ws = s.loc[inr, "w"]
            print(f"        in-period pairs {int(inr.sum())}, w used: " +
                  ", ".join(f"{v:g}:{c}" for v, c in ws.value_counts().sort_index().items()))
            print("        by period, blend MAE production -> damped: " + "   ".join(
                f"{p}: {mae(q.actual, q.prod_blend):.2f} -> {mae(q.actual, q.damped_blend):.2f}"
                for p, q in s.groupby("period", observed=True)))


if __name__ == "__main__":
    main()
