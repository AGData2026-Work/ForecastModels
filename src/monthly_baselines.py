"""
Baselines on the monthly grid, and the matched comparison with the weekly flagship
(monthly-cadence experiment, phase 2). Retrains nothing.

    python src/monthly_baselines.py
    python src/monthly_baselines.py --price-agg last   # on the last-value panel; writes
                                                       # outputs/monthly_baselines_last/

Baselines, every one point-in-time (fitted only on months up to and including the origin):
  naive           month o's average price
  seasonal_naive  P_o x SI[target month] / SI[origin month]; SI as in src/regime_blend.py
                  (food-CPI deflated, ratio to the year's mean, geometric mean over complete
                  calendar years BEFORE the origin's year, at least 3, normalised to 100)
  fews_tes        FEWS NET 2018 guidance Annex VI Holt-Winters, constants chosen to minimise
                  in-sample one-step MAD (Excel Solver's objective)
  fews_decomp     FEWS NET Annex V decomposition (2x12 centred MA, seasonal index, linear trend)

Matched comparison with the weekly flagship: a weekly grid pair (market, t, h) is matched to
monthly pair (market, o, k) when t falls in the LAST 7 DAYS of month o, so both forecasters
know about the same data, and the target week t+h falls in month o+k (k = 1, 3, 6 for
h = 4, 13, 26). Both are then scored against the monthly average of month o+k.

Writes outputs/monthly_baselines/{predictions.csv, metrics.csv, matched.csv}.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import diebold_mariano, direction_pct, mae, mape  # noqa: E402

OUT = Path("outputs/monthly_baselines")
H_WEEKS = {1: 4, 3: 13, 6: 26}
A = np.round(np.linspace(0.05, 0.95, 10), 2)
B = np.array([0, .05, .1, .2, .3])
G = np.array([0, .1, .2, .3, .5])
AA, BB, GG = [x.ravel() for x in np.meshgrid(A, B, G, indexing="ij")]


def food_cpi() -> dict:
    infl = pd.read_excel("data/external/inflation.xlsx", sheet_name="Monthly Inflation")
    infl = infl.rename(columns={"Unnamed: 0": "date"}).sort_values("date")
    infl["date"] = pd.to_datetime(infl["date"])
    infl["cpi"] = 100 * (1 + infl["Food Inflation (%)-MoM"] / 100).cumprod()
    return {(d.year, d.month): v for d, v in zip(infl["date"], infl["cpi"])}


def seasonal_index(s: pd.Series, cpi: dict, before_year: int) -> dict | None:
    s = s[s.index.year < before_year].dropna()
    if s.empty:
        return None
    df = pd.DataFrame({"p": s.values, "y": s.index.year, "m": s.index.month})
    df["r"] = df["p"] / [cpi.get((y, m), np.nan) for y, m in zip(df.y, df.m)]
    df = df.dropna()
    full = df.groupby("y")["m"].nunique()
    yrs = full[full == 12].index
    if len(yrs) < 3:
        return None
    df = df[df.y.isin(yrs)].copy()
    df["ratio"] = df["r"] / df.groupby("y")["r"].transform("mean")
    gm = df.groupby("m")["ratio"].apply(lambda x: float(np.exp(np.log(x).mean())))
    return (100 * gm / gm.mean()).to_dict()


def decomp(v: np.ndarray, months: np.ndarray):
    cma = pd.Series(v).rolling(12, center=True).mean().rolling(2).mean().shift(-1).values
    sf = v / cma
    S = {mm: np.nanmean(sf[months == mm]) for mm in range(1, 13)}
    k = np.mean(list(S.values()))
    S = {mm: S[mm] / k for mm in S}
    ok = np.isfinite(cma)
    b, a = np.polyfit(np.arange(len(v))[ok], cma[ok], 1)
    return S, a, b


def tes(v: np.ndarray, months: np.ndarray, S0: dict):
    E = np.full(AA.shape, v[:12].mean())
    T = np.full(AA.shape, (v[11] - v[0]) / 12)
    Sx = np.tile(np.array([S0[mm] for mm in range(1, 13)]), (len(AA), 1))
    err = np.zeros(AA.shape)
    for i in range(12, len(v)):
        mi = months[i] - 1
        err += np.abs(v[i] - (E + T) * Sx[:, mi])
        En = AA * (v[i] / Sx[:, mi]) + (1 - AA) * (E + T)
        Tn = BB * (En - E) + (1 - BB) * T
        Sx[:, mi] = GG * (v[i] / E) + (1 - GG) * Sx[:, mi]
        E, T = En, Tn
    j = int(np.argmin(err))
    return E[j], T[j], Sx[j]


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-agg", choices=["mean", "last"], default="mean")
    sfx = "" if ap.parse_args().price_agg == "mean" else "_last"
    OUT = Path(f"outputs/monthly_baselines{sfx}")
    OUT.mkdir(parents=True, exist_ok=True)
    pm = pd.read_parquet(f"data/panel_monthly{sfx}.parquet")
    pm["date"] = pd.to_datetime(pm["date"])
    P = pm.pivot(index="date", columns="market", values="price")
    grid = pd.read_parquet(f"data/monthly_grid{sfx}.parquet")
    cpi = food_cpi()

    rows, n_interp, skipped = [], 0, 0
    for (m, o), g in grid.groupby(["market", "origin"]):
        s = P[m][P.index <= o]
        si = seasonal_index(s, cpi, o.year)
        y = s.loc[s.first_valid_index():]
        n_interp += int(y.isna().sum())
        y = np.exp(np.log(y).interpolate(limit_area="inside")).dropna()
        have = len(y) >= 24
        if have:
            v, mo = y.values, np.array(y.index.month)
            S, a, b = decomp(v, mo)
            E, T, Sx = tes(v, mo, S)
        else:
            skipped += len(g)
        for r in g.itertuples():
            k = r.horizon
            tm = r.target.month
            rows.append(dict(market=m, origin=o, h=k, target=r.target, actual=r.actual, naive=r.naive,
                             seasonal_naive=(r.naive * si[tm] / si[o.month]) if si else np.nan,
                             fews_tes=((E + k * T) * Sx[tm - 1]) if have else np.nan,
                             fews_decomp=((a + b * (len(v) - 1 + k)) * S[tm]) if have else np.nan))
    pred = pd.DataFrame(rows)
    pred.to_csv(OUT / "predictions.csv", index=False)
    print(f"monthly grid pairs {len(pred)}; seasonal index available {int(pred.seasonal_naive.notna().sum())}; "
          f"Holt-Winters/decomp available {int(pred.fews_tes.notna().sum())} (skipped <24 months: {skipped}); "
          f"interpolated interior months summed over fits: {n_interp}")

    cols = ["naive", "seasonal_naive", "fews_tes", "fews_decomp"]
    met = []
    print("\n== Monthly grid, each baseline on its own available pairs (MAE NGN/kg / MAPE / vs naive)")
    for h in [1, 3, 6]:
        s = pred[pred.h == h]
        line = [f"h={h}m n={len(s)}"]
        for c in cols:
            q = s[s[c].notna()]
            e, en = mae(q.actual, q[c]), mae(q.actual, q.naive)
            met.append(dict(h=h, model=c, n=len(q), MAE=e, MAPE=mape(q.actual, q[c]), vs_naive_pct=100 * (en - e) / en))
            line.append(f"{c} {e:.2f}/{mape(q.actual, q[c]):.1f}%/{100*(en-e)/en:+.1f}% (n={len(q)})")
        print("  " + " | ".join(line))
    pd.DataFrame(met).to_csv(OUT / "metrics.csv", index=False)

    # ---------------- matched comparison with the weekly flagship ----------------
    wk = pd.read_csv("outputs/build3_underfit_corrected/GRU_unconditional/predictions_paired.csv",
                     parse_dates=["origin"])
    bl = pd.read_csv("outputs/regime_blend_soft/GRU/predictions_with_blend.csv", parse_dates=["origin"])
    wk = wk.merge(bl[["market", "origin", "h", "blended"]], on=["market", "origin", "h"], how="left")
    wk["o"] = wk["origin"].dt.to_period("M")
    wk["days_to_month_end"] = (wk["o"].dt.to_timestamp(how="end").dt.normalize() - wk["origin"]).dt.days
    wk["k"] = wk["h"].map({v: k for k, v in H_WEEKS.items()})
    wk["target_m"] = (wk["origin"] + pd.to_timedelta(wk["h"] * 7, unit="D")).dt.to_period("M")
    ok = (wk["days_to_month_end"] <= 6) & (wk["target_m"] == wk["o"] + wk["k"])
    mw = wk[ok].copy()
    mw["origin_m"] = mw["o"].dt.to_timestamp()
    mt = mw.merge(pred.rename(columns={"origin": "origin_m", "h": "k", "actual": "actual_month",
                                       "naive": "naive_month"}),
                  on=["market", "origin_m", "k"], how="inner")
    mt.to_csv(OUT / "matched.csv", index=False)
    print(f"\n== Matched with the weekly flagship (weekly origin in the last 7 days of month o, target week in month o+k): "
          f"{len(mt)} pairs; scored against the target MONTH's {'average' if not sfx else 'last observed week'}")
    for k in [1, 3, 6]:
        s = mt[(mt.k == k)].dropna(subset=["seasonal_naive", "fews_tes"]).sort_values(["market", "origin_m"])
        parts = []
        for c, lab in [("GRU", "weekly GRU"), ("blended", "weekly blend"), ("naive_month", "monthly naive"),
                       ("seasonal_naive", "seasonal naive"), ("fews_tes", "FEWS Holt-Winters")]:
            parts.append(f"{lab} {mae(s.actual_month, s[c]):.2f}/{mape(s.actual_month, s[c]):.1f}%")
        print(f"  k={k}m n={len(s)} | " + " | ".join(parts))
        print(f"        weekly GRU vs its own weekly actual: {mae(s.actual, s.GRU):.2f}/{mape(s.actual, s.GRU):.1f}%;"
              f" weekly naive vs weekly actual: {mae(s.actual, s.naive):.2f}")


if __name__ == "__main__":
    main()
