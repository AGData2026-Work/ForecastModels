"""
Range experiments (2026-11-03), following Forecasting_Price_Distributions_Guide.docx (@Izzy, 2 Nov 2026). Experiment,
not production: production ranges are unchanged (src/joint_forecast.py live).

    python src/range_experiments.py --out outputs/range_experiments

Bins on the % change from the origin price, fixed before testing (guide section 2): 1 month -5/-2/+2/+5%, 3 months
-10/-3/+3/+10%; 6 months -15/-5/+5/+15%, the edges that reproduce the guide's own 6-month table (no-view ranked
probability score 0.246, log loss 1.64). On pre-2019 outcomes the smallest bin holds 5-8%, below the guide's 10% rule;
kept so the results compare with the guide's.

Pairs: the production backtest (outputs/d75_monthly_parity_pit/h{1,3,6}/predictions_paired.csv, FEWS NET observed
actuals) joined to the analyst's as-published forecasts (outputs/joint_forecast/analyst_forecasts.csv) on (market,
origin, h). Test: origins from Jan 2019 where every model has a forecast. Every model is scored on the same rows,
from our origin price; the analyst's forecast enters as his predicted change (forecast / his origin price), as in the
joint forecast.

Route B (guide section 3): each model's point forecast times its own past misses, log(actual / forecast), from
forecasts whose target month is at or before the origin (known by the forecast date), pooled over markets, per
horizon; joint misses fall back to the GRU's before the analyst's first forecast (as production). Variants, fixed in
advance (the guide's section 4 repairs):
  B0  all past misses, equal weight (the production method)
  B1  misses from the last 36 months only
  B2  all misses, weight halving every 24 months of age
  B3  all misses, each rescaled by (price volatility at the origin / volatility at that miss's origin); volatility =
      root mean square of 1-month log price changes over the 12 months to the origin, 15 scored markets pooled, from the
      published panel (data/panel_monthly_gapfilled_pit.parquet, prices to the origin month)
  B4  B2 and B3 together
No-view baseline: every past % change at that horizon (same known-by rule), the same probabilities for every row.
Route A: the GRU with a 5-bin head (configs/_checks/range_bins_gru.yaml, outputs/range_experiments/route_a/h*),
probabilities averaged over its 7 seeds; A1 adds a temperature fitted walk-forward on Route A's own earlier
forecasts whose outcome was known by the forecast date (log loss, grid 0.5 to 3.0).

Scores (guide section 4), per horizon and period (by target month): log loss (probabilities floored at 0.001; the
count of floored rows is reported), ranked probability score (RPS, mean over the 4 cumulative bins), skill vs no-view
(1 - RPS / no-view RPS on the same rows) with a Diebold-Mariano p (market-clustered, h-1 lags), calibration, and for
Route B the 80% and 90% range coverage, misses below / above and median width (% of the forecast).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import diebold_mariano  # noqa: E402

EDGES = {1: [-0.05, -0.02, 0.02, 0.05], 3: [-0.10, -0.03, 0.03, 0.10], 6: [-0.15, -0.05, 0.05, 0.15]}
BINS = ["Big fall", "Fall", "Flat", "Rise", "Big rise"]
TEST_FROM = pd.Timestamp("2019-01-01")
PERIODS = [("Test, all (targets 2019-26)", "2000-01-01", "2100-01-01"), ("2019-24", "2000-01-01", "2024-12-31"),
           ("2025-26", "2025-01-01", "2100-01-01"), ("Oct 2024 to Aug 2026", "2024-10-01", "2100-01-01")]
VARIANTS = {"B0 all past misses (production method)": dict(window=None, half_life=None, vol=False),
            "B1 last 3 years": dict(window=36, half_life=None, vol=False),
            "B2 recency weights (half-life 2 years)": dict(window=None, half_life=24, vol=False),
            "B3 volatility-scaled": dict(window=None, half_life=None, vol=True),
            "B4 recency + volatility": dict(window=None, half_life=24, vol=True)}
MODELS = {"gru": "GRU", "analyst": "Inflation Adjusted", "joint": "Joint (production)"}
FLOOR = 1e-3


def months(a, b) -> np.ndarray:
    a, b = pd.DatetimeIndex(a), pd.DatetimeIndex(b)
    return np.asarray((a.year - b.year) * 12 + (a.month - b.month))


def volatility() -> pd.Series:
    p = pd.read_parquet("data/panel_monthly_gapfilled_pit.parquet", columns=["market", "date", "price"])
    w = p.pivot_table(index="date", columns="market", values="price").sort_index()
    w = w[[m for m in w.columns if m != "Aba"]]                      # Aba is a training-only market
    d2 = np.log(w).diff() ** 2
    return np.sqrt(d2.stack().groupby(level=0).mean().rolling(12, min_periods=6).mean())


def table(h: int) -> pd.DataFrame:
    g = pd.read_csv(f"outputs/d75_monthly_parity_pit/h{h}/predictions_paired.csv", parse_dates=["origin"]).query("h == @h")
    a = pd.read_csv("outputs/joint_forecast/analyst_forecasts.csv", parse_dates=["origin"]).query("h == @h")
    x = g.merge(a[["market", "origin", "p0", "recommended"]], on=["market", "origin"], how="left")
    x["target"] = x.origin + pd.DateOffset(months=h)
    x["r_gru"] = np.log(x.GRU / x.origin_price)
    x["r_analyst"] = np.log(x.recommended / x.p0)
    x["r_joint"] = (0.5 * (x.r_gru + x.r_analyst)).fillna(x.r_gru)  # production fallback before Jan 2017
    x["y"] = np.log(x.actual / x.origin_price)                        # outcome, log change from our origin price
    for m in MODELS:
        x[f"e_{m}"] = x.y - x[f"r_{m}"]                               # past miss, log(actual / forecast)
    return x.sort_values(["origin", "market"]).reset_index(drop=True)


def wquantile(v: np.ndarray, w: np.ndarray, q) -> np.ndarray:
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, c, v)


def probs(samples: np.ndarray, w: np.ndarray, edges) -> np.ndarray:
    k = np.digitize(np.expm1(samples), edges)
    return np.bincount(k, weights=w, minlength=len(edges) + 1) / w.sum()


def rps(p: np.ndarray, k: np.ndarray) -> np.ndarray:
    o = np.zeros_like(p)
    o[np.arange(len(k)), k] = 1
    return ((np.cumsum(p, 1)[:, :-1] - np.cumsum(o, 1)[:, :-1]) ** 2).mean(1)


def route_b(x: pd.DataFrame, h: int, vol: pd.Series) -> pd.DataFrame:
    edges = EDGES[h]
    test = x[(x.origin >= TEST_FROM) & x.r_analyst.notna()]
    vol_at = x.origin.map(vol).to_numpy()
    out = []
    for org, cur in test.groupby("origin"):
        past = x[x.target <= org]
        age = months(np.full(len(past), org), past.target)
        nv = probs(past.y.to_numpy(), np.ones(len(past)), edges)
        base = dict(origin=org, market=cur.market.values, target=cur.target.values, h=h, y=cur.y.values,
                    k=np.digitize(np.expm1(cur.y.values), edges), actual=cur.actual.values, origin_price=cur.origin_price.values)
        out.append(pd.DataFrame({**base, "model": "no-view", "variant": "no-view", "n_pool": len(past),
                                 **{f"p{i}": nv[i] for i in range(5)}}))
        for m in MODELS:
            ok = past[f"e_{m}"].notna().to_numpy()
            for vname, v in VARIANTS.items():
                e, a_, va = past[f"e_{m}"].to_numpy()[ok], age[ok], vol_at[past.index.to_numpy()[ok]]
                if v["window"]:
                    keep = a_ < v["window"]
                    e, a_, va = e[keep], a_[keep], va[keep]
                w = np.ones(len(e))
                if v["half_life"]:
                    w = w * 0.5 ** (a_ / v["half_life"])
                if v["vol"]:
                    e = e * vol[org] / va
                rows = []
                for r in cur[f"r_{m}"].to_numpy():
                    s = r + e
                    q = wquantile(s, w, [0.05, 0.10, 0.90, 0.95])
                    rows.append((*probs(s, w, edges), *np.exp(q - r)))
                rows = np.array(rows)
                out.append(pd.DataFrame({**base, "model": m, "variant": vname, "n_pool": len(e),
                                         "forecast": cur.origin_price.values * np.exp(cur[f"r_{m}"].values),
                                         **{f"p{i}": rows[:, i] for i in range(5)},
                                         "lo90": rows[:, 5], "lo80": rows[:, 6], "hi80": rows[:, 7], "hi90": rows[:, 8]}))
    d = pd.concat(out, ignore_index=True)
    for c in ("lo90", "lo80", "hi80", "hi90"):                        # multipliers of the forecast -> NGN/kg
        d[c] = d[c] * d.forecast
    return d


def route_a(x: pd.DataFrame, h: int, root: Path) -> pd.DataFrame | None:
    f = root / f"h{h}" / "forecasts_bins.csv"
    if not f.exists():
        return None
    b = pd.read_csv(f, parse_dates=["origin"]).query("h == @h")
    pc = [f"p{i}" for i in range(5)]
    b = b.groupby(["origin", "market"], as_index=False)[pc].mean()       # average of the 7 seeds
    b = b.merge(x[["origin", "market", "target", "y", "actual", "origin_price", "r_analyst"]], on=["origin", "market"])
    b["k"] = np.digitize(np.expm1(b.y), EDGES[h])
    out, temps = [], []
    for org, cur in b[(b.origin >= TEST_FROM) & b.r_analyst.notna()].groupby("origin"):
        past = b[b.target <= org]
        lp = np.log(np.clip(past[pc].to_numpy(), 1e-9, 1))
        best_t = 1.0
        if len(past) >= 200:
            grid = np.arange(0.5, 3.01, 0.05)
            ll = [-(np.log(np.clip(_temper(lp, t)[np.arange(len(past)), past.k], FLOOR, 1))).mean() for t in grid]
            best_t = float(grid[int(np.argmin(ll))])
        temps.append(dict(h=h, origin=org, temperature=best_t, n_past=len(past)))
        base = dict(origin=org, market=cur.market.values, target=cur.target.values, h=h, y=cur.y.values, k=cur.k.values,
                    actual=cur.actual.values, origin_price=cur.origin_price.values, model="gru_bins", n_pool=len(past))
        raw = cur[pc].to_numpy()
        out.append(pd.DataFrame({**base, "variant": "A0 GRU bin head", **{f"p{i}": raw[:, i] for i in range(5)}}))
        tp = _temper(np.log(np.clip(raw, 1e-9, 1)), best_t)
        out.append(pd.DataFrame({**base, "variant": "A1 GRU bin head + temperature", **{f"p{i}": tp[:, i] for i in range(5)}}))
    return pd.concat(out, ignore_index=True), pd.DataFrame(temps)


def _temper(logp: np.ndarray, t: float) -> np.ndarray:
    z = logp / t
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def score(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pc = [f"p{i}" for i in range(5)]
    P = d[pc].to_numpy()
    k = d.k.to_numpy()
    pk = P[np.arange(len(d)), k]
    d = d.assign(ll=-np.log(np.clip(pk, FLOOR, 1)), floored=pk < FLOOR, rps=rps(P, k))
    nv = d[d.variant == "no-view"].set_index(["h", "origin", "market"])[["rps", "ll"]].rename(columns=lambda c: c + "_nv")
    assert nv.index.is_unique
    d = d.join(nv, on=["h", "origin", "market"])
    rows, cal = [], []
    for (h, m, v), g in d.groupby(["h", "model", "variant"]):
        for per, a, z in PERIODS:
            s = g[(g.target >= a) & (g.target <= z)].sort_values(["market", "origin"])
            if not len(s):
                continue
            r = dict(h=h, model=m, variant=v, period=per, n=len(s), log_loss=s.ll.mean(), log_loss_noview=s.ll_nv.mean(),
                     rps=s.rps.mean(), rps_noview=s.rps_nv.mean(), skill_vs_noview_pct=100 * (1 - s.rps.mean() / s.rps_nv.mean()),
                     n_floored=int(s.floored.sum()))
            if v != "no-view":
                r["dm_p_vs_noview"] = diebold_mariano(np.zeros(len(s)), s.rps.values, s.rps_nv.values, h,
                                                      groups=s.market.values)["dm_p"]
            if "lo80" in s and s.lo80.notna().any():
                for lv in (80, 90):
                    lo, hi = s[f"lo{lv}"], s[f"hi{lv}"]
                    r[f"inside{lv}_pct"] = 100 * ((s.actual >= lo) & (s.actual <= hi)).mean()
                    r[f"below{lv}_pct"] = 100 * (s.actual < lo).mean()
                    r[f"above{lv}_pct"] = 100 * (s.actual > hi).mean()
                    r[f"width{lv}_pct_median"] = 100 * ((hi - lo) / s.forecast).median()
            rows.append(r)
        s = g[g.target >= PERIODS[0][1]]
        p, o = s[pc].to_numpy().ravel(), (np.arange(5)[None, :] == s.k.to_numpy()[:, None]).ravel()
        for lo_, hi_ in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]:
            mk = (p >= lo_) & (p < hi_)
            if mk.any():
                cal.append(dict(h=h, model=m, variant=v, stated=f"{int(100 * lo_)}-{min(100, int(100 * hi_))}%", n=int(mk.sum()),
                                mean_stated_pct=100 * p[mk].mean(), happened_pct=100 * o[mk].mean()))
    return pd.DataFrame(rows), pd.DataFrame(cal), d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/range_experiments")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vol = volatility()
    parts, temps = [], []
    for h in (1, 3, 6):
        x = table(h)
        parts.append(route_b(x, h, vol))
        ra = route_a(x, h, out / "route_a")
        if ra is not None:
            parts.append(ra[0])
            temps.append(ra[1])
        print(f"h={h}: {len(x)} pairs, test origins from {TEST_FROM.date()}, route A {'in' if ra is not None else 'not run yet'}")
    s, cal, d = score(pd.concat(parts, ignore_index=True))
    s["model"] = s.model.map({**MODELS, "no-view": "no-view", "gru_bins": "GRU (bin head)"})
    cal["model"] = cal.model.map({**MODELS, "no-view": "no-view", "gru_bins": "GRU (bin head)"})
    s.round(4).to_csv(out / "scores.csv", index=False)
    cal.round(2).to_csv(out / "calibration.csv", index=False)
    d.to_csv(out / "forecast_rows.csv.gz", index=False)
    if temps:
        pd.concat(temps).to_csv(out / "route_a_temperature.csv", index=False)
    vol.rename("volatility").to_csv(out / "volatility.csv")
    pd.set_option("display.width", 250)
    cols = ["h", "model", "variant", "n", "log_loss", "rps", "skill_vs_noview_pct", "dm_p_vs_noview", "inside80_pct",
            "width80_pct_median", "inside90_pct"]
    for per in ("Test, all (targets 2019-26)", "2025-26"):
        print(f"\n{per}\n" + s[s.period == per][[c for c in cols if c in s]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
