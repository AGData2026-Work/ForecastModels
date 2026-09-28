"""
Harder naive benchmarks and the regime-aware soft blend for an AFEX run.

Formalises what D-69 (seasonal / drift / moving-average naive) and D-70
(regime-aware blend) did in throwaway scratch scripts, using the same
methods main's D-52/D-53/D-54 already documented:

- seasonal naive: origin_price * SI[target_month] / SI[origin_month], SI a
  deflated monthly seasonal index (food CPI from data/external/inflation.xlsx,
  geometric mean across complete 12-month years, normalised to average 100).
  Two columns:
    `seasonal_naive`     one pooled index for every maize series (each series
                         normalised to its own mean, then averaged). This is
                         what D-69 used: re-run against the v2 panel it lands
                         within ~0.3 NGN/kg MAE of D-69 at every horizon; the
                         scratch script's exact pooling step wasn't recorded.
                         The blend uses this column, matching D-70.
    `seasonal_naive_own` each series' own index where it has >= 3 complete
                         years, else the pooled one. A harder bar (lower
                         error than pooled on v2 by 2-5 NGN/kg); reported as a
                         sensitivity check. Which series fell back is in the
                         metadata.
  CAVEAT: the index is fitted on every complete year in the file, including
  years after a given origin. That is how D-52/D-69 built it, kept here so
  results stay comparable; it gives the benchmark a mild in-sample advantage.
- drift naive: origin_price * exp(drift * h), drift = mean weekly log change
  over the series' history strictly before the origin (expanding).
- moving-average naive: mean price over the 4 weeks ending at the origin.
- soft blend: weight = sigmoid(z / scale), z = log(trailing 13-week
  volatility / expanding median of that series' own past volatility),
  scale = std of z across maize series; blended = weight * model +
  (1 - weight) * seasonal_naive. Unknown regime (too little history) gets
  weight 1 (pure model). The expanding median at week i uses readings
  strictly before i.

Reads only an existing run's predictions_paired.csv plus the panel file;
retrains nothing.

    python src/afex_benchmarks.py --run-dir outputs/afex_operational_v3/RNN --kind RNN
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import diebold_mariano, mae, mape  # noqa: E402

TRAILING_WINDOW = 13
MIN_COMPLETE_YEARS = 3


def load_maize_prices(panel_path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(panel_path, sheet_name="Panel_Long", parse_dates=["date"])
    df = df[df["commodity"] == "Maize"].copy()
    df["series"] = df["market"] + " | Maize"
    return df[["date", "series", "price_NGN_per_kg"]].rename(columns={"price_NGN_per_kg": "price"})


def build_food_cpi(inflation_path: str | Path) -> dict:
    infl = pd.read_excel(inflation_path, sheet_name="Monthly Inflation")
    infl = infl.rename(columns={"Unnamed: 0": "date"}).sort_values("date").reset_index(drop=True)
    infl["date"] = pd.to_datetime(infl["date"])
    infl["cpi_food"] = 100.0 * (1 + infl["Food Inflation (%)-MoM"] / 100.0).cumprod()
    return {(d.year, d.month): v for d, v in zip(infl["date"], infl["cpi_food"])}


def _monthly_real(prices: pd.DataFrame, cpi: dict) -> pd.DataFrame:
    p = prices.dropna(subset=["price"]).copy()
    p["year"], p["month"] = p["date"].dt.year, p["date"].dt.month
    m = p.groupby(["year", "month"], as_index=False)["price"].mean()
    m["cpi"] = [cpi.get((y, mo), np.nan) for y, mo in zip(m["year"], m["month"])]
    m["real"] = m["price"] / m["cpi"]
    return m.dropna(subset=["real"])


def _index_from_monthly(m: pd.DataFrame) -> tuple[dict | None, list[int]]:
    counts = m.groupby("year")["month"].nunique()
    years = counts[counts == 12].index.tolist()
    if len(years) < MIN_COMPLETE_YEARS:
        return None, years
    m = m[m["year"].isin(years)].copy()
    m["ratio"] = m["real"] / m.groupby("year")["real"].transform("mean")
    gm = m.groupby("month")["ratio"].apply(lambda x: float(np.exp(np.log(x).mean())))
    return (100 * gm / gm.mean()).to_dict(), years


def build_seasonal_indices(maize: pd.DataFrame, cpi: dict) -> tuple[dict, dict, dict]:
    """Returns (own_else_pooled per series, pooled index, info)."""
    per_series, info, pooled_parts = {}, {}, []
    for s, g in maize.groupby("series"):
        m = _monthly_real(g, cpi)
        idx, years = _index_from_monthly(m)
        if idx is not None:
            per_series[s] = idx
        info[s] = {"complete_years": years}
        if len(m):
            mm = m.copy()
            mm["real"] = mm["real"] / mm["real"].mean()
            pooled_parts.append(mm)
    pooled = pd.concat(pooled_parts)
    pooled = pooled.groupby(["year", "month"], as_index=False)["real"].mean()
    pooled_idx, pooled_years = _index_from_monthly(pooled)
    if pooled_idx is None:
        raise SystemExit("pooled seasonal index has fewer than 3 complete years; cannot fall back")
    out = {}
    for s in info:
        if s in per_series:
            out[s] = per_series[s]
            info[s]["index"] = "own"
        else:
            out[s] = pooled_idx
            info[s]["index"] = "pooled"
    info["_pooled_complete_years"] = pooled_years
    return out, pooled_idx, info


def regime_z(prices: pd.Series, window: int = TRAILING_WINDOW) -> pd.Series:
    """Indexed by date. NaN-aware: needs >= 4 finite weekly changes in the
    trailing window. Expanding median uses readings strictly before each week."""
    s = prices.sort_index()
    v = s.values.astype(float)
    chg = np.full(len(v), np.nan)
    chg[1:] = np.diff(v) / v[:-1]
    trailing = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = chg[max(1, i - window + 1):i + 1]
        w = w[np.isfinite(w)]
        if len(w) >= 4:
            trailing[i] = np.std(w)
    z = np.full(len(v), np.nan)
    seen: list[float] = []
    for i in range(len(v)):
        if np.isfinite(trailing[i]) and len(seen) >= 4:
            med = float(np.median(seen))
            if med > 0 and trailing[i] > 0:
                z[i] = np.log(trailing[i] / med)
        if np.isfinite(trailing[i]):
            seen.append(float(trailing[i]))
    return pd.Series(z, index=s.index)


def drift_forecast(prices: pd.Series, origin: pd.Timestamp, h: int, p0: float) -> float:
    past = prices[prices.index < origin].dropna()
    if len(past) < 2:
        return np.nan
    lr = np.diff(np.log(past.values))
    return float(p0 * np.exp(np.mean(lr) * h)) if len(lr) else np.nan


def ma_forecast(prices: pd.Series, origin: pd.Timestamp) -> float:
    w = prices[(prices.index <= origin) & (prices.index > origin - pd.Timedelta(weeks=4))].dropna()
    return float(w.mean()) if len(w) else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--kind", required=True, choices=["RNN", "GRU"])
    ap.add_argument("--panel", default="data/external/afex_multicommodity_panel.xlsx")
    ap.add_argument("--inflation", default="data/external/inflation.xlsx")
    ap.add_argument("--out", default=None, help="defaults to --run-dir")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    out = Path(a.out) if a.out else run_dir
    out.mkdir(parents=True, exist_ok=True)

    maize = load_maize_prices(a.panel)
    cpi = build_food_cpi(a.inflation)
    si_own, si_pooled, si_info = build_seasonal_indices(maize, cpi)
    by_series = {s: g.set_index("date")["price"].sort_index() for s, g in maize.groupby("series")}
    z_by_series = {s: regime_z(p) for s, p in by_series.items()}
    z_scale = float(np.nanstd(np.concatenate([z.values for z in z_by_series.values()])))

    pred = pd.read_csv(run_dir / "predictions_paired.csv", parse_dates=["origin"])
    pred["target"] = pred["origin"] + pd.to_timedelta(pred["h"] * 7, unit="D")
    pred["seasonal_naive"] = [
        p0 * si_pooled[t.month] / si_pooled[o.month]
        for p0, o, t in zip(pred["origin_price"], pred["origin"], pred["target"])]
    pred["seasonal_naive_own"] = [
        p0 * si_own[s][t.month] / si_own[s][o.month]
        for s, p0, o, t in zip(pred["market"], pred["origin_price"], pred["origin"], pred["target"])]
    pred["drift_naive"] = [drift_forecast(by_series[s], o, h, p0) for s, o, h, p0 in
                           zip(pred["market"], pred["origin"], pred["h"], pred["origin_price"])]
    pred["ma_naive"] = [ma_forecast(by_series[s], o) for s, o in zip(pred["market"], pred["origin"])]
    pred["z"] = [z_by_series[s].get(o, np.nan) for s, o in zip(pred["market"], pred["origin"])]
    pred["weight"] = (1 / (1 + np.exp(-pred["z"] / z_scale))).fillna(1.0)
    pred["blended"] = pred["weight"] * pred[a.kind] + (1 - pred["weight"]) * pred["seasonal_naive"]
    pred.to_csv(out / "predictions_with_benchmarks.csv", index=False)

    cols = [a.kind, "naive", "seasonal_naive", "seasonal_naive_own", "drift_naive", "ma_naive", "blended"]
    rows = []
    for h in sorted(pred["h"].unique()):
        s = pred[pred["h"] == h].sort_values(["market", "origin"])
        row = {"h": int(h), "n": len(s)}
        for c in cols:
            ok = s[c].notna()
            row[f"{c}_MAE"] = mae(s.loc[ok, "actual"], s.loc[ok, c])
            row[f"{c}_MAPE"] = mape(s.loc[ok, "actual"], s.loc[ok, c])
            row[f"{c}_n"] = int(ok.sum())
        g = s["market"].values
        for label, p1, p2 in [("model_vs_seasonal", a.kind, "seasonal_naive"),
                              ("model_vs_seasonal_own", a.kind, "seasonal_naive_own"),
                              ("blend_vs_model", "blended", a.kind),
                              ("blend_vs_seasonal", "blended", "seasonal_naive"),
                              ("model_vs_naive", a.kind, "naive")]:
            d = diebold_mariano(s["actual"], s[p1], s[p2], int(h), groups=g)
            row[f"dm_stat_{label}"], row[f"dm_p_{label}"] = d["dm_stat"], d["dm_p"]
        rows.append(row)
        print(f"h={h:2d} n={len(s)}  " + "  ".join(
            f"{c}={row[f'{c}_MAE']:.2f}/{row[f'{c}_MAPE']:.2f}%" for c in cols))
    pd.DataFrame(rows).to_csv(out / "benchmark_metrics.csv", index=False)
    (out / "benchmark_metadata.json").write_text(json.dumps({
        "run_dir": str(run_dir), "kind": a.kind, "panel": a.panel, "z_scale": z_scale,
        "seasonal_index": si_info,
    }, indent=2, default=str))
    print(f"z_scale={z_scale:.3f}  seasonal index own/pooled: "
          f"{sum(1 for k, v in si_info.items() if not k.startswith('_') and v['index'] == 'own')}/"
          f"{sum(1 for k, v in si_info.items() if not k.startswith('_') and v['index'] == 'pooled')}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
