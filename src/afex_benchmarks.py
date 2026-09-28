"""
Harder naive benchmarks and the regime-aware soft blend for an AFEX run.

The soft blend here is the documented companion to the production AFEX GRU
(D-76). Methods follow main's D-52/D-53/D-54, with the look-ahead removed
per D-76:

- seasonal naive: origin_price * SI[target_month] / SI[origin_month], SI a
  deflated monthly seasonal index (food CPI from data/external/inflation.xlsx,
  geometric mean across complete 12-month years, normalised to average 100),
  pooled across maize series (each normalised to its own mean, then
  averaged). Columns:
    `seasonal_naive`            POINT-IN-TIME. The index for an origin in
                                year Y uses only complete calendar years
                                before Y. Needs MIN_COMPLETE_YEARS (2) such
                                years, else NaN. The blend uses this column.
                                2, not main's 3: the panel starts April 2021,
                                so 3 would leave only 2025-26 origins (17% of
                                pairs) with an index; 2 covers 2024 on (57%).
    `seasonal_naive_fullsample` the pre-D-76 index, fitted on every complete
                                year in the file including years after the
                                origin. Kept ONLY as a reference so results
                                stay comparable with D-69/D-74; it has an
                                in-sample advantage and is not a fair bar.
- drift naive: origin_price * exp(drift * h), drift = mean weekly log change
  over the series' history strictly before the origin (expanding).
- moving-average naive: mean price over the 4 weeks ending at the origin.
- soft blend: weight = sigmoid(z / z_scale), z = log(trailing 13-week
  volatility / expanding median of that series' own past volatility);
  z_scale = std of every maize series' z readings dated strictly before the
  origin (point-in-time; was full-sample before D-76). blended = weight *
  model + (1 - weight) * seasonal_naive. No seasonal index, or unknown regime
  (too little history), gets weight 1 (pure model); counts are in
  benchmark_metrics.csv.

Reads only an existing run's predictions_paired.csv plus the panel file;
retrains nothing.

    python src/afex_benchmarks.py --run-dir outputs/afex_operational_v3/GRU
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
MIN_COMPLETE_YEARS = 2
MIN_Z_READINGS = 52  # across all maize series, before the origin


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


def _pooled_monthly(maize: pd.DataFrame, cpi: dict, before: pd.Timestamp | None = None) -> pd.DataFrame:
    parts = []
    for _, g in maize.groupby("series"):
        if before is not None:
            g = g[g["date"] < before]
        m = _monthly_real(g, cpi)
        if len(m):
            m = m.copy()
            m["real"] = m["real"] / m["real"].mean()
            parts.append(m)
    if not parts:
        return pd.DataFrame(columns=["year", "month", "real"])
    return pd.concat(parts).groupby(["year", "month"], as_index=False)["real"].mean()


def pooled_index_fullsample(maize: pd.DataFrame, cpi: dict) -> tuple[dict, list[int]]:
    """Reference only (look-ahead); see module docstring."""
    idx, years = _index_from_monthly(_pooled_monthly(maize, cpi))
    if idx is None:
        raise SystemExit("pooled seasonal index has too few complete years")
    return idx, years


def pooled_index_before(maize: pd.DataFrame, cpi: dict, year: int) -> tuple[dict | None, list[int]]:
    """Point-in-time pooled index from complete calendar years before `year`."""
    return _index_from_monthly(_pooled_monthly(maize, cpi, before=pd.Timestamp(year, 1, 1)))


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
    ap.add_argument("--kind", default="GRU", choices=["RNN", "GRU"],
                    help="GRU is the production model (D-76); RNN only for re-scoring archived runs")
    ap.add_argument("--panel", default="data/external/afex_multicommodity_panel.xlsx")
    ap.add_argument("--inflation", default="data/external/inflation.xlsx")
    ap.add_argument("--out", default=None, help="defaults to --run-dir")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    out = Path(a.out) if a.out else run_dir
    out.mkdir(parents=True, exist_ok=True)

    maize = load_maize_prices(a.panel)
    cpi = build_food_cpi(a.inflation)
    si_full, full_years = pooled_index_fullsample(maize, cpi)
    by_series = {s: g.set_index("date")["price"].sort_index() for s, g in maize.groupby("series")}
    z_by_series = {s: regime_z(p) for s, p in by_series.items()}

    pred = pd.read_csv(run_dir / "predictions_paired.csv", parse_dates=["origin"])
    pred["target"] = pred["origin"] + pd.to_timedelta(pred["h"] * 7, unit="D")
    si_pit = {y: pooled_index_before(maize, cpi, y) for y in sorted(pred["origin"].dt.year.unique())}
    pred["seasonal_naive"] = [
        np.nan if si_pit[o.year][0] is None else p0 * si_pit[o.year][0][t.month] / si_pit[o.year][0][o.month]
        for p0, o, t in zip(pred["origin_price"], pred["origin"], pred["target"])]
    pred["seasonal_naive_fullsample"] = [
        p0 * si_full[t.month] / si_full[o.month]
        for p0, o, t in zip(pred["origin_price"], pred["origin"], pred["target"])]
    pred["drift_naive"] = [drift_forecast(by_series[s], o, h, p0) for s, o, h, p0 in
                           zip(pred["market"], pred["origin"], pred["h"], pred["origin_price"])]
    pred["ma_naive"] = [ma_forecast(by_series[s], o) for s, o in zip(pred["market"], pred["origin"])]
    pred["z"] = [z_by_series[s].get(o, np.nan) for s, o in zip(pred["market"], pred["origin"])]

    all_z = pd.concat([z.dropna() for z in z_by_series.values()]).sort_index()
    z_dates, z_vals = all_z.index.values, all_z.values
    scale = {}
    for o in pred["origin"].unique():
        n = int(np.searchsorted(z_dates, np.datetime64(o), side="left"))
        scale[o] = float(np.std(z_vals[:n])) if n >= MIN_Z_READINGS else np.nan
    pred["z_scale"] = pred["origin"].map(scale)
    no_si = pred["seasonal_naive"].isna()
    pred["weight"] = 1 / (1 + np.exp(-pred["z"] / pred["z_scale"]))
    no_regime = pred["weight"].isna() & ~no_si
    pred.loc[no_si | no_regime, "weight"] = 1.0
    pred["blended"] = np.where(no_si, pred[a.kind],
                               pred["weight"] * pred[a.kind] + (1 - pred["weight"]) * pred["seasonal_naive"])
    pred.to_csv(out / "predictions_with_benchmarks.csv", index=False)

    cols = [a.kind, "naive", "seasonal_naive", "seasonal_naive_fullsample", "drift_naive", "ma_naive", "blended"]
    rows = []
    for h in sorted(pred["h"].unique()):
        s = pred[pred["h"] == h].sort_values(["market", "origin"])
        row = {"h": int(h), "n": len(s), "n_fallback_no_index": int(no_si.loc[s.index].sum()),
               "n_fallback_no_regime": int(no_regime.loc[s.index].sum())}
        for c in cols:
            ok = s[c].notna()
            row[f"{c}_MAE"] = mae(s.loc[ok, "actual"], s.loc[ok, c])
            row[f"{c}_MAPE"] = mape(s.loc[ok, "actual"], s.loc[ok, c])
            row[f"{c}_n"] = int(ok.sum())
        for label, p1, p2 in [("model_vs_seasonal", a.kind, "seasonal_naive"),
                              ("model_vs_seasonal_fullsample", a.kind, "seasonal_naive_fullsample"),
                              ("blend_vs_model", "blended", a.kind),
                              ("blend_vs_seasonal", "blended", "seasonal_naive"),
                              ("blend_vs_naive", "blended", "naive"),
                              ("model_vs_naive", a.kind, "naive")]:
            ok = s[[p1, p2]].notna().all(axis=1)
            d = diebold_mariano(s.loc[ok, "actual"], s.loc[ok, p1], s.loc[ok, p2], int(h),
                                groups=s.loc[ok, "market"].values)
            row[f"dm_stat_{label}"], row[f"dm_p_{label}"] = d["dm_stat"], d["dm_p"]
        rows.append(row)
        print(f"h={h:2d} n={len(s)}  " + "  ".join(
            f"{c}={row[f'{c}_MAE']:.2f}/{row[f'{c}_MAPE']:.2f}%" for c in cols)
            + f"  fallback no index={row['n_fallback_no_index']} no regime={row['n_fallback_no_regime']}")
    pd.DataFrame(rows).to_csv(out / "benchmark_metrics.csv", index=False)
    (out / "benchmark_metadata.json").write_text(json.dumps({
        "run_dir": str(run_dir), "kind": a.kind, "panel": a.panel, "point_in_time": True,
        "min_complete_years": MIN_COMPLETE_YEARS, "min_z_readings": MIN_Z_READINGS,
        "pit_index_years_by_origin_year": {int(y): v[1] for y, v in si_pit.items()},
        "fullsample_index_years": full_years,
    }, indent=2, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
