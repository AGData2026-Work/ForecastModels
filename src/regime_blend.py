"""Regime-aware soft blend: RNN/GRU forecasts weighted against a per-market
seasonal-naive baseline by how volatile each market currently looks.

Exploratory (see docs/DECISIONS.md): not adopted as the production forecast,
built to test whether the models' well-established calm-period losses to
naive (D-52) can be reduced without new data. Everything here uses only the
maize price panel and the food inflation series already in data/external/ --
no new data source, per the same finding as D-52's own reasoning.

    python src/regime_blend.py --run-dir outputs/build3_underfit_corrected/RNN_unconditional --kind RNN --out outputs/regime_blend_soft/RNN
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from metrics import diebold_mariano, mae, mape

SCORED_MARKETS = [
    "Biu", "Damaturu", "Dandume", "Giwa", "Gombe", "Gujungu", "Gwandu, Dodoru",
    "Ibadan, Bodija", "Kano, Dawanau", "Kaura Namoda", "Lagos, Mile 12",
    "Maiduguri", "Mubi", "Potiskum", "Saminaka",
]
TRAILING_WINDOW = 13  # weeks


def build_food_cpi(inflation_path: Path) -> dict:
    """Monthly food CPI, compounded from month-on-month food inflation.
    Base value is arbitrary (100 at the first observed month); only ratios
    within and across years are ever used, so the base cancels out."""
    infl = pd.read_excel(inflation_path, sheet_name="Monthly Inflation")
    infl = infl.rename(columns={"Unnamed: 0": "date"}).sort_values("date").reset_index(drop=True)
    infl["date"] = pd.to_datetime(infl["date"])
    mom = infl["Food Inflation (%)-MoM"] / 100.0
    infl["cpi_food"] = 100.0 * (1 + mom).cumprod()
    infl["year"] = infl["date"].dt.year
    infl["month"] = infl["date"].dt.month
    return infl.set_index(["year", "month"])["cpi_food"].to_dict()


def build_seasonal_index(panel: pd.DataFrame, market: str, cpi_lookup: dict) -> dict | None:
    """Per-market deflated seasonal index, D-52's method: monthly nominal
    price deflated by food CPI, ratio to that year's own real average,
    geometric mean across complete (12-month) years, normalised to average
    100 across the twelve months. Returns None if fewer than 3 complete
    years exist for this market (none currently do, per D-52)."""
    df = panel[panel["market"] == market].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"], df["month"] = df["date"].dt.year, df["date"].dt.month
    monthly = df.groupby(["year", "month"], as_index=False)["price"].mean().rename(columns={"price": "N"})
    monthly["CPI"] = monthly.apply(lambda r: cpi_lookup.get((r["year"], r["month"]), np.nan), axis=1)
    monthly["R"] = monthly["N"] / monthly["CPI"]
    complete_years = monthly.groupby("year")["month"].nunique()
    complete_years = complete_years[complete_years == 12].index.tolist()
    if len(complete_years) < 3:
        return None
    m = monthly[monthly["year"].isin(complete_years)].copy()
    m["ratio"] = m["R"] / m.groupby("year")["R"].transform("mean")
    gm = m.groupby("month")["ratio"].apply(lambda x: np.exp(np.log(x).mean()))
    K = gm.mean()
    return (100 * gm / K).to_dict()


def seasonal_naive_forecast(origin_price: float, origin_month: int, target_month: int, si: dict) -> float:
    return origin_price * (si[target_month] / si[origin_month])


def compute_regime_z(panel: pd.DataFrame, market: str, window: int = TRAILING_WINDOW) -> pd.Series:
    """log(trailing `window`-week volatility / expanding median of that same
    market's own past volatility readings). Positive = more volatile than
    this market's own history to date; negative = calmer. Uses only price
    data at or before each date -- the expanding median at date i is built
    from readings strictly before i, so nothing here can see its own future,
    let alone the target the eventual forecast is being scored against."""
    s = panel[panel["market"] == market].sort_values("date").reset_index(drop=True)
    prices = s["price"].values
    pct_change = np.diff(prices) / prices[:-1]
    trailing_vol = np.full(len(prices), np.nan)
    for i in range(len(prices)):
        window_changes = pct_change[max(0, i - window):i]
        if len(window_changes) >= 4:
            trailing_vol[i] = np.std(window_changes)
    expanding_median = np.full(len(prices), np.nan)
    seen: list[float] = []
    for i in range(len(prices)):
        if len(seen) >= 8:
            expanding_median[i] = np.median(seen)
        if not np.isnan(trailing_vol[i]):
            seen.append(trailing_vol[i])
    valid = (trailing_vol > 0) & (expanding_median > 0)
    z = np.full(len(prices), np.nan)
    z[valid] = np.log(trailing_vol[valid] / expanding_median[valid])
    return pd.Series(z, index=s["date"].values)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="an existing run's output dir (predictions_paired.csv)")
    ap.add_argument("--kind", required=True, choices=["RNN", "GRU"])
    ap.add_argument("--panel", default="data/panel_weekly.parquet")
    ap.add_argument("--inflation", default="data/external/inflation.xlsx")
    ap.add_argument("--window", type=int, default=TRAILING_WINDOW)
    ap.add_argument("--mode", default="soft", choices=["soft", "hard"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    run_dir, out = Path(a.run_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(a.panel)
    panel = panel[panel["market"].isin(SCORED_MARKETS)].copy()
    cpi_lookup = build_food_cpi(Path(a.inflation))

    seasonal_index, z_by_market = {}, {}
    for mkt in SCORED_MARKETS:
        si = build_seasonal_index(panel, mkt, cpi_lookup)
        if si is None:
            raise SystemExit(f"{mkt}: fewer than 3 complete years, no fallback implemented -- "
                              f"see D-52, this has not happened for any of the 15 scored markets so far")
        seasonal_index[mkt] = si
        z_by_market[mkt] = compute_regime_z(panel, mkt, a.window)

    pred = pd.read_csv(run_dir / "predictions_paired.csv")
    pred["origin"] = pd.to_datetime(pred["origin"])
    pred["target_date"] = pred["origin"] + pd.to_timedelta(pred["h"] * 7, unit="D")
    pred["origin_month"], pred["target_month"] = pred["origin"].dt.month, pred["target_date"].dt.month

    pred["seasonal_naive"] = pred.apply(
        lambda r: seasonal_naive_forecast(r["origin_price"], r["origin_month"], r["target_month"],
                                          seasonal_index[r["market"]]), axis=1)

    z_scale = float(np.nanstd(np.concatenate([z_by_market[m].values for m in SCORED_MARKETS])))

    def lookup_z(row):
        z = z_by_market[row["market"]].get(row["origin"], np.nan)
        return z
    pred["z"] = pred.apply(lookup_z, axis=1)

    if a.mode == "soft":
        pred["weight"] = sigmoid(pred["z"] / z_scale)
        pred["weight"] = pred["weight"].fillna(1.0)  # unknown regime (too little history) -> full model weight
    else:
        pred["weight"] = np.where(pred["z"].isna(), 1.0, (pred["z"] > 0).astype(float))

    pred["blended"] = pred["weight"] * pred[a.kind] + (1 - pred["weight"]) * pred["seasonal_naive"]
    pred.to_csv(out / "predictions_with_blend.csv", index=False)

    rows = []
    for h in sorted(pred["h"].unique()):
        s = pred[pred.h == h].sort_values(["market", "origin"])
        row = dict(h=int(h), n=len(s), mode=a.mode, z_scale=z_scale,
                  model_MAE=mae(s["actual"], s[a.kind]), model_MAPE=mape(s["actual"], s[a.kind]),
                  naive_MAE=mae(s["actual"], s["naive"]), naive_MAPE=mape(s["actual"], s["naive"]),
                  seasonal_naive_MAE=mae(s["actual"], s["seasonal_naive"]), seasonal_naive_MAPE=mape(s["actual"], s["seasonal_naive"]),
                  blend_MAE=mae(s["actual"], s["blended"]), blend_MAPE=mape(s["actual"], s["blended"]))
        d_vs_naive = diebold_mariano(s["actual"], s["blended"], s["naive"], h, groups=s["market"].values)
        d_vs_seasonal = diebold_mariano(s["actual"], s["blended"], s["seasonal_naive"], h, groups=s["market"].values)
        row["dm_p_vs_naive"], row["dm_p_vs_seasonal_naive"] = d_vs_naive["dm_p"], d_vs_seasonal["dm_p"]
        rows.append(row)
        print(f"h={h}: blend MAE={row['blend_MAE']:.2f} MAPE={row['blend_MAPE']:.2f}%  "
              f"(model MAE={row['model_MAE']:.2f} MAPE={row['model_MAPE']:.2f}%)  "
              f"dm_p vs naive={row['dm_p_vs_naive']:.4f}  vs seasonal-naive={row['dm_p_vs_seasonal_naive']:.4f}")

    out_metrics = pd.DataFrame(rows)
    out_metrics.to_csv(out / "blend_metrics.csv", index=False)
    (out / "run_metadata.json").write_text(json.dumps({
        "run_dir": str(run_dir), "kind": a.kind, "mode": a.mode, "window": a.window, "z_scale": z_scale,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
