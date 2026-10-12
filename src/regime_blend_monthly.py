"""
Point-in-time soft blend for the monthly GRU (D-75), the monthly counterpart of src/regime_blend.py.

    python src/regime_blend_monthly.py --run-root outputs/d75_monthly_gapfilled_pit \
        --filled ~/Downloads/20261012_FEWSNET_Maize_Monthly_GapFilled_Panel_v3.xlsx \
        --out outputs/regime_blend_soft_monthly_pit

forecast = w x GRU + (1 - w) x seasonal naive, w = 1 / (1 + exp(-z / z_scale)):
- seasonal naive: origin price x index(target month) / index(origin month); index per market from
  FEWS NET observed months only, food-CPI deflated, complete calendar years before the origin's year
  (at least 3), as regime_blend.build_seasonal_index.
- z: log(volatility / its usual level). Volatility = standard deviation of the last 6 monthly % changes
  of the panel price up to the origin month (at least 4); usual level = median of that market's earlier
  readings (at least 8). A zero volatility gives no reading (as regime_blend).
- z_scale: standard deviation of every market's z readings dated before the origin (at least 24).
- No index or no reading: the GRU alone (w = 1).
Reads one run per horizon (<run-root>/h1, h3, h6).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import regime_blend as RB  # noqa: E402
from metrics import diebold_mariano, direction_pct, mae, mape  # noqa: E402

VOL_WINDOW, VOL_MIN, MED_MIN, ZSCALE_MIN = 6, 4, 8, 24


def blend(d: pd.DataFrame, L: pd.DataFrame, inflation: Path) -> pd.DataFrame:
    d = d.copy()
    d["target"] = [o + pd.DateOffset(months=int(h)) for o, h in zip(d.origin, d.h)]
    L = L.copy()
    L["date"] = pd.PeriodIndex(L.month, freq="M").to_timestamp()
    obs = L[L.flag == "observed"].rename(columns={"price_NGN_per_kg": "price"})[["market", "date", "price"]]
    cpi = RB.build_food_cpi(inflation)
    si, sn = {}, []
    for m, o, t, p0 in zip(d.market, d.origin, d.target, d.naive):
        k = (m, o.year)
        if k not in si:
            si[k] = RB.build_seasonal_index(obs, m, cpi, before_year=o.year)
        sn.append(np.nan if si[k] is None else RB.seasonal_naive_forecast(p0, o.month, t.month, si[k]))
    d["seasonal_naive"] = sn
    pp = L.pivot(index="date", columns="market", values="price_NGN_per_kg").sort_index()
    ch = pp.pct_change(fill_method=None)
    Z = {}
    for m in pp.columns:
        vol = ch[m].rolling(VOL_WINDOW, min_periods=VOL_MIN).std()
        med = vol.expanding(min_periods=MED_MIN).median().shift(1)
        Z[m] = np.log(vol.where(vol > 0) / med.where(med > 0))
    Z = pd.DataFrame(Z)
    zs = Z.stack().dropna()
    zdates = zs.index.get_level_values(0)
    d["z"] = [Z.loc[o, m] if o in Z.index and m in Z.columns else np.nan for m, o in zip(d.market, d.origin)]
    scale = {pd.Timestamp(o): (zs[zdates < o].std() if (zdates < o).sum() >= ZSCALE_MIN else np.nan) for o in d.origin.unique()}
    d["z_scale"] = d.origin.map(scale)
    w = 1 / (1 + np.exp(-d.z / d.z_scale))
    ok = d.seasonal_naive.notna() & w.notna()
    d["weight"] = np.where(ok, w, 1.0)
    d["blended"] = np.where(ok, d.weight * d.GRU + (1 - d.weight) * d.seasonal_naive, d.GRU)
    d["fallback"] = np.where(d.seasonal_naive.isna(), "no seasonal index", np.where(w.isna(), "no regime reading", ""))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--filled", required=True)
    ap.add_argument("--inflation", default="data/external/inflation.xlsx")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(f"{a.run_root}/h{h}/predictions_paired.csv", parse_dates=["origin"]).query("h==@h")
                   for h in (1, 3, 6)], ignore_index=True)
    d = blend(d, pd.read_excel(a.filled, "Long_panel"), Path(a.inflation))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "predictions_with_blend.csv", index=False)
    rows = []
    for h, s in d.groupby("h"):
        s = s.sort_values(["market", "origin"])
        n = mae(s.actual, s.naive)
        rows.append(dict(h=int(h), n=len(s), mean_weight_on_GRU=round(s.weight.mean(), 3),
                         fallback_no_index=int((s.fallback == "no seasonal index").sum()),
                         fallback_no_regime=int((s.fallback == "no regime reading").sum()),
                         GRU_MAE=mae(s.actual, s.GRU), blend_MAE=mae(s.actual, s.blended), naive_MAE=n,
                         GRU_MAPE=mape(s.actual, s.GRU), blend_MAPE=mape(s.actual, s.blended),
                         GRU_direction=direction_pct(s.actual, s.GRU, s.naive), blend_direction=direction_pct(s.actual, s.blended, s.naive),
                         blend_vs_naive_pct=100 * (n - mae(s.actual, s.blended)) / n,
                         dm_p_blend_vs_GRU=diebold_mariano(s.actual.values, s.blended.values, s.GRU.values, int(h), groups=s.market.values)["dm_p"]))
    M = pd.DataFrame(rows)
    M.to_csv(out / "blend_metrics.csv", index=False)
    json.dump({"run_root": a.run_root, "filled": a.filled, "vol_window_months": VOL_WINDOW,
               "min_complete_years": RB.MIN_COMPLETE_YEARS, "zscale_min_readings": ZSCALE_MIN}, open(out / "run_metadata.json", "w"), indent=2)
    print(M.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
