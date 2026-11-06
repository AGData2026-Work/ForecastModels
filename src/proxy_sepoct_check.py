"""
Do the FEWSNET forecasts' error rates hold up on Sep and Oct 2026 (owner request, 2026-11-04)?

    python src/proxy_sepoct_check.py --out outputs/proxy_20261104

FEWS NET had not released Sep or Oct 2026 by 4 Nov 2026, so the "actual" here is the proxy value in gap-filled
panel v8 (20261104_FEWSNET_Maize_Monthly_GapFilled_Panel_v8.xlsx): AFEX and collector quotes, level-adjusted to
FEWS NET by the prior-overlap ratio. These are not FEWS NET prices; every row carries its source and a tracking grade.

Forecasts scored (every one made from information available at its origin):
  issued   origin Aug 2026, 1 month -> Sep: the production file (GRU, Inflation Adjusted, joint, 80% range)
  rerun    origins Jun and Jul 2026, 3 months -> Sep and Oct: the production GRU checkpoint (cut 2026-06-01, trained on
           targets to May 2026) on production panel v3 cut at the origin; the analyst's forecast for the same
           origin; joint = the production combination; 80% range = the production method (quantiles of all past joint
           misses whose target is at or before the origin)
  analyst  origins Mar and Apr 2026, 6 months -> Sep and Oct: the analyst's model only (the GRU checkpoint that would
           have served those origins was not saved)
  proxy    origin Sep 2026 (itself a proxy month), 1 month -> Oct: GRU only, from the v8 panel
Backtest reference: outputs/joint_forecast/backtest/summary.csv (MAPE on FEWS NET actuals, same models).

Sensitivity (actual_harvest): the panel level-adjusts AFEX by the median FEWS NET / AFEX ratio over ALL overlap months,
but in Sep and Oct FEWS NET has sat further below AFEX's old-crop quotes (Dandume 0.81 against 0.96, Saminaka 0.83
against 0.95). actual_harvest re-adjusts AFEX-sourced cells with the Sep-Oct-only ratio where there are at least 5
such overlap months; collector-sourced cells (no harvest-month overlap with FEWS NET) are left as they are.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from joint_forecast import combine  # noqa: E402
from range_experiments import table  # noqa: E402

V8 = "/Users/augmentumadvisory/Downloads/20261104_FEWSNET_Maize_Monthly_GapFilled_Panel_v8.xlsx"
RUNS = "outputs/proxy_20261104"
# how well the proxy source tracked FEWS NET where both exist (2024 to Aug 2026; monthly changes correlation)
GRADE = {"Saminaka": "strong (AFEX same market, 0.99)", "Dandume": "strong (AFEX same market, 0.93)",
         "Giwa": "fair (Sep collector, no overlap; Oct AFEX Tundun Saibu, 0.93)", "Biu": "fair (Lashe Money, 0.65)",
         "Potiskum": "fair (collector same market, 0.52)", "Damaturu": "fair (collector Potiskum, 0.63)",
         "Mubi": "fair (collector same market, 0.69, unadjusted)", "Kano, Dawanau": "weak (AFEX Dawanau, 0.40)",
         "Kaura Namoda": "weak (AFEX Talata Mafara, 0.66)"}


def harvest_ratios(sources: str) -> dict:
    """(target market, AFEX source market) -> (Sep-Oct median FEWS NET / AFEX ratio, n), same AFEX inputs as v8."""
    from build_gapfilled_panel import load_inputs
    dl = Path("/Users/augmentumadvisory/Downloads")
    inp = load_inputs(sources, None, "2026-10", dl / "_chat 2.txt", dl / "20261030_AFEX_Maize_Observed_AllMarkets_v3.xlsx")
    F, A = inp["F"], inp["src"]["AFEX"]
    out = {}
    for tgt in F.columns:
        for sm in A.columns:
            x = pd.concat([F[tgt], A[sm]], axis=1, keys=["f", "a"]).dropna()
            x = x[x.index.str[5:].isin(["09", "10"])]
            out[(tgt, sm)] = (float((x.f / x.a).median()) if len(x) else np.nan, len(x))
    return out


def ranges(model: str, h: int, origin: pd.Timestamp, point: pd.Series) -> tuple[pd.Series, pd.Series]:
    x = table(h)
    e = x.loc[x.target <= origin, f"e_{model}"].dropna().to_numpy()
    return point * np.exp(np.quantile(e, 0.10)), point * np.exp(np.quantile(e, 0.90))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=RUNS)
    ap.add_argument("--sources", required=True, help="folder with the panel source files (as build_gapfilled_panel --dl)")
    a = ap.parse_args()
    hr = harvest_ratios(a.sources)
    v8 = pd.read_excel(V8, "Long_panel")
    act = v8[v8.month.isin(["2026-09", "2026-10"]) & v8.price_NGN_per_kg.notna()].copy()
    act["target"] = pd.to_datetime(act.month + "-01")
    act = act.rename(columns={"price_NGN_per_kg": "actual"})[["market", "target", "actual", "source", "source_market",
                                                               "source_value", "ratio_applied"]]
    h = [hr.get((m, s), (np.nan, 0)) if src == "AFEX" else (np.nan, 0)
         for m, s, src in zip(act.market, act.source_market, act.source)]
    act["ratio_harvest"] = [r if n >= 5 else np.nan for r, n in h]
    act["n_harvest_overlap"] = [n for _, n in h]
    act["actual_harvest"] = np.where(act.ratio_harvest.notna(), act.source_value * act.ratio_harvest, act.actual)
    d = forecasts().merge(act, on=["market", "target"], how="inner")
    _score_and_write(d, act, a.out)


def forecasts() -> pd.DataFrame:
    """Every forecast scored here, one row per set and market, before any actual is attached."""
    an = pd.read_csv("outputs/joint_forecast/analyst_forecasts.csv", parse_dates=["origin", "target"])
    rows = []
    # issued: production joint file, origin Aug 2026
    p = pd.read_csv("outputs/joint_forecast/predictions/2026-10-12.csv", parse_dates=["origin"]).query("h == 1")
    for _, r in p.iterrows():
        rows.append(dict(set="issued Aug -> Sep (1 month)", market=r.market, origin=r.origin, h=1,
                         origin_price=r.origin_price_gru, gru=r.gru, analyst=r.inflation_adjusted, joint=r.joint,
                         lo80=r.lo80, hi80=r.hi80))
    # rerun: origins Jun and Jul, 3 months
    for o in ("06", "07"):
        g = pd.read_csv(f"{RUNS}/gru_serving_origin_2026{o}_v3/predictions/2026-11-04.csv", parse_dates=["origin"]).query("h == 3")
        x = g.merge(an[an.h == 3][["market", "origin", "p0", "recommended"]], on=["market", "origin"], how="left")
        x["joint"] = combine(x.forecast_price, x.origin_price, x.recommended, x.p0)
        x["lo80"], x["hi80"] = ranges("joint", 3, x.origin.iloc[0], x.joint)
        for _, r in x.iterrows():
            rows.append(dict(set=f"rerun {r.origin:%b} -> {r.origin + pd.DateOffset(months=3):%b} (3 months)", market=r.market,
                             origin=r.origin, h=3, origin_price=r.origin_price, gru=r.forecast_price,
                             analyst=r.recommended, joint=r.joint, lo80=r.lo80, hi80=r.hi80))
    # analyst only: origins Mar and Apr, 6 months
    for o in ("2026-03-01", "2026-04-01"):
        x = an[(an.h == 6) & (an.origin == o)].copy()
        x["lo80"], x["hi80"] = ranges("analyst", 6, pd.Timestamp(o), x.recommended)
        for _, r in x.iterrows():
            rows.append(dict(set=f"analyst only {r.origin:%b} -> {r.target:%b} (6 months)", market=r.market, origin=r.origin,
                             h=6, origin_price=r.p0, gru=np.nan, analyst=r.recommended, joint=np.nan, lo80=r.lo80, hi80=r.hi80))
    # proxy origin Sep, 1 month, GRU only
    g = pd.read_csv(f"{RUNS}/gru_serving_origin_202609/predictions/2026-11-04.csv", parse_dates=["origin"]).query("h == 1")
    g["lo80"], g["hi80"] = ranges("gru", 1, pd.Timestamp("2026-09-01"), g.forecast_price)
    for _, r in g.iterrows():
        rows.append(dict(set="proxy origin Sep -> Oct (1 month, GRU only)", market=r.market, origin=r.origin, h=1,
                         origin_price=r.origin_price, gru=r.forecast_price, analyst=np.nan, joint=np.nan, lo80=r.lo80, hi80=r.hi80))
    d = pd.DataFrame(rows)
    d["origin"] = pd.to_datetime(d.origin)
    d["target"] = pd.to_datetime([o + pd.DateOffset(months=int(h)) for o, h in zip(d.origin, d.h)])
    return d


def _score_and_write(d: pd.DataFrame, act: pd.DataFrame, out: str) -> None:
    d["grade"] = d.market.map(GRADE)
    d["headline"] = np.where(d.joint.notna(), d.joint, np.where(d.gru.notna(), d.gru, d.analyst))
    for sfx, col in [("", "actual"), ("_h", "actual_harvest")]:
        y = d[col]
        for c in ["naive", "gru", "analyst", "joint", "headline"]:
            src = d.origin_price if c == "naive" else d[c]
            d[f"ape_{c}{sfx}"] = 100 * (src - y).abs() / y
        d[f"err_headline_pct{sfx}"] = 100 * (d.headline / y - 1)
        d[f"inside80{sfx}"] = (y >= d.lo80) & (y <= d.hi80)
        d[f"position{sfx}"] = np.where(y < d.lo80, "below", np.where(y > d.hi80, "above", "inside"))
    out = Path(out)
    d.round(2).to_csv(out / "sepoct_rows.csv", index=False)

    def summ(x: pd.DataFrame, sfx: str) -> pd.Series:
        return pd.Series({"n": len(x), "naive_mape": x[f"ape_naive{sfx}"].mean(), "gru_mape": x[f"ape_gru{sfx}"].mean(),
                          "analyst_mape": x[f"ape_analyst{sfx}"].mean(), "joint_mape": x[f"ape_joint{sfx}"].mean(),
                          "headline_bias_pct": x[f"err_headline_pct{sfx}"].mean(), "inside80_pct": 100 * x[f"inside80{sfx}"].mean()})
    parts = []
    for sfx, actual in [("", "panel v8 (all-month ratio)"), ("_h", "harvest-month ratio for AFEX")]:
        for name, sub in [("all proxied", d), ("strong proxies only", d[d.grade.str.startswith("strong")])]:
            parts.append(sub.groupby("set").apply(summ, sfx).assign(actual=actual, markets=name))
    s = pd.concat(parts)
    s.round(1).to_csv(out / "sepoct_summary.csv")
    pd.set_option("display.width", 250)
    print(s.round(1).to_string())
    print(d[["set", "market", "headline", "lo80", "hi80", "actual", "actual_harvest", "err_headline_pct",
             "err_headline_pct_h", "position", "position_h"]].round(1).to_string(index=False))
    print(act[["market", "target", "source", "source_market", "source_value", "ratio_applied", "ratio_harvest",
               "n_harvest_overlap", "actual", "actual_harvest"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
