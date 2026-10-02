"""
Scores the monthly GRU per docs/MONTHLY_PROTOCOL.md. Retrains nothing.

    python src/monthly_compare.py
    python src/monthly_compare.py --run outputs/build3_monthly_last/GRU_unconditional \
        --baselines outputs/monthly_baselines_last

1. Primary, on the monthly grid: monthly GRU vs monthly naive by the D-45 rule
   (DM p < .05 and a margin larger than one seed standard deviation of MAE), plus
   seasonal naive and FEWS NET Holt-Winters for context.
2. Head-to-head with the weekly flagship on matched pairs (built by
   src/monthly_baselines.py): both scored against the target month's average;
   DM, seed-paired t (seed s of each model), direction.
3. Both by target period.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from metrics import diebold_mariano, direction_pct, mae, mape  # noqa: E402

RUN = Path("outputs/build3_monthly/GRU_unconditional")
WEEKLY = Path("outputs/build3_underfit_corrected/GRU_unconditional")
BASE = Path("outputs/monthly_baselines")


def period(ts: pd.Series) -> pd.Series:
    return pd.cut(ts.dt.year, [2014, 2019, 2022, 2025], labels=["2015-19", "2020-22", "2023-24"])


def main() -> None:
    global RUN, BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--baselines", default=str(BASE))
    a = ap.parse_args()
    RUN, BASE = Path(a.run), Path(a.baselines)
    print(f"run {RUN}  baselines {BASE}")
    mp = pd.read_csv(RUN / "predictions_paired.csv", parse_dates=["origin"])
    mf = pd.read_csv(RUN / "forecasts.csv", parse_dates=["origin"])
    bp = pd.read_csv(BASE / "predictions.csv", parse_dates=["origin", "target"])
    k = ["market", "origin", "h"]
    g = bp.merge(mp[k + ["GRU", "actual"]].rename(columns={"actual": "actual_run"}), on=k, how="left")
    print(f"monthly grid pairs {len(bp)}; with a monthly GRU forecast {int(g.GRU.notna().sum())}")
    assert np.allclose(g.dropna(subset=["GRU"]).actual, g.dropna(subset=["GRU"]).actual_run)

    seed_mae = mf.groupby(["seed", "h"]).apply(lambda d: mae(d.actual, d.pred)).unstack("h")
    print("\n== 1. Monthly grid: monthly GRU vs monthly naive (D-45: DM p < .05 and margin > 1 seed SD)")
    for h in [1, 3, 6]:
        s = g[(g.h == h) & g.GRU.notna()].sort_values(["market", "origin"])
        e, en = mae(s.actual, s.GRU), mae(s.actual, s.naive)
        d = diebold_mariano(s.actual, s.GRU, s.naive, h, groups=s.market.values)
        sd = float(seed_mae[h].std(ddof=1))
        passes = d["dm_p"] < .05 and (en - e) > sd and e < en
        sn = s[s.seasonal_naive.notna()]
        print(f"  h={h}m n={len(s)}  GRU {e:.2f}/{mape(s.actual, s.GRU):.1f}%  naive {en:.2f}/{mape(s.actual, s.naive):.1f}%  "
              f"margin {100*(en-e)/en:+.1f}%  DM {d['dm_stat']:+.2f} p={d['dm_p']:.4f}  seed SD {sd:.2f}  -> {'PASS' if passes else 'fail'}")
        print(f"         direction GRU {direction_pct(s.actual, s.GRU, s.naive):.1f}%  | context: seasonal naive "
              f"{mae(sn.actual, sn.seasonal_naive):.2f} vs GRU {mae(sn.actual, sn.GRU):.2f} on its n={len(sn)};  "
              f"FEWS Holt-Winters {mae(s.dropna(subset=['fews_tes']).actual, s.dropna(subset=['fews_tes']).fews_tes):.2f}")

    print(f"\n== 2. Matched head-to-head with the weekly flagship (scored against the target month's "
          f"{'last observed week' if BASE.name.endswith('_last') else 'average'})")
    mt = pd.read_csv(BASE / "matched.csv", parse_dates=["origin", "origin_m"])
    mt = mt.merge(mp[["market", "origin", "h", "GRU"]].rename(columns={"origin": "origin_m", "h": "k", "GRU": "GRU_monthly"}),
                  on=["market", "origin_m", "k"], how="left")
    mt = mt.dropna(subset=["GRU_monthly", "seasonal_naive", "fews_tes"])
    wf = pd.read_csv(WEEKLY / "forecasts.csv", parse_dates=["origin"])
    for kk in [1, 3, 6]:
        s = mt[mt.k == kk].sort_values(["market", "origin_m"])
        ew, em = mae(s.actual_month, s.GRU), mae(s.actual_month, s.GRU_monthly)
        d = diebold_mariano(s.actual_month, s.GRU_monthly, s.GRU, kk, groups=s.market.values)
        ds = []
        for sd in sorted(mf.seed.unique()):
            a = mf[(mf.seed == sd) & (mf.h == kk)][["market", "origin", "pred"]].rename(columns={"origin": "origin_m", "pred": "pm"})
            b = wf[(wf.seed == sd) & (wf.h == {1: 4, 3: 13, 6: 26}[kk])][["market", "origin", "pred"]].rename(columns={"pred": "pw"})
            j = s[["market", "origin", "origin_m", "actual_month"]].merge(a, on=["market", "origin_m"]).merge(b, on=["market", "origin"])
            ds.append(mae(j.actual_month, j.pm) - mae(j.actual_month, j.pw))
        ds = np.array(ds); t = ds.mean() / (ds.std(ddof=1) / np.sqrt(len(ds)))
        win = d["dm_p"] < .05 and t < -2.447
        print(f"  k={kk}m n={len(s)}  weekly GRU {ew:.2f}/{mape(s.actual_month, s.GRU):.1f}%  monthly GRU {em:.2f}/{mape(s.actual_month, s.GRU_monthly):.1f}%  "
              f"DM {d['dm_stat']:+.2f} p={d['dm_p']:.4f}  seed-paired t {t:+.2f} ({(ds < 0).sum()}/7 seeds favour monthly)  -> "
              f"{'MONTHLY WINS' if win else ('weekly better' if em > ew else 'no significant difference')}")
        print(f"         direction (vs month-average change from month o): weekly {direction_pct(s.actual_month, s.GRU, s.naive_month):.1f}%  "
              f"monthly {direction_pct(s.actual_month, s.GRU_monthly, s.naive_month):.1f}%")

    print("\n== 3. By target period, MAPE")
    g2 = g[g.GRU.notna()].copy(); g2["period"] = period(g2.target)
    for h in [1, 3, 6]:
        s = g2[g2.h == h]
        print(f"  monthly grid h={h}m  " + "   ".join(
            f"{p}: GRU {mape(q.actual, q.GRU):.1f} / naive {mape(q.actual, q.naive):.1f}" for p, q in s.groupby("period", observed=True)))
    mt["period"] = period(mt.origin_m + pd.to_timedelta(mt.k * 30, unit="D"))
    for kk in [1, 3, 6]:
        s = mt[mt.k == kk]
        print(f"  matched k={kk}m      " + "   ".join(
            f"{p}: weekly {mape(q.actual_month, q.GRU):.1f} / monthly {mape(q.actual_month, q.GRU_monthly):.1f}" for p, q in s.groupby("period", observed=True)))


if __name__ == "__main__":
    main()
