"""
FEWS NET weekly feeder: the month's white maize price from FEWS NET's weekly quotes, updated as each week is released
(owner request, 2026-11-04).

    python src/fewsnet_weekly_feeder.py --out data/external/<YYYYMMDD>_FEWSNET_WeeklyFeed_Monthly_v1.csv
    python src/fewsnet_weekly_feeder.py --weekly <existing fetch_fewsnet_weekly.py CSV> --out <...>   # no API call

Why: FEWS NET's monthly price is the mean of its weekly quotes (equal in 1,791 of 1,791 market-months, 2015 to Jun 2026),
and the weekly quotes reach the API about 4 days after each week ends, so the month's value can be followed while the
month is still running. Correction (4 Nov 2026, later): for a complete month the monthly value was NOT slower; Sep 2026
monthly and the last Sep week were both entered on 4 Oct. The earlier finding 'monthly absent on 4 Nov' came from a
multi-month API query that came back without the latest month that day (the same query was complete on 5 Nov; cause
not confirmed; src/fewsnet_monitor.py now requests month by month and cross-checks). The feeder's value is
the part-month view, and a fallback when the monthly series is late.

Per market and month, from the weekly quotes released so far:
  price_ngn_per_kg   mean of the weekly quotes in the month (keeps averaging as weeks arrive)
  weeks_in / weeks_expected   quotes released / Wednesdays in the month (FEWS NET dates its weeks on Wednesdays;
                     472 of 481 quotes since Jan 2025, the rest Tuesdays, counted in the week they belong to)
  status             complete: every expected week is in; closed_with_gaps: a later month's week is in but some of this
                     month's weeks never arrived (FEWS NET has moved on); partial: still waiting
  last_entered       latest API `created` timestamp among the month's quotes (when it entered FEWS NET's database)
Markets FEWS NET no longer quotes weekly (Giwa, Gombe, Gujungu, Gwandu, Lagos, Saminaka: last week Jan 2025) are absent.
src/build_gapfilled_panel.py --fews-weekly-feed reads this file. Every month is written, so history can be checked
against the monthly series (`--months` limits the output).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def wednesdays(month: pd.Period) -> int:
    d = pd.date_range(month.start_time, month.end_time.normalize(), freq="D")
    return int((d.dayofweek == 2).sum())


def monthly_feed(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    w["period_date"] = pd.to_datetime(w["period_date"])
    # a Tuesday quote belongs to that week's Wednesday
    w["week"] = w.period_date + pd.to_timedelta((2 - w.period_date.dt.dayofweek) % 7, unit="D")
    w["month"] = w.week.dt.to_period("M")
    w = w.drop_duplicates(["market", "week"], keep="last")
    last_week = w.groupby("market").week.max()
    rows = []
    for (mk, mo), g in w.groupby(["market", "month"]):
        exp = wednesdays(mo)
        n = g.week.nunique()
        status = "complete" if n >= exp else ("closed_with_gaps" if last_week[mk] > mo.end_time else "partial")
        rows.append(dict(market=mk, month=str(mo), price_ngn_per_kg=round(float(g.price_ngn_per_kg.mean()), 2), weeks_in=n,
                         weeks_expected=exp, status=status, last_week=g.week.max().date(),
                         last_entered=g["created"].max() if "created" in g else None))
    return pd.DataFrame(rows).sort_values(["month", "market"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--weekly", default=None, help="use an existing src/fetch_fewsnet_weekly.py CSV instead of calling the API")
    ap.add_argument("--months", type=int, default=None, help="write only the latest N months")
    a = ap.parse_args()
    if a.weekly:
        w = pd.read_csv(a.weekly)
    else:
        from fetch_fewsnet_weekly import SERIES, fetch
        w = pd.concat([fetch(m, s)[0] for m, s in SERIES.items()], ignore_index=True)
        raw = Path(a.out).with_name(Path(a.out).name.replace("WeeklyFeed_Monthly", "WeeklyFeed_RawQuotes"))
        w.to_csv(raw, index=False)
        print(f"raw weekly quotes -> {raw}")
    f = monthly_feed(w)
    if a.months:
        f = f[f.month.isin(sorted(f.month.unique())[-a.months:])]
    f.to_csv(a.out, index=False)
    latest = f[f.month == f.month.max()]
    print(f"{len(f)} market-months -> {a.out}; latest month {f.month.max()}: "
          + ", ".join(f"{r.market} {r.weeks_in}/{r.weeks_expected} {r.status}" for r in latest.itertuples()))


if __name__ == "__main__":
    main()
