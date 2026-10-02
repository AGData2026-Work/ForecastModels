"""
Panel loading, evaluation-grid consumption, and window construction.

Design notes that matter for correctness
----------------------------------------
1.  The evaluation grid is READ FROM the baseline forecast file, never
    regenerated.  The baseline grid is per-market 4-weekly and STAGGERED across
    markets (see docs/DATA_AUDIT.md).  Regenerating it as one global 4-weekly
    sequence -- which the lost original did -- keeps only 558 of 1601 pairs and
    caused the grid artefact recorded in the prior session's notes.

2.  For grid rows, the origin price and the target price are taken from the
    baseline file's own `naive` and `actual` columns, not from the panel.  The
    baseline's `naive` column IS the price at the origin week.  Using it
    guarantees that challenger and incumbent are scored against byte-identical
    actuals, so any MAE difference is model, not bookkeeping.

3.  Everything the network sees is expressed RELATIVE TO THE ORIGIN in log
    space.  A window carries no price level.  This is the single most important
    representational choice; see docs/METHODOLOGY.md section 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEQ_BASE_CHANNELS = [
    "rel_log_price",
    "rel_log_diesel",
    "rel_log_upstream",
    "rainfall",
    "ndvi",
    "sin1",
    "cos1",
    "sin2",
    "cos2",
    "upstream_mask",
]

# Appended after SEQ_BASE_CHANNELS only when build_sequence(..., use_macro=True).
# Ported from afex-multicommodity D-53 for the FEWSNET FX test (D-60 follow-up):
# national official FX rate (log-ratio to origin) and headline inflation YoY.
MACRO_CHANNELS = [
    "rel_log_fx_rate",
    "inflation_yoy",
]


def sequence_channel_names(use_macro: bool = False) -> list[str]:
    return SEQ_BASE_CHANNELS + (MACRO_CHANNELS if use_macro else [])


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------
@dataclass
class Panel:
    dates: pd.DatetimeIndex
    markets: list[str]
    pos: dict[pd.Timestamp, int]
    midx: dict[str, int]
    price: np.ndarray          # (T, M)
    price_filled: np.ndarray   # (T, M) bool -- True where interpolated here
    diesel: np.ndarray
    upstream: np.ndarray
    rainfall: np.ndarray
    ndvi: np.ndarray
    fourier: np.ndarray        # (T, 4)
    has_upstream: np.ndarray   # (M,) bool
    log: dict
    # Steps per seasonal cycle: 52 on the weekly panel (the default, unchanged),
    # 12 on the monthly panel (monthly-cadence experiment). Drives the lag
    # anchors in build_flat and the history needed in build_training_windows.
    season_lag: int = 52
    fx_rate: np.ndarray | None = None        # (T, M), national series broadcast to every column
    # FX restructuring option 2: bounded national FX/inflation summaries per date,
    # (T, K), the same for every market; names in fx_summary_names.
    fx_summary: np.ndarray | None = None
    fx_summary_names: list | None = None
    inflation_yoy: np.ndarray | None = None  # (T, M), national series broadcast to every column


def validate_panel(p: "Panel") -> None:
    """Centralized invariant checks, called by both `load_panel` and
    `load_afex_panel` at the end of construction. Raises on the first
    violation found rather than collecting all of them, since any one of
    these means the Panel should not be used at all, not "used with a
    warning." The weekly-spacing check is enforced earlier, on the raw
    dates before the Panel is built, so it is not repeated here."""
    if len(p.markets) != len(set(p.markets)):
        raise ValueError("duplicate market name in Panel.markets")
    empty_markets = [m for j, m in enumerate(p.markets)
                     if not np.isfinite(p.price[:, j]).any()]
    if empty_markets:
        raise ValueError(f"market(s) with no observed price at any date: {empty_markets}")
    negative_or_zero = np.isfinite(p.price) & (p.price <= 0)
    if negative_or_zero.any():
        n = int(negative_or_zero.sum())
        raise ValueError(f"{n} price cell(s) are <= 0 where a value is present -- "
                         "a real price panel should never have this, it means an "
                         "upstream data or unit-conversion bug, not a natural gap")


def _interpolate_within_span(col: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate interior gaps up to `limit` long. Never extrapolates."""
    s = pd.Series(col)
    obs = s.dropna()
    filled = np.zeros(len(col), dtype=bool)
    if obs.empty:
        return col, filled
    lo, hi = obs.index[0], obs.index[-1]
    inner = s.loc[lo:hi]
    out = inner.interpolate(method="linear", limit=limit, limit_area="inside")
    filled_inner = inner.isna() & out.notna()
    s.loc[lo:hi] = out
    filled[lo:hi + 1] = filled_inner.to_numpy()
    return s.to_numpy(dtype=float), filled


def load_panel(
    path: str | Path,
    rainfall_scheme: str = "inverse_distance",
    fourier_k: int = 2,
    interp_limit: int = 13,
    frequency: str = "W",
) -> Panel:
    """`frequency` "W" (default: the weekly panel, behaviour unchanged) or "MS"
    (the monthly panel, month-start dates). `interp_limit` is in steps of that
    frequency."""
    df = pd.read_parquet(path)
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    markets = sorted(df["market"].unique())
    pos = {t: i for i, t in enumerate(dates)}
    midx = {m: j for j, m in enumerate(markets)}

    if frequency == "W":
        step = pd.Series(np.diff(dates).astype("timedelta64[D]").astype(int))
        if not (step == 7).all():
            raise ValueError(f"panel date index is not a clean weekly grid: {step.unique()}")
        season_lag = 52
    elif frequency == "MS":
        expected = pd.date_range(dates[0], dates[-1], freq="MS")
        if len(expected) != len(dates) or not (expected == dates).all():
            raise ValueError("panel date index is not a clean month-start grid")
        season_lag = 12
    else:
        raise ValueError(f"unknown frequency {frequency!r}")

    def grid(col: str) -> np.ndarray:
        return (df.pivot(index="date", columns="market", values=col)
                  .reindex(dates)[markets].to_numpy(dtype=float))

    price_raw = grid("price")
    price = price_raw.copy()
    filled = np.zeros_like(price, dtype=bool)
    for j in range(price.shape[1]):
        price[:, j], filled[:, j] = _interpolate_within_span(price_raw[:, j], interp_limit)

    rain = grid(f"rainfall_{rainfall_scheme}")
    nd = grid(f"ndvi_{rainfall_scheme}")
    diesel = grid("diesel")
    upstream = grid("upstream_price")

    fcols = []
    for k in range(1, fourier_k + 1):
        fcols += [f"sin{k}", f"cos{k}"]
    fou = (df.drop_duplicates("date").set_index("date")[fcols]
             .reindex(dates).to_numpy(dtype=float))

    up_map = df.groupby("market")["upstream_market"].first()
    has_up = np.array([pd.notna(up_map.get(m)) for m in markets])

    log = {
        "panel_path": str(path),
        "frequency": frequency,
        "season_lag": season_lag,
        "n_weeks": len(dates),
        "n_markets": len(markets),
        "first_week": str(dates[0].date()),
        "last_week": str(dates[-1].date()),
        "rainfall_scheme": rainfall_scheme,
        "fourier_k": fourier_k,
        "price_cells_observed": int(np.isfinite(price_raw).sum()),
        "price_cells_interpolated": int(filled.sum()),
        "interp_limit_weeks": interp_limit,
        "full_panel_outage_weeks": [
            str(dates[i].date()) for i in np.where(~np.isfinite(price_raw).any(axis=1))[0]
        ],
        "root_markets": [m for m, ok in zip(markets, has_up) if not ok],
    }
    panel = Panel(dates, markets, pos, midx, price, filled, diesel, upstream,
                  rain, nd, fou, has_up, log, season_lag=season_lag)
    validate_panel(panel)
    return panel


def _national_monthly_point_in_time(source_path: str | Path, sheet: str, date_col: str,
                                    value_col: str, target_dates: pd.DatetimeIndex,
                                    lag_months: int) -> tuple[np.ndarray, dict]:
    """One national monthly series on the weekly grid, POINT-IN-TIME: a week
    in calendar month m gets the value for month m - lag_months, the latest
    reading that would actually have been published by then. The AFEX
    version (afex_data._national_monthly_series) gave every week its own
    month's value, which uses a monthly average or a release that is not
    yet out. Never interpolated; weeks before the first available reading
    are NaN."""
    raw = pd.read_excel(source_path, sheet_name=sheet).rename(columns={date_col: "date"})
    raw["date"] = pd.to_datetime(raw["date"])
    s = raw.drop_duplicates("date").set_index("date")[value_col].sort_index()
    s.index = s.index.to_period("M")
    wk = target_dates.to_period("M") - lag_months
    vals = s.reindex(wk).to_numpy(dtype=float)
    # carry the last published reading forward past the end of the file
    last = s.index.max()
    tail = wk > last
    vals[tail] = float(s.loc[last])
    return vals, {"source": str(source_path), "column": value_col, "lag_months": lag_months,
                  "last_real_month": str(last), "weeks_carried_past_end": int(tail.sum()),
                  "weeks_missing": int(np.isnan(vals).sum())}


FX_SUMMARY_NAMES = ["fx_dep3_rank", "fx_dep6_rank", "fx_dep12_rank",
                    "infl_level_rank", "infl_change6_rank", "months_since_deval"]
DEVALUATION = np.log(1.10)   # a month in which the naira fell 10% or more


def _expanding_rank(v: np.ndarray, min_past: int = 24) -> np.ndarray:
    """Share of strictly earlier finite values that are <= v[i]. Bounded in [0, 1]
    and point-in-time; NaN until `min_past` earlier values exist."""
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        past = v[:i][np.isfinite(v[:i])]
        if np.isfinite(v[i]) and len(past) >= min_past:
            out[i] = float(np.mean(past <= v[i]))
    return out


def attach_fx_summary(p: Panel, fx_path: str | Path, inflation_path: str | Path,
                      fx_lag_months: int = 1, inflation_lag_months: int = 2,
                      keep: list[str] | None = None) -> None:
    """FX restructuring option 2. Six bounded flat features at each date, computed on
    the monthly national series and then looked up for the latest month that would have
    been published by that date (lags as attach_macro):
      fx_dep{3,6,12}_rank  percentile of the naira's depreciation over the last 3/6/12
                           months among every earlier month's value (history from 2004)
      infl_level_rank      percentile of headline inflation YoY among earlier months
      infl_change6_rank    percentile of its 6-month change
      months_since_deval   months since the last month with a >= 10% depreciation,
                           capped at 24 and divided by 24 (1.0 = none in two years)
    Ranks use strictly earlier values, so a record devaluation reads 1.0 rather than an
    extrapolated extreme."""
    fx = pd.read_excel(fx_path).rename(columns={"Months": "date"})
    fx["date"] = pd.to_datetime(fx["date"])
    f = fx.set_index("date")["Official Exchange Rate"].sort_index()
    f.index = f.index.to_period("M")
    lf = np.log(f)
    feats = pd.DataFrame(index=f.index)
    for L in (3, 6, 12):
        feats[f"fx_dep{L}_rank"] = _expanding_rank(lf.diff(L).to_numpy())
    dev = (lf.diff() >= DEVALUATION).to_numpy()
    since = np.full(len(f), np.nan); last = None
    for i in range(len(f)):
        if dev[i]:
            last = i
        since[i] = 24.0 if last is None else min(i - last, 24)
    feats["months_since_deval"] = since / 24.0
    inf = pd.read_excel(inflation_path, sheet_name="Monthly Inflation").rename(columns={"Unnamed: 0": "date"})
    inf["date"] = pd.to_datetime(inf["date"])
    y = inf.set_index("date")["Headline Inflation (%)-YoY"].sort_index()
    y.index = y.index.to_period("M")
    infl = pd.DataFrame({"infl_level_rank": _expanding_rank(y.to_numpy()),
                         "infl_change6_rank": _expanding_rank(y.diff(6).to_numpy())}, index=y.index)
    months = p.dates.to_period("M")
    fx_rows = feats.reindex(months - fx_lag_months).to_numpy(dtype=float)
    in_rows = infl.reindex(months - inflation_lag_months).to_numpy(dtype=float)
    X = np.column_stack([fx_rows[:, :3], in_rows, fx_rows[:, 3:]])
    names = ["fx_dep3_rank", "fx_dep6_rank", "fx_dep12_rank", "infl_level_rank",
             "infl_change6_rank", "months_since_deval"]
    assert names == FX_SUMMARY_NAMES
    if keep:  # ablations: a subset of the six features, in their standard order
        unknown = set(keep) - set(names)
        if unknown:
            raise ValueError(f"unknown fx_summary features {sorted(unknown)}")
        idx = [i for i, n in enumerate(names) if n in keep]
        X, names = X[:, idx], [names[i] for i in idx]
    p.fx_summary, p.fx_summary_names = X, names
    p.log["fx_summary"] = {"fx_lag_months": fx_lag_months, "inflation_lag_months": inflation_lag_months,
                           "devaluation_months": [str(m) for m, d in zip(f.index, dev) if d],
                           "dates_with_missing_feature": int((~np.isfinite(X)).any(axis=1).sum())}


def attach_macro(p: Panel, fx_path: str | Path, inflation_path: str | Path,
                 fx_lag_months: int = 1, inflation_lag_months: int = 2) -> None:
    fx, fx_log = _national_monthly_point_in_time(fx_path, "Sheet1", "Months",
                                                 "Official Exchange Rate", p.dates, fx_lag_months)
    inf, inf_log = _national_monthly_point_in_time(inflation_path, "Monthly Inflation", "Unnamed: 0",
                                                   "Headline Inflation (%)-YoY", p.dates,
                                                   inflation_lag_months)
    M = len(p.markets)
    p.fx_rate = np.repeat(fx[:, None], M, axis=1)
    p.inflation_yoy = np.repeat(inf[:, None], M, axis=1)
    p.log["macro"] = {"fx_rate": fx_log, "inflation_yoy": inf_log}


# ---------------------------------------------------------------------------
# evaluation grid
# ---------------------------------------------------------------------------
def load_grid(
    path: str | Path,
    horizons: list[int],
    drop_targets: list[str] | None = None,
    require_horizons: list[int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Baseline forecasts, wide by horizon, one row per (market, origin).

    Returns the grid plus a log of what was dropped and why.

    `require_horizons` defaults to None, meaning "require every horizon in
    `horizons`" -- today's behaviour, unchanged for every caller that doesn't
    pass it. Pass a subset (e.g. [4, 13] out of horizons=[4, 13, 26]) to keep
    an origin whose longer horizon has no actual yet (the data hasn't
    happened) while still scoring it at the horizons that do. This never
    changes which origins are scored at a horizon that IS required; it only
    stops an origin missing an unrequired horizon from being dropped whole.
    """
    f = pd.read_parquet(path)
    if set(horizons) - set(f["horizon"].unique()):
        raise ValueError(f"baseline file lacks horizons {set(horizons) - set(f.horizon.unique())}")
    f = f[f["horizon"].isin(horizons)].copy()

    dropped = pd.DataFrame()
    if drop_targets:
        bad = pd.to_datetime(drop_targets)
        mask = f["target"].isin(bad) | f["origin"].isin(bad)
        dropped = f[mask].copy()
        f = f[~mask]

    piv = f.pivot_table(index=["market", "origin"], columns="horizon",
                        values=["actual", "panel_fe", "naive"], aggfunc="first")
    piv.columns = [f"{a}_h{h}" for a, h in piv.columns]
    piv = piv.reset_index()
    # pivot_table drops a value column that is entirely NaN (the monthly grid has no
    # panel_fe forecasts); restore it as NaN so downstream code sees the same columns.
    # No effect on the weekly grid, where every column has values.
    for v in ("actual", "panel_fe", "naive"):
        for h in horizons:
            if f"{v}_h{h}" not in piv.columns:
                piv[f"{v}_h{h}"] = np.nan

    need = require_horizons if require_horizons is not None else horizons
    complete = piv[[f"actual_h{h}" for h in need]].notna().all(axis=1)
    incomplete = piv[~complete].copy()
    piv = piv[complete].reset_index(drop=True)

    # naive is identical across horizons for a given origin: that is the origin price
    piv["origin_price"] = piv[f"naive_h{horizons[0]}"]

    log = {
        "grid_path": str(path),
        "horizons": horizons,
        "require_horizons": need,
        "n_market_origin_pairs": int(len(piv)),
        "n_pairs_per_horizon": int(len(piv)),
        "markets": sorted(piv["market"].unique()),
        "first_origin": str(piv["origin"].min().date()),
        "last_origin": str(piv["origin"].max().date()),
        "dropped_by_date_filter": int(len(dropped)),
        "dropped_target_dates": sorted(set(str(t.date()) for t in dropped["target"])) if len(dropped) else [],
        "dropped_incomplete_horizon_set": int(len(incomplete)),
        "origins_per_market": piv.groupby("market").size().to_dict(),
    }
    return piv, log


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------
def _safe_log_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    ok = np.isfinite(num) & np.isfinite(den) & (num > 0) & (den > 0)
    out[ok] = np.log(num[ok]) - np.log(den[ok])
    return out


# ---------------------------------------------------------------------------
# first-cut driver forecasts (Task 9) -- everything computable at the origin
# ---------------------------------------------------------------------------
# Diesel method choice, validated once in src/driver_forecasts.py against
# plain carry-forward on this panel's history. See docs/DECISIONS.md D-25.
DIESEL_FORECAST_METHOD = "rw_drift"


def climatology_forecast(series: np.ndarray, origin_idx: int, target_idx: int,
                         max_years_back: int = 20) -> float:
    """Mean of series at (target_idx - 52*k) for k=1,2,..., using only indices
    STRICTLY BEFORE origin_idx (never the target's own year or later -- using
    the origin, not the target, as the anti-leakage boundary, consistent with
    build_flat's lag-52 anchors elsewhere in this module). NaN if no
    qualifying history exists."""
    vals = []
    for k in range(1, max_years_back + 1):
        a = target_idx - 52 * k
        if a < 0:
            break
        if a >= origin_idx:
            continue
        v = series[a]
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def _diesel_drift(series: np.ndarray, origin_idx: int) -> float:
    """Mean weekly log-difference over all history strictly before
    origin_idx (expanding, not rolling, consistent with this repo's
    walk-forward)."""
    hist = series[:origin_idx]
    hist = hist[np.isfinite(hist) & (hist > 0)]
    if len(hist) < 2:
        return 0.0
    return float(np.mean(np.diff(np.log(hist))))


def diesel_forecast(series: np.ndarray, origin_idx: int, target_idx: int,
                    method: str = DIESEL_FORECAST_METHOD) -> float:
    """Random walk with drift, or plain carry-forward. `method` defaults to
    the choice validated in src/driver_forecasts.py (D-25)."""
    p0 = series[origin_idx]
    if not np.isfinite(p0) or p0 <= 0:
        return float("nan")
    if method == "carry_forward":
        return float(p0)
    drift = _diesel_drift(series, origin_idx)
    return float(p0 * np.exp(drift * (target_idx - origin_idx)))


def upstream_forecast_seasonal_naive(series: np.ndarray, target_idx: int) -> float:
    """The value 52 weeks before the TARGET. target_idx - 52 <= origin_idx
    whenever h <= 52, so this is always observed at the origin: no
    look-ahead, and no chaining a price forecast into another price
    forecast (rejected explicitly in the source task)."""
    a = target_idx - 52
    if a < 0 or not np.isfinite(series[a]) or series[a] <= 0:
        return float("nan")
    return float(series[a])


def build_sequence(
    p: Panel, i: int, j: int, lookback: int, use_macro: bool = False
) -> tuple[np.ndarray, bool]:
    """(lookback, C) channel block for origin index i, market j. Window ends AT i."""
    if i - lookback + 1 < 0:
        # A negative slice start silently wraps around in numpy rather than
        # raising, and an empty resulting slice passes np.isfinite(...).all()
        # vacuously (True on an empty array), so this must be checked
        # explicitly rather than relying on the finite-value guard below to
        # catch it. Caught by tests/test_data.py, never exercised by any real
        # grid today (every real origin already has full lookback history).
        return np.empty((0, 0)), False
    sl = slice(i - lookback + 1, i + 1)
    px = p.price[sl, j]
    p0 = p.price[i, j]
    if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(px).all():
        return np.empty((0, 0)), False

    dz = p.diesel[sl, j]
    up = p.upstream[sl, j]
    up_ok = p.has_upstream[j] and np.isfinite(p.upstream[i, j]) and p.upstream[i, j] > 0

    chan = [
        np.log(px) - np.log(p0),
        _safe_log_ratio(dz, np.full(lookback, p.diesel[i, j])),
        _safe_log_ratio(up, np.full(lookback, p.upstream[i, j])) if up_ok else np.zeros(lookback),
        p.rainfall[sl, j],
        p.ndvi[sl, j],
    ]
    for k in range(p.fourier.shape[1]):
        chan.append(p.fourier[sl, k])
    chan.append(np.full(lookback, 1.0 if up_ok else 0.0))
    if use_macro:
        if p.fx_rate is None or p.inflation_yoy is None:
            raise ValueError("use_macro=True but Panel has no fx_rate/inflation_yoy "
                             "(call attach_macro first)")
        chan.append(_safe_log_ratio(p.fx_rate[sl, j], np.full(lookback, p.fx_rate[i, j])))
        chan.append(p.inflation_yoy[sl, j])

    X = np.column_stack(chan)
    if not np.isfinite(X).all():
        return np.empty((0, 0)), False
    return X, True


def build_flat(
    p: Panel, i: int, j: int, horizons: list[int],
    use_lag52: bool, use_realised_drivers: bool,
    realised_upstream: bool = True,
    driver_source: str = "realised",
    use_fx_summary: bool = False,
) -> tuple[np.ndarray, bool]:
    """Non-sequence features. Empty array when both switches are off (Build 1
    unconditional). `driver_source` picks between the CONDITIONAL arm's
    perfect-foresight realised driver values ("realised", the default) and
    Task 9's first-cut forecasts ("forecast": climatology for rainfall/NDVI,
    random walk with drift for diesel, seasonal-naive for upstream -- see
    docs/DECISIONS.md D-25). Only meaningful when use_realised_drivers=True."""
    p0 = p.price[i, j]
    vals: list[float] = []

    if use_lag52:
        # Anchor for each horizon: the price in the SAME CALENDAR WEEK one year
        # before the target. t+h-52 <= t whenever h <= 52, so this is always
        # observed at the origin -- no look-ahead. S = p.season_lag (52 weekly,
        # 12 monthly: the same calendar month one year before the target).
        S = p.season_lag
        for h in horizons:
            a = i + h - S
            if a < 0 or not np.isfinite(p.price[a, j]) or p.price[a, j] <= 0:
                return np.empty(0), False
            vals.append(float(np.log(p.price[a, j]) - np.log(p0)))
        # What the same window did last year: log(P[t+h-52]) - log(P[t-52]).
        for h in horizons:
            a, b = i + h - S, i - S
            if b < 0 or not np.isfinite(p.price[a, j]) or not np.isfinite(p.price[b, j]):
                return np.empty(0), False
            if p.price[a, j] <= 0 or p.price[b, j] <= 0:
                return np.empty(0), False
            vals.append(float(np.log(p.price[a, j]) - np.log(p.price[b, j])))

    if use_realised_drivers and driver_source == "realised":
        # CONDITIONAL arm only. Driver values over the forecast window, i.e.
        # information NOT available at the origin. Mirrors what the incumbent
        # panel FE appears to have used (docs/CONDITIONAL_CONVENTION.md).
        for h in horizons:
            t = i + h
            if t >= p.price.shape[0]:
                return np.empty(0), False
            vals.append(float(_safe_log_ratio(np.array([p.diesel[t, j]]),
                                              np.array([p.diesel[i, j]]))[0]))
            if realised_upstream:
                up_ok = p.has_upstream[j] and np.isfinite(p.upstream[i, j]) and p.upstream[i, j] > 0
                vals.append(float(_safe_log_ratio(np.array([p.upstream[t, j]]),
                                                  np.array([p.upstream[i, j]]))[0]) if up_ok else 0.0)
            seg = slice(i + 1, t + 1)
            r, n = p.rainfall[seg, j], p.ndvi[seg, j]
            vals.append(float(np.nanmean(r)) if np.isfinite(r).any() else 0.0)
            vals.append(float(np.nanmean(n)) if np.isfinite(n).any() else 0.0)

    elif use_realised_drivers and driver_source == "forecast":
        # FORECAST_DRIVERS arm (Task 9). Same four quantities as the
        # conditional arm, but every value is something computable AT THE
        # ORIGIN: climatology for rainfall/NDVI, RW+drift for diesel,
        # seasonal-naive for upstream. No perfect foresight anywhere here.
        for h in horizons:
            t = i + h
            if t >= p.price.shape[0]:
                return np.empty(0), False
            d0 = p.diesel[i, j]
            dfc = diesel_forecast(p.diesel[:, j], i, t)
            vals.append(float(_safe_log_ratio(np.array([dfc]), np.array([d0]))[0]))

            up_ok = p.has_upstream[j] and np.isfinite(p.upstream[i, j]) and p.upstream[i, j] > 0
            if up_ok:
                ufc = upstream_forecast_seasonal_naive(p.upstream[:, j], t)
                vals.append(float(_safe_log_ratio(np.array([ufc]),
                                                  np.array([p.upstream[i, j]]))[0])
                            if np.isfinite(ufc) else 0.0)
            else:
                vals.append(0.0)

            rain_fc = [climatology_forecast(p.rainfall[:, j], i, w) for w in range(i + 1, t + 1)]
            ndvi_fc = [climatology_forecast(p.ndvi[:, j], i, w) for w in range(i + 1, t + 1)]
            rain_fc = [v for v in rain_fc if np.isfinite(v)]
            ndvi_fc = [v for v in ndvi_fc if np.isfinite(v)]
            vals.append(float(np.mean(rain_fc)) if rain_fc else 0.0)
            vals.append(float(np.mean(ndvi_fc)) if ndvi_fc else 0.0)

    if use_fx_summary:
        if p.fx_summary is None:
            raise ValueError("use_fx_summary=True but Panel has no fx_summary (call attach_fx_summary)")
        vals.extend(float(v) for v in p.fx_summary[i])

    arr = np.asarray(vals, dtype=float)
    if arr.size and not np.isfinite(arr).all():
        return np.empty(0), False
    return arr, True


def flat_feature_names(horizons: list[int], use_lag52: bool, use_realised: bool,
                       realised_upstream: bool = True,
                       driver_source: str = "realised",
                       use_fx_summary: bool = False,
                       fx_summary_names: list[str] | None = None) -> list[str]:
    names = []
    if use_lag52:
        names += [f"lag52_anchor_h{h}" for h in horizons]
        names += [f"lag52_window_return_h{h}" for h in horizons]
    if use_realised:
        prefix = "realised" if driver_source == "realised" else "forecast"
        for h in horizons:
            names.append(f"{prefix}_diesel_ret_h{h}")
            if realised_upstream:
                names.append(f"{prefix}_upstream_ret_h{h}")
            names += [f"{prefix}_rain_mean_h{h}", f"{prefix}_ndvi_mean_h{h}"]
    if use_fx_summary:
        names += list(fx_summary_names or FX_SUMMARY_NAMES)
    return names


def build_training_windows(
    p: Panel, horizons: list[int], lookback: int,
    origin_max_idx: int, train_market_ids: list[int],
    use_lag52: bool, use_realised_drivers: bool,
    origin_min_idx: int | None = None,
    realised_upstream: bool = True,
    driver_source: str = "realised",
    use_macro: bool = False,
    use_fx_summary: bool = False,
):
    """Every usable (market, week) window whose LAST TARGET lands at or before
    origin_max_idx. Nothing at or after the forecast cut can enter."""
    max_h = max(horizons)
    lo = max(lookback - 1, p.season_lag) if use_lag52 else lookback - 1
    if origin_min_idx is not None:
        lo = max(lo, origin_min_idx)
    Xs, Fs, ys, meta = [], [], [], []
    for j in train_market_ids:
        for i in range(lo, origin_max_idx - max_h + 1):
            p0 = p.price[i, j]
            if not np.isfinite(p0) or p0 <= 0:
                continue
            tgt = p.price[[i + h for h in horizons], j]
            if not np.isfinite(tgt).all() or (tgt <= 0).any():
                continue
            X, ok = build_sequence(p, i, j, lookback, use_macro=use_macro)
            if not ok:
                continue
            F, ok = build_flat(p, i, j, horizons, use_lag52, use_realised_drivers,
                              realised_upstream, driver_source, use_fx_summary)
            if not ok:
                continue
            Xs.append(X)
            Fs.append(F)
            ys.append(np.log(tgt) - np.log(p0))
            meta.append((p.markets[j], j, p.dates[i], float(p0)))
    if not Xs:
        return (np.empty((0, lookback, 0)), np.empty((0, 0)),
                np.empty((0, len(horizons))), pd.DataFrame())
    md = pd.DataFrame(meta, columns=["market", "market_id", "origin", "origin_price"])
    F = np.stack(Fs) if Fs[0].size else np.zeros((len(Xs), 0))
    return np.stack(Xs), F, np.stack(ys), md


def build_grid_windows(
    p: Panel, grid: pd.DataFrame, horizons: list[int], lookback: int,
    use_lag52: bool, use_realised_drivers: bool,
    realised_upstream: bool = True,
    driver_source: str = "realised",
    require_horizons: list[int] | None = None,
    use_macro: bool = False,
    use_fx_summary: bool = False,
):
    """Windows for scored (market, origin) pairs. Origin price and actuals come
    from the baseline file so the paired comparison uses identical targets.

    `require_horizons` defaults to None, meaning "require every horizon in
    `horizons`" -- unchanged behaviour for every caller that doesn't pass it.
    Pass a subset to keep a window whose longer horizon has no actual yet
    (its target hasn't happened): `ys`/`act` carry NaN there. Nothing
    downstream trains on this function's `ys` (it's serve/score windows, not
    training windows); the caller is responsible for not scoring a NaN
    actual, since mae()/mape() do not skip NaN themselves.
    """
    need = require_horizons if require_horizons is not None else horizons
    need_idx = [horizons.index(h) for h in need]
    Xs, Fs, ys, meta, skipped = [], [], [], [], []
    for _, r in grid.iterrows():
        j = p.midx.get(r["market"])
        i = p.pos.get(pd.Timestamp(r["origin"]))
        if j is None or i is None:
            skipped.append((r["market"], r["origin"], "not_in_panel"))
            continue
        X, ok = build_sequence(p, i, j, lookback, use_macro=use_macro)
        if not ok:
            skipped.append((r["market"], r["origin"], "sequence_incomplete"))
            continue
        F, ok = build_flat(p, i, j, horizons, use_lag52, use_realised_drivers,
                          realised_upstream, driver_source, use_fx_summary)
        if not ok:
            skipped.append((r["market"], r["origin"], "flat_incomplete"))
            continue
        p0 = float(r["origin_price"])
        act = np.array([float(r[f"actual_h{h}"]) for h in horizons])
        need_vals = act[need_idx]
        if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(need_vals).all() or (need_vals <= 0).any():
            skipped.append((r["market"], r["origin"], "bad_actual_or_origin_price"))
            continue
        Xs.append(X)
        Fs.append(F)
        with np.errstate(invalid="ignore", divide="ignore"):
            ys.append(np.log(act) - np.log(p0))
        row = dict(market=r["market"], market_id=j, origin=pd.Timestamp(r["origin"]),
                   origin_price=p0)
        for h in horizons:
            row[f"actual_h{h}"] = float(r[f"actual_h{h}"])
            row[f"panel_fe_h{h}"] = float(r[f"panel_fe_h{h}"])
            row[f"naive_h{h}"] = float(r[f"naive_h{h}"])
        meta.append(row)
    md = pd.DataFrame(meta)
    sk = pd.DataFrame(skipped, columns=["market", "origin", "reason"])
    if not Xs:
        return (np.empty((0, lookback, 0)), np.empty((0, 0)),
                np.empty((0, len(horizons))), md, sk)
    F = np.stack(Fs) if Fs[0].size else np.zeros((len(Xs), 0))
    return np.stack(Xs), F, np.stack(ys), md, sk


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------
class Scaler:
    """Fit on training rows only. Sequence channels are pooled over time."""

    def __init__(self) -> None:
        self.seq_mu = self.seq_sd = self.flat_mu = self.flat_sd = None
        self.y_mu = self.y_sd = None

    def fit(self, X: np.ndarray, F: np.ndarray, y: np.ndarray | None = None) -> "Scaler":
        self.seq_mu = X.reshape(-1, X.shape[-1]).mean(0)
        self.seq_sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-8
        if F.shape[1]:
            # A flat feature that is constant across the training rows (e.g. "months
            # since the last devaluation" before 2016) would otherwise be divided by
            # ~1e-8 and explode the first time it moves. Leave such a feature unscaled
            # (sd 1). Never triggers for the flagship's features, which all vary.
            sd = F.std(0)
            self.flat_mu, self.flat_sd = F.mean(0), np.where(sd < 1e-6, 1.0, sd) + 1e-8
        if y is not None:
            self.y_mu, self.y_sd = y.mean(0), y.std(0) + 1e-8
        return self

    def transform(self, X: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xs = (X - self.seq_mu) / self.seq_sd
        Fs = (F - self.flat_mu) / self.flat_sd if F.shape[1] else F
        return Xs, Fs

    def y_forward(self, y: np.ndarray) -> np.ndarray:
        return y if self.y_mu is None else (y - self.y_mu) / self.y_sd

    def y_inverse(self, y: np.ndarray) -> np.ndarray:
        return y if self.y_mu is None else y * self.y_sd + self.y_mu
