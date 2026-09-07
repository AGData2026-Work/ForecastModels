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
) -> Panel:
    df = pd.read_parquet(path)
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    markets = sorted(df["market"].unique())
    pos = {t: i for i, t in enumerate(dates)}
    midx = {m: j for j, m in enumerate(markets)}

    step = pd.Series(np.diff(dates).astype("timedelta64[D]").astype(int))
    if not (step == 7).all():
        raise ValueError(f"panel date index is not a clean weekly grid: {step.unique()}")

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
    return Panel(dates, markets, pos, midx, price, filled, diesel, upstream,
                 rain, nd, fou, has_up, log)


# ---------------------------------------------------------------------------
# evaluation grid
# ---------------------------------------------------------------------------
def load_grid(
    path: str | Path,
    horizons: list[int],
    drop_targets: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Baseline forecasts, wide by horizon, one row per (market, origin).

    Returns the grid plus a log of what was dropped and why.
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

    complete = piv[[f"actual_h{h}" for h in horizons]].notna().all(axis=1)
    incomplete = piv[~complete].copy()
    piv = piv[complete].reset_index(drop=True)

    # naive is identical across horizons for a given origin: that is the origin price
    piv["origin_price"] = piv[f"naive_h{horizons[0]}"]

    log = {
        "grid_path": str(path),
        "horizons": horizons,
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
    p: Panel, i: int, j: int, lookback: int
) -> tuple[np.ndarray, bool]:
    """(lookback, C) channel block for origin index i, market j. Window ends AT i."""
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

    X = np.column_stack(chan)
    if not np.isfinite(X).all():
        return np.empty((0, 0)), False
    return X, True


def build_flat(
    p: Panel, i: int, j: int, horizons: list[int],
    use_lag52: bool, use_realised_drivers: bool,
    realised_upstream: bool = True,
    driver_source: str = "realised",
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
        # observed at the origin -- no look-ahead.
        for h in horizons:
            a = i + h - 52
            if a < 0 or not np.isfinite(p.price[a, j]) or p.price[a, j] <= 0:
                return np.empty(0), False
            vals.append(float(np.log(p.price[a, j]) - np.log(p0)))
        # What the same window did last year: log(P[t+h-52]) - log(P[t-52]).
        for h in horizons:
            a, b = i + h - 52, i - 52
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

    arr = np.asarray(vals, dtype=float)
    if arr.size and not np.isfinite(arr).all():
        return np.empty(0), False
    return arr, True


def flat_feature_names(horizons: list[int], use_lag52: bool, use_realised: bool,
                       realised_upstream: bool = True,
                       driver_source: str = "realised") -> list[str]:
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
    return names


def build_training_windows(
    p: Panel, horizons: list[int], lookback: int,
    origin_max_idx: int, train_market_ids: list[int],
    use_lag52: bool, use_realised_drivers: bool,
    origin_min_idx: int | None = None,
    realised_upstream: bool = True,
    driver_source: str = "realised",
):
    """Every usable (market, week) window whose LAST TARGET lands at or before
    origin_max_idx. Nothing at or after the forecast cut can enter."""
    max_h = max(horizons)
    lo = max(lookback - 1, 52) if use_lag52 else lookback - 1
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
            X, ok = build_sequence(p, i, j, lookback)
            if not ok:
                continue
            F, ok = build_flat(p, i, j, horizons, use_lag52, use_realised_drivers,
                              realised_upstream, driver_source)
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
):
    """Windows for scored (market, origin) pairs. Origin price and actuals come
    from the baseline file so the paired comparison uses identical targets."""
    Xs, Fs, ys, meta, skipped = [], [], [], [], []
    for _, r in grid.iterrows():
        j = p.midx.get(r["market"])
        i = p.pos.get(pd.Timestamp(r["origin"]))
        if j is None or i is None:
            skipped.append((r["market"], r["origin"], "not_in_panel"))
            continue
        X, ok = build_sequence(p, i, j, lookback)
        if not ok:
            skipped.append((r["market"], r["origin"], "sequence_incomplete"))
            continue
        F, ok = build_flat(p, i, j, horizons, use_lag52, use_realised_drivers,
                          realised_upstream, driver_source)
        if not ok:
            skipped.append((r["market"], r["origin"], "flat_incomplete"))
            continue
        p0 = float(r["origin_price"])
        act = np.array([float(r[f"actual_h{h}"]) for h in horizons])
        if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(act).all() or (act <= 0).any():
            skipped.append((r["market"], r["origin"], "bad_actual_or_origin_price"))
            continue
        Xs.append(X)
        Fs.append(F)
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
            self.flat_mu, self.flat_sd = F.mean(0), F.std(0) + 1e-8
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
