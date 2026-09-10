"""
Loader for the AFEX multi-commodity weekly farmgate panel: 51 series across
18 markets and 7 commodities, price only, no incumbent forecast file.

Structurally different from the FEWSNET/NADIH panel the rest of this repo is
built around, in three ways that matter:

1.  No diesel, rainfall, NDVI or upstream-market data exists for this panel.
    Those four sequence channels are zero-filled so build_sequence,
    build_flat and build_training_windows from data.py can be reused
    unchanged; the model sees them as constant, uninformative channels
    rather than missing ones. The source file's own README says to start
    price-only and add drivers later, so this is by design, not a gap.

2.  No incumbent forecast file exists, so there is nothing to read an
    evaluation grid from the way load_grid reads 07_panel_fe_forecasts.parquet
    for the main panel. generate_afex_grid below builds origins directly
    from this panel. This is not the D-01 mistake (regenerating a grid that
    an incumbent already defines) -- there is no incumbent grid here to
    diverge from.

3.  Multi-commodity: every (market, commodity) pair -- e.g. "Anchau | Maize",
    "Anchau | Sorghum" -- is one series, treated as its own embedding unit
    exactly as data.Panel already supports for markets. All 51 series train
    the pooled model; only the 16 maize series are scored. This mirrors
    DECISIONS D-03's precedent (Aba trains but is never scored).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data import Panel, build_flat, build_sequence

FOURIER_PERIOD = 52.18
FOURIER_K = 2


def load_afex_panel(path: str | Path, drop_zero_window_series: bool = True) -> Panel:
    df = pd.read_excel(path, sheet_name="Panel_Long", parse_dates=["date"])
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    step = pd.Series(np.diff(dates).astype("timedelta64[D]").astype(int))
    if not (step == 7).all():
        raise ValueError(f"AFEX panel date index is not a clean weekly grid: {step.unique()}")

    df = df.copy()
    df["series"] = df["market"] + " | " + df["commodity"]
    series = sorted(df["series"].unique())

    drop_log = []
    if drop_zero_window_series and "Dandume | Maize" in series:
        # Clears the source panel's own 112-week entry bar but has zero
        # usable 78-week (lookback 52 + max horizon 26) windows after
        # new-crop removal -- flagged in the source file's Build_Decisions
        # sheet as "should be dropped or reinstated by a narrower rule".
        # Dropped: a series that can never produce a full-horizon window
        # contributes nothing to training and cannot be scored at h=26.
        series.remove("Dandume | Maize")
        drop_log.append("Dandume | Maize: 0 usable 78-week windows (source Series_Catalogue), "
                        "cannot produce an h=26 window; dropped from training and scoring")

    pos = {t: i for i, t in enumerate(dates)}
    midx = {m: j for j, m in enumerate(series)}

    piv = df[df["series"].isin(series)].pivot_table(
        index="date", columns="series", values="price_NGN_per_kg", aggfunc="first")
    price = piv.reindex(index=dates, columns=series).to_numpy(dtype=float)

    proxy_piv = df[df["series"].isin(series)].pivot_table(
        index="date", columns="series", values="is_proxy_price", aggfunc="first")
    price_filled = (proxy_piv.reindex(index=dates, columns=series)
                    .fillna(False).to_numpy(dtype=bool))

    zeros = np.zeros_like(price)
    t = np.arange(len(dates), dtype=float)
    fourier_cols = []
    for k in range(1, FOURIER_K + 1):
        fourier_cols.append(np.sin(2 * np.pi * k * t / FOURIER_PERIOD))
        fourier_cols.append(np.cos(2 * np.pi * k * t / FOURIER_PERIOD))
    fourier = np.column_stack(fourier_cols)

    has_upstream = np.zeros(len(series), dtype=bool)

    log = {
        "panel_path": str(path),
        "n_weeks": len(dates),
        "n_series": len(series),
        "n_maize_series": sum(1 for s in series if s.endswith("| Maize")),
        "commodities": sorted(set(s.split(" | ")[1] for s in series)),
        "first_week": str(dates[0].date()),
        "last_week": str(dates[-1].date()),
        "price_cells_observed": int(np.isfinite(price).sum()),
        "price_cells_total": int(price.size),
        "series_dropped": drop_log,
        "driver_channels": "none: diesel, upstream, rainfall, NDVI all zero-filled; no source data for this panel",
    }
    return Panel(dates, series, pos, midx, price, price_filled,
                 zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(),
                 fourier, has_upstream, log)


def maize_series_ids(panel: Panel) -> list[int]:
    return [j for j, m in enumerate(panel.markets) if m.endswith("| Maize")]


def generate_afex_grid(panel: Panel, horizons: list[int], lookback: int) -> pd.DataFrame:
    """Every weekly origin, maize series only, where the lookback window and
    every horizon's target are fully observed. No incumbent file exists for
    this panel, so this is generated rather than read (see module docstring
    point 2)."""
    maize_ids = maize_series_ids(panel)
    max_h = max(horizons)
    rows = []
    for j in maize_ids:
        for i in range(lookback - 1, len(panel.dates) - max_h):
            p0 = panel.price[i, j]
            if not np.isfinite(p0) or p0 <= 0:
                continue
            tgt = panel.price[[i + h for h in horizons], j]
            if not np.isfinite(tgt).all() or (tgt <= 0).any():
                continue
            row = dict(market=panel.markets[j], origin=panel.dates[i], origin_price=float(p0))
            for h, v in zip(horizons, tgt):
                row[f"actual_h{h}"] = float(v)
            rows.append(row)
    return pd.DataFrame(rows)


def build_afex_scored_windows(panel: Panel, grid: pd.DataFrame, horizons: list[int],
                              lookback: int, use_lag52: bool):
    """Windows for the self-generated grid. Mirrors data.build_grid_windows
    but without panel_fe columns (no incumbent for this dataset); carries a
    carry-forward "naive" column as the do-nothing benchmark, computed
    directly from the origin price."""
    Xs, Fs, ys, meta, skipped = [], [], [], [], []
    for _, r in grid.iterrows():
        j = panel.midx.get(r["market"])
        i = panel.pos.get(pd.Timestamp(r["origin"]))
        if j is None or i is None:
            skipped.append((r["market"], r["origin"], "not_in_panel"))
            continue
        X, ok = build_sequence(panel, i, j, lookback)
        if not ok:
            skipped.append((r["market"], r["origin"], "sequence_incomplete"))
            continue
        F, ok = build_flat(panel, i, j, horizons, use_lag52, False)
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
        row = dict(market=r["market"], market_id=j, origin=pd.Timestamp(r["origin"]), origin_price=p0)
        for h in horizons:
            row[f"actual_h{h}"] = float(r[f"actual_h{h}"])
            row[f"naive_h{h}"] = p0
        meta.append(row)
    md = pd.DataFrame(meta)
    sk = pd.DataFrame(skipped, columns=["market", "origin", "reason"])
    if not Xs:
        return (np.empty((0, lookback, 0)), np.empty((0, 0)), np.empty((0, len(horizons))), md, sk)
    F = np.stack(Fs) if Fs[0].size else np.zeros((len(Xs), 0))
    return np.stack(Xs), F, np.stack(ys), md, sk
