"""
Loader for the AFEX multi-commodity weekly farmgate panel: 51 series across
18 markets and 7 commodities, price only, no incumbent forecast file.

Structurally different from the FEWSNET/NADIH panel the rest of this repo is
built around, in three ways that matter:

1.  No diesel data exists for this panel; that sequence channel stays
    zero-filled. Rainfall and NDVI CAN be populated (see `ndvi_path`/
    `rainfall_path` below), and the upstream-market channel can carry a
    sibling commodity (see `sibling_commodity` below). The source file's
    own README says to start price-only and add drivers later, so the
    remaining gaps are by design, not an oversight.

    `ndvi_path`/`rainfall_path` (both required together, with
    `climate_state_map_path`): UN Data Exchange dekadal (10-day) state-level
    series. Both source files identify locations only by codes NG001-NG037,
    with no state-name legend in either workbook -- flagged by whoever
    cleaned them as a "GEOGRAPHY BLOCKER" rather than guessed. Resolved via
    a THIRD file, `climate_state_map_path` (a version of the NDVI source
    that does carry a State column), which gives an unambiguous
    alphabetical NG001=Abia .. NG037=Zamfara mapping (verified: one state
    per code, 37 distinct pairs) and is applied to the rainfall file too,
    since it uses the identical code scheme. See DECISIONS D-33.
    Resampled from dekadal to this panel's weekly grid by linear
    interpolation, then forward-filled past the source's last real date
    (both sources run out a few months before this panel does) -- a real
    coverage gap, logged in `panel.log["agroclimatic"]`, not hidden.

    IMPORTANT: build_sequence has no NaN-safety on the raw rainfall/NDVI
    channels (unlike diesel, which degrades to zero via a safe log-ratio) --
    any NaN in these arrays would silently reject the whole window. These
    arrays are therefore always zero-initialised and only ever overwritten
    with real, finite values; never left as NaN.

    `sibling_commodity` (e.g. "Sorghum") repurposes the upstream channel:
    where a maize market also has that commodity's own series, the
    channel carries that series' price history over the same 52-week
    lookback as maize's own price -- never the forecast window, so this
    is not foreknowledge, exactly as available at deployment time as
    maize's own price history is. Chosen by correlation screening
    (DECISIONS D-31): sorghum co-moves with maize far more than any other
    commodity in this panel (0.86 at 13-week changes, pooled, vs 0.45 for
    soybean, the next best) and pairs with 7 of the 15 scored maize markets
    (Dandume, dropped from scoring above, would have made 8 of 16 before
    that drop). Superseded in later builds by `upstream_lag_map` below;
    the two are mutually exclusive (both populate the same channel) and
    passing both raises.

    `upstream_lag_map` (dict: follower market -> (leader market, lag in
    weeks)) is the alternative, same-commodity use of the upstream
    channel: a genuinely lagged neighbouring MAIZE market's price, sized
    empirically per market by the cross-correlation screen in DECISIONS
    D-36, not assumed. The shift moves the leader's price forward in time
    so the channel holds what the leader was doing `lag` weeks before the
    current origin -- real, already-observed information, never the
    forecast window. Positions before the shift is valid, or built from a
    leader with no price that week, are left at 0, which `build_sequence`'s
    existing `up_ok` check (requires > 0) already treats as "no data here."

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


def _dekadal_state_series(source_path: str | Path, sheet: str, value_col: str,
                          code_to_state: dict[str, str],
                          target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """UN Data Exchange dekadal (10-day) series, Location Code x Date, mapped
    to state names and resampled onto target_dates (linear interpolation
    between dekadal observations, then forward-filled past the source's
    last date to cover the tail gap -- the source runs out a few months
    before the AFEX panel does; this is a real coverage gap, logged in the
    caller, not hidden by the interpolation)."""
    raw = pd.read_excel(source_path, sheet_name=sheet)
    raw = raw.copy()
    raw["state"] = raw["Location Code"].map(code_to_state)
    raw = raw.dropna(subset=["state"])
    wide = raw.pivot_table(index="Date", columns="state", values=value_col, aggfunc="first")
    full_idx = pd.date_range(wide.index.min(),
                             max(wide.index.max(), target_dates.max()), freq="D")
    wide = wide.reindex(full_idx)
    last_real_date = wide.dropna(how="all").index.max()
    wide = wide.interpolate(method="linear", limit_area="inside").ffill()
    out = wide.reindex(target_dates)
    out.attrs["last_real_date"] = str(last_real_date.date())
    return out


def _weekly_state_diesel_series(source_path: str | Path,
                                target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """State-level diesel from the main FEWSNET/NADIH panel
    (data/panel_weekly.parquet). Already weekly on the same Wednesday grid
    as the AFEX panel (verified: both start on a Wednesday), so this only
    needs a reindex, not the dekadal-to-daily interpolation the rainfall/
    NDVI sources need. Forward-filled past the source's last real date
    (2024-09-18) to cover the ~22-month tail gap to the AFEX panel's end --
    a real coverage gap, logged by the caller, not hidden."""
    raw = pd.read_parquet(source_path, columns=["date", "state", "diesel"])
    wide = raw.drop_duplicates(["date", "state"]).pivot(index="date", columns="state", values="diesel")
    full_idx = pd.date_range(wide.index.min(), max(wide.index.max(), target_dates.max()), freq="7D")
    wide = wide.reindex(full_idx)
    last_real_date = wide.dropna(how="all").index.max()
    wide = wide.ffill()
    out = wide.reindex(target_dates)
    out.attrs["last_real_date"] = str(last_real_date.date())
    return out


def load_afex_panel(path: str | Path, drop_zero_window_series: bool = True,
                    sibling_commodity: str | None = None,
                    ndvi_path: str | Path | None = None,
                    rainfall_path: str | Path | None = None,
                    climate_state_map_path: str | Path | None = None,
                    upstream_lag_map: dict[str, tuple[str, int]] | None = None,
                    diesel_source_path: str | Path | None = None) -> Panel:
    df = pd.read_excel(path, sheet_name="Panel_Long", parse_dates=["date"])
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    step = pd.Series(np.diff(dates).astype("timedelta64[D]").astype(int))
    if not (step == 7).all():
        raise ValueError(f"AFEX panel date index is not a clean weekly grid: {step.unique()}")

    df = df.copy()
    market_state = df.groupby("market")["state"].first().to_dict()
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

    rainfall_arr = zeros.copy()
    ndvi_arr = zeros.copy()
    climate_log = {"requested": bool(ndvi_path or rainfall_path),
                  "states_covered": [], "states_missing": [], "last_real_date": {}}
    if ndvi_path or rainfall_path:
        if not (ndvi_path and rainfall_path and climate_state_map_path):
            raise ValueError("ndvi_path, rainfall_path and climate_state_map_path must all be "
                             "given together: rainfall has no state labels of its own and "
                             "needs the NDVI-with-states file's Location Code -> State mapping "
                             "(see docs/DECISIONS.md on the UN Data Exchange geography blocker)")
        code_map_df = pd.read_excel(climate_state_map_path, sheet_name="Sheet1")
        code_to_state = (code_map_df[["Location Code", "State"]].drop_duplicates()
                        .set_index("Location Code")["State"].to_dict())

        ndvi_wide = _dekadal_state_series(ndvi_path, "Cleaned Data", "NDVI", code_to_state, dates)
        rain_wide = _dekadal_state_series(rainfall_path, "Cleaned Data", "Rainfall (mm)",
                                          code_to_state, dates)
        climate_log["last_real_date"] = {"ndvi": ndvi_wide.attrs.get("last_real_date"),
                                         "rainfall": rain_wide.attrs.get("last_real_date")}

        needed_states = sorted(set(market_state.get(m.split(" | ")[0]) for m in series))
        for st in needed_states:
            if st in ndvi_wide.columns and st in rain_wide.columns:
                climate_log["states_covered"].append(st)
            else:
                climate_log["states_missing"].append(st)

        for j, m in enumerate(series):
            st = market_state.get(m.split(" | ")[0])
            if st in ndvi_wide.columns:
                ndvi_arr[:, j] = ndvi_wide[st].to_numpy(dtype=float)
            if st in rain_wide.columns:
                rainfall_arr[:, j] = rain_wide[st].to_numpy(dtype=float)

    diesel_arr = zeros.copy()
    diesel_log = {"requested": bool(diesel_source_path), "states_covered": [],
                 "states_missing": [], "last_real_date": None}
    if diesel_source_path:
        diesel_wide = _weekly_state_diesel_series(diesel_source_path, dates)
        diesel_log["last_real_date"] = diesel_wide.attrs.get("last_real_date")
        needed_states = sorted(set(market_state.get(m.split(" | ")[0]) for m in series))
        for st in needed_states:
            if st in diesel_wide.columns:
                diesel_log["states_covered"].append(st)
            else:
                diesel_log["states_missing"].append(st)
        for j, m in enumerate(series):
            st = market_state.get(m.split(" | ")[0])
            if st in diesel_wide.columns:
                diesel_arr[:, j] = np.nan_to_num(diesel_wide[st].to_numpy(dtype=float), nan=0.0)

    has_upstream = np.zeros(len(series), dtype=bool)
    upstream = zeros.copy()
    sibling_log = {"requested": sibling_commodity, "maize_markets_paired": [],
                   "maize_markets_unpaired": []}
    if sibling_commodity:
        # Repurposes the "upstream" channel exactly as the main pipeline uses
        # it: a related price series the model may find informative. Here
        # the relation is a sibling commodity at the SAME market, not a
        # different market for the same commodity. Populated for MAIZE
        # series only (the only series that get scored); channel is the
        # sibling's own price history, over the same 52-week lookback,
        # never the forecast window, so this carries no foreknowledge --
        # it is exactly as available at deployment time as maize's own
        # price history is.
        for m in sorted(set(s.split(" | ")[0] for s in series if s.endswith("| Maize"))):
            maize_col = f"{m} | Maize"
            sib_col = f"{m} | {sibling_commodity}"
            j_maize = midx.get(maize_col)
            j_sib = midx.get(sib_col)
            if j_maize is None:
                continue
            if j_sib is not None:
                upstream[:, j_maize] = price[:, j_sib]
                has_upstream[j_maize] = True
                sibling_log["maize_markets_paired"].append(m)
            else:
                sibling_log["maize_markets_unpaired"].append(m)

    upstream_lag_log = {"requested": bool(upstream_lag_map), "assigned": [], "skipped": []}
    if upstream_lag_map:
        if sibling_commodity:
            raise ValueError("sibling_commodity and upstream_lag_map both populate the "
                             "upstream channel; pass at most one")
        # Same-commodity, lagged neighbouring-market price, sized empirically
        # in DECISIONS D-36 (cross-correlation of weekly log-returns at lags
        # -13..+13 weeks). Populated for MAIZE series only. The shift moves
        # the leader's price forward in time so that at origin index i the
        # channel holds the leader's price from `lag` weeks earlier -- real,
        # already-observed information, never the forecast window. Positions
        # before the shift is valid, or where the leader itself has no
        # price, are left at 0, which build_sequence's up_ok check (requires
        # > 0) already treats as "no data here" and falls back to zero for
        # that window -- the same convention diesel/rainfall-absent markets
        # already use elsewhere in this file.
        for follower, (leader, lag) in upstream_lag_map.items():
            j_follow = midx.get(f"{follower} | Maize")
            j_lead = midx.get(f"{leader} | Maize")
            if j_follow is None or j_lead is None or lag < 0:
                upstream_lag_log["skipped"].append(follower)
                continue
            leader_price = price[:, j_lead]
            shifted = np.zeros(len(dates))
            if lag == 0:
                shifted[:] = np.nan_to_num(leader_price, nan=0.0)
            else:
                shifted[lag:] = np.nan_to_num(leader_price[:-lag], nan=0.0)
            upstream[:, j_follow] = shifted
            has_upstream[j_follow] = True
            upstream_lag_log["assigned"].append(dict(follower=follower, leader=leader, lag_weeks=lag))

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
        "sibling_commodity": sibling_log,
        "upstream_lag": upstream_lag_log,
        "agroclimatic": climate_log,
        "diesel": diesel_log,
        "driver_channels": (
            (f"upstream channel = {sibling_commodity} price at the same market, "
             f"{len(sibling_log['maize_markets_paired'])} of "
             f"{len(sibling_log['maize_markets_paired']) + len(sibling_log['maize_markets_unpaired'])} "
             "maize markets paired; " if sibling_commodity else
             f"upstream channel = lagged neighbouring-market maize price, "
             f"{len(upstream_lag_log['assigned'])} markets assigned "
             f"(D-36 lead-lag screen); "
             if upstream_lag_map else "upstream channel unused; ")
            + (f"rainfall/NDVI channels = state-level UN Data Exchange series (states covered: "
               f"{climate_log['states_covered']}, missing: {climate_log['states_missing']}, "
               f"real data through {climate_log['last_real_date']}, forward-filled after that); "
               if climate_log["requested"] else "rainfall/NDVI channels zero-filled; ")
            + (f"diesel channel = state-level FEWSNET panel series (states covered: "
               f"{diesel_log['states_covered']}, missing: {diesel_log['states_missing']}, "
               f"real data through {diesel_log['last_real_date']}, forward-filled after that)"
               if diesel_log["requested"] else "diesel still zero-filled (not requested)")
        ),
    }
    return Panel(dates, series, pos, midx, price, price_filled,
                 diesel_arr, upstream, rainfall_arr, ndvi_arr,
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
