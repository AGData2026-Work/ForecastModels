"""
Walk-forward scheduling.

The evaluation grid comes from the baseline file, so this module does NOT invent
origins. Its job is to decide, given that grid, when the network is refit and
which origins each fit serves.

Two separate cadences:

  origin cadence   -- fixed by the baseline: each market has its own 4-weekly
                      sequence, staggered against the other markets, so the
                      union of origins across markets looks weekly.
  retrain cadence  -- ours. A network refit at every origin is not affordable and
                      does not reflect deployment. Between refits the model runs
                      on a fit that is up to `retrain_every` weeks stale, which
                      is a HARDER test than refitting each time, not an easier
                      one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def retrain_cuts(origins: pd.Series, every_weeks: int) -> list[pd.Timestamp]:
    """Refit dates spanning the origin range, spaced `every_weeks` apart."""
    o = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(origins))))
    cuts, cur = [], o[0]
    while cur <= o[-1]:
        cuts.append(cur)
        cur = cur + pd.Timedelta(weeks=every_weeks)
    return cuts


def assign_to_cuts(grid: pd.DataFrame, cuts: list[pd.Timestamp]) -> pd.Series:
    """Map each grid row to the most recent cut at or before its origin."""
    cut_idx = pd.DatetimeIndex(cuts)
    o = pd.DatetimeIndex(grid["origin"])
    loc = cut_idx.searchsorted(o, side="right") - 1
    loc = np.clip(loc, 0, len(cut_idx) - 1)
    return pd.Series(cut_idx[loc], index=grid.index, name="cut")


def chronological_split(
    meta: pd.DataFrame, val_frac: float, purge_weeks: int, min_train: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split training windows by origin date, oldest to newest.

    `purge_weeks` removes the training windows immediately before the validation
    block. With a 52-week lookback and a 26-week maximum horizon, the last
    training windows and the first validation windows share up to 51 weeks of
    input and the training targets reach into the validation input period. Early
    stopping decisions made on an unpurged split are therefore partly
    self-referential. Purging costs training rows and buys an honest stopping
    signal.
    """
    order = np.argsort(meta["origin"].values, kind="stable")
    n = len(order)
    k = int(n * (1 - val_frac))
    tr_idx, va_idx = order[:k], order[k:]
    log = {"purge_weeks_requested": purge_weeks, "purge_weeks_applied": 0,
           "n_train_pre_purge": len(tr_idx), "n_val": len(va_idx)}
    if purge_weeks > 0 and len(va_idx):
        first_val = pd.Timestamp(meta["origin"].values[va_idx[0]])
        for pw in (purge_weeks, purge_weeks // 2, 0):
            cutoff = first_val - pd.Timedelta(weeks=pw)
            keep = tr_idx[pd.DatetimeIndex(meta["origin"].values[tr_idx]) < cutoff]
            if len(keep) >= min_train or pw == 0:
                tr_idx = keep
                log["purge_weeks_applied"] = pw
                break
    log["n_train"] = len(tr_idx)
    return tr_idx, va_idx, log


def recency_weights(origins: np.ndarray, half_life_weeks: int | None) -> np.ndarray:
    """Exponential recency weighting. None or 0 gives uniform weights.

    Rationale: the 2023-24 currency and fuel-subsidy regime differs enough from
    2015-19 that equal weighting asks one parameter set to serve two regimes.
    A long half-life tilts the fit toward the recent regime without discarding
    the older data, which is still the only evidence about seasonal shape.
    """
    if not half_life_weeks:
        return np.ones(len(origins))
    o = pd.DatetimeIndex(origins)
    age_weeks = (o.max() - o).days / 7.0
    return np.power(0.5, age_weeks / half_life_weeks)
