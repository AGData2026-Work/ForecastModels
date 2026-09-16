"""Shared fixtures. Everything here is synthetic and small on purpose: these
tests check logic (shapes, no-look-ahead, invariants), not model quality, so
they should run in well under a second and never touch the real data files."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import Panel  # noqa: E402


@pytest.fixture
def tiny_panel() -> Panel:
    """3 markets, 120 weeks, a clean Wednesday grid. Market 2 has no upstream
    (mirrors Dandume's role in the real panel: the root of the hierarchy)."""
    n_weeks = 120
    markets = ["MarketA", "MarketB", "MarketC"]
    dates = pd.date_range("2020-01-01", periods=n_weeks, freq="7D")
    pos = {t: i for i, t in enumerate(dates)}
    midx = {m: j for j, m in enumerate(markets)}

    rng = np.random.default_rng(0)
    t = np.arange(n_weeks)
    # a mild trend plus noise, kept strictly positive, so log() is always valid
    price = np.stack([
        100 + 0.2 * t + rng.normal(0, 3, n_weeks),
        150 + 0.1 * t + rng.normal(0, 4, n_weeks),
        80 + 0.05 * t + rng.normal(0, 2, n_weeks),
    ], axis=1)
    price = np.clip(price, 10, None)
    price_filled = np.zeros_like(price, dtype=bool)

    diesel = np.full_like(price, 500.0) + rng.normal(0, 5, price.shape)
    upstream = np.column_stack([price[:, 0] * 0.9, price[:, 1] * 0.9, np.zeros(n_weeks)])
    has_upstream = np.array([True, True, False])

    rainfall = rng.uniform(0, 50, price.shape)
    ndvi = rng.uniform(0.1, 0.6, price.shape)

    fourier_k = 2
    fcols = []
    for k in range(1, fourier_k + 1):
        fcols.append(np.sin(2 * np.pi * k * t / 52.18))
        fcols.append(np.cos(2 * np.pi * k * t / 52.18))
    fourier = np.column_stack(fcols)

    return Panel(dates, markets, pos, midx, price, price_filled, diesel, upstream,
                 rainfall, ndvi, fourier, has_upstream, log={})
