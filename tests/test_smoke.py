"""End-to-end integration test: does the whole FEWSNET pipeline (config ->
panel/grid loading -> window building -> training -> metrics -> files on
disk) actually wire together, on tiny synthetic data. This is not a model
quality check (hyperparameters here are shrunk purely for speed: a 10-week
lookback, 4 markets, 60 weeks of history) -- it exists to catch "the whole
pipeline is broken" the way the existing --smoke flag is meant to, but
automatically instead of by a human remembering to run it first."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_tiny_panel(n_weeks=60, markets=("MarketA", "MarketB", "MarketC", "MarketD")):
    dates = pd.date_range("2020-01-01", periods=n_weeks, freq="7D")
    rng = np.random.default_rng(1)
    rows = []
    for m_idx, m in enumerate(markets):
        base = 100 + m_idx * 20
        price = base + np.cumsum(rng.normal(0, 1.5, n_weeks))
        price = np.clip(price, 10, None)
        for i, d in enumerate(dates):
            rows.append(dict(
                date=d, market=m, price=float(price[i]),
                rainfall_inverse_distance=float(rng.uniform(0, 50)),
                ndvi_inverse_distance=float(rng.uniform(0.1, 0.6)),
                diesel=float(500 + rng.normal(0, 5)),
                upstream_price=float(price[i] * 0.9),
                upstream_market="MarketA" if m != "MarketA" else None,
                sin1=float(np.sin(2 * np.pi * i / 52.18)),
                cos1=float(np.cos(2 * np.pi * i / 52.18)),
            ))
    return pd.DataFrame(rows), dates, markets


def _build_tiny_grid(dates, markets, horizons=(2, 3, 4)):
    """Origins spaced so every one has full lookback (>=10 weeks in) and
    every horizon's target lands inside the panel (<= last week)."""
    # actual/panel_fe/naive below are self-consistent synthetic values, not
    # tied to _build_tiny_panel's own price series -- this test checks
    # plumbing, not numerical agreement between the two fixtures.
    max_h = max(horizons)
    origin_idx = list(range(15, len(dates) - max_h, 5))
    rows = []
    rng = np.random.default_rng(2)
    for m in markets:
        for oi in origin_idx:
            origin = dates[oi]
            origin_price = float(100 + rng.normal(0, 5))
            for h in horizons:
                rows.append(dict(
                    market=m, origin=origin, horizon=h,
                    actual=origin_price + float(rng.normal(0, 3)),
                    panel_fe=origin_price + float(rng.normal(0, 3)),
                    naive=origin_price,
                ))
    return pd.DataFrame(rows)


def test_full_pipeline_runs_end_to_end(tmp_path):
    panel_df, dates, markets = _build_tiny_panel()
    grid_df = _build_tiny_grid(dates, markets)

    panel_path = tmp_path / "panel.parquet"
    grid_path = tmp_path / "grid.parquet"
    panel_df.to_parquet(panel_path)
    grid_df.to_parquet(grid_path)

    cfg = {
        "build": {"name": "smoke_test_build", "note": "integration test fixture"},
        "data": {"panel": str(panel_path), "baseline_forecasts": str(grid_path),
                 "train_markets": "all_in_panel", "drop_target_dates": [],
                 "price_interp_limit_weeks": 13},
        "protocol": {"horizons": [2, 3, 4], "gate_horizons": [2, 3], "gate_threshold_pct": 5.0},
        "features": {"climate_scheme": "inverse_distance", "fourier_k": 1, "explicit_lag52": False},
        "model": {"lookback": 10, "hidden": 8, "market_embedding_dim": 4,
                 "head_dropout": 0.0, "input_dropout": 0.0},
        "training": {"seeds": [0, 1], "epochs": 4, "batch_size": 32, "lr": 0.003,
                    "patience": 5, "grad_clip": 1.0, "weight_decay": 0.0,
                    "lr_schedule": "none", "huber_delta": 1.0, "val_fraction": 0.2,
                    "purge_weeks": 2, "recency_half_life_weeks": None,
                    "scale_targets": True, "retrain_every_weeks": 10,
                    "min_train_windows": 5, "seed_aggregation": "median"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "run.py"),
         "--config", str(config_path), "--kind", "GRU", "--convention", "unconditional",
         "--smoke", "--out", str(out_dir), "--device", "cpu"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    assert (out_dir / "run_metadata.json").exists()
    assert (out_dir / "paired_metrics.csv").exists()
    assert (out_dir / "predictions_paired.csv").exists()

    import json
    meta = json.loads((out_dir / "run_metadata.json").read_text())
    assert meta["smoke"] is True

    paired = pd.read_csv(out_dir / "paired_metrics.csv")
    assert set(paired["h"]) == {2, 3, 4}
    assert np.isfinite(paired["challenger_MAE"]).all()
