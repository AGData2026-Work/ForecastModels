"""
Proxy sensitivity: rescore every v2 run excluding rows whose TARGET actual was
proxy-filled in the panel rather than observed, and report the shift in every
headline figure against the full-grid figure.

    python src/proxy_sensitivity.py

Reuses error_analysis.py's run-loading and panel-joining (is_proxy_target is
already computed there) so the two scripts cannot disagree on what counts as
proxy-filled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from error_analysis import aggregate, attach_panel, load_runs  # noqa: E402

SHIFT_COLS = [
    "challenger_MAE", "challenger_MAPE", "challenger_bias",
    "challenger_direction_pct", "panel_fe_MAE", "naive_MAE",
    "vs_panel_fe_MAE_pct", "vs_naive_MAE_pct",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--panel", default="data/panel_weekly.parquet")
    ap.add_argument("--out", default="outputs/proxy_sensitivity.csv")
    a = ap.parse_args()

    raw, _ = load_runs(Path(a.outdir))
    df, _ = attach_panel(raw, a.panel)

    full = aggregate(df)
    clean = df[~df["is_proxy_target"].fillna(False)]
    excl = aggregate(clean)

    n_dropped = len(df) - len(clean)
    key = ["build", "model", "convention", "h_weeks"]
    m = full.merge(excl, on=key, suffixes=("_full", "_excl_proxy"))
    for col in SHIFT_COLS:
        m[f"{col}_shift"] = m[f"{col}_excl_proxy"] - m[f"{col}_full"]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(a.out, index=False)

    print(f"{n_dropped} of {len(df)} scored forecast rows are proxy-filled at target "
          f"({n_dropped / len(df) * 100:.2f}%); excluded from the _excl_proxy columns.")
    show = ["build", "model", "convention", "h_weeks", "n_full", "n_excl_proxy",
            "challenger_MAE_full", "challenger_MAE_excl_proxy", "challenger_MAE_shift",
            "challenger_direction_pct_full", "challenger_direction_pct_excl_proxy",
            "challenger_direction_pct_shift"]
    print(m[show].round(3).to_string(index=False))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
