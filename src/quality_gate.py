"""
The post-D-36 replacement for `metrics.change_control()`'s retired
5%-vs-incumbent rule (workplan item 5.1). `change_control()` itself is
untouched and still runs -- this does not replace it in the codebase,
only in what this project treats as its own pass/fail bar, per D-36.

A horizon passes if BOTH:
  1. The margin over naive is statistically real: Diebold-Mariano
     p < 0.05 (already computed into every run's diebold_mariano.csv).
  2. The margin over naive survives the model's own seed-to-seed spread:
     margin_over_naive_over_seed_sd > 1 (D-20/D-34's own bar, reusing
     seed_analysis.analyse_run_horizon rather than reimplementing it).

Both conditions answer different questions -- (1) is this specific
deployed forecast's historical track record distinguishable from chance,
(2) would a re-trained model likely reproduce it -- and this project's
own work today (D-46, D-49, D-50) found real cases where a result passes
one and not the other. Requiring both is deliberately more conservative
than either alone.

    python src/quality_gate.py --run-dir outputs/build3_underfit_corrected/GRU_unconditional
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from seed_analysis import analyse_run_horizon


def evaluate(run_dir: Path, gate_horizons: list[int]) -> pd.DataFrame:
    forecasts = pd.read_csv(run_dir / "forecasts.csv", parse_dates=["origin"])
    dm = pd.read_csv(run_dir / "diebold_mariano.csv")
    dm_naive = dm[dm["vs"] == "naive"].set_index("h")

    rows = []
    for h in gate_horizons:
        g = forecasts[forecasts["h"] == h]
        seed_stats = analyse_run_horizon(g)
        dm_p = float(dm_naive.loc[h, "dm_p"]) if h in dm_naive.index else float("nan")
        dm_stat = float(dm_naive.loc[h, "dm_stat"]) if h in dm_naive.index else float("nan")
        dm_passes = dm_p < 0.05 and dm_stat < 0  # negative stat favours the challenger
        seed_ratio = seed_stats["margin_over_naive_over_seed_sd"]
        seed_passes = seed_ratio > 1
        rows.append(dict(
            h=h, dm_p=dm_p, dm_stat=dm_stat, dm_passes=dm_passes,
            margin_over_naive_over_seed_sd=seed_ratio, seed_passes=seed_passes,
            gate_passes=bool(dm_passes and seed_passes),
        ))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gate-horizons", type=int, nargs="+", default=[4, 13])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    result = evaluate(run_dir, a.gate_horizons)
    print(result.to_string(index=False))

    overall = bool(result["gate_passes"].all())
    print(f"\nquality gate (vs naive, D-36/workplan 5.1): "
         f"{'PASS' if overall else 'FAIL'} -- {run_dir.name}")

    out_path = Path(a.out) if a.out else run_dir / "quality_gate.json"
    out_path.write_text(json.dumps(dict(
        gate_horizons=a.gate_horizons, overall_pass=overall,
        by_horizon=result.to_dict("records"),
    ), indent=2, default=str))


if __name__ == "__main__":
    main()
