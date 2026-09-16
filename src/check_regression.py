"""
Flags an unexplained MAE swing between two runs of the same build/kind
(the automated version of the manual comparison this whole project has
been doing by hand, run after run, all day).

    python src/check_regression.py --history outputs/run_history.csv --threshold-pct 10

Reads `outputs/run_history.csv` -- an append-only log that `run.py` and
`run_afex.py` each add one row to per (build, model, convention, h) at
the end of every real (non-smoke) run, via `append_run_history` below.
For every (build, model, convention, h) with two or more logged runs,
compares the two most recent by timestamp and warns if MAE moved by more
than --threshold-pct. This is a warning tool, not a gate: it does not
fail the run it's called from, it prints what a human (or the next entry
in docs/DECISIONS.md) should account for. A group with only one logged
run so far (its first time being run) is silently skipped -- there is
nothing to compare it against yet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HISTORY_COLUMNS = ["timestamp", "build", "model", "convention", "h", "challenger_MAE"]


def append_run_history(history_path, build: str, model: str, convention: str,
                       paired_metrics: pd.DataFrame) -> None:
    """Called once per real (non-smoke) run, after paired_metrics.csv is
    written. Appends one row per horizon; never overwrites, so this is
    the actual history check_regression.py needs and paired_metrics.csv
    (overwritten every run) cannot provide on its own."""
    history_path = Path(history_path)
    rows = [dict(timestamp=pd.Timestamp.now().isoformat(), build=build, model=model,
                convention=convention, h=int(h), challenger_MAE=float(mae))
            for h, mae in zip(paired_metrics["h"], paired_metrics["challenger_MAE"])]
    new_rows = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    if history_path.exists():
        existing = pd.read_csv(history_path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    history_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(history_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="outputs/run_history.csv")
    ap.add_argument("--threshold-pct", type=float, default=10.0,
                    help="flag a (build, model, convention, h) whose MAE moved by "
                         "more than this percent between its two most recent runs")
    a = ap.parse_args()

    history_path = Path(a.history)
    if not history_path.exists():
        print(f"[check_regression] no history file at {history_path} yet -- "
             "nothing to compare (this is expected before any run has completed).")
        return

    df = pd.read_csv(history_path, parse_dates=["timestamp"])
    any_flagged = False
    for key, g in df.groupby(["build", "model", "convention", "h"]):
        if len(g) < 2:
            continue
        g = g.sort_values("timestamp")
        prev_mae, latest_mae = g["challenger_MAE"].iloc[-2], g["challenger_MAE"].iloc[-1]
        if prev_mae == 0:
            continue
        pct_change = (latest_mae - prev_mae) / prev_mae * 100
        if abs(pct_change) >= a.threshold_pct:
            any_flagged = True
            build, model, convention, h = key
            print(f"[check_regression] {build} {model} {convention} h={h}: "
                 f"MAE moved {prev_mae:.2f} -> {latest_mae:.2f} ({pct_change:+.1f}%). "
                 f"If this is expected, confirm a docs/DECISIONS.md entry explains "
                 f"why before trusting the new number.")

    if not any_flagged:
        print(f"[check_regression] no (build, model, convention, h) moved more "
             f"than {a.threshold_pct}% between its two most recent logged runs.")


if __name__ == "__main__":
    main()
