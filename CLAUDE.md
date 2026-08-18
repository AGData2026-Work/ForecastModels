# CLAUDE.md

Context for Claude Code working in this repo.

## What this is

A challenger evaluation for a production forecasting model, not a research
playground. Two neural architectures (RNN, GRU) are being tested against an
incumbent panel fixed-effects model on a frozen protocol. The protocol is governed
by `PROJECT_HANDOFF.md` sections 5.2 (evaluation) and 5.3 (change control), which
the owner has marked as locked decisions.

Domain: weekly wholesale white maize prices, 15 scored markets in Nigeria, NGN per
kilogram, 2015-2024. Partners are the Gates Foundation and NADIH.

## Hard rules

**Do not regenerate the evaluation grid.** Read `(market, origin, horizon)` from
`07_panel_fe_forecasts.parquet`. The grid looks weekly but is per-market 4-weekly
and staggered across markets. Regenerating it as one global sequence keeps 558 of
1,601 pairs and produces a result that moves with the grid rather than the model.
This is the single largest error in the prior session; `docs/DECISIONS.md` D-01 has
the detail.

**Do not change locked protocol values without being asked.** Horizons are 4, 13, 26
weeks. Gate horizons are 4 and 13. Threshold is 5% MAE. Walk-forward is expanding,
not rolling. These come from §5.2 and §5.3 of the handoff.

**Always recompute the incumbent on the same pairs as the challenger.** Never compare
an MAE from one grid to an MAE from another. `metrics.paired_table` does this
correctly; do not bypass it.

**Never let a test window, or a statistic derived from one, reach a fit.** Scalers fit
on training rows only. A fit at cut `c` uses only windows whose last target lands at
or before `c - 1` weeks.

**Read `change_control.json` from the conditional runs.** See
`docs/CONDITIONAL_CONVENTION.md`. The incumbent appears to have been measured with
drivers known in advance, so only the conditional arm is comparable to it.

**Log rather than smooth.** Every fill, drop, skip and fallback goes into
`cleaning_log.json` or `split_log.csv` with a count and a reason. The owner checks
outputs for errors themselves and needs edge cases surfaced, not tidied.

**Never present a conditional figure as achievable accuracy.** It assumes perfect
foresight of rainfall, diesel and upstream prices.

## Units and conventions

- Prices are **NGN per kilogram**. Do not convert to metric tonnes; the owner does
  all unit conversions.
- MAE is in NGN/kg and is the primary metric.
- `vs_panel_fe_pct` and `vs_naive_pct`: **positive means the challenger is better.**
- Deliverable filenames follow `YYYYMMDD_Source_Type_Description_vN.ext`.

## Layout

    RUNBOOK.md             step-by-step execution instructions
    configs/build1.yaml    replication of the lost original
    configs/build2.yaml    the same architectures given their best chance
    src/data.py            panel loading, grid consumption, window construction, scaling
    src/model.py           RecurrentForecaster (RNN|GRU), train_one, predict
    src/walkforward.py     retrain cuts, purged splits, recency weights
    src/metrics.py         MAE/MAPE/direction, paired tables, Diebold-Mariano, the gate
    src/run.py             the runner
    src/report.py          reads finished runs, retrains nothing
    src/summary_table.py   consolidated MAPE/MAE table across all runs
    src/audit.py           reproduces docs/DATA_AUDIT.md
    src/diagnose_convention.py  reproduces docs/CONDITIONAL_CONVENTION.md

The two builds share one source tree. `--kind RNN|GRU` selects the recurrent cell and
nothing else, so an RNN-versus-GRU difference is attributable to gating alone.

## Commands

    python src/audit.py
    python src/diagnose_convention.py
    python src/run.py --config configs/build1.yaml --kind RNN --convention unconditional --smoke
    python src/run.py --config configs/build2.yaml --kind GRU --convention conditional --device mps
    python src/report.py
    python src/report.py --gate
    python src/summary_table.py --csv outputs/20260818_MAPE_summary.csv

`--smoke` runs 3 cuts, 1 seed, 4 epochs. It proves the pipeline, not the model, and
tags `smoke: true` in metadata so nothing downstream mistakes it for a result.

## Data paths

Both configs default to `data/`. `run.py` takes `--panel` and `--baseline` overrides.
Files needed:

- `panel_weekly.parquet` (10,352 rows, 27 columns; the white-maize panel)
- `07_panel_fe_forecasts.parquet` (4,803 rows; carries both `panel_fe` and `naive`)

`panel_weekly_yellow.parquet` is a variety split the incumbent did not use.
`climate_weekly_by_market.parquet` is already merged into the panel.

## Things that will bite

1. **`pivot_table` silently drops all-null rows**, which shortens the date index and
   makes `get_loc` fail on 15 full-panel outage weeks. `load_panel` reindexes onto an
   explicit weekly index. Keep it that way.
2. **Two market names contain commas**: `Gwandu, Dodoru`, `Ibadan, Bodija`,
   `Kano, Dawanau`, `Lagos, Mile 12`. Anything that flattens market names into a
   comma-joined string will corrupt them.
3. **Dandume has no upstream market.** Its `upstream_price` is null at every date by
   construction. Channel 10 is the availability mask; a zeroed upstream channel
   without the mask reads as "upstream did not move," which is a different claim.
4. **`config.yaml` versus outputs.** Owner decision D-25 of 2026-08-12 made panel FE
   production, replacing the earlier C8 ensemble, but older documents still call C8
   production. Panel FE is the incumbent here. Do not reintroduce C8 comparisons.
5. **The context file's "26,300 parameters" is wrong.** It cannot be identical for an
   RNN and a GRU at equal hidden size. Report measured counts; do not target it.
6. **The purge in Build 2 falls back** from 78 to 39 to 0 weeks when it would leave
   too little training data. The applied value is in `split_log.csv`. If it is
   falling back at most cuts, say so rather than reporting the requested value.

## When something looks too good

The prior session's worst error produced a headline that moved from "+14%" to "−25%"
without the model changing. The rule that came out of it: **if a result swings and
the model barely moved, suspect the measuring stick, not the model.** Check n first,
then whether the incumbent was recomputed on the same pairs, then seed spread.

## Writing style for any documents produced here

Plain and direct. No em-dashes; use semicolons or rewrite. Avoid "comprehensive",
"successfully", "strategically", "demonstrated". Explain why a choice exists, not
only what it does. Cite specific files, columns and row counts rather than describing
things in the abstract. Flag what is uncertain instead of smoothing it over.
