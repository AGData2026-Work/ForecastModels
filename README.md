# Maize price forecasting: RNN and GRU challenger rebuild

Rebuild of the lost neural challenger for the Nigerian maize weekly panel, in two
builds, evaluated against the incumbent panel fixed-effects model and naive
persistence.

**Nothing in this repo has been run at full scale yet.** The `outputs/` directory
contains smoke runs only, which exist to prove the pipeline is correct end to end.
Every run writes `smoke: true` into `run_metadata.json` and `src/report.py` prints
a warning when it finds one. Delete `outputs/` before the real runs.

## Read in this order

| Document | What it covers |
|---|---|
| `RUNBOOK.md` | step-by-step execution on a Mac through Claude Code. **Start here to run anything** |
| `docs/PLAIN_ENGLISH_BUILD2.md` | Build 2's changes with no maths. Start here for a non-technical reader |
| `docs/DECISIONS.md` | D-01 to D-15: every choice, its reason, its cost |
| `docs/DATA_AUDIT.md` | what is actually in the panel and the grid, measured not remembered |
| `docs/CONDITIONAL_CONVENTION.md` | why the incumbent's accuracy is conditional, and what that means for the verdict |
| `docs/METHODOLOGY.md` | full technical specification |
| `CLAUDE.md` | handoff for Claude Code |

## The two builds

**Build 1** (`configs/build1.yaml`) replicates the lost original as closely as the
surviving `run_protocol.py` allows: 52-week lookback, 10 sequence channels,
8-dimensional market embedding, direct multi-horizon output, log-return target,
Huber loss, Adam at 5e-3, 80 epochs, patience 15, no dropout, refit every 26 weeks,
2 seeds. Two deliberate departures, both about measurement: the evaluation grid is
read from the incumbent's own forecast file rather than regenerated (D-01), and nine
unverifiable rows are dropped (D-02).

**Build 2** (`configs/build2.yaml`) gives the same two architectures their best
realistic chance: explicit per-horizon lag-52 anchors, per-horizon target scaling, a
purged validation split, dropout and weight decay, recency weighting, refit every 13
weeks, 7 seeds aggregated by median, and Aba added as a sixteenth training series
while still scoring fifteen. Thirteen numbered changes, each with a stated reason and
a stated cost in `docs/DECISIONS.md`.

Both are the same source tree driven by different configs, rather than two copies of
the code. The original kept byte-identical RNN and GRU folders to make the comparison
controlled; one tree with a `--kind` flag achieves the same thing without the risk of
the copies drifting apart.

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

On Apple silicon, pass `--device mps` for the GPU backend. Check it first:

    python -c "import torch; print(torch.backends.mps.is_available())"

Put `panel_weekly.parquet` and `07_panel_fe_forecasts.parquet` in `data/`. Every
config and script defaults there, so nothing needs editing. `run.py` also takes
`--panel` and `--baseline` overrides.

## Running

Verify the pipeline before committing hours to it:

    python src/audit.py                    # reproduces every number in DATA_AUDIT.md
    python src/diagnose_convention.py      # reproduces CONDITIONAL_CONVENTION.md
    python src/run.py --config configs/build1.yaml --kind RNN --convention unconditional --smoke

The full matrix is eight runs, two builds by two architectures by two driver
conventions:

    bash run_all.sh                        # resumable; skips completed runs. Or individually:
    python src/run.py --config configs/build1.yaml --kind RNN --convention unconditional --device mps
    python src/run.py --config configs/build1.yaml --kind RNN --convention conditional   --device mps
    python src/run.py --config configs/build1.yaml --kind GRU --convention unconditional --device mps
    python src/run.py --config configs/build1.yaml --kind GRU --convention conditional   --device mps
    python src/run.py --config configs/build2.yaml --kind RNN --convention unconditional --device mps
    python src/run.py --config configs/build2.yaml --kind RNN --convention conditional   --device mps
    python src/run.py --config configs/build2.yaml --kind GRU --convention unconditional --device mps
    python src/run.py --config configs/build2.yaml --kind GRU --convention conditional   --device mps

Build 2 costs roughly seven times Build 1 per unit of grid: 3.5x the seeds (D-11) and
2x the refits (D-14).

## Reading results

    python src/summary_table.py      # MAPE by horizon, every run against the incumbent
    python src/summary_table.py --metric MAE
    python src/report.py             # headline table, all runs, with the change-control gate
    python src/report.py --h 13      # per-market at one horizon
    python src/report.py --market Gombe
    python src/report.py --gate
    python src/report.py --export    # -> outputs/comparison.csv

Two things to check before believing any margin:

1. **`seed_variance.csv` and the `pred_sd` column.** The prior session found seed
   spread at h=26 larger than the differences under discussion. A margin narrower
   than the spread is not a result.
2. **`diebold_mariano.csv`.** A 5% MAE gap that fails a DM test is noise. The
   change-control rule does not require this check; it is here because a bare
   threshold can be tripped by chance.

**Read `change_control.json` from the conditional runs, not the unconditional ones.**
The incumbent's published accuracy appears to assume drivers known in advance
(`docs/CONDITIONAL_CONVENTION.md`), so only the conditional arm is a like-for-like
comparison. The unconditional arm is the honest statement of deployment accuracy and
has no incumbent counterpart to be compared against.

## Output layout

    outputs/<build>/<KIND>_<convention>/
      forecasts.csv             per seed, origin, market, horizon
      predictions_paired.csv    seed-aggregated, joined to panel_fe and naive
      paired_metrics.csv        headline, incumbent recomputed on identical pairs
      metrics_by_market.csv
      seed_variance.csv
      diebold_mariano.csv
      change_control.json
      training_log.csv          epochs, best validation loss, train size, params per cut and seed
      split_log.csv             requested and applied purge per cut
      cleaning_log.json         every fill, drop and skip with counts
      run_metadata.json         resolved config, panel audit, grid audit, param breakdown

`src/summary_table.py` consolidates across runs. Its reference rows are computed from
the baseline file directly, so the table is complete before any challenger finishes;
unrun challengers appear blank rather than being omitted.

## Expected outcome

Build 2 should improve most at h=26, where the missing level anchor was doing the
most damage, and probably still fail the gate. §8.5 of `PROJECT_HANDOFF.md` predicted
this before any of the work started, on the grounds that recurrent forecasters need
hundreds of series and this panel has sixteen. Confirming a prediction that was
written down in advance is a governance result worth recording, and it gives the next
person who proposes a neural model here a documented answer rather than a hunch.

## Open items

1. **The panel FE script has not been supplied.** The driver convention is inferred
   from a replication, not read from code. Re-run `src/diagnose_convention.py` if it
   appears.
2. **No prediction intervals.** §5.5 requires empirically calibrated intervals from a
   rolling window of prior errors. Not implemented. Required before deployment if
   either build ever cleared the gate.
3. **Hidden size is untuned** (D-13), deliberately.
4. **The incumbent's own accuracy may need restating unconditionally.** This affects
   the production model, not just these challengers.
5. **`02_lead_lag_pairs.csv` puts Dandume at the root of the market hierarchy, not
   Dawanau.** §8.3 of the handoff assumed the opposite. Worth flagging to whoever
   wrote it.
