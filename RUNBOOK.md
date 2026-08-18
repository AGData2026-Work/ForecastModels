# Runbook: running this on a Mac through Claude Code

Follow in order. Steps 1 to 6 take about fifteen minutes and prove the pipeline
before you commit hours to it. Step 7 is the long run.

Every command assumes you are in the repo root with the virtual environment active.

---

## Step 1. Put the repo somewhere permanent

Not Downloads. Something you will still have in three months, because these results
feed a quarterly review.

```bash
mkdir -p ~/work/maize-forecasting
cd ~/work/maize-forecasting
tar -xzf ~/Downloads/20260818_Maize_RNN_GRU_Challenger_Rebuild_v1.tar.gz
cd 20260818_Maize_RNN_GRU_Challenger_Rebuild_v1
```

Optional but worth it, since the long run produces a lot of files and you will want
to know what changed:

```bash
git init && git add -A && git commit -m "Rebuild v1: source and docs, no runs yet"
```

`.gitignore` already excludes `outputs/`, `logs/`, `data/` and `.venv/`, so the repo
tracks code and documents only.

---

## Step 2. Put the two data files in `data/`

```bash
mkdir -p data
cp /path/to/panel_weekly.parquet data/
cp /path/to/07_panel_fe_forecasts.parquet data/
ls -la data/
```

Both configs and all four scripts default to `data/`, so nothing needs editing. If
you would rather keep the parquet files elsewhere, every script takes an override:
`--panel`, `--baseline` on `run.py`, `--panel`, `--grid` on `audit.py` and
`diagnose_convention.py`.

`climate_weekly_by_market.parquet` and `panel_weekly_yellow.parquet` are not needed;
the first is already merged into the panel, the second is a variety split the
incumbent did not use.

---

## Step 3. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check what backend you have:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
sysctl -n hw.perflevelcount hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.ncpu
```

Note the core count. You will use it in step 6.

---

## Step 4. Audit the data before modelling anything

```bash
python src/audit.py | tee logs/audit.txt
```

Check three things in the output against `docs/DATA_AUDIT.md`:

- panel shape `(10352, 27)`, 647 weeks, 2012-05-02 to 2024-09-18
- grid `4803` rows, `1601` pairs per horizon, `432` origins, Aba absent
- per-market origin spacing showing `{28: ...}` for every market, which is what
  confirms the grid is 4-weekly and staggered rather than weekly

If any of those differ, stop; the files are not the ones this was built against.

---

## Step 5. Reproduce the convention finding

```bash
python src/diagnose_convention.py | tee logs/convention.txt
```

You are looking for the `replica_vs_published_MAD` column: roughly 20.5 and 27.4 for
the conditional rows at h=13 and h=26 against roughly 32.0 and 57.3 for the
unconditional rows. That factor-of-two gap is the evidence that the incumbent was
measured with drivers known in advance. `docs/CONDITIONAL_CONVENTION.md` has the
argument.

Takes about a minute. If the panel FE script ever turns up, rerun this and check the
conclusion still holds.

---

## Step 6. Smoke test, then calibrate the runtime

Smoke runs use 3 retrain cuts, 1 seed and 4 epochs. They prove the pipeline end to
end and are not results; every one tags `smoke: true` in its metadata and shows a
SMOKE label in the summary table.

```bash
python src/run.py --config configs/build1.yaml --kind RNN --convention unconditional --smoke
python src/run.py --config configs/build2.yaml --kind GRU --convention conditional  --smoke
python src/summary_table.py
```

The summary table should show `panel_fe` at 9.58 / 15.16 / 17.78 with n=1592, and two
SMOKE rows at n of a couple of hundred. Now delete them so they cannot contaminate
anything:

```bash
rm -rf outputs logs/*.log
```

### Decide CPU or MPS

Do not assume the GPU is faster. These are small networks, 52 timesteps at hidden 64,
and PyTorch has no fused recurrent kernel on Metal, so MPS often loses to CPU on
Apple silicon because of per-kernel launch overhead. Measure it:

```bash
mkdir -p logs
for dev in cpu mps; do
  echo "=== $dev ==="
  /usr/bin/time -p python src/run.py --config configs/build1.yaml --kind GRU \
    --convention unconditional --smoke --device $dev --out /tmp/bench_$dev 2>&1 | tail -4
done
rm -rf /tmp/bench_cpu /tmp/bench_mps
```

Use whichever is faster in step 7. If they are close, use `cpu`: it is bit-reproducible
across runs, MPS is not, and reproducibility matters more here than a few percent of
speed.

### Expected runtime

Measured on this build: at the largest training set (5,964 windows), one epoch takes
about 3.6 seconds for the RNN and 1.6 seconds for the GRU on a single CPU thread.
The full matrix is 1,124 fits:

| build | cuts | seeds | fits per architecture and convention | fits total |
|---|---|---|---|---|
| 1 | 18 | 2 | 36 | 144 |
| 2 | 35 | 7 | 245 | 980 |

That works out to roughly 26 single-thread hours, most of it Build 2. On 8 to 12
cores expect **4 to 7 hours** for everything, with Build 1 finishing in well under
an hour.

Treat that as an order of magnitude, not a promise. Calibrate it yourself: run one
full Build 1 configuration, note the wall time it prints, and scale.

```bash
python src/run.py --config configs/build1.yaml --kind RNN --convention unconditional \
  --device cpu 2>&1 | tee logs/calibration.txt
```

Build 1 has 36 fits per configuration and Build 2 has 245, so multiply that run's
time by 4 for all of Build 1 and by about 27 for all of Build 2.

---

## Step 7. The full run

`run_all.sh` is resumable. Any run whose `predictions_paired.csv` already exists is
skipped, so if your laptop sleeps or you interrupt it, rerun the identical command and
it picks up where it stopped. Each run writes its own log under `logs/`.

Build 1 first, so you have a complete result inside an hour:

```bash
caffeinate -i -s bash -c 'DEVICE=cpu BUILDS="configs/build1.yaml" bash run_all.sh'
```

Then Build 2, which is the long one:

```bash
caffeinate -i -s bash -c 'DEVICE=cpu BUILDS="configs/build2.yaml" bash run_all.sh'
```

Or everything in one go:

```bash
caffeinate -i -s bash run_all.sh
```

`caffeinate -i -s` stops the Mac idling or sleeping mid-run. Without it a lid close
will suspend the process, and while the resume logic will recover, you will lose the
partial run.

To pin thread count instead of letting PyTorch use every core:

```bash
THREADS=8 DEVICE=cpu bash run_all.sh
```

Useful if you want to keep the machine usable. Below about 4 threads the run gets
slow enough to matter.

`run_all.sh` prints the summary tables and the change-control gate when it finishes,
and lists any failed runs.

---

## Step 8. Read the results

```bash
python src/summary_table.py                                   # MAPE, all runs vs incumbent
python src/summary_table.py --metric MAE
python src/summary_table.py --csv outputs/$(date +%Y%m%d)_MAPE_summary.csv
python src/report.py --gate                                   # change-control verdict
python src/report.py --h 13                                   # per-market at 3 months
python src/report.py --market Gombe
```

**Read the gate from the conditional rows, not the unconditional ones.** The incumbent
has no unconditional counterpart, so only the conditional arm is a like-for-like
comparison. The unconditional arm is the only honest statement of deployment accuracy.

Before believing any margin, check two things:

```bash
cat outputs/build2_best_practice/GRU_conditional/seed_variance.csv
cat outputs/build2_best_practice/GRU_conditional/diebold_mariano.csv
```

A margin narrower than the seed spread is not a result. A 5% MAE gap that fails the
Diebold-Mariano test is noise. The change-control rule does not require the DM test;
it is there because a bare threshold can be tripped by chance.

Also worth a look, because they are where problems show up:

```bash
python -c "import json;print(json.dumps(json.load(open('outputs/build2_best_practice/GRU_conditional/cleaning_log.json')),indent=1))" | head -30
column -s, -t outputs/build2_best_practice/GRU_conditional/split_log.csv
head -5 outputs/build2_best_practice/GRU_conditional/training_log.csv
```

- `cleaning_log.json`: `scored_pairs_per_horizon` should equal
  `expected_pairs_per_horizon`, 1592. If it does not, windows were skipped and
  `windows_skipped` says which and why.
- `split_log.csv`: `purge_weeks_applied` against `purge_weeks_requested`. If the
  purge fell back from 78 at most cuts rather than a few early ones, D-08 is not
  doing its job and the result needs that caveat attached.
- `training_log.csv`: if `epochs_run` clusters at the patience ceiling, or
  `best_val` swings widely between seeds at the same cut, early stopping is firing
  on noise and the seed count needs raising.

---

## Step 9. Archive the run

```bash
STAMP=$(date +%Y%m%d)
mkdir -p ../archive
tar -czf "../archive/${STAMP}_Maize_RNN_GRU_Results_v1.tar.gz" outputs logs
git add -A && git commit -m "Results run $STAMP"
```

The `outputs/` tree carries `run_metadata.json` per run with the fully resolved
config, the panel audit and the grid audit, so an archived run is self-describing
and reproducible from seeds and grid alone.

---

## If something goes wrong

**A run fails partway.** Rerun the same `run_all.sh` command. Completed runs are
skipped, the failed one restarts. Check its log under `logs/` first.

**`no forecasts produced; check the config paths`.** The parquet files are not where
the config says. Run `ls data/`.

**`panel date index is not a clean weekly grid`.** The panel supplied is not the one
this was built against. Do not work around it; the window logic assumes contiguous
weekly Wednesdays.

**`KeyError: Timestamp(...)`.** Something rebuilt the date index with
`pivot_table`, which silently drops the 15 all-null weeks. `load_panel` reindexes to
avoid this; do not replace it with a bare pivot.

**MPS gives different numbers between runs.** Expected. MPS is not bit-reproducible.
Use `--device cpu` for anything that has to be reproducible.

**Everything is slower than the estimate.** Check `THREADS`. With `THREADS=1` the
matrix takes roughly 26 hours rather than 5.

---

## What to hand to whom

| Audience | Give them |
|---|---|
| Non-technical reader, or anyone asking why neural models were tried | `docs/PLAIN_ENGLISH_BUILD2.md` |
| Quarterly review | `outputs/<date>_MAPE_summary.csv`, `report.py --gate` output, `docs/CONDITIONAL_CONVENTION.md` |
| Next analyst | this runbook, `CLAUDE.md`, `docs/DECISIONS.md` |
| Anyone questioning a specific choice | `docs/DECISIONS.md`, cited by D-number |

Two items belong on the review agenda regardless of what the models do: whether the
production model's published accuracy should be restated unconditionally
(`docs/CONDITIONAL_CONVENTION.md`), and that the market hierarchy puts Dandume at the
root rather than Dawanau, which inverts the prior stated in §8.3 of the handoff.
