# Decision log

Every choice that could have gone another way, with the reason it went this way
and what it cost. Numbered so results can cite them.

Two categories:

- **D-01 to D-05** apply to both builds. They are about measuring honestly, and
  fix defects in the lost original rather than changing the models.
- **D-06 to D-14** are what makes Build 2 different from Build 1.

---

## D-01. The evaluation grid is read from the baseline file, never regenerated

**Decision.** `src/data.py:load_grid` reads `(market, origin, horizon)` straight
out of `07_panel_fe_forecasts.parquet`. `src/walkforward.py` contains no function
that invents origins.

**Why.** The baseline grid is not what it looks like. Its 432 distinct origin
dates appear weekly, but each market has its own sequence spaced exactly 28 days
apart, and the markets are staggered against each other so the union looks
weekly. Per-market counts: 111 or 112 origins for the twelve markets whose record
starts in 2015-09, and 98 for Biu, Damaturu, Gombe, Mubi and Potiskum, whose
price record starts a year later. Only six distinct market-sets occur across all
432 origins, rotating on a 4-week cycle.

The lost original built one global 4-weekly sequence for all markets at once
(`make_protocol_origins(df["date"], spacing, ...)` in the surviving
`run_protocol.py`). Joined against the baseline that keeps 558 of 1,601 pairs, and
the prior session's own notes record the consequence: a headline that moved from
"+14% versus the incumbent" to "−25%" while the model barely changed, because the
comparison set moved underneath it.

**Cost.** None. Reading the grid is strictly less code than generating it.

**Effect.** 1,592 of 1,601 pairs per horizon are scored, with zero windows
skipped for missing inputs (verified in `cleaning_log.json`). The remaining nine
go under D-02.

---

## D-02. Nine rows with target 2019-04-24 are dropped

**Decision.** `drop_target_dates: ["2019-04-24"]` in both configs.

**Why.** 2019-04-24 is one of fifteen weeks where every market's price is null in
`panel_weekly.parquet`. The baseline file nonetheless reports an `actual` for the
nine market-horizon rows whose target lands there, so those actuals came from
somewhere the panel does not contain. Scoring against a target we cannot verify
is worse than losing a fraction of a percent of the grid.

The other fourteen outage weeks (2014-01-01, a run from 2014-04-30 to 2014-07-16,
2014-10-01, 2015-04-01) all sit before the first origin and never reach the
evaluation.

**Cost.** The pair set is kept rectangular across horizons, so dropping nine rows
also drops the two sibling horizons of the same nine `(market, origin)` pairs:
27 of 4,803 rows, 0.56%. Ragged pairs would let a market's h=4 and h=13 numbers
rest on different origin sets, which is the D-01 error in miniature.

**Note for the record.** This makes the incumbent's MAE in this repo differ in
the third significant figure from a figure computed on the full 1,601. Both are
correct for their own grid. `paired_metrics.csv` recomputes the incumbent on the
same 1,592 pairs as the challenger, every time, so no comparison in this repo
ever crosses grids.

---

## D-03. Aba trains in Build 2 but is never scored

**Decision.** `train_markets: scored_only` in Build 1 (15 markets);
`train_markets: all_in_panel` in Build 2 (16 markets, 15 scored).

**Why.** Section 8.5 of `PROJECT_HANDOFF.md` is the governing prediction about
this whole line of work: for neural forecasting the binding constraint is the
number of series, not the number of rows. Aba is a sixteenth series with 430
observed weeks sitting unused in the panel. A pooled model with a market
embedding can absorb it at the cost of eight embedding parameters, and it is
excluded from scoring, so it cannot flatter the comparison. If the number-of-
series argument is right, adding a series is the cheapest available test of it.

Build 1 excludes it to stay faithful to the original.

**Cost.** Aba's price record has a 139-week interior gap and 46 proxy-filled
weeks, the worst of any market. If it is noisy enough to hurt, that will show as
Build 2 doing worse at h=4 where the pooled fit matters most.

---

## D-04. Where the context file and the surviving code disagree, the code wins

The prior session's context file describes the original build in prose. Three of
its claims are contradicted by `run_protocol.py`, which is the code that ran.

| Claim in the context file | What the code shows | Resolution |
|---|---|---|
| "Every origin retrained from scratch on data strictly before it" (§2.3) | `--retrain-every 26` default, and a docstring saying per-origin retraining "is not affordable" | Build 1 refits every 26 weeks (`retrain_every_weeks: 26`). Not leakage; a stale fit is the harder test |
| "Per-horizon scaler fit on training data only" (§2.3) | `standardise()` is applied to inputs only; `ytr` reaches `train_one` raw | Build 1 leaves targets unscaled (`scale_targets: false`). Build 2 scales them (D-07) |
| "26,300 parameters" for both RNN and GRU (§2.1) | impossible: at equal hidden size a GRU has three times the recurrent parameters of a tanh RNN | Not reproduced. This repo reports actual counts: 9,859 (Build 1 RNN), 20,739 (Build 2 GRU with flat features). See D-05 |

---

## D-05. Parameter counts are reported, not targeted

**Decision.** Take the documented architecture (lookback 52, hidden 64,
embedding 8) and report whatever parameter count follows.

**Why.** The context file's 26,300 figure cannot be right for both
architectures simultaneously, so it cannot be used as a reconstruction target.
Chasing it would mean inventing capacity the original may not have had.

**Effect.** The parameters-per-training-window ratio, which is the quantity §8.5
actually cares about, is now computed from the run rather than asserted.
`run_metadata.json` carries `n_params` and `param_breakdown` for every run, and
`training_log.csv` carries `n_train` per cut, so the ratio is auditable at every
refit rather than quoted once.

---

## D-06. Build 2 adds explicit lag-52 anchors, one per horizon

**Decision.** `explicit_lag52: true`. Six flat features:
`log(P[t+h-52]) - log(P[t])` for each of h = 4, 13, 26, plus
`log(P[t+h-52]) - log(P[t-52])` for each.

**Why.** This is the largest single expected gain, and the reason is structural
rather than empirical.

The incumbent uses a lag-52 regressor with its own coefficient, so it cannot fail
to use last year's same-week price as a level anchor. At h = 26 that anchor is
still *observed data from before the origin*, which is why the incumbent keeps a
seasonal foothold at six months while a recurrent network extrapolates from
hidden state.

The networks did have lag-52 inside their 52-week lookback. What they lacked was
any pressure to treat it as an anchor. To use it, the year-ago value has to
survive 51 subsequent hidden-state updates in a cell whose memory decays by
construction, and be legible in a fixed-size final state that also has to carry
the 51 more recent weeks. As a flat feature the gradient path length is one.

The multi-horizon head sharpens this. All three horizons are emitted from the
same final hidden state, but each horizon's seasonal anchor sits at a different
position in the window:

| horizon | target | year-ago anchor | position in the 52-week window |
|---|---|---|---|
| h=4 | P(t+4) | P(t−48) | step 4 |
| h=13 | P(t+13) | P(t−39) | step 13 |
| h=26 | P(t+26) | P(t−26) | step 26 |

One shared vector would have to encode three differently-positioned anchors and
each head learn to read the right one. Three explicit scalars hand each head its
own.

Fourier terms are not a substitute and were never removed. `sin1, cos1, sin2,
cos2` are deterministic functions of calendar position, identical in 2017 and
2024; they encode the average seasonal shape in four parameters. Lag-52 carries
the realised value from that specific week last year: that year's harvest timing,
that market's particular shock, the level it was operating at. Two years with
identical Fourier values can have lag-52 values 200 NGN/kg apart.

**No look-ahead.** `t + h − 52 ≤ t` whenever `h ≤ 52`, so every anchor is
observed at the origin. The check is in `build_flat`, which returns a failure flag
rather than a value when the anchor index is out of range.

**Cost.** Six parameters. Windows now need 52 weeks of history plus a year before
that for the `t-52` term, which pushes the earliest usable training window later;
`min_train_windows` guards against a cut being served by too small a fit.

---

## D-07. Build 2 scales targets per horizon

**Decision.** `scale_targets: true`. Mean and standard deviation per horizon, fit
on training windows only, inverted before prices are recovered.

**Why.** The loss is Huber over a three-element output vector. Log-returns at
h = 26 have a much wider spread than at h = 4, simply because six months of price
movement exceeds one month's. With unscaled targets a single loss treats a large
h=26 error as more important than a large h=4 error in the same proportion as the
raw variances, so the gradient budget is allocated by horizon variance rather
than by anything anyone chose. Worse, Huber's delta of 1.0 is a fixed
threshold in target units: with unscaled log-returns almost every h=4 residual is
inside the quadratic region and a larger share of h=26 residuals fall in the
linear region, so the two horizons are effectively fitted under different loss
functions.

Standardising per horizon makes delta mean the same thing at every horizon.

**Cost.** One more object to invert correctly. `Scaler.y_inverse` is applied
before `p0 * exp(yhat)`; the smoke run confirms recovered prices land in the right
magnitude.

---

## D-08. Build 2 purges the validation split

**Decision.** `purge_weeks: 78`, with automatic fallback to 39 then 0 if purging
would leave fewer than `min_train_windows` training rows. Applied value is logged
per cut in `split_log.csv`.

**Why.** Both builds split training windows chronologically, 85% train and 15%
validation, and early stopping is the load-bearing regulariser in Build 1: it is
the only one the original configured. That makes the honesty of the validation
signal important.

With a 52-week lookback and a 26-week maximum horizon, an unpurged split leaks in
two directions at once. The last training windows and the first validation
windows share up to 51 weeks of identical input. And the last training windows'
targets land up to 26 weeks after their origin, which is inside the validation
block's input period. So the epoch chosen as "best" is chosen partly on data the
fit has already seen. 78 weeks is 52 + 26: enough to break both overlaps.

**Cost.** Real, and worst at the early cuts where training data is thinnest. In
the smoke run the fallback engaged at 39 weeks for the first three cuts. That is
why the fallback exists and why it is logged rather than silent.

---

## D-09. Build 2 adds dropout and weight decay

**Decision.** `head_dropout: 0.2`, `input_dropout: 0.1`, `weight_decay: 1e-4`.

**Why.** Build 1 has exactly one deliberate regulariser, early stopping. Its
other two protections are incidental: capacity is limited by design choices made
for other reasons (pooling across markets rather than fitting 15 separate models;
an 8-dimensional embedding rather than 15 independent fits), and the log-return
target makes memorising price levels structurally impossible. Dropout appears
nowhere in the surviving code or the context file's architecture description.

The ratio D-05 exposes is the reason it matters. Roughly four training windows
per parameter is not a regime where one early-stopping criterion, evaluated on a
partly self-referential validation slice (D-08), can be relied on alone.

**Cost.** Two more knobs. Both are set to conventional values rather than tuned,
because tuning them against the walk-forward is precisely the overfitting the
change-control rule exists to prevent (§8.5).

---

## D-10. Build 2 uses cosine learning-rate decay and a lower initial rate

**Decision.** `lr: 3e-3` with `lr_schedule: cosine` over `epochs: 200`, patience
25. Build 1 keeps `lr: 5e-3` flat, 80 epochs, patience 15.

**Why.** 5e-3 is high for a recurrent net on ~6,000 windows; with a flat rate and
patience 15 the run can stop while still bouncing rather than because it has
converged. Decaying to near zero lets the late epochs settle. The lower start
plus longer patience is the standard trade: more epochs, less chance the stopping
decision is noise. The prior session recorded seed spread at h=26 exceeding the
margins under discussion, which is the signature of stopping on noise.

**Cost.** Roughly two and a half times the epochs per fit.

---

## D-11. Build 2 uses seven seeds and aggregates by median

**Decision.** `seeds: [0..6]`, `seed_aggregation: median`. Build 1 keeps two
seeds and the mean.

**Why.** The prior session's own recommendation was five to ten seeds because
seed spread at h=26 was larger than the differences being argued about. Two seeds
cannot distinguish a model difference from an initialisation difference. Median
rather than mean because a single diverged seed shifts a two-point mean by half
its error and a seven-point median not at all.

`seed_variance.csv` reports per-seed metrics and `predictions_paired.csv` carries
`pred_sd` per row, so the spread is reported rather than averaged away.

**Cost.** 3.5x the compute of Build 1 per cut, compounding with D-14's tighter
cadence.

---

## D-12. Build 2 weights recent windows more heavily

**Decision.** `recency_half_life_weeks: 156` (three years). Exponential weight in
the loss, uniform in Build 1.

**Why.** Prices in the panel run from 28.4 to 1,221.6 NGN/kg and most of the
climb is 2023-24. Diesel over the same panel runs from 101.68 to 1,930.79, a
consequence of the subsidy removal. Equal weighting asks one parameter set to
describe both the pre-2020 regime and the post-2023 one. A three-year half-life
tilts the fit toward the recent regime while keeping the older data, which is
still the only evidence about seasonal shape and therefore cannot be discarded.

Note the log-return target already removes the *level* problem. This addresses
the separate question of whether return *dynamics* changed, which differencing
does not fix.

**Cost.** Effective sample size falls. If the older data was in fact informative
about dynamics, this hurts, and it will show up as Build 2 losing ground at h=4
where short-run dynamics dominate.

---

## D-13. Hidden size is held at 64 in both builds

**Decision.** `hidden: 64` everywhere, exposed as `--hidden` for a manual sweep.

**Why.** Capacity is a plausible lever, but sweeping it inside the walk-forward
would make Build 2's result partly a product of architecture search against the
evaluation set. Holding it fixed keeps Build 2 versus Build 1 a clean test of the
other twelve changes. If a sweep is wanted it should be run as a separate
exercise on a training-only holdout and reported as such.

**Cost.** Build 2 may be leaving accuracy on the table. Stated rather than
quietly captured.

---

## D-14. Build 2 refits every 13 weeks instead of 26

**Decision.** `retrain_every_weeks: 13`.

**Why.** Halving staleness is the cheapest remaining improvement and it aligns
the refit cadence with the quarterly review cadence in §5.2, so the evaluation
mirrors how the model would actually be maintained.

**Cost.** Doubles the number of fits. Combined with D-11 that is 7x Build 1's
compute per unit of grid.

---

## D-15. Both driver conventions are run, and only one of them can be compared to the incumbent

**Decision.** Every build runs twice, `--convention unconditional` and
`--convention conditional`. Under `conditional`, twelve flat features carry
driver movement over the forecast window: diesel and upstream-price log-returns
from origin to target, and mean rainfall and NDVI over the interval.

**Why, and why it matters more than it looks.** The script that produced
`07_panel_fe_forecasts.parquet` was not available, so which convention the
incumbent used had to be inferred. `src/diagnose_convention.py` rebuilds panel FE
both ways and compares. The evidence points clearly to conditional; the numbers
and the caveats are in `docs/CONDITIONAL_CONVENTION.md`.

The consequence is a governance point, not a modelling one. The unconditional arm
is the only one that describes deployment, but the incumbent has no unconditional
counterpart, so the change-control gate can only be adjudicated on the
conditional arm. `paired_metrics.csv` is produced for both; `change_control.json`
should be read from the conditional arm until an unconditional incumbent exists.

**Cost.** Doubles the run matrix to eight runs. Stated in
`docs/CONDITIONAL_CONVENTION.md` rather than buried: a conditional number is not
an achievable accuracy, it is an upper bound conditional on perfect driver
foresight.

---

## D-16. Version 1 discontinued, 28 August 2026

**Decision.** No further runs of either architecture at version 1. Version 2 only
going forward.

**Why.** Version 1 exists to reproduce a lost original and that job is done. Its
results are recorded in the Week 0 status note and its outputs are archived.
Running it further consumes compute without answering a live question.

**What is retained and why.** `configs/build1.yaml`, the four version 1 output
folders, and the version 1 figures in the status note. These are the only
evidence of what the nine version 2 changes did, and one of those effects is
still open: version 1 calls direction correctly at three months 68.5 and 68.9
per cent of the time against version 2's 60.3 and 62.1. That gap is the basis
for the build 3 hypothesis, that version 2's regularisation flattened its
predictions. Discarding the evidence would leave the hypothesis unsupported.

**Cost.** None to compute. If the directional question is ever settled against
build 3, version 1 becomes reproducible from the archived config.

---

## D-17. Origin date filter closed: `drop_target_dates` now drops by origin too

**Decision.** `load_grid` in `src/data.py` masks on `f["target"].isin(bad) |
f["origin"].isin(bad)`, not `target` alone. The log key is renamed
`dropped_by_date_filter` (was `dropped_by_target_filter`) so it does not claim to
be target-only when it is not; `src/run.py`'s reader was updated in the same edit.

**Why.** 2019-04-24 is a full-panel outage week (D-02). The target-only mask
caught the nine rows whose *target* lands there, but missed six more whose
*origin* week is 2019-04-24: those forecasts are issued from a week with no
recorded panel price, so the input side of those windows is exactly as
unverifiable as the target side was. `load_grid('data/07_panel_fe_forecasts.parquet',
[4,13,26], ['2019-04-24'])` now reports 1,590 pairs and 15 dropped rows, up from
1,592 and 9.

**Cost.** Six more of 1,601 pairs dropped, 0.37 percentage points on top of D-02's
0.56%. `audit.py` reports the full 1,601-row grid and is not itself affected by
`drop_target_dates`; it was re-run only to confirm `load_grid` still executes
without error after the mask change, which it does. The expected effect on a
full walk-forward, not yet re-run at time of writing, is MAE rising by 0.018,
0.033 and 0.046 NGN/kg at h=4, 13, 26 with reported figures unchanged to two
decimal places; this is a defensibility fix, not an accuracy fix, and a result
that barely moves after it is the expected outcome, not a sign the fix failed.

---

## D-18. Training loss logged alongside validation loss

**Decision.** `train_one` in `src/model.py` now records `{epoch, train_loss,
val_loss, train_loss_subsampled}` every epoch and returns it as `history` in the
info dict. `run.py` writes it to `epoch_log.csv` in each run directory, one row
per (cut, seed, epoch). Training-loss evaluation subsamples to a fixed 4,000
rows, drawn once per fit, whenever `n_train > 8000`; `train_loss_subsampled`
records when that happened so a downstream reader knows the two losses in a row
are not always computed on the same-sized set.

**Why.** Build 1's Week 0 diagnosis (under-learning versus overfitting, the
premise behind the Build 3 hypothesis in D-16) had to be inferred from
validation loss alone, because training loss was never recorded. The two losses
answer different questions: a gap between them with training falling and
validation rising is overfitting, both high and flat is under-learning. Without
`train_loss`, the diagnosis rested on the shape of validation loss alone, which
is consistent with either failure mode.

**Verified.** Smoke run of `build2.yaml` RNN unconditional produced
`epoch_log.csv` with columns `cut, seed, epoch, train_loss, val_loss,
train_loss_subsampled`, four rows per (cut, seed) matching the smoke epoch
count.

**Incident during verification, and the fix.** Running that smoke-test
verification with the command given in the work order writes to
`outputs/build2_best_practice/RNN_unconditional/`, the same path the real
35-cut production run already occupied, with no smoke-specific suffix. It
overwrote that run's `forecasts.csv`, `training_log.csv`,
`predictions_paired.csv` and related files with the 3-cut, 1-seed, 4-epoch
smoke output. `outputs/` and `logs/` are both gitignored, so there was no git
history to recover from; the other three build2 run directories were
unaffected (confirmed by directory size and modification time). The original
console log survived at `logs/build2_best_practice_RNN_unconditional.log` and
was copied to `..._pre_D17.log` before the rerun; it records the pre-D17
headline: h=4/13/26 challenger MAE 17.76/34.46/50.59, direction 56.4/60.3/57.4%,
n=1592, 35 cuts. The run was redone in full
(`--config configs/build2.yaml --kind RNN --convention unconditional --device
cpu --threads 0`) to restore it; it is not bit-identical to the original
because D-17 changed the grid under it (1,590 pairs, not 1,592), so the new
numbers carry both the D-17 grid fix and the D-18 logging addition, not D-18
alone. Anyone diffing this run against the Week 0 status note should expect
that combined effect, not D-18 in isolation.

**`run.py` should gain a smoke-specific output suffix** so this cannot recur;
not implemented here because expanding scope beyond Task 2's stated change was
not asked for. Flagged for whoever picks this up next.

**Cost.** One column set per epoch, four training-set forward passes at the
worst (largest-cut, most-seeds) point instead of one, since the training loss
now needs a fresh forward pass every epoch on top of the existing validation
pass; capped by the 4,000-row subsample so it does not scale with the largest
cuts' full training set (nearly 5,000 windows by cut 35).

---

## D-19. Directional accuracy at h=26 unconditional is void against the always-up benchmark

**Decision/finding.** `src/directional_benchmark.py` compares each run's
directional accuracy against the share of windows in which the price actually
rose (or fell), not against 50%. Prices trend upward across the panel (65.4%
of h=26 windows rose in the pairs `RNN_unconditional`/`GRU_unconditional`
score against), so 50% was never the right reference.

**Result.** At h=26, both unconditional-arm models score below the always-up
share: GRU 54.3% directional accuracy against a 65.4% always-up share, RNN
57.3% against 65.35%. An "always predict up" rule with no model at all calls
direction better than either network does at six months. h=4 and h=13
unconditional, and all three horizons conditional, clear their respective
always-up/always-down benchmarks with margins of 0.75 to 16.1 percentage
points. Full breakdown, including per-period and per-market scopes, in
`outputs/directional_benchmark.csv`.

**Consequence.** The unconditional arm is the one that describes deployment
(D-15); any six-month directional claim made from it is void as currently
reported and needs correcting wherever it appears, including any prior
reference to build 2's 60.3/62.1% h=13 direction being read as evidence of
skill without the corresponding always-up share alongside it. The h=13
figures do clear their benchmark (60.4% and 62.1% against a 59.6% always-up
share) but only by 0.75 and 2.5 points, on the seed-median prediction. No
seed-level spread of directional accuracy itself has been computed (D-18's
`seed_variance.csv` carries per-seed MAE, not per-seed direction), so whether
a 0.75-point margin survives seed choice is open and should be checked before
the h=13 unconditional claim is quoted without qualification.

**Cost.** None to compute; this is a five-minute check against data already
produced. Not run earlier because directional accuracy had not been
questioned against anything but 50%.

---

## D-20. Seed-to-seed spread of the headline metric, and per-market verdict stability

**Decision/finding.** `src/seed_analysis.py` computes MAE per seed (not the
spread of individual predictions, which `pred_sd` already reports and which
is a different, smaller number), the median-ensemble MAE, a margin-to-noise
ratio (margin over naive divided by the standard deviation of per-seed MAE),
a seed-count curve, and, per market, whether the beats-naive verdict changes
depending on which single seed's fit is used.

**Result.** Both unconditional-arm h=4 runs have a margin over naive below
one seed-MAE standard deviation: GRU 0.86, RNN 0.63. Every other run/horizon
clears 1 comfortably (2.1 to 24.9). The seed-count curve has flattened by
k=7 everywhere (mean ensemble MAE moved under 0.5% from k=6 to k=7), so
seven seeds looks adequate for the aggregate metric.

Per-market verdict stability is worse than the aggregate numbers suggest. At
h=4 unconditional, 9 of 15 markets (GRU) and 10 of 15 (RNN) change their
beats-naive verdict depending on which single seed is used; at h=26
unconditional, 9 of 15 (GRU) and 7 of 15 (RNN). Conditional is far more
stable: 0 to 3 of 15 markets flip at any horizon. Full detail in
`outputs/seed_analysis.csv` and `outputs/seed_count_curve.csv`.

**Consequence.** A per-market ranking or claim ("this market beats the
incumbent") drawn from the unconditional arm at h=4 or h=26 is not defensible
as reported: a majority of markets' verdicts are a coin flip over which of
the seven fitted seeds happened to be looked at. The seed-median aggregation
already in `run.py` (D-11) is the right response to this, and is being used;
the finding here is that per-market publication needs the same treatment
applied per market, not inferred safe because the pooled run uses it. h=13
unconditional and the whole conditional arm are comparatively stable (0 to 6
flips of 15).

**Cost.** None to compute beyond reading data already on disk. The seed-count
enumeration is exact (all C(7,k) subsets, at most 35) rather than a
random sample, so it is exhaustive at this seed count and would need to
change if the seed count grows past roughly 10-12, where C(n,k) stops being
cheap to enumerate in full.

---

## D-21. Reproducibility confirmed; proxy-target sensitivity is small

**Reproducibility.** `build2.yaml` RNN unconditional, smoke settings, run
twice from a `git worktree` checked out at the `pre-fixes` tag (the code as
it stood before any Task 1-2 fix in this session), same data, same seeds,
default (unpinned) thread count. `diff -rq` between the two output
directories: zero differences, every file byte-identical, including
`forecasts.csv`'s raw per-seed floating point predictions. The determinism
claim holds, at least under CPU execution with the thread configuration used
here; this does not test GPU/MPS determinism, which was not exercised.

**Proxy-target sensitivity.** `src/proxy_sensitivity.py` reuses
`error_analysis.py`'s existing panel join and reruns `aggregate()` with rows
where `is_proxy_target` is true excluded, then reports the shift per run and
horizon. 132 of 1,592 (or 1,590 for `RNN_unconditional`, D-17) scored rows
per run have a proxy-filled target actual, 528 across the four runs, 2.76%
of all scored rows. Excluding them shifts challenger MAE by +0.13 to +0.31
NGN/kg (always worse, i.e. proxy-filled actuals are on average slightly
easier to hit) and directional accuracy by at most 0.28 percentage points
in either direction. Full table in `outputs/proxy_sensitivity.csv`.

**Note on the "233" figure in the work order.** That count does not match
`is_proxy_target` alone (132 per run). It is close to rows proxy-filled at
target *or* origin (231 measured here), which is a different, wider
definition than the task's stated instruction to exclude on `is_proxy_target`.
The rescoring here follows the instruction as written (target only); anyone
reconciling this against the "233" figure should check which definition it
was computed under.

**Consequence.** Neither check changes a headline figure. Proxy sensitivity
is small enough that no reported MAE or direction figure needs a proxy
caveat beyond what is already logged in `Cleaning_Log`.

**Cost.** None; both are read-only checks against data already produced.

---

## D-22. New convention: `conditional_exog`, realised weather and fuel without upstream price

**Decision.** `--convention conditional_exog` in `src/run.py`. `build_flat` in
`src/data.py` gained `realised_upstream: bool = True`; when false, the
realised-driver block still supplies diesel log-return and mean rainfall/NDVI
over the forecast window, but not the neighbouring market's realised price
return. `flat_feature_names` takes the same parameter so
`run_metadata.json` cannot misreport what the model saw.
`use_realised = convention.startswith("conditional")`,
`realised_upstream = (convention == "conditional")`, so the existing
`conditional` arm is unchanged and `conditional_exog` is the new middle arm.

**Why.** The upstream feature in the `conditional` arm is another market's
maize price over the forecast window, which is closer to supplying part of
the answer than to supplying next season's rainfall. Separating it from the
other three realised drivers (fuel, rainfall, NDVI, none of which are maize
prices) tells us what share of the foreknowledge advantage the upstream
price alone was providing versus genuinely exogenous weather/fuel foresight.

**Verified before training.** Flat feature counts: unconditional 6,
`conditional_exog` 15, `conditional` 18, matching the specification exactly.
Smoke run confirmed `run_metadata.json`'s `flat_feature_names` lists the
9 realised names (diesel + rain + NDVI per horizon, no upstream) and
`n_flat_features: 15`; head parameter count rose from 10,243
(unconditional, 6 flat) to 10,819 (`conditional_exog`, 15 flat), consistent
with 9 extra scalar inputs into a 64-wide head layer.

**Cost.** Two more full runs (RNN, GRU) at the full grid, ~40 min each on
this machine per the original estimate for a comparable run.

**Result, both architectures run, and it is not what "what share of the
advantage" implies.** `conditional_exog` is worse than `unconditional` at
every horizon for both architectures, not merely worse than full
`conditional`:

| h | RNN uncond | RNN exog | RNN cond | GRU uncond | GRU exog | GRU cond |
|---|---|---|---|---|---|---|
| 4 | 17.78 | 18.68 | 14.65 | 17.64 | 18.65 | 14.57 |
| 13 | 34.49 | 40.93 | 25.52 | 34.24 | 40.49 | 25.27 |
| 26 | 50.63 | 67.13 | 37.95 | 53.69 | 60.83 | 38.25 |

(challenger MAE, NGN/kg, full grid). The gap between `exog` and
`unconditional` grows with horizon and is largest for RNN at h=26 (67.13 vs
50.63, 32% worse). The question this task set out to answer -- "what share
of the foreknowledge advantage was the upstream price alone providing" --
does not have a percentage-split answer, because the premise that
`conditional_exog` sits between `unconditional` and `conditional` is false
here. The upstream feature is not contributing part of the benefit; without
it, the other three realised drivers (diesel, rainfall, NDVI) make the
model worse than having no forecast-window driver information at all.

**Interpretation offered, not asserted.** Diesel and rainfall/NDVI realised
values are weaker, noisier predictors of a maize price than another maize
price is. Upstream may have been doing double duty: supplying its own
information and anchoring the other three so the head does not overfit
three weak, forecast-window signals in the same 15-feature-vs-64-hidden-unit
regime that D-05/D-09 already flag as parameter-thin. This is offered as a
plausible mechanism, not confirmed; distinguishing it from an interaction
with build 2's regularisation (calibrated against the 18-feature
`conditional` set, not 15-feature `conditional_exog`) would need a separate
run and is not done here.

---

## D-23. Build 3: remove the overfitting safeguards, raise capacity

**Decision.** `configs/build3.yaml`, copied from `build2.yaml` with six
changes and nothing else: `hidden: 128` (was 64), `head_dropout: 0.0` (was
0.2), `input_dropout: 0.0` (was 0.1), `weight_decay: 0.0` (was 1e-4),
`recency_half_life_weeks: null` (was 156, uniform weighting confirmed by
`walkforward.recency_weights`'s existing `None`/`0` -> uniform branch),
`purge_weeks: 26` (was 78).

**Why.** Four of build 2's changes (D-08 purge, D-09 dropout/weight decay,
D-12 recency weighting) were added against overfitting. The Week 0
diagnostics point at under-learning instead: 7-11% of price movement
explained, predictions varying a third to two thirds as much as reality, and
more training data not helping. Removing them together is deliberate, not
six independent bets: all six were justified by the same overfitting premise,
so testing that premise means removing all six at once. Doing them one at a
time would take six runs to answer one question that one run can answer
directly, at the cost of not knowing which of the six mattered if the result
changes.

**Success criterion, fixed before looking.** Build 3 succeeds only if
directional accuracy at h=13 rises above build 2's 60.3% (RNN) and 62.1%
(GRU) *while average error does not worsen*. Error improvement alone is not
success: a flatter forecast lowers MAE while sinking direction, which is the
failure mode already visible in build 2 (D-19).

**Verified before training.** Smoke run, RNN unconditional: purge applied at
the full 26 weeks with no fallback (`split_log.csv`), 36,739 parameters (up
from 10,243 at hidden=64), pipeline runs end to end.

**Cost.** Two full runs (RNN, GRU unconditional), ~50 min each per the
original estimate, longer than build 2 because capacity is doubled. If
direction recovers, `conditional_exog` runs follow for build 3 as well
(not run here yet).

**Result: success, both architectures, by the criterion fixed above.**

| h | RNN b2 MAE | RNN b3 MAE | RNN b2 dir% | RNN b3 dir% | GRU b2 MAE | GRU b3 MAE | GRU b2 dir% | GRU b3 dir% |
|---|---|---|---|---|---|---|---|---|
| 4 | 17.78 | 16.91 | 56.47 | 59.51 | 17.64 | 16.47 | 56.97 | 59.94 |
| 13 | 34.49 | 31.71 | 60.37 | **61.96** | 34.24 | 30.88 | 62.07 | **63.01** |
| 26 | 50.63 | 49.07 | 57.30 | 55.67 | 53.69 | 48.74 | 54.30 | 56.06 |

At h=13, the horizon the success criterion is fixed on: both architectures'
direction rises above their own build 2 figure (RNN 60.37% -> 61.96%, GRU
62.07% -> 63.01%) while MAE falls, not rises, at every horizon for both
architectures. h=26 direction is mixed (RNN falls 57.30% -> 55.67%, GRU
rises 54.30% -> 56.06%), but the criterion was fixed on h=13 specifically,
before looking, and both architectures clear it there.

**Consequence.** Per the source task's own trigger -- "if direction
recovers, run `conditional_exog` for build 3 as well" -- that follow-up is
launched (`configs/build3.yaml`, `--convention conditional_exog`, RNN and
GRU). Given D-22's `conditional_exog` result for build 2 (worse than
unconditional for both architectures), whether that holds, reverses, or
is unaffected under build 3's higher capacity and lighter regularisation is
now an open, answerable question rather than an assumption carried over
from build 2.

**Answered: the pattern holds under build 3 too.** RNN build3
`conditional_exog` MAE 17.20/34.52/58.57 versus `unconditional`
16.91/31.71/49.07 -- worse at every horizon. GRU build3 `conditional_exog`
17.54/35.56/54.88 versus `unconditional` 16.47/30.88/48.74 -- worse at
every horizon again. Four of four architecture/build combinations now show
`conditional_exog` underperforming `unconditional`. This rules out D-22's
tentative "regularisation calibrated for the wrong feature count"
explanation as the sole cause, since build 3 removes exactly that
regularisation and the pattern is unchanged: something about the
diesel/rainfall/NDVI realised-driver features without the upstream anchor
is a net negative for this architecture family regardless of capacity or
regularisation. Worth investigating on its own terms rather than folded
into the capacity/regularisation question build 3 was designed to answer.

---

## D-24. Split conformal intervals, and 2023-24 coverage breaks down as expected

**Decision.** `src/intervals.py`, per run and horizon: origins split
chronologically (never shuffled) into calibration (first 60% of distinct
origin dates) and test (the rest); residuals normalised by origin price
before the finite-sample split-conformal quantile is taken
(`k = ceil((n+1)(1-alpha))/n`); interval = pred +/- qhat * origin_price.
**The 60/40 calibration/test split is a default I chose, not specified in
the source task** -- it is not the same split as the three reporting
regimes (2015-19/2020-22/2023-24) below, which are a separate, coarser
bucketing applied on top for the regime-uncertainty report.

**Headline (test portion, out-of-sample).** Realised coverage tracks nominal
reasonably across all twelve run/horizon combinations: 72.1-93.6% against an
80% nominal at alpha=0.20, 87.3-93.6% against 90% at alpha=0.10. Mean width
in NGN/kg rises with horizon as expected (roughly 70-100 at h=4 to 210-315
at h=26) and is wider for the unconditional arm than conditional at every
horizon, since the model is doing more work without driver foreknowledge.

**Regime breakdown, the more informative view.** 2015-19 is 100%
calibration by construction (it is the earliest data, so the chronological
split puts nearly all of it before the cutoff) and 2023-24 is 0% calibration
everywhere, fully out-of-sample. Coverage there is the real test of the
method, and it fails for the unconditional arm exactly where the Week 0
diagnostics would predict: GRU-unconditional h=26 realised coverage 60.06%
against 80% nominal (20-point gap), h=13 67.94% (12-point gap), RNN-
unconditional h=4 64.71% (15-point gap). The conditional arm's 2023-24
coverage stays close to nominal throughout (76.9-86.5%), consistent with
driver foreknowledge absorbing part of the regime shift that the
unconditional arm cannot see coming. Full table in
`outputs/intervals_by_regime.csv`; reported as found, not widened to hide
the gap, per the task's own instruction.

**Three uncertainty components, reported separately.**
- *Model* (seed spread, from D-20/Task 4): 0.13 to 2.08 NGN/kg standard
  deviation of MAE across seeds -- small next to the other two.
- *Data* (conformal width): 71 to 315 NGN/kg mean interval width depending
  on run and horizon -- the dominant term by roughly two orders of
  magnitude over model uncertainty.
- *Regime* (coverage by period): holds near nominal in 2015-19 and 2020-22,
  breaks down in 2023-24 for the unconditional arm as above.
Collapsing these into one number would have hidden that regime uncertainty,
not model uncertainty, is where this method's assumptions are weakest.

**Per-market caution honoured.** No per-market quantile is fit; every
market uses the pooled (run, horizon) qhat. `outputs/intervals_by_market.csv`
reports each market's realised coverage under that pooled interval as a
diagnostic, not a separately calibrated guarantee -- a per-market scale
factor (mentioned as the intended refinement in the source task) is not
implemented here.

**Cost.** None to compute beyond reading existing run outputs; no
retraining. The 2023-24 finding is a reason for caution about deployment in
the current or a future regime shift, not a defect in the interval method
itself -- split conformal's marginal coverage guarantee is unconditional
over the calibration distribution and is not expected to hold under
distribution shift, which 2023-24 is.

---

## D-25. Driver forecasts, first cut, and a mandated method that loses to carry-forward

**Decision.** `src/data.py` gains `climatology_forecast`, `diesel_forecast`
(random walk with drift or carry-forward), and
`upstream_forecast_seasonal_naive`, all computable at the origin with no
look-ahead (climatology uses only years strictly before the origin;
seasonal-naive for upstream uses `target - 52 <= origin` whenever h <= 52,
so it never reaches past the origin). `build_flat` gained
`driver_source: "realised" | "forecast"`; a new `--convention
forecast_drivers` in `run.py` uses the forecast path for all four driver
quantities (diesel, upstream, rainfall, NDVI), the same shape as
`conditional` (18 flat features) but with every value something the model
could actually have at deployment time. `flat_feature_names` prefixes
`forecast_` instead of `realised_` under this convention so
`run_metadata.json` cannot misreport a forecast as a realised value (a bug
caught and fixed during this task, before any run used it).

**Diesel method, validated as instructed.** Random walk with drift beats
plain carry-forward at every horizon and pooled: MAE 49.196 vs 53.157
NGN/kg pooled, winning at h=4 (22.82 vs 23.11), h=13 (47.24 vs 50.39), and
h=26 (77.52 vs 85.97). `DIESEL_FORECAST_METHOD = "rw_drift"` in `data.py`
matches this result; `src/driver_forecasts.py` reruns the comparison and
prints a reminder to check the two stay consistent if the panel changes.

**Rainfall and NDVI climatology, confirmed strongly seasonal.** Climatology
beats carry-forward by a wide margin at every horizon: rainfall MAE 4.3-5.0
vs 10.2-37.8; NDVI 0.018-0.019 vs 0.059-0.252. Consistent with the task's
own expectation that strongly seasonal variables should beat carry-forward
substantially.

**Finding not asked for but discovered: upstream seasonal-naive loses to
carry-forward, badly, at every horizon.** MAE 60.7 vs 20.1 (h=4), 67.0 vs
39.0 (h=13), 80.4 vs 56.3 (h=26) NGN/kg -- carry-forward is 2-3x more
accurate. The source task specifies seasonal-naive for upstream without
asking for the carry-forward comparison it required for diesel; that
comparison was run anyway because the two methods were already being
computed side by side for the report, and the result is one-sided enough to
flag rather than bury. Plausible reason: maize prices carry a strong trend
(2023-24 inflation and fuel-subsidy removal put current levels far above a
year-ago level), so "52 weeks ago" is a stale anchor exactly where the panel
has moved the most, while carry-forward at least starts from the current
level. **Implemented as specified (seasonal-naive) regardless**, since the
task fixed the method rather than asking for validation, but this is a
material caveat on `forecast_drivers`: its upstream input is measurably
worse than the simplest possible alternative, so any shortfall in
`forecast_drivers` versus `conditional_exog` should be checked against this
before being read as "driver forecasting doesn't help" -- some of the gap
may be an avoidably bad upstream method rather than a ceiling on
forecastability itself.

**Verified before training.** Feature counts and names checked:
`forecast_drivers` produces 18 flat features named `forecast_*` (not
`realised_*`); smoke run trains end to end, 11,011 parameters.

**Cost.** Two full runs (RNN, GRU, `forecast_drivers`, full grid) to
complete the four-row headline table (operational / forecast_drivers /
conditional_exog / foreknowledge) that the source task asks for; not
complete at time of writing this entry.

**Result: the first cut sizes a negative prize.** Both runs complete.
Challenger MAE, NGN/kg, full grid:

| h | RNN operational | RNN forecast_drivers | RNN conditional_exog | RNN foreknowledge |
|---|---|---|---|---|
| 4 | 17.78 | 18.13 | 18.68 | 14.65 |
| 13 | 34.49 | 37.14 | 40.93 | 25.52 |
| 26 | 50.63 | 63.93 | 67.13 | 37.95 |

| h | GRU operational | GRU forecast_drivers | GRU conditional_exog | GRU foreknowledge |
|---|---|---|---|---|
| 4 | 17.64 | 18.71 | 18.65 | 14.57 |
| 13 | 34.24 | 38.01 | 40.49 | 25.27 |
| 26 | 53.69 | 65.29 | 60.83 | 38.25 |

`forecast_drivers` is worse than `operational` at every horizon, both
architectures. The gap between rows one and two -- what better driver
forecasts were supposed to buy in practice -- is negative: -0.35 to -13.30
NGN/kg depending on horizon and architecture. The gap between rows two and
four -- what remains on the table -- is therefore larger than the gap
between rows one and four (operational to foreknowledge), because the
first-cut driver forecasts actively subtract value rather than recovering
part of the foreknowledge gain. `forecast_drivers` beats `conditional_exog`
for RNN at every horizon but is mixed against it for GRU (better at h=13,
worse at h=4 and h=26).

**Why, most likely: the mandated upstream method (D-25 above).** Upstream
seasonal-naive was already shown to lose to plain carry-forward by 2-3x
(60.7 vs 20.1 NGN/kg at h=4). `forecast_drivers` is built exactly as
specified, seasonal-naive included, so a materially worse upstream forecast
than the simplest alternative is baked into the one run that was supposed
to size the prize. This does not mean the prize is genuinely negative; it
means this first cut cannot distinguish "driver forecasting doesn't help"
from "this specific first-cut method, using a mandated component already
shown to underperform carry-forward, doesn't help." Re-running
`forecast_drivers` with carry-forward substituted for upstream (a one-line
change, `realised_upstream`-style branch already exists in `build_flat`) is
the natural next check before concluding anything about the underlying
prize -- not done here because the task specified seasonal-naive rather
than asking for this validation the way it did for diesel.

**Each driver forecast's own accuracy, from `src/driver_forecasts.py`**
(against realised, pooled across the scored grid): diesel RW+drift MAE
22.8/47.2/77.5 at h=4/13/26 (beats carry-forward, D-25 above); rainfall
climatology 4.3-5.0 (beats carry-forward by 2-8x); NDVI climatology
0.018-0.019 (beats carry-forward by 3-13x); upstream seasonal-naive
60.7-80.4 (loses to carry-forward by 2-3x, the outlier). Three of four
driver forecasts are genuinely informative; the run result is nonetheless
negative, consistent with the fourth (mandated, not chosen) component being
weak enough to dominate the outcome.

**Consequence for the "commission the multi-week version" decision.** As
measured, this first cut argues against committing further multi-week work
to rainfall/fuel/neighbouring-price forecasting models on the strength of
this result alone -- but the result is confounded by one mandated,
already-known-weak component. The task's own purpose for the first cut
("sizes the prize... before anyone commits") is only partly served: it
correctly identifies that naive driver-forecast substitution is not a free
win, but it cannot yet separate "the prize is small or negative" from "the
upstream method needs fixing first." Recommend the one-line carry-forward
substitution as a cheap follow-up before treating this as the final answer
on whether to commission further driver-forecasting work.

---

## D-26. Grid consistency completed: GRU unconditional rerun with D-17, operational vs safeguard-removed compared in full

**Decision.** `GRU_unconditional` (build2) reran with the D-17 origin-date
fix already in the code (it had been sitting on the pre-fix 1,592-pair
grid since D-17 landed after its original run). All four "operational"
(build2 unconditional) and "safeguard-removed" (build3 unconditional) runs
-- RNN and GRU, both settings -- are now on the same 1,590-pair grid.
`RNN_conditional` and `GRU_conditional` (build2, foreknowledge) remain on
the pre-fix 1,592-pair grid; left as is, since foreknowledge is not the
current priority and re-running them was explicitly out of scope for this
pass.

**Full comparison, MAE and MAPE, all four runs, same grid:**

| h | RNN operational | RNN safeguard-removed | GRU operational | GRU safeguard-removed |
|---|---|---|---|---|
| 4 (MAE) | 17.78 | 16.91 | 17.66 | 16.47 |
| 13 (MAE) | 34.49 | 31.71 | 34.28 | 30.88 |
| 26 (MAE) | 50.63 | 49.07 | 53.74 | 48.74 |
| 4 (MAPE) | 9.65 | 9.45 | 9.78 | 9.27 |
| 13 (MAPE) | 18.01 | 17.57 | 18.15 | 16.96 |
| 26 (MAPE) | 24.85 | 26.42 | 25.81 | 25.11 |

**Safeguard-removed vs operational, same architecture (MAE delta, negative
is an improvement):** RNN -0.87/-2.78/-1.57 at h=4/13/26; GRU
-1.19/-3.40/-5.00. Both architectures improve at every horizon under
safeguard-removed settings on this panel, GRU by a larger margin at h=26.
Matches D-23's finding, now confirmed on the grid-consistent data.

**GRU vs RNN, same setting (MAE delta, negative is GRU better):**
operational -0.12/-0.22/+3.11 (GRU better at h=4/13, RNN better at h=26 by
3.11 NGN/kg); safeguard-removed -0.44/-0.83/-0.33 (GRU better at every
horizon, though by less than one NGN/kg at h=4 and h=26). No architecture
dominates cleanly; GRU has a slight, mostly small edge, except operational
h=26 where RNN is meaningfully ahead. MAPE tells a mixed story too:
RNN safeguard-removed has the highest h=26 MAPE (26.42%) despite a lower
MAE than GRU operational (53.74) at that horizon, since MAPE and MAE do not
always rank the same way (CLAUDE.md's own caution, restated by
`error_analysis.py`'s README: MAPE weights the low-price years more).

Full table: `outputs/taskA_operational_vs_safeguard_removed_comparison.csv`.

**Not pushed.** Per instruction, none of this session's commits are pushed
to GitHub until reviewed.

---

## D-27. AFEX multi-commodity farmgate panel: new workstream, price-only, no incumbent

**Decision.** New loader (`src/afex_data.py`) and runner (`src/run_afex.py`)
for `20260824_AFEX_MultiCommodity_Weekly_Panel_v1.xlsx`: 51 series across 18
markets and 7 commodities, 277 weeks (2021-04-07 to 2026-07-22), price only.
Built as a separate pipeline rather than bent into the FEWSNET-specific one,
because the two datasets differ in three structural ways: no exogenous
driver data exists here at all (the source file's own README says to start
price-only and add drivers later); no incumbent forecast file exists, so
the evaluation grid is generated directly from the panel
(`generate_afex_grid`), not read from a baseline file (this is not the D-01
mistake, since there is no incumbent grid here to diverge from); and every
(market, commodity) pair is treated as its own embedding series (e.g.
"Anchau | Maize", "Anchau | Sorghum"), so all 51 series train the pooled
model but only the 16 maize series are scored (mirrors D-03's precedent:
Aba trains, is not scored).

**Data handling decisions.**
- Diesel, upstream, rainfall and NDVI sequence channels are zero-filled so
  `build_sequence`/`build_flat`/`build_training_windows` from `data.py`
  reuse unchanged; the model sees these as constant, uninformative channels.
- `Dandume | Maize` clears the source panel's own 112-week entry bar but
  has zero usable 78-week (lookback 52 + max horizon 26) windows after
  new-crop removal, flagged in the source file's own Build_Decisions sheet
  as needing a decision. Dropped from training and scoring: a series that
  can never produce an h=26 window contributes nothing and cannot be scored
  at that horizon.
- Fourier terms computed directly from the date index (period 52.18, k=2),
  matching the convention in the original `build_panel.py` this project
  descends from, since this panel carries no pre-computed sin/cos columns.

**Two settings run, both architectures, full grid:**

| Run | h=4 MAE | h=13 MAE | h=26 MAE | h=4 MAPE | h=13 MAPE | h=26 MAPE |
|---|---|---|---|---|---|---|
| RNN operational | 59.93 | 131.20 | 179.91 | 13.00 | 25.24 | 34.53 |
| GRU operational | 63.50 | 135.85 | 162.67 | 13.72 | 26.31 | 31.07 |
| RNN safeguard-removed | 62.59 | 153.53 | 233.74 | 13.47 | 29.68 | 46.21 |
| GRU safeguard-removed | 65.01 | 155.02 | 223.29 | 14.18 | 30.39 | 44.13 |

Naive (carry-forward) benchmark: 57.60 / 135.37 / 177.74 MAE at h=4/13/26.

**Finding: the main panel's safeguard-removed result reverses here.**
Safeguard-removed is worse than operational at every horizon for both
architectures on this panel -- the opposite of D-23's result on the main
FEWSNET panel, where it improved MAE everywhere. Plausible reading: D-23's
diagnosis (under-learning, so remove the anti-overfitting safeguards and
raise capacity) was reached on a panel with roughly 5,000 training windows
per cut by the later years; this panel's pooled training set peaks around
2,850 windows at the very last cut and is far smaller earlier. A larger,
less-regularised model on a fifth of the data (by span) and a much thinner
per-series history looks like it is overfitting rather than under-learning.
Operational settings (build2-equivalent) are the better choice on this
panel as measured, which is itself informative: the right amount of
regularisation is a function of how much data is available, not a fixed
property of the RNN/GRU architecture family.

Neither setting clearly beats carry-forward: operational is worse at h=4,
roughly level at h=13, and RNN operational beats naive at h=26 (179.91 vs
177.74, essentially a tie) while GRU operational is genuinely better there
(162.67 vs 177.74, +8.5%). This is a much harder result than the main panel
produced, consistent with 277 weeks of history being a fraction of the
main panel's twelve years.

**A split-logic interaction worth flagging.** Several operational-setting
cuts show a validation set larger than the training set (e.g. RNN
operational cut 11: train=212, val=397). `chronological_split` fixes
validation at 15% of the pre-purge window count and only prunes training
windows near the cutoff; with `purge_weeks=78` on a panel this short, and
per-series histories that start and end unevenly across the five-year span
(several maize series stop reporting in 2024), purge can remove a very
large share of the training pool while validation stays fixed. This never
happened on the main 12-year panel, where the training pool was always
large relative to a 78-week purge. Early-stopping decisions in the affected
cuts should be read cautiously. Safeguard-removed (`purge_weeks=26`) does
not show this issue in any cut.

**Deliverable.**
`outputs/20260910_AFEX_PredictedVsActual_OperationalVsSafeguardRemoved_v1.xlsx`:
headline MAE/MAPE by horizon, by calendar quarter, by maize series, and a
full predicted-vs-actual detail sheet (10,740 rows) for every scored
forecast across all four runs.

**Roadblocks and open decisions, for the owner.**
1. No exogenous variables decided yet (explicitly deferred by the owner);
   this run is price-only throughout.
2. The panel file itself lives outside the repo
   (`~/Downloads/20260824_AFEX_MultiCommodity_Weekly_Panel_v1.xlsx`,
   referenced by absolute path in the two new configs) and has not been
   copied into `data/`. Worth deciding whether it should be, for
   reproducibility on another machine.
3. Giwa | Maize is flagged by the source panel itself as a near-duplicate
   of Anchau | Maize (correlation +0.947 after an 81-week repair); it is
   currently kept in both training and scoring, unfiltered, matching the
   source file's own default. A decision to exclude it would be a one-line
   change in `afex_data.py`.
4. `RNN_conditional`/`GRU_conditional` on the main FEWSNET panel remain on
   the pre-D17 grid (D-26); not touched in this pass.
5. **Not pushed.** All commits in this entry and D-26 are local only, per
   instruction, pending review.

---

## D-28. AFEX workstream: safeguard-removed discontinued, operational only going forward

**Decision.** No further AFEX runs under `configs/afex_safeguard_removed.yaml`.
`configs/afex_operational.yaml` is the only setting used from here on for
this panel. This mirrors D-16's precedent on the main panel (version 1
discontinued, its config and outputs retained as evidence): both
`afex_safeguard_removed` run folders and its config are kept, unchanged,
as the only evidence of what that setting did here.

**Why.** D-27 already found safeguard-removed worse than operational at
every horizon, both architectures, on this panel -- the reverse of the
main panel's result. The owner's read, which this decision follows: real
need for the four overfitting safeguards on a panel this size, so there is
no live question left for safeguard-removed to answer here. Running it
further consumes compute without informing a decision. If the panel grows
substantially (more history, or exogenous variables that change the
effective sample size per cut), that premise could change and
safeguard-removed could be worth revisiting -- the config is kept for
exactly that reason.

**Everything from here (price direction analysis, per-market error
analysis) uses the operational runs only** (`outputs/afex_operational/RNN`
and `outputs/afex_operational/GRU`), reading their existing saved
forecasts rather than retraining.

---

## D-29. Price direction analysis: operational models mostly fail against the panel's own move share

**Decision/finding.** `src/afex_directional.py` scores each operational
run's directional accuracy against the panel's own always-up/always-down
share (the correct benchmark, not 50% and not the do-nothing forecast --
see below), per horizon and per market, maize only.

**Why "vs the do-nothing variant" needed reframing.** The do-nothing
(carry-forward) forecast predicts exactly zero change at every origin by
construction, so it never asserts a direction at all: scored the same way
the model is, its directional accuracy is 0% on every window where price
actually moved. That is not a real comparison, it is a restatement that a
flat forecast has no opinion. The comparison that means something, and the
one already used for the main panel (D-19), is against the share of
windows where price actually rose or fell.

**Overall result, maize, all markets pooled:**

| Run | h | Model direction % | Always-up share % | Beats always-up/down? |
|---|---|---|---|---|
| RNN operational | 4 | 40.82 | 59.66 | No |
| RNN operational | 13 | 61.42 | 58.66 | **Yes** (+2.8 pts) |
| RNN operational | 26 | 42.89 | 64.47 | No |
| GRU operational | 4 | 35.15 | 59.66 | No |
| GRU operational | 13 | 52.42 | 58.66 | No |
| GRU operational | 26 | 56.55 | 64.47 | No |

**5 of 6 combinations fail.** Only RNN at h=13 clears the benchmark.
GRU at h=4 scores 35.15%, below a 50% coin flip and 24.5 points below the
59.66% always-up share -- worse than useless at calling direction at that
horizon, not merely unhelpful. RNN is the stronger architecture for
direction on this panel at every horizon (40.8/61.4/42.9 vs GRU's
35.2/52.4/56.6), the reverse of the error-rate picture in D-27 where
neither architecture dominated cleanly.

**Per-market detail makes the always-up bar itself informative.** Several
of the thinner, earlier-truncated maize series (Ikara, Ikara ends
2025-10-08; Garbabi, Gazabu, Jalingo, Leggal, all ending in 2024) show
always-up shares of 70-100% at h=26, because their last scored windows sit
in a narrow, consistently-rising late stretch of a short series. Beating a
90%+ base rate is a high bar by construction, not a fair test of the
model at those series specifically; the pooled figures above are a fairer
read than any single thin market's row. Full detail in
`outputs/afex_directional_by_market.csv`.

**Consequence.** As it stands, neither operational model has a
directional edge worth reporting outside of RNN at three months. This
sits alongside D-27's error-rate finding (GRU edges out on MAE at h=26,
RNN roughly ties on MAE elsewhere): the two metrics do not point at the
same architecture, which is itself worth keeping in mind before picking
one over the other for this panel.

---

## D-30. Per-market error analysis, operational runs, maize only

**Decision.** Consolidated `metrics_by_market.csv` from both operational
runs into `outputs/afex_operational_per_market_error.csv`, RNN and GRU
side by side, per horizon. No commodity breakdown beyond maize: the other
six commodities train the pooled embedding but are never scored, so there
is no actual-vs-predicted to compare them against (see D-27).

**Result.** At h=26, GRU has lower MAE than RNN at every one of the 14
maize markets, matching D-27's aggregate finding that GRU pulls ahead at
the longest horizon. At h=4 and h=13 the picture is mixed and RNN wins in
most markets, though not uniformly.

`Ikara | Maize` is the weakest market at every horizon by a wide margin
(MAE 333.53 GRU / 366.76 RNN at h=26, roughly double the next-weakest
market), and also the thinnest (24 scored windows, series ends
2025-10-08). `Pambegua | Maize` looks like the strongest market at every
horizon, but rests on only 5 scored windows -- too few to trust as a
genuine result rather than a favourable sample. Per this project's own
established caution (`error_analysis.py`'s README note on the main panel:
"treat per-market rankings as indicative, not decisive"), the same applies
here more strongly, since several AFEX maize series carry under 15 scored
windows per horizon.

**Cost.** None; reused existing saved forecasts, no retraining.

---

## D-31. Sibling-commodity correlation screen: sorghum is the clear choice for maize

**Decision/finding.** Computed correlation between maize log-returns and
each co-located sibling commodity's log-returns, at 1, 4 and 13-week
change horizons (matching the source panel's own Independence-sheet
methodology), for every market that has both series. Pooled by commodity
(n-weighted mean across markets):

| Commodity | 1-week | 4-week | 13-week | Markets paired with maize |
|---|---|---|---|---|
| Sorghum | 0.538 | 0.703 | **0.859** | 7 of 15 scored (Anchau, Dawanau, Ikara, Kumo, Pambegua, Saminaka, Tundun Saibu) |
| Millet | 0.189 | 0.431 | 0.682 | 1 (Dawanau only) |
| Paddy_Rice | 0.356 | 0.477 | 0.642 | 3 (Jengre, Pambegua, Saminaka) |
| Cowpea | 0.390 | 0.488 | 0.597 | 2 (Kumo, Leggal) |
| Soybean | 0.239 | 0.388 | 0.454 | 14 (present at nearly every maize market) |
| Sesame | 0.132 | 0.108 | 0.143 | 2 (Dawanau, Jalingo) |

**Sorghum wins clearly at every horizon**, by a wide margin over the
second-best (soybean, which is more widely available but much less
informative). Per-market detail: `outputs/afex_maize_sibling_correlation_detail.csv`.
Individual pairs go as high as 0.93 (Dandume, dropped from scoring but
still informative as a data point) and 0.89 (Dawanau). Plausible reading:
maize and sorghum are grown across an overlapping growing season in this
region and are partial demand substitutes, so they carry a shared
regional supply/demand signal that soybean, on a different cycle, does
not track as closely.

**Consequence.** Sorghum was chosen as the exogenous sibling-commodity
driver for the next build (D-32), not soybean, despite soybean's wider
market coverage -- correlation strength was the deciding factor per the
owner's ask ("which sibling commodity would have the best effect"), not
coverage. This leaves 8 of 15 scored maize markets without a sibling
driver at all (Bali, Danja, Garbabi, Gazabu, Giwa, Jalingo, Jengre,
Leggal); a future build could revisit soybean or a market-specific choice
for those.

**Cost.** None; read-only analysis over the panel already loaded.

---

## D-32. Sorghum exogenous driver built and run: a clean negative result

**Decision.** `configs/afex_operational_sorghum.yaml`, identical to
`afex_operational.yaml` except `data.sibling_commodity: Sorghum`.
`afex_data.py`'s `load_afex_panel` gained a `sibling_commodity` parameter
that repurposes the existing upstream channel: for the 7 of 15 scored
maize markets that also have a sorghum series (Anchau, Dawanau, Ikara,
Kumo, Pambegua, Saminaka, Tundun Saibu), the channel carries that market's
own sorghum price history over the same 52-week lookback as maize's own
price -- origin-time only, never the forecast window, so this is
deployable information, not foreknowledge. No architecture change: the
model's input channel count is unchanged (10 channels, as before); only
what channel 2 actually contains differs. Sorghum chosen over every other
commodity by the correlation screen in D-31.

**Result: worse, not better, exactly where it was applied.** Restricting
to the 7 paired markets (`outputs/afex_sorghum_vs_operational_paired_split.csv`),
MAE rises at every horizon for both architectures when sorghum is added:

| | h=4 | h=13 | h=26 |
|---|---|---|---|
| RNN, no sibling | 54.01 | 126.11 | 202.93 |
| RNN, + sorghum | 55.30 | 128.76 | **213.84** |
| GRU, no sibling | 58.99 | 129.80 | 192.07 |
| GRU, + sorghum | 61.05 | 134.21 | **201.04** |

In the 8 unpaired markets, where the input is unchanged and only the
shared model's weights differ, the shift is smaller and mixed (GRU
improves slightly at h=4 and h=26, worsens at h=13; RNN worsens
throughout but by less than the paired markets do) -- consistent with
ordinary training-run variation from a shared pooled model, not a real
effect of the new information.

**Why a strongly-correlated series still hurts: a documented data-quality
gap, not a modelling contradiction.** The source panel's own
Build_Decisions sheet states plainly, under "Not done": maize needed six
separate defects fixed (emoji labels, glued naira, non-hybrid read as
hybrid, price before weight, dropped digits, interpolation across a
programme discontinuity); sorghum and the other five commodities "have
had only a plausibility band applied," and the same classes of fault
should be expected in them. Sorghum's 0.86 correlation with maize (D-31)
describes the SIGNAL two cleaner series would share; the series actually
fed into this model carries whatever the plausibility band let through
uncleaned. On a model already shown to be data-constrained on this panel
(D-27's operational-vs-safeguard-removed reversal), adding a correlated
but noisier input looks to cost more in injected noise than it returns in
shared signal.

**Consequence.** Do not adopt sorghum as an exogenous driver on this data
as it stands. The correlation screen (D-31) picked the right commodity by
the criterion asked for (co-movement with maize); the result here is a
statement about this particular sorghum series' data quality, not
evidence that a cleaner cross-commodity signal would fail the same way.
If sorghum (or another sibling commodity) is revisited, cleaning it to
the same standard maize received first is the more promising next step
than trying a different commodity or a different way of feeding the same
raw series in.

**Cost.** Two full runs (RNN, GRU), reusing the existing walk-forward
machinery unchanged.

---

## D-33. Rainfall/NDVI geography blocker resolved; agroclimatic build launched

**Decision.** Both `UN_DataExchange_NDVI_cleaned.xlsx` and
`UN_DataExchange_Rainfall_cleaned.xlsx` identify their 37 locations by
code (`NG001`-`NG037`) only. Whoever cleaned these files had already
flagged this correctly as a "GEOGRAPHY BLOCKER" in both workbooks' own
Cleaning Log sheets and declined to guess an ordering. Before proceeding,
I searched independently for a public, verifiable code-to-state mapping
(confirmed the code scheme matches UN OCHA's Nigeria COD-AB pcode system;
found partial admin2-count evidence for 2 of 37 codes via FAO microdata
that was suggestive but not conclusive) and did not find one I was
willing to act on without guessing. The owner then supplied a third file,
`20260806_NDVI with States_v0.xlsx`, which carries the same Location Code
alongside an explicit State column. Verified: 37 distinct codes, exactly
one state per code, no ambiguity (`NG001`=Abia ... `NG037`=Zamfara,
alphabetical). Applied to the rainfall file too, since it uses the
identical code scheme and is described in the NDVI file's own quality
scorecard as "the companion UN Rainfall source."

**Why this took a real check rather than an assumption.** Guessing wrong
here (e.g. an alphabetical ordering that happened to be off by one, or a
regional grouping) would have silently attached the wrong state's
rainfall to a market -- a corruption exactly as damaging as, and far
harder to detect than, D-01's grid-regeneration mistake. The blocker is
now resolved with a verified source, not an assumption.

**Merge.** `afex_data.py`'s `load_afex_panel` gained `ndvi_path`/
`rainfall_path`/`climate_state_map_path` (all required together).
Dekadal (10-day) series resampled to this panel's weekly grid by linear
interpolation, then forward-filled past each source's last real
observation (NDVI: 2026-05-11, rainfall: 2026-06-01) to cover the ~7-11
week tail gap to the panel's 2026-07-22 end -- a real, logged coverage
gap (`panel.log["agroclimatic"]`), not hidden. All 6 states the AFEX
maize markets sit in (Gombe, Kaduna, Kano, Katsina, Plateau, Taraba) are
covered, so every one of the 15 scored maize markets gets a real channel
-- unlike the sorghum build (D-32), which only reached 7 of 15.

**A code-level caution acted on, not just noted.** `build_sequence` has
no NaN-safety on the raw rainfall/NDVI channels (unlike diesel, which
degrades to zero via a safe log-ratio): any NaN in these arrays would
silently reject the entire window rather than degrade gracefully. The
arrays are zero-initialised and only ever overwritten with real, finite
values, verified by direct check before running anything
(`np.isfinite(...).all()` confirmed true for both channels across the
full panel).

**Build.** `configs/afex_operational_agroclimatic.yaml`: operational
hyperparameters, rainfall and NDVI as origin-time exogenous channels
(never the forecast window, so not foreknowledge), sibling-commodity
crosscheck (D-31/D-32) deliberately left out of this build -- one change
at a time, per instruction, so this result cannot be confounded with
sorghum's already-negative one. Smoke-verified: window counts identical
to the plain operational baseline (no windows silently dropped).

**Cost.** Two full runs (RNN, GRU), launched next; results in a
following entry once complete.

---

## D-34. Agroclimatic build result: small, mixed, and outlier-masked at h=13

**Decision/finding.** Both runs (`configs/afex_operational_agroclimatic.yaml`,
RNN and GRU) complete. All 15 scored maize markets got a real channel this
time (D-33), so no paired/unpaired split is needed the way D-32 required
for sorghum.

**Overall MAE, NGN/kg:**

| | h=4 | h=13 | h=26 |
|---|---|---|---|
| RNN, no exog | 59.93 | 131.20 | 179.91 |
| RNN, + agroclimatic | 60.84 | 130.74 | 182.39 |
| GRU, no exog | 63.50 | 135.85 | 162.67 |
| GRU, + agroclimatic | 62.87 | 134.56 | 167.39 |

Small shifts throughout, an order of magnitude smaller than sorghum's
effect (D-32). Both architectures improve slightly at h=13 (RNN -0.46,
GRU -1.29). At h=4 the architectures disagree (RNN +0.91 worse, GRU -0.63
better); at h=26 both worsen (RNN +2.48, GRU +4.72).

**The h=13 aggregate understates how often this actually helps.** Per
market (`outputs/afex_agroclimatic_vs_operational_per_market.csv`), 9 of
14 maize markets improve for RNN and 9 of 14 for GRU at h=13. The
aggregate is pulled back toward neutral by one extreme outlier per
architecture: `Giwa | Maize` for RNN (+14.87 NGN/kg worse) and
`Ikara | Maize` for GRU (+15.71 worse). Both are markets already flagged
independently, before this build, as unreliable: Giwa is the source
panel's own documented near-duplicate of Anchau (D-27/source
Build_Decisions, 0.947 correlation after an 81-week repair), and Ikara is
the thinnest, noisiest maize series in this panel (D-30, 24 scored
windows, consistently the weakest market at every horizon regardless of
build). A single bad cut at either market, amplified by how few windows
each contributes, is enough to swing the pooled average.

**Consequence.** Unlike sorghum (D-32), this is not a clean negative.
Agroclimatic data helps at the majority of individual markets at h=13 for
both architectures; the pooled headline number understates that because
two already-known-unreliable markets absorb a large, possibly noise-
driven hit. Worth revisiting with Giwa and/or Ikara excluded or
down-weighted before drawing a final conclusion on this horizon. h=4 and
h=26 do not show the same pattern and remain genuinely mixed or negative.

**Cost.** Two full runs (RNN, GRU), reusing the existing walk-forward
machinery unchanged.

---

## D-35. Rescored excluding Giwa and Ikara, across every non-sorghum build

**Decision.** `src/afex_exclude_outliers.py` rescores every saved forecast
(operational, safeguard-removed, agroclimatic; sorghum excluded per
instruction, since D-32 was already a clean result this would only
confound) excluding `Giwa | Maize` and `Ikara | Maize`, chosen over a
down-weighting scheme for consistency with this project's existing
precedent of outright exclusion for a documented data-quality reason
(Dandume, D-27). No retraining; reads `predictions_paired.csv` as already
saved. Full table: `outputs/afex_excl_outliers_comparison.csv`.

**Effect is horizon-dependent, and not uniformly in the expected
direction.** At h=4, excluding these two markets makes pooled MAE
slightly WORSE for every one of the six runs (e.g. RNN operational 59.93
-> 61.17). Giwa and Ikara are not bad markets at every horizon; whatever
makes them unreliable shows up at longer horizons, not short ones. At
h=13 and h=26, exclusion improves MAE across every run, most sharply at
h=26 (up to -7.67 NGN/kg for GRU agroclimatic, -60.56 in the opposite
direction on the safeguard-removed delta below). This matches Ikara's
already-documented h=26 MAE of 333-367 NGN/kg (D-30), roughly double the
next-weakest market, dominating the pooled average at that horizon in
particular.

**Agroclimatic's case gets stronger with the exclusion, not weaker.**
Within the excluded-outlier scope, agroclimatic beats operational at
h=13 by a wider margin than the full-market comparison showed (RNN -1.55
vs -0.46 before, GRU -2.16 vs -1.29 before), and its h=26 degradation
shrinks substantially (RNN +0.25 vs +2.48 before, GRU +2.54 vs +4.72
before). h=4 stays mixed (RNN slightly worse, GRU better), unchanged in
direction from the full-market result. D-34's read holds and sharpens:
this is a real, if modest, effect being partly obscured by two already-
flagged markets, not overturned by them.

**Safeguard-removed's underperformance is robust, not an artifact of
these two markets.** The gap versus operational is if anything larger
within the excluded scope (h=26: RNN +55.96, GRU +60.56 NGN/kg worse)
than the full-market comparison implied. D-27's conclusion needs no
revision.

**A genuine tension, flagged rather than resolved.** At h=13, MAE
improves when Giwa and Ikara are excluded, but directional accuracy gets
slightly WORSE for operational and agroclimatic (RNN operational: 61.42%
-> 60.12%; GRU operational: 52.42% -> 50.49%). These two markets were
apparently easier to call the direction of than average at that horizon
even while being costly in absolute error. MAE and directional accuracy
do not move together here, consistent with this project's repeated
finding (D-19 vs the main panel's own MAE story) that the two metrics can
disagree and neither should be read as a proxy for the other.

**Consequence.** Recommend treating the excluded-outlier figures as the
more decision-relevant ones for h=13 and h=26 MAE reporting on this
panel, while carrying the directional caveat above alongside them.
h=4 conclusions should stay based on the full 15-market scope, since
exclusion moves that horizon's numbers the wrong way.

**Cost.** None; rescoring existing forecasts only.

---

## D-36. Upstream lead-lag analysis: mostly 1-week transmission, two real cycles, one weak outlier

**Decision.** Model left unchanged (owner instruction: analysis only,
nothing retrained). `src/afex_upstream_lag_analysis.py` cross-correlates
every ordered pair of maize markets' weekly log-return series at lags -13
to +13 weeks (returns, not price levels, for the same reason as D-31's
sibling-commodity screen: levels share a common inflation trend that would
swamp any real lead-lag signal). For each market, the best upstream
candidate is the market with the highest correlation at a strictly
positive lag (a real lead, not a contemporaneous or reverse relationship).
This sizes a possible future exogenous variable analogous to the original
FEWSNET/NADIH panel's own `upstream_price`/`upstream_market`/
`upstream_lag_weeks` design, discovered from this panel's data rather than
assumed. Full detail: `outputs/afex_upstream_lag_full_detail.csv`
(every pair, every lag); per-pair best lag:
`outputs/afex_upstream_lag_best_per_pair.csv`; the practical result, one
row per market: `outputs/afex_upstream_lag_tree.csv`.

**Result: 14 of 16 markets' best lead is exactly 1 week.** Correlations
range 0.33 to 0.55 at that lag, comfortably above noise given 88-206
overlapping weeks per pair. One market (`Ikara`, best upstream `Bali`)
resolves at 2 weeks. One result is a clear outlier and should not be
trusted at face value: `Anchau`'s best positive-lag correlate is
`Dandume` at 12 weeks, correlation 0.25 -- both notably weaker (half the
typical correlation) and thinner (n=88, versus 111-206 for every other
pair) than the rest of the table, and a 12-week transmission lag has no
obvious logistics-based explanation the way a 1-2 week lag (typical
travel time between nearby wholesale markets) does. Flagged, not used.

**Not a clean tree: two real cycles at the core.** Checked for cycles of
any length, not just mutual pairs. Two found: `Leggal -> Jengre -> Giwa ->
Leggal` (each leads the next by 1 week) and `Bali <-> Ikara` (Bali leads
Ikara by 1 week at 0.47 correlation, Ikara leads Bali by 2 weeks at 0.31 --
the two markets move together closely enough that a single best-predecessor
metric cannot cleanly rank them). The rest of the panel branches off these
two cores as a directed forest: `Leggal` also leads `Dandume`, `Kumo` and
`Jalingo`; `Dandume` leads `Anchau` (the weak edge above); `Anchau` leads
`Pambegua` and `Tundun Saibu`; `Tundun Saibu` leads `Dawanau`; `Jengre`
leads `Saminaka`; `Ikara` also leads `Danja` and `Gazabu`; `Gazabu` leads
`Garbabi`. Read as: two tightly-coupled local market clusters (plausibly
geographically close, fast bidirectional price transmission) each feeding
a chain of more distant markets at the same 1-week step, not a strict
single-root hierarchy.

**Consequence, for a future build.** A 1-week-lagged neighbouring-market
maize price is the most defensible first exogenous candidate this
analysis supports -- short, consistent, well above noise for 15 of 16
markets. This is structurally the same idea already tried and found
harmful for a DIFFERENT source (sorghum, D-32); whether a same-commodity,
correctly-lagged neighbour price fares better is an open, answerable
question for the next build, not assumed from D-32's result. The
`Leggal`/`Jengre`/`Giwa` and `Bali`/`Ikara` cycles mean a single
market cannot always be picked as "the" upstream source for those
markets without an arbitrary tie-break; both pair members carry real
information about each other.

**Cost.** None; read-only correlation analysis over data already in hand.

**Visual.** Published as an artifact for review: https://claude.ai/code/artifact/0e984c0a-0737-4578-b1f3-c5454030b8fe
(private link; not part of the repo).

---

## D-37. Full-exogenous build: rainfall + NDVI + lagged upstream maize price

**Decision.** `configs/afex_operational_full_exog.yaml`: operational
hyperparameters, all three exogenous channels combined for the first
time -- rainfall and NDVI (D-33) plus a genuinely lagged neighbouring-
market maize price (`upstream_lag_map`, sized empirically by D-36's
cross-correlation screen, not assumed). `afex_data.py`'s `load_afex_panel`
gained `upstream_lag_map`, mutually exclusive with `sibling_commodity`
(both populate the same channel; passing both raises). The shift moves
each leader market's price forward in time so the channel holds what the
leader was doing `lag` weeks before the current origin -- real,
already-observed information at every origin, never the forecast window.

**Anchau deliberately excluded from the upstream channel.** Its best
candidate (D-36) correlates at only 0.24-0.25, roughly half the strength
of every other market's assignment, and the stronger-looking 12-week
version has no defensible logistics rationale. Given a noisy exogenous
input already cost more than it returned once on this panel (sorghum,
D-32), a weak, arguable pairing was not worth adding here. Anchau keeps
no upstream channel in this build, the same treatment an unpaired market
got in the sorghum build.

**Verified before training.** `Ikara`'s upstream channel at index 10
equals `Bali`'s own price at index 8, confirming the 2-week shift lands
exactly where intended. All three channels (`rainfall`, `ndvi`,
`upstream`) checked finite across the whole panel before running
anything. Smoke run: window counts identical to every earlier AFEX build
(656/886/1089 training windows at the same three cuts), confirming
nothing was silently dropped by combining three exogenous sources at
once.

**Cost.** Two full runs (RNN, GRU), launched next; compared against
operational (no exog) and agroclimatic (rainfall/NDVI only) to isolate
what the upstream-lag channel adds on top of D-34's result. Findings in a
following entry once complete.

---

## D-38. Full-exogenous result: combining sources helps more than either alone

**Decision.** Both runs (`configs/afex_operational_full_exog.yaml`, RNN
and GRU) complete. Compared against operational (no exog, D-27) and
agroclimatic (rainfall/NDVI only, D-34), all three now on the same 15
maize markets.

**Overall MAE, NGN/kg, all 15 markets:**

| | h=4 | h=13 | h=26 |
|---|---|---|---|
| RNN operational | 59.93 | 131.20 | 179.91 |
| RNN agroclimatic | 60.84 | 130.74 | 182.39 |
| RNN full-exog | 61.25 | **130.11** | **177.91** |
| GRU operational | 63.50 | 135.85 | 162.67 |
| GRU agroclimatic | 62.87 | 134.56 | 167.39 |
| GRU full-exog | **62.03** | **132.40** | 167.46 |

Bold marks the best of the three variants in each cell. Full-exog is
best or effectively tied-for-best in 4 of 6 architecture/horizon cells
(RNN h=13, RNN h=26, GRU h=4, GRU h=13), and at RNN h=26 it is the first
exogenous variant tried on this panel to land ahead of plain operational,
not just close to it (177.91 vs 179.91, and now essentially at the naive
benchmark of 177.74 rather than behind it as agroclimatic-alone was).
The one clear loss is RNN h=4 (61.25 vs operational's 59.93), and GRU
h=26 is a near-tie with agroclimatic (167.46 vs 167.39), both still
behind plain operational there (162.67).

**Adding the upstream-lag channel on top of agroclimatic helps in 5 of 6
cells**, comparing full-exog against agroclimatic-alone directly: RNN
h=13 (130.74 -> 130.11), RNN h=26 (182.39 -> 177.91, the largest single
improvement in this comparison), GRU h=4 (62.87 -> 62.03), GRU h=13
(134.56 -> 132.40); only GRU h=26 is essentially flat (167.39 -> 167.46).
This is the first exogenous addition on this panel where the effect is
consistently positive rather than mixed (agroclimatic alone, D-34) or a
clean negative (sorghum, D-32) -- suggesting the two exogenous
signals are complementary rather than redundant, though not proven
additive in a formal sense.

**Same pattern holds excluding Giwa and Ikara**
(`outputs/afex_full_exog_vs_all_comparison.csv` carries both scopes): full-
exog stays competitive-to-best in the same cells within the 13-market
scope, so this is not an artifact of those two flagged markets.

**Consequence.** Of everything tried on this panel so far, full-exog
(rainfall + NDVI + D-36's lagged upstream maize price) is the strongest
overall exogenous candidate: never the worst variant by a wide margin at
any horizon, and the clear best at more horizon/architecture
combinations than either operational-alone or agroclimatic-alone. Worth
treating as the leading candidate for a future production build on this
panel, with the caveat that RNN h=4 and the GRU h=26/operational gap
remain open questions rather than settled ones.

**Cost.** Two full runs (RNN, GRU), reusing the existing walk-forward
machinery unchanged; no new engineering beyond D-33's and D-36's already-
built channels.

---

## D-39. Final selection for this AFEX exploration: GRU full-exog (accuracy), RNN operational (direction)

**Decision.** Two builds carried forward as the outcome of this workstream;
every other build's config and run outputs removed from the repo. Kept:

- **`configs/afex_operational_full_exog.yaml`, kind GRU** -- accuracy
  pick. Rainfall + NDVI (D-33) + D-36's lagged upstream maize price
  (D-37). Best MAE-vs-naive performer of every candidate tried, and the
  only build where combining exogenous sources produced a consistently
  positive effect rather than a mixed or negative one (D-38).
- **`configs/afex_operational.yaml`, kind RNN** -- direction pick. No
  exogenous data at all. Best directional margin against the panel's own
  always-up/down base rate of any candidate, and by far the tightest
  cross-horizon consistency (std of vs-naive MAE 3.18 points, versus
  6.81 for GRU full-exog) -- see the scoring below.

**Removed:** `configs/afex_safeguard_removed.yaml`,
`configs/afex_operational_sorghum.yaml`,
`configs/afex_operational_agroclimatic.yaml`, and every run output under
`outputs/afex_safeguard_removed/`, `outputs/afex_operational_sorghum/`,
`outputs/afex_operational_agroclimatic/`, plus the GRU run under
`outputs/afex_operational/` and the RNN run under
`outputs/afex_operational_full_exog/` (GRU operational and RNN full-exog
specifically, the two architecture/build combinations not carried
forward). Nothing here is undocumented: D-27 (safeguard-removed
discontinued), D-32 (sorghum, clean negative), D-34 (agroclimatic alone,
mixed), and D-38 (the full six-way comparison) carry the actual numbers
and reasoning for every removed build. The cross-build comparison CSVs in
`outputs/` (per-market, directional, MAPE-across-variants, the scoring
components) are left in place as the evidence trail; only the model
configs and their raw per-seed/per-cut outputs were removed.

**Scoring rationale, in full.** Six live candidates (RNN/GRU x
operational/agroclimatic/full-exog; safeguard-removed and sorghum
already excluded on independent, mechanistic grounds) scored on a
weighted composite: MAE-vs-naive (60%, per this project's own stated
convention that MAE is the primary metric), directional margin against
the actual always-up/down base rate (25%), and cross-horizon consistency
-- the standard deviation of the MAE score across h=4/13/26 (15%). MAE
scored per D-35's own scope rule (h=4 on all 15 markets, h=13/h=26
excluding Giwa and Ikara). All three raw components min-max normalised
across the six candidates before weighting.

| Rank | Candidate | MAE score | Direction score | Consistency score | Composite |
|---|---|---|---|---|---|
| 1 | GRU full-exog | 100.0 | 9.9 | 39.1 | **68.4** |
| 2 | RNN operational | 26.9 | 100.0 | 100.0 | 56.2 |
| 3 | GRU agroclimatic | 53.0 | 0.0 | 20.5 | 34.9 |
| 4 | GRU operational | 18.8 | 87.8 | 0.0 | 33.2 |
| 5 | RNN agroclimatic | 11.1 | 37.0 | 77.1 | 27.5 |
| 6 | RNN full-exog | 0.0 | 36.0 | 73.6 | 20.0 |

**The ranking is not weight-independent, and that is disclosed rather
than hidden.** Under MAE-dominant or MAE-only weighting, GRU full-exog
wins clearly every time. Under equal-thirds or direction-weighted
schemes, RNN operational overtakes it, on the strength of its
consistency and directional showing. The 60/25/15 split above was chosen
because MAE is this project's own declared primary metric, not because
it was the split that produced a preferred answer; the sensitivity
itself is the reason both candidates, not one, are being carried
forward rather than declaring a single global winner.

**Two caveats carried forward with the winners, not smoothed away.**
(1) Every one of the six candidates, including both survivors, loses to
plain carry-forward at h=4 -- "best available" describes a group that
has not yet cleared the shortest-horizon bar, not a model that is
unambiguously good there. (2) GRU full-exog depends on three external
data sources (the AFEX panel, the UN Data Exchange rainfall/NDVI feed,
and a static upstream-lag map derived from a correlation snapshot that
could drift and need re-deriving); RNN operational depends on none of
these. That operational-simplicity difference is a real form of
robustness this scoring does not capture numerically and is worth
weighing before either is treated as production-ready.

**Cost.** None to compute (scoring only); disk space freed by removing
four discontinued run directories and their configs.

---

## D-40. Diesel added to full-exog: partial coverage, real staleness, both logged

**Decision.** `afex_data.py` gains `diesel_source_path`, reusing the main
FEWSNET panel's own state-level diesel series (`data/panel_weekly.parquet`)
rather than sourcing anything new. Both panels use the same Wednesday
weekly grid (verified directly), so this only needs a reindex, not the
dekadal interpolation rainfall/NDVI required. New config
`afex_operational_full_exog_diesel.yaml` adds diesel on top of the
already-combined full-exog build (D-37/38) directly, rather than isolating
diesel alone first -- the question today is whether a fourth real signal
helps the current best model further; a solo-diesel isolation is the
natural next step if this result is mixed.

**Two real gaps, neither hidden.** Diesel covers only 4 of 6 AFEX states
(Gombe, Kaduna, Kano, Katsina -- 10 of 15 scored maize markets); Plateau
and Taraba markets (Jengre, Bali, Garbabi, Gazabu, Jalingo) get no diesel
channel at all. And the source only runs to 2024-09-18; the ~22 months
after that (to the panel's 2026-07-22 end) are forward-filled, meaning
diesel is a stale, carried-forward value for most of the panel's most
recent history -- exactly the period the model is most likely to be used
against in practice.

**Verified before training.** Covered/missing states match the design
exactly; every value finite; a spot-checked market's diesel channel is
flat (correctly forward-filled) across the last five weeks and varies
correctly across the first five. Smoke run: window counts identical to
every other AFEX build (656/886/1089), confirming nothing silently
dropped.

**Cost.** Two full runs (RNN, GRU), launched next.

---

## D-41. FX rate and inflation: checked, not yet integrated -- scoped for tomorrow

**Finding, not a build.** Two national-level macro series checked for
integration today: `Official FX rate_2004-2026.xlsx` (monthly, 2004-01 to
2026-02) and `Complete Inflation data.xlsx` (monthly headline/food, MoM
and YoY, 2000-01 to 2026-05). Both are clean, single-value-per-month,
well within range of both panels; both would need only a short forward-fill
tail (FX: ~5 months to 2026-07-22; inflation: ~2.5 months), far smaller
than diesel's 22-month gap (D-40).

**Why this isn't built today.** Diesel, rainfall, NDVI and upstream-lag
all reused channel slots that already existed in `data.Panel` and
`build_sequence` but sat zero-filled for AFEX -- additive, low-risk changes
confined to `afex_data.py`. FX and inflation have no existing slot: adding
them means extending `SEQ_BASE_CHANNELS` and `build_sequence` in
`src/data.py` itself, which is shared by the main FEWSNET pipeline and
AFEX both. That is a materially different, higher-risk change (touches
code the governed evaluation depends on) than anything else done today,
and it deserves its own careful pass rather than being rushed in while two
other training queues are mid-run.

**Plan for the next session.** Add `fx_rate` and `inflation_yoy` (or
similar) as two new optional sequence channels, zero-filled by default so
every existing config's behaviour is unchanged unless explicitly opted
in; national values, so no per-market or per-state matching is needed,
just a monthly-to-weekly forward-fill onto each panel's own date grid.
Cheapest of the untried data additions to build once done carefully; also
the first candidate that could apply to the main FEWSNET evaluation, not
only AFEX, since neither panel currently carries a macro/currency signal.

---

## D-42. Diesel on top of full-exog: helps GRU at every horizon, mixed for RNN

**Decision/finding.** Both runs (`afex_operational_full_exog_diesel.yaml`,
RNN and GRU) complete.

| | h=4 | h=13 | h=26 |
|---|---|---|---|
| RNN full-exog (D-38) | 61.25 | 130.11 | 177.91 |
| RNN full-exog + diesel | 61.78 | 131.08 | **177.13** |
| GRU full-exog (D-38) | 62.03 | 132.40 | 167.46 |
| GRU full-exog + diesel | **61.42** | **132.04** | **166.90** |

GRU improves at every horizon with diesel added, consistently if by a
small margin (-0.61, -0.36, -0.56 NGN/kg). RNN is mixed: slightly worse at
h=4 and h=13 (+0.53, +0.97), slightly better at h=26 (-0.78). Both effects
are small, in the same range as agroclimatic-alone's effect (D-34), not
the scale of upstream-lag's clear contribution (D-38) or sorghum's clear
harm (D-32).

**Read with the coverage gaps in mind (D-40).** Diesel here covers only 10
of 15 scored maize markets and is stale (forward-filled) for the panel's
most recent ~22 months. A cleaner test -- full geographic coverage, real
rather than carried-forward recent values -- might show a larger or
different effect; this result characterises diesel as currently
integrated, not diesel's ceiling.

**Consequence.** For GRU, worth keeping: it improves the already-best
build with no new data collection, only a partial-coverage, partly-stale
version of data the FEWSNET side already had. For RNN, inconclusive at
this scale of effect -- not worth changing the RNN operational
recommendation (D-39) over it. Not yet re-run through D-39's exclusion-
of-outliers or directional lens; both would be quick follow-ups reusing
existing scripts before treating this as final.

**Cost.** Two full runs, reusing infrastructure built earlier today.

**Follow-up checks, as flagged above.** Directional: both architectures
fail the always-up/down base rate at every horizon (RNN 39.8/56.0/42.9%
vs 59.7/58.7/64.5% share; GRU 40.7/52.0/46.0% vs the same shares) --
consistent with D-29's pattern, diesel doesn't change the directional
story. Outlier exclusion: h=13/h=26 improve with Giwa/Ikara excluded
(e.g. GRU h=26: 166.90 -> 162.11), h=4 worsens (GRU: 61.42 -> 62.38) --
exactly D-35's established pattern, holding for a fifth build variant now.
No new surprises; both checks confirm consistency rather than revising
the read above.

---

## D-43. Width-capacity bracket applied to GRU full-exog too

**Decision.** Same bracket used on the FEWSNET side (D-29/D-30) applied
here for direct comparability: `hidden=96` and `hidden=192`, otherwise
identical to `afex_operational_full_exog.yaml` (baseline `hidden=64`).
GRU only, per instruction. No code changes needed -- `hidden` was already
a plain config value on this branch.

**Verified before training.** Smoke run (`hidden=96`): 42,451 parameters
(up from full-exog's smaller count at hidden=64), window counts identical
to every other AFEX build (656/886/1089), runs end to end.

**Cost.** Two full runs (hidden=96, hidden=192), launched next. Result in
a following entry.

---

## D-44. AFEX capacity bracket result: the opposite of FEWSNET's, and consistently so

**Decision/finding.** Both runs (`hidden=96`, `hidden=192`) complete,
compared against `afex_operational_full_exog.yaml`'s baseline `hidden=64`.

| Hidden | h=4 MAE | h=13 MAE | h=26 MAE |
|---|---|---|---|
| 64 (baseline) | 62.03 | 132.40 | 167.46 |
| 96 | **61.65** | **130.00** | 170.80 |
| 192 | 63.19 | 132.07 | 180.54 |

At h=26, MAE gets monotonically WORSE as width increases (167.46 -> 170.80
-> 180.54) -- the exact opposite direction from FEWSNET's GRU build3
bracket (D-30), where wider was monotonically better at the same two
longer horizons. At h=4 and h=13, `hidden=96` is the best of the three
(marginal improvement over 64), but `hidden=192` is close to or worse
than the baseline at every horizon, not just h=26.

**This is not a contradiction, it is the same principle from two different
sides.** D-23 found removing overfitting safeguards and raising capacity
helped on the FEWSNET panel (12 years, thousands of training windows per
cut) because that panel was under-learning. D-27 found the identical
safeguard-removal setting *hurt* on the AFEX panel (277 weeks, at most a
few thousand training windows even pooling 51 series) because that panel
is comparatively data-poor. This capacity bracket is the same story told
through width alone, holding everything else fixed: more capacity is
only good relative to how much data is actually available to fill it.
AFEX at `hidden=192` (158,227 parameters) is being asked to learn from a
fraction of FEWSNET's data with over four times FEWSNET's best-performing
parameter count (D-30's `hidden=192` on FEWSNET: not yet at a ceiling).

**Consequence.** `hidden=96` is a small, plausible improvement over the
current `hidden=64` default for GRU full-exog at the two shorter
horizons, but the finding that matters more is qualitative: capacity
should not be tuned the same way across the two evaluations, because the
two panels sit in different positions on the same underfitting/overfitting
spectrum. Any future capacity change on AFEX should be checked against
this panel's own data volume, not carried over from what worked on
FEWSNET.

**Cost.** Two full runs, complete.

---

## D-45. hidden=96 adopted as the GRU full-exog default

**Decision.** `configs/afex_operational_full_exog.yaml`'s `hidden` changed
from 64 to 96, promoting D-43/D-44's bracket result into the canonical
build rather than leaving it as a side comparison. `outputs/
afex_operational_full_exog/GRU/` now holds the hidden=96 run's actual
output files (copied from `outputs/afex_operational_full_exog_hidden96/
GRU/`, not re-run -- the result was already verified and re-running it
would only reproduce the same numbers at real compute cost). The
superseded hidden=64 output is kept, not deleted, at
`outputs/afex_operational_full_exog/GRU_hidden64_superseded/`, following
this project's own precedent (D-16) of retaining superseded evidence
rather than discarding it.

**Why 96 and not 64 or 192, restated plainly.** hidden=96 beats hidden=64
at h=4 (61.65 vs 62.03) and h=13 (130.00 vs 132.40); hidden=192 is worse
than hidden=64 at every horizon. 96 is the best of the three tried, not
merely "not worse."

**What this does not change.** RNN is no longer a full-exog finalist
(D-39: RNN operational is the direction pick instead), so this capacity
change was validated for GRU only and is not claimed to be right for RNN
if anyone reruns this config with `--kind RNN`.

**Cost.** None; reuses an already-completed, already-verified run.

---

## D-46. Seed-variance check finally applied to AFEX: both adopted candidates are largely noise

**Decision.** `src/seed_analysis.py` (the D-20 method) ported to this
branch. It required one compatibility fix: it hard-required a
`convention` column that only exists on the FEWSNET side, since AFEX has
no conditional/unconditional split. Fixed with a fallback (`"n/a"` where
the column is absent) rather than forking a second script -- one line in
`load_runs`, backward compatible with every existing FEWSNET use of the
same function. No AFEX-side seed-variance check had been run before this;
every capacity and driver decision since D-27 (including D-39's final
selection and D-45's hidden=96 adoption) was made on single-seed-median
numbers only.

**Finding: neither of the two adopted candidates clears its own margin
over naive by a comfortable amount, and per-market stability is worse
than anything seen on FEWSNET.**

| Candidate | h | margin/naive/seed-sd | verdict flips (of 14 markets) |
|---|---|---|---|
| RNN operational (direction pick) | 4 | -1.48 (confidently worse than naive) | 6 |
| RNN operational | 13 | +1.49 (barely clears 1.0) | **12** |
| RNN operational | 26 | **-0.11 (noise)** | 13 |
| GRU full-exog h96 (accuracy pick) | 4 | -3.19 (confidently worse than naive) | 4 |
| GRU full-exog h96 | 13 | **+0.66 (noise, below 1.0 bar)** | **13** |
| GRU full-exog h96 | 26 | +1.04 (barely clears 1.0) | 12 |

**This lands harder than the FEWSNET version of the same check (D-35).**
FEWSNET's worst per-market flip rate was 12 of 15 (RNN h=4, one horizon).
Here, both candidates flip on 12-13 of 14 markets at the *other* two
horizons -- the ones each candidate was actually selected for. GRU
full-exog's h=13 margin (the horizon D-38/D-39 leaned on to call it the
accuracy pick) does not clear the 1-seed-sd bar at all. RNN operational's
h=26 margin over naive is statistically indistinguishable from zero in
either direction.

**One seed is doing a lot of the damage, and it is a real seed effect,
not a bug.** RNN operational h=26 per-seed MAE: seed 0 at 139.18, the
other six spanning 165.7-199.0. That single seed drags the mean down and
inflates the reported spread; it is a genuine random-init outcome, traced
directly in `outputs/afex_operational/RNN/forecasts.csv`, not a
computation error.

**What this does not do.** It does not overturn D-39's selection outright
-- RNN operational's *directional* hit-rate finding (61.4% at h=13,
beating both naive baselines, D-29/D-39) is a different metric from the
MAE-margin check here and has not itself been seed-checked yet; that is
next. It also does not mean AFEX has learned nothing (both models clearly
lose to naive at h=4, which is at least a consistent, confident signal,
just a negative one). What it does mean: neither candidate's claimed edge
over naive at the horizon it was picked for should be presented as
established without this caveat attached.

**Consequence.** No config or selection change from this entry alone.
Flagging this as the single most important AFEX finding of the day before
continuing the rest of the game plan (the h=13/26 directional seed-check,
the retroactive capacity-bracket seed-check, and the rest), since it
changes how much weight the existing D-39 selection can bear.

**Cost.** None to compute; reused seven-seed outputs already on disk.
One-line source fix in `src/seed_analysis.py`, backward compatible.

---

## D-47. RNN operational's directional finding checked at the seed level: real, but only just converged at 7 seeds

**Decision.** D-46 checked MAE margins; the actual basis for D-39's "RNN
operational is the direction pick" call was a different metric --
directional hit-rate against both naive always-up/always-down baselines,
at h=13. That metric had never been seed-checked either. Direct check,
one-off script against `outputs/afex_operational/RNN/forecasts.csv`
(not yet folded into a reusable src/ tool -- flagged for later if this
becomes a recurring need).

**Per-seed picture: mostly weak.** Of the 7 individual seeds, only one
(seed 0, 60.4%) beats both the always-up (58.66%) and always-down
(40.67%) baselines on its own. The other six range 48.0-58.4% -- five of
them do not even clear the always-up baseline alone. Read naively, this
would suggest D-39's finding is one lucky seed, not a property of the
model.

**Seed-count curve tells a different, more reassuring story.** Enumerating
every subset of the 7 seeds, k=1 through 7, and taking the median-ensemble
direction call at each subset size:

| k (seeds ensembled) | mean direction hit-rate | % of subsets beating both baselines |
|---|---|---|
| 1 | 52.8% | 14.3% |
| 2 | 54.2% | 9.5% |
| 3 | 56.3% | 31.4% |
| 4 | 57.6% | 51.4% |
| 5 | 59.0% | 61.9% |
| 6 | 60.5% | 71.4% |
| 7 (all seeds, the reported number) | 61.0% | 100% |

This is a smooth, monotonic climb with shrinking increments (1.4, 2.1,
1.2, 1.4, 1.5, 0.6 points) -- the signature of averaging away symmetric
prediction noise and converging toward a real underlying signal, not the
signature of one lucky draw. A single seed's prediction can be noisy
enough to land on the wrong side of a near-zero directional call; median-
ensembling several seeds cancels that noise and reveals the sign more
reliably, which is exactly why this project aggregates by seed-median
everywhere (D-11) rather than reporting a single fit.

**But it converged right at the edge of the seed count actually used.**
At k=6, only 71.4% of possible subsets already clear both baselines; the
finding is not yet fully stable even one seed short of what was run. This
is a weaker form of the D-20 seed-count-curve check (there, k=6 to k=7
moved the MAE metric under 0.5%; here, going from 6 to 7 seeds is what
finally pushes every remaining subset over the line). More seeds (10-12,
per D-20's own suggested cap before exhaustive enumeration gets
expensive) would confirm whether this keeps climbing toward a firmer
plateau or has already arrived at one.

**Consequence.** D-39's directional pick for RNN operational is on
firmer ground than D-46 alone would suggest -- it is a real, converging
signal, not an artifact of the seed-median aggregation working against
the model. It is also not yet a settled result: seven seeds is the bare
minimum at which it appears, not a comfortable margin above it. Worth a
10-12 seed re-run before this is presented as fully robust.

**Cost.** None to compute; reused the same seven-seed forecasts already
on disk. Exhaustive enumeration of all C(7,k) subsets, same method as
D-20's seed-count curve.

---

## D-48. hidden=64->96 adoption (D-45) does not survive a paired seed check either

**Decision.** Same paired-by-seed test as D-34 (FEWSNET's hidden=256
check), applied to the exact comparison D-45 used to promote hidden=96
over hidden=64 for GRU full-exog. All three width configs (64, 96, 192)
share the same seven seeds, so each seed pairs directly across configs.

| h | mean diff (64 minus 96, positive = 96 better) | sd | t | significant? |
|---|---|---|---|---|
| 4 | +0.67 | 2.24 | +0.79 | no |
| 13 | +1.55 | 6.88 | +0.60 | no |
| 26 | -6.15 | 10.16 | -1.60 | no |

No horizon clears even the 10% threshold (needs \|t\|>1.94). The per-seed
differences flip sign with no consistent pattern -- h=13's seven pairs are
+5.35, -9.46, +0.43, +9.30, +7.11, +3.72, -5.59, arguably the clearest
case of noise in this whole session's work. **This is the same failure
mode as D-34's GRU finding on FEWSNET**, now found independently on a
different panel: a capacity change adopted on a single-seed-median
comparison (D-44/D-45) does not survive being checked against its own
seed-to-seed spread.

**Consequence.** D-45's adoption of hidden=96 as the GRU full-exog
default is not supported by this test. It is not necessarily wrong
either -- 96 is not shown to be *worse* than 64, only that the specific
margin used to justify picking it over 64 is indistinguishable from seed
noise. Reverting to 64 would rest on equally thin evidence. Flagging
this rather than acting on it unilaterally: three of this session's
capacity-adoption decisions (FEWSNET D-29-33's GRU result, and now this
one) have made the same mistake of trusting a single-seed-median
comparison, which suggests the fix belongs in the process (seed-pair
every capacity comparison before adopting one, not after), not in
re-litigating each one individually.

**Cost.** None to compute; reused seven-seed forecasts already on disk.

---

## D-49. Diebold-Mariano test: a genuinely different check, and it partly rehabilitates GRU full-exog

**Decision.** D-46/D-48's seed-pairing checks ask "would a re-trained
model reproduce this margin." That is not the only test this project
already has built: `src/metrics.diebold_mariano` (Newey-West HAC-corrected,
already used on the FEWSNET side) tests a different question -- given the
specific, already-deployed median-ensemble forecast, is its paired error
differential against a reference significantly different from zero over
the actual historical test record. Never run on AFEX before. Applied to
the real production forecasts (`predictions_paired.csv`, the seven-seed
median, not a single seed) for both adopted candidates, against naive and
against each other, maize-only, all horizons:

| Comparison | h=4 | h=13 | h=26 |
|---|---|---|---|
| GRU full-exog vs naive | stat +5.39, p<.0001 (GRU **worse**) | stat -2.27, **p=.023 (GRU better)** | stat -0.50, p=.62 (n.s.) |
| RNN operational vs naive | stat +4.34, p<.0001 (RNN **worse**) | stat -1.86, p=.063 (n.s., borderline) | stat +0.32, p=.75 (n.s.) |
| GRU vs RNN head-to-head | stat +3.00, **p=.003 (RNN better)** | stat -0.59, p=.56 (n.s.) | stat -0.87, p=.39 (n.s.) |

**This is the redeeming result for GRU full-exog.** Its 3-month edge over
naive -- the exact horizon it was picked for in D-39 -- clears p<.05 on a
standard, established forecast-comparison test, using the full realized
error record rather than a 7-point seed sample. Per-market breakdown at
h=13 (12 of 14 markets with enough observations to test): every single
one has a negative DM statistic, i.e. numerically favours GRU, though none
individually reaches significance alone at typical per-market sample
sizes (24-121). The pooled significance is coming from a consistent
direction spread broadly across nearly the whole panel, not from one or
two markets carrying it -- the same "broad rather than concentrated"
pattern the FEWSNET side found reassuring in its own three-month result.

**This does not contradict D-46; it answers a different question.** D-46
says: retrain this model with a different random seed and the aggregate
margin might not reappear (training-stochasticity risk). D-49 says: over
the actual historical period this specific forecast was tested on, its
track record against naive is not attributable to chance (sampling risk
in the paired-loss series). Both are real, independent forms of
uncertainty; a result can pass one and fail the other, and both belong in
front of reviewers rather than either one silently overriding the other.

**Why GRU and RNN show so little disparity, on direct evidence rather
than speculation.** Three things line up:
1. Both architectures predict log-return (proportional change), never a
   price level -- confirmed in `src/afex_data.py`: `np.log(act) - np.log(p0)`,
   identical design choice to FEWSNET. This is the same reason the
   FEWSNET side never found a clean GRU-vs-RNN gap either: gating exists
   to solve a long-raw-memory problem that this representational choice
   already mostly removes.
2. D-44 already showed GRU's *extra* capacity (widening past 64) actively
   hurts on this data-poor, 277-week panel. The head-to-head DM result
   here suggests the same constraint applies to GRU's *structural* extra
   capacity relative to RNN, not only to explicit widening: three gates'
   worth of parameters have less data to earn their keep here than on
   FEWSNET's 647-week panel.
3. Direct evidence, not inference: the DM head-to-head test finds no
   significant GRU edge at any horizon, and a significant **RNN** edge at
   h=4 (p=.003). There is no hidden GRU advantage being masked by noise --
   the standard test that would detect one does not find one.

**Consequence.** GRU full-exog keeps a real, defensible claim to being
the accuracy pick at h=13 specifically (D-49), even though the capacity
choice that produced its current hidden size does not (D-48) and its
seed-level margin is thin (D-46). These are three separate, non-
contradictory findings about three different things, and belong together
in whatever goes to reviewers, not resolved into one verdict.

**Cost.** None to compute; reused `predictions_paired.csv` and an
existing, already-validated function. No new training.
