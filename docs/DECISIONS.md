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

## D-27. Prioritizing 2 of the deployable candidates going forward: both Build 3 architectures

**Decision.** Applied the same scoring rigor used for the AFEX workstream
to this project's own deployable candidates. Result, unlike AFEX, is
unanimous rather than weight-dependent: **GRU build3 (unconditional)**
first, **RNN build3 (unconditional)** second, both by a clear margin
under every weighting scheme tried. Build 2's two architectures rank
third and fourth, also unanimously.

**Scope of the comparison, and why conditional/conditional_exog/
forecast_drivers are not in it.** This is a governed evaluation, not an
open exploration like AFEX, so the arms left out of the scoring are not
being discarded, only excluded from "which to prioritize for further
refinement" on grounds already established elsewhere in this log:

- **Conditional (foreknowledge)** is never a deployment candidate by
  definition -- CLAUDE.md's own hard rule: "Never present a conditional
  figure as achievable accuracy." It remains governance-required
  regardless (CLAUDE.md: "Read change_control.json from the conditional
  runs... only the conditional arm is comparable to the incumbent"), so
  its outputs stay exactly as they are; it is simply not a candidate for
  "which model goes forward."
- **Conditional_exog** shares the same non-deployability problem
  (forecast-window driver values, just missing the upstream one) and is
  additionally shown worse than unconditional at every horizon for both
  architectures, both builds (D-22, D-23) -- doubly excluded.
- **Forecast_drivers** is deployable in principle but was a clean
  negative at every horizon, both architectures (D-25), traced to one
  mandated, weaker-than-carry-forward component (the upstream seasonal-
  naive method) rather than the general approach. Excluded from this
  round on the same "mechanistically explained failure" basis AFEX
  applied to sorghum, not because the underlying idea is dead.

That leaves the unconditional family: build2 (operational) and build3
(safeguard-removed / capacity-raised) unconditional, both architectures
-- four candidates, the same shape of comparison AFEX ran.

**Scoring.** Weighted composite: vs-incumbent MAE (`vs_panel_fe_pct`,
60% -- the metric this whole evaluation exists to answer, more directly
relevant here than vs-naive since there is a real incumbent to beat),
directional margin against the panel's own always-up/down base rate
(25%, D-19/D-29-style, refreshed to include build3 -- not previously
computed for it), and cross-horizon consistency (15%, std of
`vs_panel_fe_pct` across h=4/13/26). All three min-max normalised across
the four candidates.

| Rank | Candidate | vs-incumbent score | Direction score | Consistency score | Composite |
|---|---|---|---|---|---|
| 1 | GRU build3 (safeguard-removed) | 100.0 | 100.0 | 60.5 | **94.1** |
| 2 | RNN build3 (safeguard-removed) | 80.4 | 65.9 | 80.0 | **76.7** |
| 3 | RNN build2 (operational) | 22.1 | 11.6 | 100.0 | 31.1 |
| 4 | GRU build2 (operational) | 0.0 | 0.0 | 0.0 | 0.0 |

**Unlike AFEX, this ranking does not flip under any tested weighting**
(vs-incumbent-only, equal-thirds, consistency-weighted, direction-
weighted all pick GRU build3 first, RNN build3 second). Build 3 beats
Build 2 on every axis, both architectures, which is the strongest form
of evidence this workstream has produced for a "go forward" pick.

**The headline number this does not change: nothing here beats the
incumbent by the 5% gate threshold, and only one cell beats it at all.**
`GRU build3` at h=4 is the single positive `vs_panel_fe_pct` across all
twelve candidate/horizon rows (+1.90%), still short of the 5% change-
control threshold and still a `change_control` FAIL. Every other
candidate at every other horizon loses to the incumbent, several by a
wide margin (h=26 losses of 36-50%). "Prioritize going forward" means
these two are the strongest base to keep refining, not that either is
ready to recommend for deployment.

**Consequence.** Recommend GRU build3 and RNN build3 (unconditional) as
the two architectures to carry forward for further development on the
main FEWSNET/NADIH evaluation. Build 2's two runs, and the conditional/
conditional_exog/forecast_drivers arms across both builds, are retained
in full -- none of this is a pruning exercise the way the AFEX cleanup
was; the governance record needs all of it, and D-16's own precedent
(discontinue further runs, retain existing outputs as evidence) is the
model being followed here, not deletion.

**Cost.** None to compute (scoring only); `directional_benchmark.py`
rerun to bring build3 into the directional comparison for the first time
(D-19 predates build3's existence).

---

## D-28. Ensembling build2 with build3 does not help; build3 alone stays best

**Decision/finding.** `src/ensemble_build2_build3.py` averaged build2 and
build3's already-trained predictions (same architecture, same grid) --
free, no retraining. Result: the ensemble is worse than build3 alone in
every one of six cells (both architectures, all three horizons). Build2 is
uniformly worse than build3 on `vs_panel_fe_pct` (D-27), so blending the
two doesn't cancel independent errors, it just dilutes build3's better fit
toward build2's worse one. Ensembling only helps when the errors being
combined are meaningfully uncorrelated; here they share the same data,
architecture, and lookback, differing only in regularisation strength, so
the errors move together more than they diverge.

**Consequence.** No further work on this specific ensemble. Confirms
build3 alone, not a blend, is the right base for the two prioritized
architectures (D-27 stands unchanged). Cost: near zero, worth trying
before spending real compute on architecture changes.
