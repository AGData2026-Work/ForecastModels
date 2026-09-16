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

---

## D-29. Bracketing build3's capacity choice: hidden=96/192 and a second layer

**Decision.** `RecurrentForecaster` gains `num_layers` (default 1, backward
compatible with every existing config). Three new configs, each changing
exactly one thing versus build3.yaml: `build3_hidden96.yaml` (width down),
`build3_hidden192.yaml` (width up), `build3_2layer.yaml` (depth instead of
width, hidden held at 128). `forward()`'s `h[-1]` already takes the top
layer's final state regardless of layer count, so no other code changed.

**Why.** Build2 (hidden=64) to build3 (hidden=128) was one jump, never
bracketed on either side, and every build in this project has used exactly
one recurrent layer. This checks two different questions with one small
code addition: was 128 the right amount of width, and does adding depth
(a second stacked layer) behave differently from adding width.

**Verified before training.** `num_layers=2` smoke run: 69,763 parameters,
consistent with roughly doubling the recurrent layer's weight matrix; runs
end to end with no other code path affected.

**Cost.** Six full runs (RNN + GRU x three configs), queued in background.
Results in a following entry once complete.

---

## D-30. Capacity bracket complete: width beats depth, GRU still has room at 192

**Decision/finding.** All six new runs complete (D-29). Full comparison,
MAE and `vs_panel_fe_pct`:

| Run | h=4 MAE | h=13 MAE | h=26 MAE | h=4 vs-fe | h=13 vs-fe | h=26 vs-fe |
|---|---|---|---|---|---|---|
| GRU hidden=96 | 16.46 | 31.09 | 50.14 | +1.96 | -6.86 | -40.20 |
| GRU hidden=128 (baseline) | 16.47 | 30.88 | 48.74 | +1.90 | -6.12 | -36.29 |
| GRU hidden=192 | **16.55** | **30.22** | **48.14** | +1.41 | **-3.85** | **-34.63** |
| GRU 2-layer (128) | 16.90 | 31.03 | 49.56 | -0.65 | -6.65 | -38.59 |
| RNN hidden=96 | 16.89 | 31.54 | 49.04 | -0.62 | -8.41 | -37.13 |
| RNN hidden=128 (baseline) | 16.91 | 31.71 | 49.07 | -0.73 | -8.98 | -37.21 |
| RNN hidden=192 | 16.94 | 31.64 | **48.39** | -0.87 | -8.75 | **-35.32** |
| RNN 2-layer (128) | 16.93 | 31.54 | 48.94 | -0.82 | -8.38 | -36.85 |

**GRU's result is the clean one: MAE improves monotonically with width at
h=13 and h=26** (31.09 -> 30.88 -> 30.22; 50.14 -> 48.74 -> 48.14 across
96 -> 128 -> 192), and `vs_panel_fe_pct` improves the same way at both
horizons (-6.86 -> -6.12 -> -3.85; -40.20 -> -36.29 -> -34.63). No ceiling
found yet at 192 -- capacity may still be the binding constraint for GRU
specifically, more than build3's own choice of 128 assumed. h=4 moves the
other way (96 and 128 both beat 192 there), a real trade-off, not noise
in one direction only.

**RNN is messier but not contradictory.** h=4 and h=13 barely move across
96/128/192 (all within about 0.2 NGN/kg of each other -- plausibly seed
noise, not a real capacity effect at this horizon for this architecture).
h=26 shows the same direction as GRU: hidden=192 clearly best (48.39 vs
~49.05 for both 96 and 128), a genuine, if smaller, capacity-helps signal
at the longest horizon.

**Depth loses to width for both architectures, decisively for GRU.** The
2-layer variant (same 128 hidden units as build3's baseline, stacked
instead of widened) is worse than the 1-layer baseline at every horizon
for GRU, and is the only capacity variant that loses to the incumbent at
h=4 (-0.65%, versus +1.90% and +1.41% for the two 1-layer widths). For
RNN, 2-layer roughly ties the 1-layer baseline rather than beating or
badly trailing it. Stacking a second recurrent layer is not a substitute
for a wider single layer on this data, at this scale.

**Consequence.** `hidden=192` (1 layer) is now a stronger candidate than
build3's original `hidden=128` for GRU specifically, and worth adopting
or extending further (`hidden=256`?) as a follow-up, given no plateau was
found yet. Not being adopted as a new "final" recommendation in this
entry without checking seed variance first (D-20's own caution: several
of the margins here, especially at h=4, are the same order of magnitude
as this project's typical per-seed MAE spread) -- that check is the
natural next step before revising D-27's pick.

**Cost.** Six full runs, all complete.

---

## D-31. Continuing the width search: hidden=256

**Decision.** `configs/build3_hidden256.yaml`, extending D-29/D-30's
bracket one step further since no plateau was found at 192. Smoke-verified:
276,099 parameters (up from 192's ~158K), runs end to end.

**Plan.** Run both architectures; if the h=13/h=26 improvement continues,
try a further step (384?); if it plateaus or reverses, that is the
realistic stopping point for width alone on this panel, and stop there
rather than searching indefinitely.

**Cost.** Two full runs, launched next.

---

## D-32. hidden=256 still improving, no plateau at any horizon

**Decision/finding.** Both hidden=256 runs (D-31) complete. Full comparison
against the hidden=192 rows from D-30:

| Run | h=4 MAE | h=13 MAE | h=26 MAE | h=4 vs-fe | h=13 vs-fe | h=26 vs-fe |
|---|---|---|---|---|---|---|
| GRU hidden=192 | 16.55 | 30.22 | 48.14 | +1.41 | -3.85 | -34.63 |
| GRU hidden=256 | 16.33 | 30.13 | 47.91 | +2.77 | -3.55 | -33.97 |
| RNN hidden=192 | 16.94 | 31.64 | 48.39 | -0.87 | -8.75 | -35.32 |
| RNN hidden=256 | 16.20 | 30.83 | 46.53 | +3.53 | -5.94 | -30.13 |

Both architectures improve at every horizon, not just h=13/h=26. GRU's
move is smaller and consistent with the 96->128->192 trend continuing.
RNN's move is larger than anything seen in the bracket so far: h=4 flips
from negative vs-fe at 192 (-0.87%) to clearly positive at 256 (+3.53%),
and h=26 vs-fe closes by 5.2 points (-35.32 -> -30.13), more than it moved
across the entire 96/128/192 bracket in D-30. No plateau or reversal at
either architecture or any horizon.

**Consequence.** Per D-31's plan, this does not stop the search. Params:
276,099 (GRU) / 138,883 (RNN) at 256, both comfortably under the ~5,708
max training rows seen at the final cut, so overfitting-by-parameter-count
is not an obvious concern yet on its own, though this has not been
seed-variance-checked (D-20's standing caution still applies, and applies
more now that the moves are bigger, not smaller). `configs/build3_hidden384.yaml`
created, same single-variable-change pattern as every prior step in this
bracket. Two full runs queued next.

**Note.** The RNN hidden=256 launch failed twice before succeeding, for
tooling reasons unrelated to the model: the background shell had no
`python` on `PATH` (plain `python not found`, exit code 1 on the first
attempt; `source .venv/bin/activate` did not fix it on the second,
suggesting the activation script's PATH change did not propagate to that
shell). Fixed by invoking `.venv/bin/python` directly. Caught before any
run was trusted -- both failed attempts show up as an immediate one-line
error in their output logs, not a completed result, so nothing here was
close to being silently accepted as a real hidden=256 RNN number.

**Cost.** Two full runs, all complete. Two more (384) launched next.

---

## D-33. hidden=384 reverses for both architectures: 256 is the realistic stopping point

**Decision/finding.** Both hidden=384 runs complete. Full comparison
against hidden=256 (D-32):

| Run | h=4 MAE | h=13 MAE | h=26 MAE |
|---|---|---|---|
| GRU hidden=256 | 16.33 | 30.13 | 47.91 |
| GRU hidden=384 | 16.79 | 30.53 | 47.93 |
| RNN hidden=256 | 16.20 | 30.83 | 46.53 |
| RNN hidden=384 | 16.98 | 32.85 | 52.13 |

Both architectures get worse at 384, at every horizon. RNN's reversal is
severe and unambiguous (h=26 alone worsens by 5.6 NGN/kg, more than the
entire 96->256 improvement it had shown up to that point). GRU's is
milder -- h=4 and h=13 clearly worse, h=26 essentially flat (47.91 ->
47.93, a 0.02 NGN/kg move that is almost certainly noise) -- but it is
not a continuation of the 96->128->192->256 trend by any reading.

**Per D-31's own stated stopping rule** ("if it plateaus or reverses,
that is the realistic stopping point for width alone on this panel, and
stop there rather than searching indefinitely"): this is that point.
Hidden=256 is where the width search stops, for both architectures. 384
is not being extended to a further step; there is no signal here that
would justify one.

**What this is and is not.** This settles "how far does widening help,"
not "should build3's production hidden size change from 128 to 256."
D-20's standing caution still applies and has still not been checked at
this scale: every comparison in D-29 through this entry is a single fit
per configuration, and the moves between adjacent widths (especially
128->192->256) are not yet known to be larger than this project's own
seed-to-seed spread. Before recommending hidden=256 as a new production
default, the same seven-seed variance check D-20 ran for build3 at
hidden=128 needs to be run at hidden=256, for both architectures. That
is the natural next step if this is worth taking further; it has not
been done yet, and build3.yaml's hidden=128 default is unchanged.

**Cost.** Two full runs, all complete. No further width runs queued.

---

## D-34. Seed-variance check on hidden=256: real for RNN, not distinguishable from noise for GRU

**Decision/finding.** D-20's own caution, finally applied at this scale.
Both hidden=128 (build3 default) and hidden=256 were fit with the same
seven seeds (0-6), so each seed can be paired across the two configs --
a stronger test than comparing independent spreads, because it removes
whatever that seed's random init/order contributed to both runs and
isolates what changed because of `hidden` alone. Paired MAE difference
(128 minus 256; positive means 256 is better) per seed, one-sample t-test
against zero, df=6:

| Run | h | mean diff | sd | t | significant? |
|---|---|---|---|---|---|
| GRU | 4 | -0.01 | 0.37 | -0.08 | no |
| GRU | 13 | +0.21 | 1.44 | +0.38 | no |
| GRU | 26 | -0.19 | 1.43 | -0.36 | no |
| RNN | 4 | +0.81 | 0.62 | **+3.44** | **yes (p<.05)** |
| RNN | 13 | +0.65 | 1.06 | +1.63 | no |
| RNN | 26 | +2.52 | 2.13 | **+3.14** | **yes (p<.05)** |

**GRU's apparent width improvement (D-29 through D-33) does not survive
this test at any horizon.** The per-seed differences are small and flip
sign about as often as not (e.g. h=26: -1.63, -1.69, +2.51 among others),
consistent with noise, not a real effect of widening from 128 to 256.
The clean, monotonic-looking curve reported across D-29/D-30/D-32/D-33
was real at the level of the seed-median aggregate number, but that
single aggregate number was never itself checked against how much it
moves for reasons unrelated to hidden size -- this is that check, and it
does not hold up for GRU.

**RNN's improvement at h=4 and h=26 is real by the same test** -- both
comfortably clear significance, and the direction is consistent within
each seed's own pair (6 of 7 seeds favour 256 at h=4, 6 of 7 at h=26).
h=13 does not reach significance (t=1.63, needs ~1.94).

**This creates a genuine conflict with a design principle stated in
CLAUDE.md's own project layout**: "`--kind RNN|GRU` selects the recurrent
cell and nothing else, so an RNN-versus-GRU difference is attributable
to gating alone." That guarantee depends on both architectures sharing
one `hidden` value in `build3.yaml`. Adopting hidden=256 for RNN only
would break it; adopting it for GRU too would mean shipping a change for
GRU that this same test says is not distinguishable from seed noise.
Neither option is taken here. `build3.yaml`'s `hidden=128` default is
left unchanged for both architectures pending a decision on which matters
more: architecture-parity in the shared comparison, or RNN's own
strongest available result on its own terms.

**Consequence.** No config change, no promotion to production, nothing
pushed as a result of this entry. This is a documentation-only commit.

**Cost.** None to compute; reused `seed_variance.csv` already on disk
from the D-31/D-33 runs.

---

## D-35. Two safeguards re-run against build3: per-market seed stability, and intervals

**Decision.** `src/seed_analysis.py` and `src/intervals.py` were last run
against build2 only (D-20, D-24), before D-27 made build3 the recommended
candidate. Neither had been rerun since. Both rerun now, scoped to the
FEWSNET output tree only (`outputs/*afex*` excluded via a filtered
`--outdir`, since that tree shares this project's gitignored `outputs/`
directory with the AFEX branch's runs and its files don't carry the same
`convention` column). `outputs/seed_analysis.csv`, `seed_count_curve.csv`,
`intervals_summary.csv`, `intervals_by_regime.csv` and
`intervals_by_market.csv` are now the build3-inclusive versions.

**Finding 1: build3's aggregate margin over naive clears seed noise
everywhere, unlike build2's.** All six build3 unconditional cells
(GRU/RNN x h=4/13/26) now have `margin_over_naive_over_seed_sd` between
1.88 and 11.34 -- comfortably above the 1.0 bar D-20 flagged build2 for
failing at h=4 (GRU 0.88, RNN 0.63). Build3's aggregate result is not a
seed-noise artefact at any horizon, for either architecture.

**Finding 2: per-market verdict stability is not uniformly better, and
RNN h=4 is worse than build2 was.** Markets where the beats-naive verdict
flips depending on which single seed is used, of 15:

| Run | h=4 | h=13 | h=26 |
|---|---|---|---|
| GRU build3 unconditional | 5 | 0 | 2 |
| RNN build3 unconditional | **12** | 1 | 4 |
| GRU build2 unconditional (D-20, for reference) | 9 | -- | -- |
| RNN build2 unconditional (D-20, for reference) | 10 | -- | -- |

GRU improved (9->5 of 15 flipping at h=4). RNN got worse (10->12): at
h=4, 12 of RNN build3's 15 markets have a beats-naive call that depends
on which of the seven fitted seeds you happen to look at. h=13 and h=26
are stable for both architectures (0-4 of 15). Any per-market claim drawn
from build3's h=4 unconditional results is not defensible as reported,
same conclusion D-20 reached for build2, now confirmed current for the
actual recommended build.

**Finding 3: build3's prediction intervals exist now, and 2023-24
undercoverage is real but smaller than build2's was.** At alpha=0.20 (80%
nominal), 2023-24 (fully out-of-sample by construction, 0% calibration):

| Run | h | n | coverage | gap vs 80% nominal | mean width |
|---|---|---|---|---|---|
| GRU unconditional | 4 | 255 | 72.5% | -7.5pp | 103 NGN/kg |
| GRU unconditional | 13 | 287 | 78.7% | -1.3pp | 220 NGN/kg |
| GRU unconditional | 26 | 333 | 74.8% | -5.2pp | 287 NGN/kg |
| RNN unconditional | 4 | 255 | 69.4% | **-10.6pp** | 105 NGN/kg |
| RNN unconditional | 13 | 287 | 77.4% | -2.6pp | 224 NGN/kg |
| RNN unconditional | 26 | 333 | 77.8% | -2.2pp | 310 NGN/kg |

D-24's build2 numbers for the same regime were worse at every comparable
cell: GRU h=26 60.06% (20-point gap), h=13 67.94% (12-point gap), RNN h=4
64.71% (15-point gap). Build3's worst gap (RNN h=4, -10.6pp) is smaller
than build2's best-case gap in that regime. The 2023-24 shock still
breaks calibration for the unconditional arm -- this is not claimed to be
fixed -- but it breaks it less than it used to, consistent with build3's
higher capacity giving it more room to track a regime it cannot see
coming.

**What this does and does not settle before expert review.** Build3 now
has both safeguards current rather than stale. It is not a clean pass:
RNN's h=4 per-market instability is real and worse than build2's, and no
regime here reaches nominal coverage. Both are being handed to reviewers
as known, current limitations, not smoothed into the headline numbers.

**Cost.** Two script reruns, no new training. `outputs/` is gitignored;
only this entry is a tracked-file change.

---

## D-36. Owner instruction: drop the incumbent, naive is the only benchmark

**Decision.** Owner instruction, 2026-09-16: this workstream no longer compares
GRU/RNN build3 against `panel_fe` (the panel fixed-effects incumbent). Panel FE is
its own workstream, managed internally by the owner. The naive (do-nothing)
forecast is the sole benchmark for this project going forward -- for headline
reporting, for any pass/fail judgment, and for how results are framed in
conversation and in documents produced from here on.

**Why this reverses a founding premise of the project, not a small tweak.**
Everything from `CLAUDE.md`'s original framing ("a challenger evaluation for a
production forecasting model") through `change_control.json`'s 5%-vs-incumbent
gate through D-27's own scoring of which candidate to prioritize ("vs-incumbent MAE
... 60%, the metric this whole evaluation exists to answer") was built around
beating panel_fe. That framing is retired as of this entry. It is not being erased
from the record -- every prior decision that reasoned about the incumbent was a
correct description of the evaluation as it was designed at the time, and stays as
written (D-16's own precedent: retain superseded work, do not rewrite it).

**What prompted it, stated plainly.** Reviewing today's fresh Diebold-Mariano
checks (this session, same day): build3 GRU/RNN are statistically indistinguishable
from panel_fe at h=4 and h=13 (p=.40-.80), and significantly *worse* than panel_fe
at h=26 (p<.001, both architectures) -- while both architectures show a real,
DM-confirmed edge over naive at h=4 and h=13. The owner's read: with the incumbent
already covered by its own internally-managed workstream, holding this project's
models to a bar they are not shown to clear (and are shown to lose on at h=26) is
not the useful comparison; whether they beat doing nothing is.

**What actually changes.**
- `CLAUDE.md`'s "What this is", the "recompute the incumbent" hard rule, the
  `change_control.json` rule, the units/conventions line on `vs_panel_fe_pct`, the
  "things that will bite" panel-FE item, the "when something looks too good"
  checklist, and the `src/metrics.py` layout line are all updated to reflect this.
  `RUNBOOK.md`, `docs/METHODOLOGY.md`, `docs/CONDITIONAL_CONVENTION.md` and
  `docs/DATA_AUDIT.md` each get a short banner note pointing here, without
  rewriting their substantive (and still factually accurate) content.
- Nothing is deleted from the codebase. `metrics.paired_table` still recomputes
  `panel_fe` on the same pairs (D-01/D-04's grid-integrity guarantee is a data
  question, not a reporting one, and stays); `change_control()` still exists and
  still runs; `vs_panel_fe_pct` still lands in every `paired_metrics.csv`. Removing
  working, harmless code because a reporting decision changed would be a second,
  unrelated decision, and was not asked for. This entry governs what gets
  *reported and judged*, not what the pipeline is capable of computing.
- Every historical decision entry (D-01 through D-35) that discusses the
  incumbent is left exactly as written. They are accurate history of an evaluation
  that was, at the time, designed around beating panel_fe.

**Consequence for reading anything from D-37 onward.** Any figure phrased as "beats
naive by X%" is this workstream's live comparison. Any figure that would have been
phrased as "beats/loses to panel_fe by X%" should not appear as a headline claim
going forward; if panel_fe numbers are computed for some other internal reason (the
owner's separate workstream may still need them), they are not this project's
result.

**Cost.** None to compute; a documentation and framing change only.

---

## D-37. RNN's real hidden=256 gain, and whether GRU-128 already makes RNN redundant

**Decision/finding.** Two questions asked before deciding whether to give up
RNN's confirmed hidden=256 gain (D-34) to protect architecture parity: how
big is that gain, and does GRU-128 already beat RNN-128 on both accuracy
and direction, which would make the whole question moot.

**How big is RNN's 128->256 gain (D-34 already confirmed it is real at
h=4 and h=26):**

| h | MAE 128 | MAE 256 | improvement | vs-naive 128 | vs-naive 256 |
|---|---|---|---|---|---|
| 4 | 16.91 | 16.20 | 0.72 NGN/kg (4.2%) | +5.9% | +9.9% (+4.0pp) |
| 13 | 31.71 | 30.83 | 0.88 NGN/kg (2.8%) | +17.2% | +19.5% (+2.3pp) |
| 26 | 49.07 | 46.53 | 2.53 NGN/kg (5.2%) | +13.7% | +18.1% (+4.5pp) |

Not huge in absolute terms, but real at h=4 and h=26 (D-34's t-test), and a
meaningful jump in vs-naive terms at those two horizons.

**Does GRU-128 already beat RNN-128 on both accuracy and direction --
tested properly this time, not by the old composite score (D-27), which
answers a different question (which to prioritize overall) than "is GRU
simply better."**

*Accuracy, head-to-head DM test on the actual production forecasts:*

| h | GRU MAE | RNN MAE | dm p | verdict |
|---|---|---|---|---|
| 4 | 16.47 | 16.91 | **.014** | **GRU significantly better** |
| 13 | 30.88 | 31.71 | .159 | tied, not significant |
| 26 | 48.74 | 49.07 | .832 | tied, not significant |

*Direction, McNemar's test on paired correct/incorrect calls (the right
test for two models scored on the identical set of forecasts, not DM
which is for continuous errors):*

| h | GRU correct | RNN correct | McNemar chi2 | verdict |
|---|---|---|---|---|
| 4 | 52.1% | 51.8% | 0.11 | tied |
| 13 | 60.0% | 59.0% | 1.24 | tied |
| 26 | 54.1% | 53.7% | 0.15 | tied |

**GRU does not "beat both" across the board.** It has one real, confirmed
edge: 1-month accuracy. Everywhere else -- 3-month accuracy, 6-month
accuracy, and direction at all three horizons -- the two architectures are
statistically indistinguishable, not GRU-wins. The premise that GRU-128
already makes RNN redundant is not supported by this test; RNN is
carrying real, distinct capability (rough parity on 2 of 3 accuracy
horizons and all of direction, plus its own confirmed capacity headroom)
rather than being a strictly dominated architecture.

**Consequence.** This does not resolve D-34's parity question -- it
answers a different, narrower one (is dropping RNN a free simplification)
with "no, not on this evidence." The choice between protecting parity and
giving RNN its own hidden=256 variant is still open and still the owner's
call.

**Cost.** None to compute; reused predictions_paired.csv already on disk.

---

## D-38. RNN hidden=256 not adopted; build3.yaml stays at hidden=128 for both architectures

**Decision.** Owner decision, 2026-09-16: RNN's hidden=256 result (D-34)
is not adopted. `build3.yaml`'s `hidden: 128` default is unchanged for
both GRU and RNN. This closes the parity-vs-performance question D-34
raised.

**Why, on the evidence.** D-37 tested the actual question that mattered:
does RNN-256 clearly beat the current GRU-128 pick, independent of the
parity principle. It does not. Head-to-head against GRU-128: accuracy
favours RNN-256 numerically at all three horizons but clears significance
at none of them (p=.16/.94/.16); direction is tied at h=4 and h=26 but
**GRU-128 significantly beats RNN-256 at h=13** (60.0% vs 56.7% correct,
McNemar chi2=14.45) -- widening RNN to 256 cost real 3-month directional
accuracy, the horizon this project's own naive-benchmark work (D-24, the
pre-D-36 significance testing) had already flagged as the one place a
result reliably held up. RNN-256 was never a confirmed win being declined
for a procedural reason; it was a mixed result, unconfirmed on the
dimension it improved and confirmed worse on the dimension that mattered
most, once actually tested against the model already in production.

**Consequence.** Both FEWSNET architectures stay at hidden=128, sharing
one `build3.yaml`, preserving the "same everything but the recurrent
cell" guarantee this project's design depends on. `configs/build3_hidden96/
192/256/384.yaml` and `build3_2layer.yaml` remain on disk as documented,
tested, not-adopted alternatives (D-16's retain-don't-delete precedent),
available to revisit if a future change to the data or protocol changes
the calculus.

**Cost.** None; a documentation-only decision.

---

## D-39. Two engineering hygiene fixes from the deployment-readiness audit

**Decision.** First two items of the ML Test Score gap-bridging workplan,
mirrored from `afex-multicommodity`'s D-58 since `requirements.txt` and
`train_one` were byte-identical between branches before this change.

1. **`requirements.txt` pinned to exact versions** (was `>=` ranges):
   torch==2.8.0, numpy==2.0.2, pandas==2.3.3, pyarrow==21.0.0,
   pyyaml==6.0.3, openpyxl==3.1.5. `RUNBOOK.md` documents how to
   intentionally bump a pin (smoke-verify before and after).

2. **`train_one` (`src/model.py`) now fails loudly on a non-finite loss**
   instead of continuing silently. Verified: a normal smoke run
   (`build3.yaml`, GRU) completes unaffected; the detection logic itself
   was verified directly on the AFEX branch (injected NaN input correctly
   raised `RuntimeError`) before being mirrored here, since `train_one`
   is identical code in both places. `run_label` is a new optional
   parameter (default `""`); no existing call site needed to change.

**Cost.** Under ten minutes; a direct mirror of an already-verified fix.

---

## D-40. Test suite mirrored from afex-multicommodity's D-60, plus the same latent bug fixed here too

**Decision.** `tests/` (`conftest.py`, `test_data.py`, `test_model.py`)
and `requirements-dev.txt` copied from `afex-multicommodity`. The suite
is written to be portable across both branches: macro-channel-specific
tests (`TestSequenceChannelNames`, `test_use_macro_*`) detect via
`inspect.signature`/`try: import` whether this branch's `data.py` has
that feature and skip cleanly if not; `num_layers`-specific tests in
`test_model.py` do the same check in the other direction. Result here:
23 passed, 4 skipped (the macro tests, correctly, since that feature is
afex-multicommodity-only); the complementary picture to that branch's 25
passed, 2 skipped.

**The same latent `build_sequence` bug existed here too, since this
branch's version of the function predates today's AFEX-only changes and
has the identical `slice(i - lookback + 1, i + 1)` logic.** Fixed with
the same explicit bounds check, before copying the tests over (so the
copied `test_insufficient_history_returns_not_ok` would actually pass
rather than just being skipped or expected to fail). Confirmed with a
smoke run of `build3.yaml`, GRU: identical numbers (6.86/28.51/48.60 MAE)
to the smoke run in D-39. Same as the AFEX side, this path is never
reached by any real FEWSNET grid, so nothing behind any decision in this
log was ever affected by it.

**Cost.** About 20 minutes: copy, one source fix, one smoke-test
re-verification, full suite passing.

---

## D-41. Integration test and CI mirrored from afex-multicommodity's D-61

**Decision.** `tests/test_smoke.py` and `.github/workflows/tests.yml`
copied unchanged (the fixture and subprocess call are branch-agnostic --
`src/run.py`'s only difference between branches is the pre-existing
`num_layers` config read, which this test's minimal config doesn't
exercise). `requirements-dev.txt` gains `pytest-timeout`.

**Full suite here: 24 passed, 4 skipped** (the macro-channel tests,
correctly, since that feature is afex-multicommodity-only) -- the
complementary picture to that branch's 26 passed, 2 skipped.

**Cost.** About five minutes; a direct, unmodified mirror.

---

## D-42. Equity slice: no evidence the model favours producing over consumption markets, or vice versa

**Decision.** The audit flagged that nobody had asked, of a food-security
tool, whether accuracy is systematically worse for one class of market.
The panel already carries `market_kind` (11 consumption, 5 producing
markets); checked directly against GRU build3, unconditional.

**vs-naive advantage by group, and whether the gap between groups is
itself significant** (Welch's t-test on the per-forecast advantage,
producing minus consumption):

| h | producing vs-naive | consumption vs-naive | group-difference significant? |
|---|---|---|---|
| 4 | +11.3% (DM p=.008) | +6.8% (DM p=.024) | no (t=+1.33) |
| 13 | +21.8% (DM p=.024) | +18.2% (DM p=.005) | no (t=+0.98) |
| 26 | +12.6% (DM p=.340) | +15.1% (DM p=.200) | no (t=-0.52) |

**No group clears the bar at every horizon consistently, and no
horizon shows a significant gap between the two groups.** Both classes
of market do about equally well or badly at the same horizons; there is
no evidence in this test that the model systematically shortchanges
consumption markets (generally poorer, more numerous) in favour of
producing ones, or the reverse. This is a "run and disclose" finding, not
a fix -- reported as found, no action follows from it.

**Cost.** About 15 minutes.

---

## D-43. Model persistence: workplan item 9, the real prerequisite for everything serving-related

**Decision.** Confirmed before starting: no `torch.save` existed anywhere
in this codebase. Every model ever trained behind every entry in this log
was discarded the moment its evaluation script finished. Nothing about
serving, a canary process, or rollback (the audit's Infrastructure tests
#4/#6/#7) was possible until this existed.

**`save_checkpoint`/`load_checkpoint` added to `src/model.py`**: bundles
a model's `state_dict`, its `Scaler`'s fitted numpy state (mean/sd for
sequence channels, flat features, and targets -- a saved model is
useless without the exact scaling it was trained under), and metadata
(kind, hidden, channel/flat counts, market list, `num_layers`) into one
`torch.save`'d file. `load_checkpoint` returns the raw pieces; the caller
reconstructs the actual `RecurrentForecaster`/`Scaler` objects, so
loading never silently depends on those classes' current implementation
matching what produced the file.

**Wired into `src/run.py` behind `--save-latest-cut-models`, off by
default.** When set, only the most recent retrain cut's seven seed
checkpoints are kept (the previous cut's directory is deleted before the
new one is written), so a full 35-cut run doesn't accumulate hundreds of
checkpoints by default -- this answers 2.1's own storage caveat directly
rather than leaving it as an open question.

**Verified three ways, in increasing order of realism.** (1) A synthetic
unit test (`tests/test_model.py::TestCheckpointRoundTrip`): train a tiny
network, save it, reload it, assert the reloaded model's predictions are
bit-for-bit identical to the original's. (2) A real smoke run of
`build3.yaml` GRU *without* the new flag: identical numbers
(6.86/28.51/48.60 MAE) to every prior smoke run today, confirming zero
default-behaviour change. (3) The same smoke run *with* the flag: exactly
one `models/cut_<date>/` directory exists afterward (the last of the
three smoke cuts, not all three), and the reloaded checkpoint's metadata
and parameter count (72,579, matching `hidden=128`) are exactly right.

**Not yet done, and explicitly out of scope for this entry**: `run_afex.py`
does not have this wired in yet (same pattern, not applied); the
"which saved model counts as *the* current one" convention (workplan
2.2) and the serving-cadence decision (3.1) are still open and are what
this unlocks, not what it resolves.

**Cost.** About an hour: the save/load functions, the run.py wiring, one
new unit test, three real verifications.

---

## D-63. Process error, and the finding it surfaced: D-47's directional result does not replicate

**Decision.** Launching more seeds for RNN operational's h=13 directional
result (the workplan's own next step after D-47) was run with
`--seeds 0 1 ... 11` but **no `--out` override**, so it wrote to the
default path and overwrote `outputs/afex_operational/RNN/` -- the
original 7-seed production output -- instead of writing alongside it.
This breaks D-16's own retained-evidence precedent, which had otherwise
been followed carefully everywhere else today (D-45's superseded-output
handling, D-50's correction, D-56's ablation). No entry already written
in this log depends on the raw file surviving, but the exact 7-seed
`forecasts.csv`/`predictions_paired.csv` D-47 was computed from no
longer exists, and since these AFEX runs use `--device mps` -- which
this project's own `RUNBOOK.md` already documents as not bit-reproducible
run to run, unlike `cpu` -- rerunning seeds 0-6 does not reliably recover
the original numbers. Flagged plainly rather than smoothed over.

**What the new 12-seed run actually shows, and it is the most important
finding to come out of today's AFEX work.** Recomputing D-47's exact
seed-count-curve method on the new data:

| k (seeds) | mean direction hit-rate | % of subsets beating both baselines |
|---|---|---|
| 1 | 47.6% | 0.0% |
| 6 | 49.7% | 0.1% |
| 12 (all) | 49.7% | 0.0% |

**This does not replicate D-47's finding at all.** D-47 found a smooth,
monotonic climb from 52.8% (k=1) to 61.0% (k=7, all seeds), read as
noise cancelling out to reveal a real signal. The new, larger sample
shows no such climb -- it sits flat around 48-50% (a coin flip) at every
seed-count, and the fraction of subsets beating both naive baselines
never exceeds 2.7% at any k, against D-47's 100% at k=7. Checked directly:
seeds 0-6 *in this new file* (the same seed numbers D-47 used) give a
median direction hit-rate of 47.4%, not 61.0% -- confirming this is not
just "different seeds happened to land differently," the same seed
numbers produced a materially different result on MPS this time.

**Consequence for D-39's selection.** RNN operational was picked as the
"direction pick" specifically on the strength of D-47's h=13 result. That
result does not hold up under a larger, independent seed sample. Of the
two things this AFEX exploration had going for it after today's full
rigor pass (D-49's GRU full-exog DM significance, and D-47's RNN
directional signal), **only one survives**: GRU full-exog's h=13 edge
over naive (D-49) is unaffected by any of this, since it is a
Diebold-Mariano test on the actual historical forecast record, not a
seed-retraining question, and its underlying file was never touched
today. RNN operational's own selection rationale is now unconfirmed,
not merely "not yet firmed up" as D-47 concluded.

**What this does not do.** It does not prove RNN operational is worse
than GRU full-exog at direction, or that no real signal exists -- only
that the specific evidence claimed for one did not survive a proper
retest. No config change is made here; this is a finding, and D-39's
standing selection is now flagged as resting on weaker ground for its
RNN half than previously believed.

**Process fix going forward.** Any future exploratory run that is not
meant to become the new production artifact gets an explicit `--out`
override to a clearly-named side directory, full stop, regardless of how
routine the run seems.

**Cost.** One seven-seed production artifact is gone and not
recoverable; the replacement finding is more informative than what was
lost.

---

## D-44. Monitoring workplan items 10-12: data invariants, a regression-warning tool, training-cost tracking

**Decision.** Three small, independent additions, each opt-in or
additive, each verified to leave every existing behaviour unchanged.

1. **`validate_panel(panel)`** (`src/data.py`), called at the end of
   `load_panel`: no duplicate market names, no market with zero observed
   prices at any date, no non-positive price where a value is present
   (a real gap is `NaN`, never `<=0`; a `<=0` value means an upstream
   data or unit bug). The pre-existing weekly-spacing check stays where
   it was, on the raw dates before the `Panel` is built. Verified against
   the real panel (`data/panel_weekly.parquet` loads and validates
   cleanly) and four new unit tests.

2. **`src/check_regression.py`**: flags a `(build, model, convention, h)`
   whose MAE moved by more than a threshold (default 10%) since its
   previous run. This required a real design fix mid-build: the first
   draft compared the two most recent `paired_metrics.csv` files on disk,
   but each run *overwrites* its own file at a fixed path, so there is
   never more than one snapshot per config to compare against -- caught
   before shipping it, not after. Fixed with an actual history:
   `append_run_history` (same file) appends one row per horizon to
   `outputs/run_history.csv` at the end of every real (non-smoke)
   `run.py` invocation, and `check_regression.py` reads that log. A
   warning tool, not a gate -- it never fails the run it's called from.
   Six new unit tests.

3. **Training wall-clock time recorded**: `run_metadata.json` gains
   `wall_seconds`; every real (non-smoke) run also appends a row
   (timestamp, build, kind, hidden, params, device, wall_seconds) to
   `outputs/training_time_log.csv`, so a future slowdown is visible as a
   trend rather than only felt anecdotally.

**Verified**: a smoke run of `build3.yaml` GRU produces identical MAE
numbers to every prior smoke run today (6.86/28.51/48.60), correctly
does *not* write to `run_history.csv` or `training_time_log.csv` (both
are non-smoke-only by design), and does report `wall_seconds` in its own
metadata. Full suite: 35 passed, 4 skipped.

**Cost.** About 90 minutes, including the mid-build redesign of
`check_regression.py`.

---

## D-45. Workplan 5.1: a real pass/fail bar for the post-D-36 world

**Decision.** D-36 retired `change_control()`'s 5%-vs-incumbent gate for
this workstream without replacing it with anything -- there was no
answer to "how would we know a future result is good enough to act on,"
only "how would we know if it beats naive at all." `src/quality_gate.py`
answers that, as a standalone post-hoc tool in the same family as
`intervals.py`/`seed_analysis.py`, not a change to `run.py`'s live loop:

**A horizon passes only if both:**
1. Diebold-Mariano p<0.05 vs naive (already computed into every run's
   `diebold_mariano.csv`) -- is this specific deployed forecast's
   historical track record distinguishable from chance.
2. `margin_over_naive_over_seed_sd` > 1 (D-20/D-34's own bar, reusing
   `seed_analysis.analyse_run_horizon` rather than reimplementing it) --
   would a re-trained model likely reproduce this margin.

**Why both, not either.** Today's own work found real cases where a
result passes one and fails the other -- D-46's GRU full-exog (passed
seed-variance, initially miscomputed as failing DM until D-50's
correction) and D-49 (passed DM, seed-variance found separately in D-50)
being the clearest example. Requiring both is deliberately the more
conservative reading of a result, not the more convenient one.

**Verified against the actual current models**: `python
src/quality_gate.py --run-dir outputs/build3_underfit_corrected/
GRU_unconditional --gate-horizons 4 13` reports PASS at both gate
horizons (dm_p=.001/.0005, seed ratios 5.84/10.36), matching every DM
and seed-variance number already on record from today's work exactly.
RNN build3 also PASSes both. Three new unit tests, including one that
required fixing a genuine bug in the *test fixture itself* before it
would pass: independently noisy predictions from several seeds beat a
single naive guess through ensembling's own variance reduction even with
zero true architectural edge, so a fair "this model has no real skill"
fixture has to track naive's own guess, not just be noisy around the
truth at naive's noise level -- getting this fixture right was itself a
small, concrete demonstration of the exact ensembling-reduces-noise
effect this project's own `seed_aggregation: median` setting (D-11)
relies on.

**Cost.** About an hour, including debugging the test fixture. `main`
mirror queued next.

---

## D-46. A cheap GRU+RNN ensemble trial: not a confirmed win, not a loss either

**Decision.** D-28 found ensembling build2+build3 doesn't help. Different
question, not yet asked: does a plain 50/50 average of the two *current*
build3 architectures (GRU, RNN) do any better than either alone. No new
training -- reuses `predictions_paired.csv` already on disk for both.

| h | GRU MAE | RNN MAE | 50/50 ensemble MAE | naive MAE |
|---|---|---|---|---|
| 4 | 16.47 | 16.91 | 16.49 | 17.97 |
| 13 | 30.88 | 31.71 | 30.79 | 38.30 |
| 26 | 48.74 | 49.07 | **47.88** | 56.83 |

**DM test, ensemble vs. each architecture and vs. naive:**

| h | vs GRU | vs RNN | vs naive |
|---|---|---|---|
| 4 | p=.81 (tied) | p=.78 (tied) | **p=.001 (real)** |
| 13 | p=.78 (tied) | p=.78 (tied) | **p=.0005 (real)** |
| 26 | p=.33 (tied) | (not separately tested, same direction) | p=.115 (not quite) |

**Reading it straight.** The ensemble is never significantly better than
either architecture alone -- it is not a confirmed improvement. But it is
never worse either, and it is numerically the best of the three at h=26
(47.88, beating both GRU's 48.74 and RNN's 49.07 by more than either beats
the other). This is a mild, plausible-but-unconfirmed candidate, not a
result on the strength of D-28's own bar. Not adopted; noted as a cheap,
low-risk option if simplicity of a single architecture is not a
requirement.

**Cost.** None to compute; reused existing files.

---

## D-47. Four remaining workplan decisions, made and documented rather than left open

**Decision.** The audit's four decision-gated items (2.2, 3.1, 5.1's
sibling questions 3.3/3.4), resolved now that the engineering
prerequisites (D-43's persistence, D-45's gate) exist to act on them.

**2.2, which saved model counts as "current":** the most recent retrain
cut's artifact, not a separately-promoted one. This project already
retrains on a fixed cadence (13 weeks) rather than on-demand; adding a
separate promotion step would be process overhead this team's size and
cadence doesn't need yet. Convention: `outputs/<build>/<kind>_<convention>/
models/` (D-43) always holds exactly the latest cut by construction
(older cuts are deleted as each new one is saved), so "current" is simply
"whatever's there" -- no separate pointer file needed, the directory
already only ever contains one answer.

**3.1, what "serving" means:** a weekly batch job, not an on-demand
service. Matches the actual data cadence (weekly prices, 13-week
retrains) and needs no uptime/latency engineering a request-driven
service would. `src/predict.py` (new) implements this: load the current
checkpoint, load the latest available data, produce one batch of
h=4/13/26 forecasts, write them to a dated file, refuse to run on stale
input rather than silently forecasting from it.

**3.3, canary convention:** `predict.py` always writes the current
model's forecasts alongside the immediately-previous cut's, for at least
one full retrain cycle, so a newly promoted model is never trusted alone
before there is something to compare it against.

**3.4, rollback convention -- corrected while writing RUNBOOK.md, not
left as first drafted.** The first draft of this entry claimed rollback
was free because "D-43 already keeps checkpoints on disk." That is not
what D-43 actually built: it deletes the *previous* cut's checkpoints
before saving each new one, specifically to avoid disk bloat, which
means there is no history on disk to roll back to by design. Caught
before committing rather than after. The honest convention, written into
`RUNBOOK.md`: rolling back means either re-running training up to the
desired earlier cut (real compute cost, since nothing was kept), or
manually copying a checkpoint elsewhere before the next save would
overwrite it, a step someone has to remember to take, not one this code
does automatically. This is a real, acknowledged gap between "avoid
accumulating hundreds of checkpoints" and "support instant rollback" --
both are reasonable goals and this entry does not pretend they were
reconciled for free.

**Cost.** Four decisions plus `src/predict.py`, described in the next
entry.

---

## D-48. `src/predict.py` built and verified against a real checkpoint

**Decision.** The actual inference path (D-47's 3.1/3.2/3.3), built and
tested against a real (smoke-scale) checkpoint before being trusted, not
just the synthetic unit fixtures. Loads the one checkpoint D-43's save
convention keeps, the same panel path the training run used, reuses
`build_sequence`/`build_flat`/`Scaler` unchanged (train/serve skew is
not possible by construction, not by a separate check), and writes one
forecast per scored market per horizon from the panel's most recent
available week.

**Three real behaviours verified directly, not assumed:**
1. **Basic prediction**: ran against a real smoke checkpoint, produced
   48 rows (16 markets x 3 horizons) with plausible price levels and the
   correct origin date (the panel's actual last date, 2024-09-18, not
   today's date -- confirms it is forecasting *from* the data's edge, not
   confusing "when this script runs" with "what the data supports").
2. **Staleness gate**: with a realistic 2-week limit, correctly refused
   with exit code 1 and a clear message (the real panel is 104 weeks
   stale relative to today, so this is the actually-correct behaviour
   right now, not a hypothetical).
3. **Canary comparison**: with a synthetic prior prediction file placed
   in the same directory, correctly found and merged it, printing both
   forecasts side by side.

**Cost.** About 90 minutes including the rollback-claim correction
above. Closes the workplan's Phase 3 to the extent it can be closed
without a live serving environment to deploy into, which does not exist
and was never in scope for today.
entry once built and verified.
