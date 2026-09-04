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
