# Methodology

**As of D-36 (2026-09-16, see `CLAUDE.md`), this workstream's only benchmark is the
naive (do-nothing) forecast.** References below to the incumbent/`panel_fe` describe
the evaluation as it was designed and run before that date; they are accurate
history, not current comparison targets.

Technical specification of both builds. Decisions are cited as D-nn against
`DECISIONS.md`, where the reasoning and cost of each sits.

## 1. Task

Given the weekly price history of a Nigerian maize market up to week `t`, forecast
the wholesale price at `t+4`, `t+13` and `t+26` weeks (approximately 1, 3 and 6
months; §5.2 and §8.4 of `PROJECT_HANDOFF.md`). Primary metric MAE in NGN/kg.
Secondary: MAPE, directional accuracy. Change-control gate at h = 4 and h = 13
(§5.3).

## 2. Evaluation protocol

The grid is read from `07_panel_fe_forecasts.parquet`, not regenerated (D-01).
1,592 `(market, origin)` pairs per horizon after D-02, across 15 scored markets and
432 origin dates, 2015-09-16 to 2024-03-20. Each market's own origins are spaced 28
days; markets are staggered against each other.

Origin price and target actuals are taken from the baseline file's `naive` and
`actual` columns. `naive` at any origin is the price at that origin, so this
guarantees challenger and incumbent are scored against identical values and any
MAE difference is attributable to the model.

Walk-forward is expanding: training data grows at each refit and nothing is
dropped (§5.2). A fit at cut `c` uses only windows whose **last target** lands at
or before `c - 1` weeks, enforced in `build_training_windows` via
`origin_max_idx = cut_i - 1` combined with the `i + max_h <= origin_max_idx`
loop bound. No test window, and no statistic derived from one, reaches any fit.

Refit cadence is separate from origin cadence: every 26 weeks in Build 1, every 13
in Build 2 (D-14). Between refits the model forecasts from a fit that is up to that
many weeks stale, which is harder than refitting per origin, not easier.

## 3. Target representation

The single most important choice, carried over from the original.

Predict log-returns relative to the origin:

    y_h = log(P[t+h]) - log(P[t])

Recover a price by `P_hat[t+h] = P[t] * exp(y_hat_h)`.

Two reasons. **Range.** Prices in the panel run 28.4 to 1,221.6 NGN/kg and most of
the climb is 2023-24. A network trained on levels through 2022 cannot represent
2024 levels without extrapolating past everything it has seen, and saturating
activations mean it flattens instead. **Pooling.** A 10% move in Gombe at ~200
NGN/kg and a 10% move in Lagos at ~600 are the same number in log-return space, so
one shared pattern is learned instead of fifteen market-specific ones.

Every input channel derived from a price or cost is expressed the same way, as a
log ratio to its own value at the origin. A window therefore contains no price
level anywhere. This is scale invariance by construction rather than by scaling.

## 4. Inputs

### 4.1 Sequence channels (52 weeks, 10 channels, both builds)

| # | channel | form |
|---|---|---|
| 1 | own price | `log(P[t-k]) - log(P[t])` |
| 2 | diesel | `log(D[t-k]) - log(D[t])`, state-level |
| 3 | upstream price | `log(U[t-k]) - log(U[t])`, zeroed when unavailable |
| 4 | rainfall | inverse-distance weighted, level |
| 5 | NDVI | inverse-distance weighted, level |
| 6-9 | Fourier | `sin1, cos1, sin2, cos2` at period 52.18, K=2 |
| 10 | upstream mask | 1 if the market has an upstream, else 0 |

Rainfall and NDVI stay in levels because they are already comparable across
markets and have no trend to remove; a log ratio to the origin week would be
meaningless for rainfall, which is frequently zero.

Channel 10 exists because Dandume is the hierarchy root and has no upstream price
at any date. Without the mask a zeroed channel 3 reads as "upstream price did not
move," which is a different claim from "there is no upstream price."

Excluded per owner decisions of 2026-08-11: FX (§8.2 option C); own-market price
lags as flat exogenous features (§8.1 option a, though they enter through the
recurrent channel, which is the point of using a sequence model at all); fertiliser.

### 4.2 Flat features

Concatenated to the recurrent summary and the market embedding before the output
head. Dimensions by configuration:

| build | unconditional | conditional |
|---|---|---|
| Build 1 | 0 | 12 |
| Build 2 | 6 | 18 |

**Build 2's six (D-06).** For each h in {4, 13, 26}:

    lag52_anchor_h        = log(P[t+h-52]) - log(P[t])
    lag52_window_return_h = log(P[t+h-52]) - log(P[t-52])

`t + h - 52 <= t` for all h <= 52, so both are observed at the origin.
`build_flat` returns a failure flag rather than a value if an index falls out of
range, and the window is dropped and logged.

**Conditional arm's twelve (D-15).** For each h: diesel log-return origin to
target, upstream log-return origin to target, mean rainfall over `(t, t+h]`, mean
NDVI over the same interval. This is the information the incumbent appears to have
had.

### 4.3 Scaling

`Scaler` fits on training windows only, per cut. Sequence channels are pooled over
time before computing per-channel mean and standard deviation. Flat features are
scaled per column. Targets are scaled per horizon in Build 2 and left raw in
Build 1 (D-07, D-04).

## 5. Architecture

One pooled model over all training markets, with a learnable 8-dimensional
embedding per market. Not fifteen separate models: at roughly 500 usable
observations per market, a per-market fit would carry more parameters than
observations.

    x (batch, 52, C) -> input dropout -> RNN or GRU, 1 layer, hidden 64
                     -> final hidden state h (batch, 64)
    concat[h, embedding(market), flat] -> Linear(64) -> ReLU -> dropout
                                       -> Linear(3)

Output is direct multi-horizon: all three horizons from one forward pass, not
recursive. Recursive multi-step would feed a predicted h=4 back in to reach h=13,
compounding its error; direct prediction pays no compounding cost but must learn
three mappings from one state, which is what D-06 addresses.

The only structural difference between `--kind RNN` and `--kind GRU` is the
recurrent cell. Embedding, head, loss, optimiser, scaling and splitting are shared,
so any difference in results is attributable to gating alone.

Parameter counts are reported rather than targeted (D-05). Measured: 9,859 for
Build 1 RNN with zero flat features, 20,739 for Build 2 GRU with 18. Counts and a
per-tensor breakdown go into `run_metadata.json` for every run.

## 6. Training

Adam, Huber loss with delta 1.0 applied per horizon then averaged, gradient-norm
clipping at 1.0, batch 256. Chronological train/validation split by origin date;
early stopping on validation loss, restoring the best state.

| | Build 1 | Build 2 |
|---|---|---|
| seeds | 2 | 7 (D-11) |
| aggregation | mean | median (D-11) |
| epochs | 80 | 200 |
| learning rate | 5e-3 flat | 3e-3 cosine (D-10) |
| patience | 15 | 25 |
| weight decay | 0 | 1e-4 (D-09) |
| head dropout | 0 | 0.2 (D-09) |
| input dropout | 0 | 0.1 (D-09) |
| validation purge | none | 78 weeks, with fallback (D-08) |
| recency half-life | uniform | 156 weeks (D-12) |
| target scaling | none | per horizon (D-07) |
| training markets | 15 | 16, still scoring 15 (D-03) |
| refit cadence | 26 weeks | 13 weeks (D-14) |

Recency weights enter the loss as sample weights, not by resampling, so every
window still contributes gradient.

## 7. Reported statistics

`paired_metrics.csv` is the headline: per horizon, the challenger's MAE, MAPE and
directional accuracy alongside `panel_fe` and `naive` recomputed on the identical
pairs, plus percentage differences where positive means the challenger is better.

`change_control.json` adjudicates §5.3 for a single review: >5% better than
`panel_fe` at both h=4 and h=13. The rule also requires the result to hold across
two consecutive quarterly reviews, which one run cannot establish, and the file
says so in its verdict string.

`diebold_mariano.csv` tests whether an MAE difference is distinguishable from zero,
using a Newey-West correction with `h-1` lags because overlapping forecast windows
make the error differentials autocorrelated by construction. A 5% MAE gap that
fails a DM test is not a result. The change-control rule does not require this
test; it is reported because a threshold rule with no significance check can be
tripped by noise, which is the failure mode §5.3 exists to prevent.

`seed_variance.csv` and the `pred_sd` column of `predictions_paired.csv` report
initialisation spread. Read these before believing any margin narrower than the
spread.

`metrics_by_market.csv` gives per-market MAE against the incumbent with a
`beats_panel_fe` flag, since an aggregate can hide a model that wins in three
markets and loses badly in twelve.

## 8. What is logged

`cleaning_log.json` per run: interpolated price cells and the limit applied,
full-panel outage weeks, grid rows dropped and why, every skipped window with its
reason, and scored pairs against expected pairs so a silent coverage loss cannot
pass unnoticed.

`split_log.csv` per cut: requested and applied purge, training and validation
counts. The applied purge differs from the requested one at early cuts and needs to
be visible.

`training_log.csv` per cut and seed: epochs run, best validation loss, training
size, parameter count. Epochs clustering at the patience boundary, or best
validation loss varying widely across seeds at the same cut, both indicate that
early stopping is firing on noise.

`run_metadata.json`: the entire resolved config, panel audit, grid audit, channel
and flat-feature names, seeds, parameter breakdown.

## 9. Known limitations

1. **The incumbent's convention is inferred, not read.** See
   `CONDITIONAL_CONVENTION.md`. Strong evidence, not proof. If the panel FE script
   appears, re-run `src/diagnose_convention.py` and revisit which arm carries the
   §5.3 verdict.
2. **No prediction intervals.** §5.5 requires empirically calibrated intervals
   from a rolling window of prior errors. Not implemented; the challenger is
   assessed on point accuracy only. If either build ever cleared the gate, intervals
   would be required before deployment.
3. **Hidden size is not tuned** (D-13). Build 2 may be under-capacity or
   over-capacity.
4. **Conditional flat features are a reasonable guess at the incumbent's
   information set, not a reconstruction of it.** Mean rainfall over the interval
   is one of several defensible encodings.
5. **282 interpolated price cells** enter training and 1,592 grid windows may each
   contain up to three. The incumbent's lag features draw on the same panel, so this
   is shared rather than differential, but it is not zero.
6. **Nickell bias** is inherent to lagged-dependent-variable panels and applies to
   the incumbent, not to these networks, which have no fixed effects to bias. Noted
   because it affects reading the incumbent's coefficients, not its forecasts.
