# Which driver convention did the incumbent use?

This is the most consequential open question in the comparison, and it had to be
inferred because the script that produced `07_panel_fe_forecasts.parquet` was not
available.

## Why it decides the comparison

Both models need values for diesel, upstream price, rainfall and NDVI over the
forecast window. There are two conventions:

- **Unconditional.** Only information available at the origin. This is what
  deployment looks like: on 16 September 2015 nobody knows March 2016 rainfall.
- **Conditional.** Realised driver values plugged in as though known. This
  measures the model's mapping from drivers to price, not its forecasting
  accuracy.

Conditional numbers are always better, and the gap widens with horizon. Charging
a challenger for driver uncertainty while the incumbent was handed realised
drivers would make it lose for a reason that has nothing to do with the model.
§8.4 of the prior session's notes flagged this and left it open.

## Method

`src/diagnose_convention.py` rebuilds the incumbent both ways from the documented
specification: own-price lags 1, 2, 4, 13, 52; market fixed effects; no time
effects; diesel, upstream price, inverse-distance rainfall and NDVI; Fourier K=2;
direct h-step; expanding walk-forward refit at every origin on the real grid,
training only on windows whose target had already been realised by that origin.

The test is not which replica forecasts better. It is which replica's
**predictions sit closer to the published ones**, because that is what identifies
the convention rather than the quality of the fit.

## Result

    convention      h     n   replica_MAE  published_panel_fe_MAE  naive_MAE  replica_vs_published_MAD  corr
    unconditional   4  1469         18.71                   15.96      17.87                     10.51  0.99
    unconditional  13  1466         36.45                   27.08      38.38                     31.98  0.96
    unconditional  26  1469         53.28                   31.53      57.10                     57.30  0.95
    conditional     4  1463         13.78                   16.02      17.93                      8.79  0.99
    conditional    13  1472         18.30                   26.99      38.24                     20.54  0.98
    conditional    26  1469         21.82                   31.50      57.08                     27.38  0.98

Two readings, both pointing the same way.

**Agreement with the published predictions.** At h=13 the conditional replica sits
20.54 NGN/kg from the published forecasts against the unconditional replica's
31.98. At h=26 it is 27.38 against 57.30, a factor of two. If the incumbent had
been built from origin-time information only, the unconditional replica should
have been the closer of the two, and it is not.

**Achievability.** The unconditional replica cannot get near the published MAE at
long horizons: 53.28 against 31.53 at h=26, 69% worse. The conditional replica
comfortably exceeds it, 21.82 against 31.50. The published figures sit inside the
conditional envelope and well outside the unconditional one.

## Verdict

**The incumbent was almost certainly produced conditionally, on realised driver
values.**

Strength of evidence: strong, not conclusive. This is a crude ordinary-least-
squares replication, not the original code. It uses no regularisation, no ±3 SD
clipping (the §8.2 recommendation), and levels rather than logs, because logs
performed markedly worse. The conditional replica *beating* the published figures
by a wide margin suggests the real incumbent is more constrained than this
replica in some way, which is consistent with clipping or shrinkage. Any of those
differences could move the numbers; none of them plausibly reverses a factor-of-
two gap in prediction agreement at h=26.

The original script would settle it in one reading. Until it appears, treat this
as the working assumption and re-run the diagnostic if the script turns up.

## What this changes about reporting

**The change-control gate can only be adjudicated on the conditional arm.**
Section 5.3 compares a challenger against the incumbent. The incumbent exists only
as a conditional artefact. Comparing an unconditional challenger against it is not
a comparison of models; it is a comparison of information sets.

So:

| arm | what it is for | comparable to the incumbent? |
|---|---|---|
| conditional | the Section 5.3 verdict | yes |
| unconditional | the only honest statement of deployment accuracy | no |

Both are produced for every build and architecture. `change_control.json` should
be read from the conditional runs. The unconditional runs are reported alongside
and the gap between them is itself the finding: it quantifies how much of the
incumbent's advertised accuracy depends on knowing drivers in advance.

## What to say out loud

Never present a conditional figure as an accuracy anyone will get. The honest
sentence is: *"With rainfall, diesel and upstream prices known in advance, the
model achieves X. Without them, which is the deployment case, it achieves Y. The
production model has only ever been measured the first way."*

The second half of that sentence is a live governance issue for the incumbent, not
just for these challengers. Two related items to raise at the next review:
whether the production model's published accuracy should be restated
unconditionally, and whether Phase 2 should retain the conditional convention at
all.
