# Monthly-cadence evaluation: protocol (draft for owner sign-off)

**Status.** Draft, 2026-10-02. Phases 0 to 2 done (data, this protocol, baselines). No
model trained. Phase 3 (the monthly GRU) starts only after the owner signs off on this
document. Phase 4 (post-September-2024 data, live forecasts) is on hold by owner decision.

## Owner decisions this protocol follows (2026-10-02)

1. A **separate evaluation**, not a change to the weekly flagship's protocol. The weekly
   flagship (D-59) stays production; the frozen weekly grid and its locked values
   (§5.2/§5.3, D-01) are untouched.
2. **Same period as the flagship:** data through 2024-09-18 only.
3. **No pooling** of other commodities.
4. **Same inputs as the flagship, no FX.** Cadence is the only difference.

## Data (phase 0, `src/build_monthly_panel.py`)

- `data/panel_monthly.parquet`: 16 markets x 149 months (2012-05 to 2024-09), 2,384 rows.
  A month's price is the mean of its non-missing weekly prices, proxy-filled weeks
  included because the flagship trains on them too; at least 2 observed weeks are
  needed, otherwise NaN (9 months set NaN on this rule). Every other input is the
  monthly mean of the weekly values, so units match. Fourier terms use month of year.
  September 2024 has 3 weekly observations in every market.
- `data/monthly_grid.parquet`: the monthly evaluation grid, built once and frozen like
  the weekly one. An origin is the end of a completed month o; naive is month o's
  average; targets are the averages of months o+1, o+3 and o+6. Each market's origins
  span the same months as its weekly grid origins (first 2015-09, last 2024-03). An
  origin is kept only if its own price and all three targets exist (none dropped).
  **1,477 pairs per horizon** (weekly grid: 1,590).
- Counts behind both files: `data/panel_monthly.log.json`.

## The model (phase 3, not started)

`configs/build3.yaml` translated to months, nothing else changed:

| setting | weekly flagship | monthly |
|---|---|---|
| horizons | 4 / 13 / 26 weeks | 1 / 3 / 6 months |
| lookback | 26 weeks | 6 months |
| seasonal anchors | lag 52 weeks | lag 12 months |
| purge | 26 weeks (falls back 13, 0) | 6 months (falls back 3, 0) |
| retrain cadence | every 13 weeks | every 3 months |
| price interpolation limit | 13 weeks | 3 months |
| hidden, embedding, dropout, LR, epochs, seeds | 128, 8, 0, 3e-3, 200, 7 | same |
| inputs | price, diesel, upstream, rainfall, NDVI, 4 Fourier, upstream mask, 6 lag anchors | same, monthly |

Training windows shrink about fourfold (about 1,400 at the last cut versus 5,708). Hidden
128 is kept because the owner asked for cadence to be the only difference; it is more
capacity per window than the weekly model has, which is a known risk on this panel
(D-34, D-61), and is reported as such rather than tuned away.

Code needed: the weekly assumptions in `src/data.py` (the 7-day spacing check, lag 52
in `build_flat`), `src/walkforward.py` (retrain and purge in weeks) and `src/run.py`
(weekly horizon arithmetic) become frequency-aware, with weekly behaviour unchanged by
default and checked by the existing tests.

## How it is judged

1. **Primary, on the monthly grid:** the monthly GRU against monthly naive, by the D-45
   rule (DM p < .05 and a margin larger than one seed standard deviation), at 1, 3 and
   6 months.
2. **Head-to-head with the weekly flagship, on matched pairs:** a weekly pair (market,
   t, h) is matched to monthly pair (market, o, k) when t falls in the last 7 days of
   month o (so both forecasters know about the same data) and the target week t+h falls
   in month o+k. Both are scored against month o+k's average. 1,009 matched pairs in
   total, about 255 to 265 per horizon once origins without a seasonal index are set
   aside. The monthly model "wins" a horizon only with DM p < .05 on these pairs and a
   seed-paired t beyond 2.447, the same bar every change to the flagship has had to clear.
3. Also reported: MAPE, direction, results by period (2015-19, 2020-22, 2023-24).

**Known biases in the matched comparison, stated up front.** A month's average is easier
to forecast than a single week (the weekly GRU's 1-month MAE is 19.90 against its own
weekly actuals and 15.90 against the monthly average on the same pairs). The weekly
forecast targets a single late-month week while it is scored against the whole month,
which slightly handicaps it. The weekly model may know up to 6 more days of data than
the month-end, or a few fewer.

## The bar (phase 2, `src/monthly_baselines.py`, all point-in-time)

Full monthly grid, MAE NGN/kg / MAPE (each baseline on the pairs where it exists):

| | 1 month | 3 months | 6 months |
|---|---|---|---|
| monthly naive (n=1,477) | 16.37 / 8.6% | 36.16 / 17.5% | 54.95 / 22.8% |
| seasonal naive (n=1,131) | 17.81 / 8.6% | 36.68 / 16.0% | 60.17 / 21.9% |
| FEWS NET Holt-Winters (n=1,458) | 16.45 / 9.0% | 34.17 / 17.7% | 50.56 / 24.4% |
| FEWS NET decomposition (n=1,458) | 65.02 / 37.6% | 77.31 / 41.6% | 100.13 / 47.3% |

Matched pairs, scored against the target month's average:

| | 1 month (n=264) | 3 months (n=253) | 6 months (n=261) |
|---|---|---|---|
| **weekly GRU flagship** | **15.90 / 7.9%** | **31.98 / 16.5%** | **44.62 / 21.2%** |
| weekly corrected blend | 15.62 / 7.6% | 33.02 / 16.4% | 45.90 / 19.0% |
| monthly naive | 19.91 / 8.9% | 43.10 / 20.0% | 57.30 / 19.9% |
| seasonal naive | 19.63 / 9.3% | 37.82 / 18.0% | 58.27 / 21.8% |
| FEWS NET Holt-Winters | 19.15 / 9.1% | 39.59 / 19.4% | 53.34 / 23.1% |

The monthly GRU's target is the first row: the weekly flagship already beats every
monthly baseline on matched pairs by 17 to 26% MAE.
