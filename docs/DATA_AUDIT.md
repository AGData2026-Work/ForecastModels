# Data audit

**As of D-36 (2026-09-16, see `CLAUDE.md`), this workstream's only benchmark is the
naive forecast; incumbent/`panel_fe` figures below are data-provenance history, not
a live comparison target.**

Everything below was measured from the supplied files, not carried over from
notes. Reproduce with `python src/audit.py`.

## Files used

| File | Role |
|---|---|
| `panel_weekly.parquet` | the modelling panel. 10,352 rows, 27 columns |
| `07_panel_fe_forecasts.parquet` | incumbent and naive forecasts, and the evaluation grid |
| `panel_weekly_yellow.parquet` | not used. See "Which panel" below |
| `climate_weekly_by_market.parquet` | not used directly; already merged into the panel by `build_panel.py` |
| `02_lead_lag_pairs.csv` | not used at runtime; the evidence trail for the upstream hierarchy |
| `build_panel.py` | read for provenance, not executed |

## Which panel the incumbent used

Joining the baseline back onto each candidate panel settles it:

| Panel | `actual` matches `price` at target | `naive` matches `price` at origin |
|---|---|---|
| `panel_weekly.parquet` | 99.81% | 99.88% |
| `panel_weekly_yellow.parquet` | 31.42% | 31.36% |

`panel_weekly.parquet` is wholesale white maize, per `build_panel.py:load_prices`
reading the config `primary_series.maize_type`. The yellow panel is a variety
split carrying only Fourier K=1 and K=2; the white panel carries K=1 to K=4. Both
builds use K=2 per the owner decision of 2026-08-11, so the extra harmonics in the
white panel are present but unused.

## Panel shape

- 647 weeks, 2012-05-02 to 2024-09-18, every date a Wednesday, clean 7-day
  spacing with no missing weeks. Verified in `load_panel`, which raises if the
  index is not a clean weekly grid.
- 16 markets across 13 states.
- Price observed in 8,895 of 10,352 cells (85.9%). Range 28.4 to 1,221.6 NGN/kg.

Per-market price record:

| market | state | kind | first price | observed weeks | proxy-filled | longest interior gap |
|---|---|---|---|---|---|---|
| Aba | Abia | consumption | 2012-05-02 | 430 | 46 | 139 |
| Biu | Borno | producing | 2015-01-21 | 494 | 3 | 9 |
| Damaturu | Yobe | consumption | 2015-01-21 | 494 | 6 | 9 |
| Dandume | Katsina | consumption | 2012-10-31 | 551 | 21 | 55 |
| Giwa | Kaduna | producing | 2013-11-27 | 548 | 41 | 12 |
| Gombe | Gombe | producing | 2012-05-02 | 631 | 1 | 12 |
| Gujungu | Jigawa | consumption | 2012-05-02 | 569 | 11 | 75 |
| Gwandu, Dodoru | Kebbi | consumption | 2012-05-02 | 567 | 5 | 77 |
| Ibadan, Bodija | Oyo | consumption | 2012-05-02 | 631 | 35 | 12 |
| Kano, Dawanau | Kano | consumption | 2012-10-31 | 550 | 2 | 55 |
| Kaura Namoda | Zamfara | consumption | 2013-11-27 | 549 | 9 | 12 |
| Lagos, Mile 12 | Lagos | consumption | 2012-05-02 | 631 | 5 | 12 |
| Maiduguri | Borno | producing | 2012-05-02 | 631 | 9 | 12 |
| Mubi | Adamawa | consumption | 2015-01-21 | 494 | 31 | 9 |
| Potiskum | Yobe | consumption | 2015-01-21 | 494 | 4 | 9 |
| Saminaka | Kaduna | producing | 2012-05-02 | 631 | 4 | 12 |

Eleven of sixteen markets are consumption-zone. §8.8 of `PROJECT_HANDOFF.md` set a
decision rule on exactly this count: keep the simple top-5-producing-state average
if 3 to 4 of 16 are consumption markets, build the supply-shed version if closer
to half. Eleven of sixteen is past that line, and the inverse-distance columns in
the panel are that supply-shed version. Measured effect:

| scheme | distinct values per week across markets |
|---|---|
| `rainfall_simple_average` | 3.73 |
| `rainfall_inverse_distance` | 12.73 |
| `ndvi_simple_average` | 4.00 |
| `ndvi_inverse_distance` | 13.00 |

The simple average gives the entire consumption bloc one shared number, so within
that bloc the variable has no cross-market variation. Inverse-distance restores
it. Both builds use inverse-distance; the simple average remains available as a
sensitivity arm via `features.climate_scheme`.

## Full-panel outage weeks

Fifteen weeks have no observed price in any market:

2014-01-01, 2014-04-30, 2014-05-07, 2014-05-14, 2014-05-28, 2014-06-04,
2014-06-11, 2014-06-18, 2014-06-25, 2014-07-02, 2014-07-09, 2014-07-16,
2014-10-01, 2015-04-01, 2019-04-24.

Fourteen precede the first origin and never reach the evaluation. 2019-04-24 does,
and is handled under D-02.

These are also invisible in a naive `pivot_table`, which silently drops all-null
rows and shortens the date index. `load_panel` reindexes onto the explicit weekly
index to prevent that.

## Price gap handling

`load_panel` linearly interpolates interior gaps up to 13 weeks, never
extrapolates past a market's first or last observation, and records every filled
cell. Across the whole panel this fills 282 cells.

Why any filling at all: a 52-week lookback needs 52 contiguous values. Measured at
the 1,601 grid origins before filling, price was complete in 83.4% of windows and
never worse than 49 of 52. Refusing to fill would discard 16.6% of the grid and
break the paired comparison; filling at most three points in a 52-week window is
consistent with the temporal-proxy convention in §5 of the NADIH methodology note.

The other four channels need no filling at all. Measured across all 1,601 grid
origins:

| channel | windows with all 52 values present |
|---|---|
| diesel | 100% |
| `rainfall_inverse_distance` | 100% |
| `ndvi_inverse_distance` | 100% |
| `upstream_price` | 77.7% (minimum 0) |

The upstream minimum of zero is Dandume, the hierarchy root, which has no upstream
market by construction. That is what channel 10, the upstream-availability mask,
exists for: the upstream channel is zeroed and the mask flags it, so the network
can learn a different mapping for the root series rather than reading a zero as a
price that did not move.

## Diesel is state-level, not national

12.76 distinct values per week across 16 markets in 13 states. `build_panel.py:
splice_diesel` explains why: state-level monthly data from June 2015 onward,
back-extended to 2010 by scaling the national AGO landing cost by a per-state
ratio measured over the overlap window.

This matters for reading §8.2 of the handoff, which frames diesel as national and
therefore unidentifiable without interactions. At state level it varies
cross-sectionally, so the identification problem §8.2 describes is smaller than it
assumed. 25.7% of diesel cells are interpolated, flagged in
`is_interpolated_diesel`; the November 2023 month is empty for every state and is
interpolated over time.

## Upstream hierarchy

`upstream_market` and `upstream_lag_weeks` are in the panel, derived empirically
from cross-correlation (`02_lead_lag_pairs.csv`, 240 ordered pairs with best lag
and correlation) rather than assumed.

**Dandume is the sole root.** §8.3 of the handoff guessed that Dawanau, as the
largest grain market in West Africa, would sit at the top. The empirical result
puts Dawanau's upstream as Dandume with a 1-week lag. Worth flagging to whoever
wrote §8.3, since it inverts a stated prior.

## The evaluation grid

4,803 rows in the baseline file: 1,601 `(market, origin)` pairs at each of h = 4,
13, 26. 432 distinct origin dates, 2015-09-16 to 2024-03-20. 15 markets; Aba is
absent.

**The grid is per-market 4-weekly and staggered, not weekly.** Every market's own
origin sequence is spaced exactly 28 days. Only the union across markets looks
weekly. Mean markets per origin is 3.7, maximum 6, minimum 1, and only six
distinct market-sets occur across all 432 origins, rotating on a 4-week cycle.
This is the single most important thing to know about the grid; D-01 explains what
went wrong when the original regenerated it instead of reading it.

Target dates are exactly origin + 28, 91 and 182 days, with no exceptions. The
last h=26 target is 2024-09-18, the final week of the panel, so the grid uses the
data to its limit.

Proxy-filled prices are not excluded from the grid: 44 of the h=4 origins carry
`is_proxy_price` true. Kept, to stay consistent with the incumbent.

Incumbent performance on the full 1,601-pair grid:

| h weeks | panel_fe MAE | naive MAE | panel_fe vs naive |
|---|---|---|---|
| 4 | 16.72 | 17.89 | +6.6% |
| 13 | 29.01 | 38.12 | +23.9% |
| 26 | 35.62 | 56.51 | +37.0% |

These differ from the 16.86 / 27.91 / 31.30 recorded in the prior session's notes
because those were computed on that session's 558-pair subset. Both are correct
for their own grid, which is the whole lesson of D-01. Every table this repo
produces recomputes the incumbent on the same pairs as the challenger.
