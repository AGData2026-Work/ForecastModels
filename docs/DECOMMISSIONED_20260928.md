# Decommissioning record, 2026-09-28

Owner decision, 2026-09-28: **one model per workstream, GRU, plus its soft
blend**. Everything else is documented and retired. The decision and its
evidence are in `docs/DECISIONS.md`: D-59 on `main` (FEWSNET) and D-76 on
`afex-multicommodity` (AFEX). This file lists what stayed, what moved and
where it went, so any path cited in an older decision-log entry can still
be found.

Nothing was deleted. `outputs/` is git-ignored, so these moves exist only
on the local disk; this file is their only record.

## What stays live

| Workstream | Role | Path |
|---|---|---|
| FEWSNET | production GRU, extended grid (main D-56) | `outputs/build3_h4h13_extended/GRU_unconditional/` |
| FEWSNET | same GRU, canonical 1,590-pair grid | `outputs/build3_underfit_corrected/GRU_unconditional/` |
| FEWSNET | conditional arm, the only one comparable to panel FE (see `docs/CONDITIONAL_CONVENTION.md`) | `outputs/build3_underfit_corrected/GRU_conditional/` |
| FEWSNET | point-in-time soft blend on the production GRU | `outputs/regime_blend_soft/GRU/` |
| AFEX | production GRU plus point-in-time benchmarks and soft blend | `outputs/afex_operational_v3/GRU/` |
| both | run logs appended by the runners | `outputs/run_history.csv`, `outputs/training_time_log.csv` |

Configs: `configs/build3.yaml` (FEWSNET, lookback 26, hidden 128) and
`configs/afex_operational_full_exog.yaml` (AFEX, lookback 52, hidden 64).
Blend scripts: `src/regime_blend.py` (main) and `src/afex_benchmarks.py`
(afex-multicommodity).

## Where retired outputs went

Rule: `outputs/<path>` is now `outputs/_archive/20260928_decommissioned/<path>`,
same relative path. Two exceptions, both the look-ahead blend outputs
replaced by the point-in-time versions:

- `outputs/regime_blend_soft/` (RNN and GRU, look-ahead) is now
  `outputs/_archive/20260928_decommissioned/regime_blend_soft_lookahead/`
- `outputs/afex_operational_v3/GRU/{predictions_with_benchmarks.csv,benchmark_metrics.csv,benchmark_metadata.json}`
  (look-ahead) are now in `outputs/_archive/20260928_decommissioned/afex_GRU_benchmarks_lookahead/`

FEWSNET run folders (17):
- `build2_best_practice`
- `build3_2layer`
- `build3_h4h13_extended/RNN_unconditional`
- `build3_hidden192`
- `build3_hidden256`
- `build3_hidden384`
- `build3_hidden96`
- `build3_underfit_corrected/GRU_conditional_exog`
- `build3_underfit_corrected/GRU_unconditional_lookback52_superseded`
- `build3_underfit_corrected/RNN_conditional`
- `build3_underfit_corrected/RNN_conditional_exog`
- `build3_underfit_corrected/RNN_unconditional`
- `build3_underfit_corrected/RNN_unconditional_lookback52_superseded`
- `fewsnet_afex_extension`
- `fewsnet_lookback_bracket`
- `regime_blend_hard`
- `regime_blend_soft_preD56extension_superseded`

AFEX run folders (15):
- `afex_lookback_bracket`
- `afex_operational`
- `afex_operational_20260916update`
- `afex_operational_20260916update_final`
- `afex_operational_agroclimatic_h96`
- `afex_operational_full_exog`
- `afex_operational_full_exog_diesel`
- `afex_operational_full_exog_hidden192`
- `afex_operational_full_exog_hidden96`
- `afex_operational_full_exog_macro`
- `afex_operational_v3/GRU_lookback26_superseded`
- `afex_operational_v3/RNN`
- `afex_operational_v3_maizeonly`
- `afex_rnn_capacity_bracket`
- `afex_v3_brackets`

Smoke runs (4):
- `_smoke_afex_extension`
- `_smoke_h4h13`
- `_smoke_v3_gru`
- `_smoke_v3_rnn`

Top-level aggregate tables and spreadsheets (34). These summarise
comparisons between models that are now retired; they are historical records,
not current results:
- `20260818_MAPE_summary.csv`
- `20260907_Maize_Forecast_PerMarket_ErrorAnalysis_v1.xlsx`
- `20260910_AFEX_Operational_DirectionAndPerMarketError_v1.xlsx`
- `20260910_AFEX_PredictedVsActual_OperationalVsSafeguardRemoved_v1.xlsx`
- `afex_agroclimatic_vs_operational_overall.csv`
- `afex_agroclimatic_vs_operational_per_market.csv`
- `afex_directional_by_market.csv`
- `afex_directional_overall.csv`
- `afex_excl_outliers_comparison.csv`
- `afex_final_scoring_composite.csv`
- `afex_final_scoring_mae_component.csv`
- `afex_full_exog_vs_all_comparison.csv`
- `afex_intervals_by_market.csv`
- `afex_intervals_summary.csv`
- `afex_maize_sibling_correlation_detail.csv`
- `afex_mape_all_variants.csv`
- `afex_operational_per_market_error.csv`
- `afex_sorghum_vs_operational_paired_split.csv`
- `afex_upstream_lag_best_per_pair.csv`
- `afex_upstream_lag_full_detail.csv`
- `afex_upstream_lag_tree.csv`
- `capacity_bracket_comparison.csv`
- `directional_benchmark.csv`
- `driver_forecast_accuracy_detail.csv`
- `ensemble_build2_build3.csv`
- `fewsnet_final_scoring_candidates.csv`
- `fewsnet_final_scoring_composite.csv`
- `intervals_by_market.csv`
- `intervals_by_regime.csv`
- `intervals_summary.csv`
- `proxy_sensitivity.csv`
- `seed_analysis.csv`
- `seed_count_curve.csv`
- `taskA_operational_vs_safeguard_removed_comparison.csv`

## Retired configs

Moved with `git mv` to `configs/_archive/`, so history is kept:
- main: `build1.yaml`, `build2.yaml`, `build3_2layer.yaml`,
  `build3_hidden96.yaml`, `build3_hidden192.yaml`, `build3_hidden256.yaml`,
  `build3_hidden384.yaml`, `build3_lookback26.yaml`, `build3_lookback104.yaml`
- afex-multicommodity: every `configs/afex_*.yaml` except
  `afex_operational_full_exog.yaml` (20 files, including the RNN finalist
  `afex_operational.yaml`)

## Retired branches

Tagged locally, not deleted:
- `regime-blend-exploration` as `archive/regime-blend-exploration`. Its soft
  blend moved to main with the look-ahead fixed; its hard-switch variant is
  retired.
- `fewsnet-afex-market-extension` as `archive/fewsnet-afex-market-extension`.

## What was not retired

- **The RNN code path.** `--kind RNN` still works in `src/model.py`,
  `src/run.py` and `src/run_afex.py`, so archived runs can be reproduced. No
  config or script defaults to it.
- **The incumbent.** Panel FE and naive columns in
  `07_panel_fe_forecasts.parquet` are protocol inputs, not challenger models.
