# Model status, FEWSNET workstream

Which model is production and which are experiments. Updated 2026-10-29 (D-78: the joint forecast becomes production).
The decision log, `docs/DECISIONS.md`, is the authority; this page only points into it.

## Production: the joint forecast (D-78)
| Component | Code | Outputs |
|---|---|---|
| GRU (D-77, below, unchanged) | `src/predict.py --run-dir outputs/production_monthly_parity_serving` | `outputs/production_monthly_parity_serving/predictions/<date>.csv` |
| Inflation Adjusted (analyst v4, as published) | `src/analyst_forecast.py` on `vendor/analyst_v4/` | `outputs/joint_forecast/analyst_forecasts.csv` |
| Joint forecast with 80% / 90% ranges | `src/joint_forecast.py live` | `outputs/joint_forecast/predictions/<date>.csv` |

Joint = our origin price x geometric mean of the two predicted changes; GRU alone (flagged) where the analyst's model
has no forecast. Backtest on shared forecasts (Jan 2017 to Aug 2026), MAPE joint / GRU / naive: all years 6.45 / 7.65 /
8.35, 14.69 / 15.26 / 18.30, 18.18 / 19.88 / 23.83 at 1 / 3 / 6 months; Oct 2024 to Aug 2026 6.64 / 8.16 / 7.29,
16.04 / 17.89 / 22.51, 25.00 / 27.18 / 42.70. D-45 PASS at every horizon. Checks and known limits in
`docs/DECISIONS.md` (D-78 and the entry before it); results in `outputs/joint_forecast/`.

## GRU component (production on its own from D-77 until D-78)
| Config | Decision | Outputs |
|---|---|---|
| `configs/build3_monthly_fxdaily_gapfilled_pit_parity.yaml` | D-77 | Evaluation `outputs/d75_monthly_parity_pit/h{1,3,6}`; serving model `outputs/production_monthly_parity_serving/` (`src/predict.py`, no `--blend`) |

Monthly GRU, forecast on the 15th of the month after the latest price month, on the gap-filled FEWS NET
panel v3 (`src/build_gapfilled_panel.py`, `src/build_monthly_gapfilled.py --published`), daily CBN FX scores
read on the 14th, diesel/rainfall/NDVI one month back, the upstream market's price for the same month (no lag), plus the import-parity gap. No soft blend (it lowers
the 2024-26 result once parity is in). Owner-accepted: recovered May-Aug 2025 values and the fills (D-75).

Current figures, vs naive at 1 / 3 / 6 months (corrected 2026-10-19 after the `load_grid` fix, see
`docs/DECISIONS.md`): Oct 2024 to Aug 2026 -17.1 / +21.9 / +34.5% (total +24.8%, 418 forecasts); all years +8.4 /
+23.6 / +23.7% (total +21.2%). The D-77 decision quoted -17.1 / +17.1 / +42.6% (total +28.2%) and +8.4 / +23.2 /
+25.0%, computed before the fix on fewer 3- and 6-month forecasts. D-45 still passes at every horizon. The pre-fix
3- and 6-month evaluations are kept in `outputs/_archive/20261019_pre_gridfix/`.

## Shadow, not production
| Config | Decision | Outputs |
|---|---|---|
| `configs/build3_monthly_parity_fdiesel_l0_lb3.yaml` | D-77 | Serving model `outputs/shadow_monthly_parity_fdiesel_l0_lb3_serving/`; variant E: FEWS NET-only diesel of the origin month, lookback 3 |

Run alongside production on every new month (`src/predict.py` on both run directories, then
`src/compare_shadow.py --prod outputs/production_monthly_parity_serving --shadow
outputs/shadow_monthly_parity_fdiesel_l0_lb3_serving`). Switch only if its lead over production holds on
new months. After the evaluation fix its 2024-26 lead is gone: +25.0% vs naive in total against production's
+24.8% (p .73), better only at 1 month (-13.2% against -17.1%, p .048); before the fix it read +29.9% against
+28.2% (p .016). Re-scored 3- and 6-month runs in `outputs/d77_checks/gridfix/shadowE/`.

## Reference, not production
| Config | Decision | Why it is kept |
|---|---|---|
| `configs/build3_fxdaily_fxonly_pit.yaml` | D-68 | Weekly flagship until D-77; better than monthly on 2015-24 at 1 and 3 months (fresher weekly data) |
| `configs/build3_monthly_fxdaily_gapfilled_pit.yaml` + `src/regime_blend_monthly.py` | D-75 | Monthly setup before the parity gap |
| `configs/build3.yaml` | D-59 | Older flagship; the no-FX reference |

## Ongoing experiments on main: not production
Every config below starts with a `# STATUS: ONGOING EXPERIMENT` line. Their code is on main
but switched off unless the config turns it on.

| Config | Decision | Where it stands |
|---|---|---|
| `configs/build3_pit.yaml` | D-67 | Previous flagship with inputs as published; the fair comparison for D-68 |
| `configs/build3_fxdaily_fxonly.yaml`, `configs/build3_fxdaily_fxonly_extended.yaml` | D-66 | Daily FX before the look-ahead fix; superseded by production |
| `configs/build3_fxsummary.yaml`, `configs/build3_fxsummary_fxonly.yaml` | D-65 | FX scores from the monthly file; worse weekly at 6 months |
| `configs/build3_monthly*.yaml` | D-64 to D-66 | Monthly cadence; pinned by the owner |
| `configs/build3_monthly_fxdaily_rtfp.yaml` | D-74 | Monthly GRU on World Bank RTFP estimates; not a real-price result |
| `configs/build3_monthly_fxdaily_gapfilled.yaml` | D-75 | Monthly daily-FX GRU on the gap-filled FEWS NET panel |
| `configs/build3_monthly_fxdaily_gapfilled_pit_obsonly.yaml` | D-75 | Sensitivity: as published, FEWS NET observed prices only |
| `configs/build3_fxdaily_fxonly_pit_oos2026.yaml` | D-70 | Production model, frozen at Sep 2024, out of sample 2024-26: worse than naive |
| `configs/build3_fxdaily_fxonly_pit_oos2026_partial.yaml` | D-71, D-72 | Same, retrained quarterly (D-71) and with partial targets (D-72): still worse than naive |
| `configs/build3_fxdaily_fxonly_pit_{real,balance,real_balance}.yaml` and `..._oos2026_{real,balance,real_balance}.yaml` | D-73 | Real-price target / direction-balanced training; not adopted |

## Ongoing experiments on their own branches: not production
Each branch has an `EXPERIMENT.md` at its root saying the same.

| Branch | Decision | Where it stands |
|---|---|---|
| `experiment/fewsnet-relprice` | D-60 | Rice and gasoline relative prices; not adopted, follow-ups (12 seeds, rice only) not yet run |
| `experiment/fewsnet-per-horizon` | D-61 | One GRU per horizon; step 1 worse, step 2 not recommended |
| `experiment/fewsnet-fx` | D-62 | Raw FX and inflation inputs; not adopted, superseded by D-66 |
| `experiment/fewsnet-monthly` | D-64 to D-66 | Monthly cadence and the FX work that led to D-68; pinned |

Retired work: `configs/_archive/`, `docs/DECOMMISSIONED_20260928.md`, and the archive tags
`archive/regime-blend-exploration` and `archive/fewsnet-afex-market-extension`. AFEX is a
separate workstream on `afex-multicommodity` (D-76).
