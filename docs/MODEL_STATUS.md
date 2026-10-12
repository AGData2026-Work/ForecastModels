# Model status, FEWSNET workstream

Which model is production and which are experiments. Updated 2026-10-12 (D-77). The
decision log, `docs/DECISIONS.md`, is the authority; this page only points into it.

## Production
| Config | Decision | Outputs |
|---|---|---|
| `configs/build3_monthly_fxdaily_gapfilled_pit_parity.yaml` | D-77 | Evaluation `outputs/d75_monthly_parity_pit/h{1,3,6}`; serving model `outputs/production_monthly_parity_serving/` (`src/predict.py`, no `--blend`) |

Monthly GRU, forecast on the 15th of the month after the latest price month, on the gap-filled FEWS NET
panel v3 (`src/build_gapfilled_panel.py`, `src/build_monthly_gapfilled.py --published`), daily CBN FX scores
read on the 14th, diesel/rainfall/NDVI one month back, plus the import-parity gap. No soft blend (it lowers
the 2024-26 result once parity is in). Owner-accepted: recovered May-Aug 2025 values and the fills (D-75).

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
