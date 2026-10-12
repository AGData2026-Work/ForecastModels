# Model status, FEWSNET workstream

Which model is production and which are experiments. Updated 2026-10-05 (D-68). The
decision log, `docs/DECISIONS.md`, is the authority; this page only points into it.

## Production
| Config | Decision | Outputs |
|---|---|---|
| `configs/build3_fxdaily_fxonly_pit.yaml` | D-68 | `outputs/production_fxdaily_pit_extended/GRU_unconditional/`, blend in `outputs/regime_blend_soft_production/GRU/` |

The GRU with daily CBN FX scores (D-66) and exogenous inputs as published (D-67).

## Reference, not production
| Config | Decision | Why it is kept |
|---|---|---|
| `configs/build3.yaml` | D-59 | Previous flagship; the no-FX reference every experiment is compared with |

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
| `configs/build3_monthly_fxdaily_gapfilled_pit.yaml` | D-75 | Same, with publication lags (forecast on the 15th of the next month) |
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
