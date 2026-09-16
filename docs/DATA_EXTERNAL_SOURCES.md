# External data sources

The actual files live in `data/external/`, gitignored (covered by `data/` in
`.gitignore`, same as the FEWSNET parquet files) since they are large,
licensed source data, not code. This document is tracked in git precisely
because `data/` is not -- git will not let a file inside an ignored
directory be un-ignored on its own, so the file inventory lives here
instead. `data/external/` is the one canonical location for every AFEX-side
source file that used to be a hardcoded path into a personal Downloads
folder; every `configs/afex_*.yaml` file now points there instead.

| File | Config key(s) | What it is | Refresh |
|---|---|---|---|
| `afex_multicommodity_panel.xlsx` | `data.panel` | The AFEX weekly farmgate panel, 51 series, 18 markets, 7 commodities | Re-export from AFEX's own source system when a newer cut is needed; same sheet structure (`Panel_Long`) expected |
| `ndvi_cleaned.xlsx` | `data.ndvi_path` | UN Data Exchange dekadal NDVI, state-level | Re-download from UN Data Exchange; same `Cleaned Data` sheet and columns expected |
| `rainfall_cleaned.xlsx` | `data.rainfall_path` | UN Data Exchange dekadal rainfall, state-level | Same source as NDVI above |
| `ndvi_state_map.xlsx` | `data.climate_state_map_path` | Location Code -> State mapping (the NDVI source file's own geography key) | Only needs refreshing if UN Data Exchange changes its location codes |
| `fx_rate.xlsx` | `data.fx_rate_path` | National Official Exchange Rate, monthly, 2004-2026 | Re-pull from the original FX rate source when new months are needed |
| `inflation.xlsx` | `data.inflation_path` | National headline/food inflation, annual and monthly sheets, 2000-2026 | Re-pull from the original inflation data source |

Diesel (`data.diesel_source_path`) is not listed here: it already points at
`data/panel_weekly.parquet`, the FEWSNET panel, reused directly rather than
duplicated.
