# Data fragility: risks outside the workstream's control (recorded 2026-10-12)

The monthly production setup (D-75) depends on data the project does not produce. These risks are
documented, not fixed: no modelling choice removes them. Each item says what happened, what it does to
the forecast, and what the pipeline does when it happens.

## 1. FEWS NET coverage shrank in 2025
- Seven of the 16 markets have no FEWS NET maize price after Jan 2025: Aba, Giwa, Gombe, Gujungu,
  Gwandu (Dodoru), Lagos (Mile 12), Saminaka. The other nine (Biu, Damaturu, Dandume, Ibadan, Kano,
  Kaura Namoda, Maiduguri, Mubi, Potiskum) have 18 observed months between Oct 2024 and Aug 2026.
- No market has a FEWS NET value for Oct and Nov 2024, Feb and Mar 2025, or Oct 2025.
- Effect: Gombe, Gujungu, Gwandu, Lagos and Aba cannot be forecast after early 2025; their 2024-26
  results rest on a few months. Giwa and Saminaka continue only through fill sources (item 3).
- Pipeline: a market with a missing price anywhere in its 6-month input window gets no forecast
  (`build_sequence`), and `src/predict.py` refuses a panel older than 2.5 months.

## 2. FEWS NET published some 2025 months 10 to 13 months late, and revises values
- Jan 2025 appeared in Dec 2025, Apr 2025 in May 2026, Sep 2025 in Oct 2026; May to Aug 2025 exist only
  as `value_one_year_ago` on the 2026 rows (owner decision, D-75). `docs/PUBLICATION_LAGS.md`.
- A re-pull on 2026-10-05 changed 342 values against the pull of 2026-10-02 (D-70).
- Effect: a live forecast can lack the latest month, and history is a later vintage than a forecaster
  would have seen at the time.

## 3. Three non-FEWS NET sources carry the filled markets
| Market | Fill source, Oct 2024 to Aug 2026 | Months |
|---|---|---|
| Saminaka | AFEX (Saminaka, Pambegua); Kaduna/Kano other source (Pambegua) | 21 + 1 |
| Giwa | Kaduna/Kano other source (Giwa); AFEX (Tundun Saibu) | 12 + 9 |
| Dandume | AFEX (Dandume, Bakori); NAERLS (old crop) | 4 + 1 |
| Kaura Namoda | NAERLS (Tsafe, old crop) | 1 |
- Each source can stop, change format or change markets. NAERLS starts Sep 2025 and ends Jun 2026 in the
  file held; the Kaduna/Kano source starts Sep 2025; AFEX's Gombe markets end Sep 2024.
- Pipeline: a month a source does not supply stays missing (no carry-forward of a fill); gaps of 2 months
  or less between FEWS NET values are interpolated for history only.

## 4. Licences and terms
- FEWS NET: every row labelled "Public"; attribution required ("Famine Early Warning Systems Network
  (FEWS NET), Nigeria, Price (weekly)"); no accuracy guarantee.
- AFEX, NAERLS, the Kaduna/Kano source: terms not confirmed. Needed before their values sit inside a
  published forecast.

## 5. Driver series
- NBS state diesel ends Oct 2024; from Oct 2024 diesel is FEWS NET retail diesel scaled to the state
  series (D-70), with FEWS NET's gaps and delays.
- Rainfall (CHIRPS) and NDVI (MODIS) come through WFP on HDX; recent values are revised (CHIRPS
  preliminary to final, NDVI smoothing). Either feed can pause.
- The naira input is CBN's official rate, pegged before June 2023; no parallel-market series is held.
- NBS CPI (blend only) was delayed in early 2025 by the rebasing.

## 6. What the workstream can and cannot do
- Can: log every gap, fill and late month (`price_flag`, `is_proxy_price`, `fill_kind`); refuse stale
  input; re-pull and re-score when a late month appears.
- Cannot: make a source publish on time, restore a dropped market, or confirm terms for third-party data.
