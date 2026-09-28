# AFEX outstanding work

Open items on the AFEX workstream (branch `afex-multicommodity`) as of
2026-09-28, after the v3 panel swap (D-71 to D-75). Each item cites the
decision-log entry it comes from in `docs/DECISIONS.md`. Strike an item
through, or delete it, when it closes, and cite the D-number that closed it.

Production model (D-76): GRU on `configs/afex_operational_full_exog.yaml`
(hidden 64, lookback 52), with the point-in-time soft blend from
`src/afex_benchmarks.py`. Outputs in `outputs/afex_operational_v3/GRU/`.
The RNN finalist and everything else were decommissioned by D-76; see
`docs/DECOMMISSIONED_20260928.md`.

## Left over from the v3 overhaul

1. **Hidden-size brackets around the GRU finalist.** D-74's brackets ran
   the GRU at lookback 26, before D-75 reverted it to 52. Re-run hidden
   96 and 192 at lookback 52.
2. **Re-screen `upstream_lag_map` on v3.** The map was chosen on v2 (D-36).
   Two leaders changed under v3: Leggal (leader for Kumo and Jalingo), the
   most-corrected maize market, and Gazabu (leader for Garbabi), which lost
   a bad quote.
3. **Analyses still on v2 data:** direction accuracy (D-29), per-market
   error (D-30), rescoring without Giwa and Ikara (D-35), prediction
   intervals (D-62).
4. ~~**Point-in-time seasonal index.**~~ Closed by D-76. `src/afex_benchmarks.py` fits the
   seasonal index on every complete year, including years after each
   origin (D-74 caveat). An expanding version removes the benchmark's
   in-sample advantage.
5. ~~**Hard-switch blend for AFEX.**~~ Moot after D-76 (soft blend only). The benchmark script builds only the
   soft blend; FEWSNET has both (regime-blend-exploration D-55).
6. **Wire the blend into the pipeline.** D-76 adopted the point-in-time
   soft blend as the GRU's companion, but it still runs as a separate
   post-hoc script after `run_afex.py`. Its gain also flips by year
   (helps 2024, hurts 2025-26; D-76), so re-check it as data accrues.
7. **AFEX zone metric gaps (D-73).** Pass rates at 3 and 6 months and the
   too-high vs. too-low overshoot split were done for FEWSNET (main D-58)
   but not AFEX.

## Older open questions

8. **Diesel into the GRU?** D-42 found it helped on v2; D-52 found it was
   never merged and left it open. Needs a fresh v3 test at current settings.
9. **Keep or drop the lagged-neighbour input?** D-57 found it mostly
   redundant with own-price history; D-56 found removing it doesn't clearly
   hurt. Owner's call.
10. ~~**RNN as the direction pick.**~~ Moot: RNN decommissioned (D-76). Chosen for direction (D-39), barely stable
    at 7 seeds (D-47), did not replicate at 12 (D-63). Not re-checked on v3.
11. **Seed count.** D-46: both margins over naive are largely seed noise.
    D-47 suggested 10 to 12 seeds. Everything still runs on 7.

## Data-side (the v3 file itself)

12. **v3 ends one week earlier** (2026-09-09 vs. v2's 2026-09-16). Flagged
    in D-71; not confirmed with the AFEX team.
13. **Unused v3 columns.** `quotes_on_assumed_100kg` could screen series
    that lean on the 100 kg bag-weight assumption (can be about 10% off);
    `proxy_type` is also ignored.
14. **Dandume maize is dropped** for having no usable windows. The v3 README
    mentions a separate, narrower new-crop-rule file that might bring it back.
15. **Mutumbiyu maize** is in the panel but absent from the scored output.
    Unexplained.
16. **Seven market/commodity pairs** were too new to add in the September
    merge (D-68). Revisit once they have more history.
