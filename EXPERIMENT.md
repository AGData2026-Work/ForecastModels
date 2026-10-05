# ONGOING EXPERIMENT: NOT PRODUCTION

**Branch:** `experiment/fewsnet-monthly`
**What it tests:** the same GRU on monthly data (mean or last week of each month), and the FX work done on it (raw FX, bounded FX scores, daily CBN FX)
**Decision:** D-64, D-65, D-66
**Where it stands:** Monthly cadence pinned by the owner (2026-10-04): passes against monthly naive but is behind weekly at 3 and 6 months. The weekly daily-FX work from this branch became production (D-68) and is already on main.
**Based on:** main at D-68

Production is the GRU on `configs/build3_fxdaily_fxonly_pit.yaml` on branch `main`
(D-68, 2026-10-05). Nothing on this branch is production, and no figure from it should be
reported as production accuracy. See `docs/MODEL_STATUS.md` and `docs/DECISIONS.md` on main.
