# ONGOING EXPERIMENT: NOT PRODUCTION

**Branch:** `experiment/fewsnet-per-horizon`
**What it tests:** a separate GRU for each horizon (--model-horizons) instead of one shared model
**Decision:** D-61
**Where it stands:** Step 1 worse at every horizon; step 2 (per-horizon tuning) not recommended.
**Based on:** main as of D-59 (0771e2c); does not include D-62 to D-68

Production is the GRU on `configs/build3_fxdaily_fxonly_pit.yaml` on branch `main`
(D-68, 2026-10-05). Nothing on this branch is production, and no figure from it should be
reported as production accuracy. See `docs/MODEL_STATUS.md` and `docs/DECISIONS.md` on main.
