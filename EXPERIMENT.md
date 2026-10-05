# ONGOING EXPERIMENT: NOT PRODUCTION

**Branch:** `experiment/fewsnet-fx`
**What it tests:** the official monthly exchange rate and headline inflation as raw GRU input channels
**Decision:** D-62
**Where it stands:** Not adopted: much worse at every horizon because the inputs left the training range. Superseded by the bounded daily-FX scores (D-66) now in production.
**Based on:** main as of D-59 (0771e2c); does not include D-60 to D-68

Production is the GRU on `configs/build3_fxdaily_fxonly_pit.yaml` on branch `main`
(D-68, 2026-10-05). Nothing on this branch is production, and no figure from it should be
reported as production accuracy. See `docs/MODEL_STATUS.md` and `docs/DECISIONS.md` on main.
