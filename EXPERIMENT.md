# ONGOING EXPERIMENT: NOT PRODUCTION

**Branch:** `experiment/fewsnet-relprice`
**What it tests:** rice (5% broken, wholesale) and gasoline prices relative to maize as extra GRU inputs
**Decision:** D-60
**Where it stands:** Not adopted: small, non-significant gains at 1 and 3 months, worse at 6. Follow-ups (12 seeds, rice only, delayed activation) not yet run.
**Based on:** main as of D-59 (0771e2c); does not include D-60 to D-68

Production is the GRU on `configs/build3_fxdaily_fxonly_pit.yaml` on branch `main`
(D-68, 2026-10-05). Nothing on this branch is production, and no figure from it should be
reported as production accuracy. See `docs/MODEL_STATUS.md` and `docs/DECISIONS.md` on main.
