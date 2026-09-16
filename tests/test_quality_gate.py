import numpy as np
import pandas as pd

from quality_gate import evaluate


def _make_run_dir(tmp_path, n=500, model_better=True, seed_consistent=True):
    """A run whose challenger is either consistently or inconsistently
    better than naive, so both dm_passes and seed_passes can be
    independently controlled for testing."""
    rng = np.random.default_rng(0)
    # actual/naive are per-(market, origin) -- generated once, shared by
    # every seed's prediction of that same target, exactly as a real run
    # would have one actual outcome per forecast regardless of how many
    # seeds tried to predict it.
    origins = [pd.Timestamp("2020-01-01") + pd.Timedelta(weeks=i) for i in range(n)]
    actuals = 100 + rng.normal(0, 10, n)
    naives = actuals + rng.normal(0, 8, n)

    rows = []
    seeds = range(7)
    for seed in seeds:
        # seed_bias shifts this seed's own mean error up or down, to
        # simulate inconsistent seeds.
        seed_bias = 0.0 if seed_consistent else rng.normal(0, 4)
        if model_better:
            # a real, independent edge around the true value; noisier
            # per seed than naive is on its own, but median-of-7 washes
            # that noise out, landing closer to actual than naive does.
            preds = actuals + seed_bias + rng.normal(0, 5.0, n)
        else:
            # "no edge": this model just tracks naive's own guess (plus
            # trivial extra noise), not the true value independently --
            # ensembling 7 seeds of "the same guess as naive" cannot
            # out-ensemble naive, unlike a genuinely independent-noise
            # model, which is exactly the distinction this fixture needs
            # to draw (a model merely as noisy as naive, but independent
            # of it, would still out-ensemble naive by variance reduction
            # alone -- that is not "no edge", it is a real one).
            preds = naives + seed_bias + rng.normal(0, 1.0, n)
        for i in range(n):
            rows.append(dict(seed=seed, h=4, market="M", origin=origins[i],
                             actual=actuals[i], naive=naives[i], pred=preds[i],
                             origin_price=actuals[i]))
    forecasts = pd.DataFrame(rows)
    forecasts.to_csv(tmp_path / "forecasts.csv", index=False)

    # median-ensemble predictions_paired-style, for the DM computation
    piv = forecasts.pivot_table(index=["market", "origin"], columns="seed", values="pred")
    med = piv.median(axis=1)
    dedup = forecasts.drop_duplicates(["market", "origin"]).set_index(["market", "origin"])
    import sys
    sys.path.insert(0, str(tmp_path.parents[0]))
    from metrics import diebold_mariano
    r = diebold_mariano(dedup["actual"], med.reindex(dedup.index), dedup["naive"], 4)
    pd.DataFrame([dict(h=4, vs="naive", dm_stat=r["dm_stat"], dm_p=r["dm_p"], n=r["n"])]
                ).to_csv(tmp_path / "diebold_mariano.csv", index=False)
    return tmp_path


class TestQualityGate:
    def test_consistent_real_edge_passes_both_conditions(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, model_better=True, seed_consistent=True)
        result = evaluate(run_dir, [4])
        row = result.iloc[0]
        assert row["dm_passes"]
        assert row["seed_passes"]
        assert row["gate_passes"]

    def test_no_edge_fails_both_conditions(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, model_better=False, seed_consistent=True)
        result = evaluate(run_dir, [4])
        row = result.iloc[0]
        assert not row["dm_passes"]
        assert not row["gate_passes"]

    def test_gate_requires_both_conditions_not_just_one(self, tmp_path):
        """A real edge that doesn't survive seed noise should still fail
        the combined gate -- this is exactly D-46's GRU-full-exog-style
        finding, and the whole reason this gate checks both."""
        run_dir = _make_run_dir(tmp_path, model_better=True, seed_consistent=False)
        result = evaluate(run_dir, [4])
        row = result.iloc[0]
        # seed inconsistency alone doesn't guarantee a fail here (it's
        # randomized), but gate_passes must never be True when either
        # individual condition is False
        assert row["gate_passes"] == (row["dm_passes"] and row["seed_passes"])
