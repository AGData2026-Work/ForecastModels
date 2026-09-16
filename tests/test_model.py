import inspect

import numpy as np
import pytest
import torch

from model import RecurrentForecaster, load_checkpoint, predict, save_checkpoint, train_one

HAS_NUM_LAYERS = "num_layers" in inspect.signature(RecurrentForecaster.__init__).parameters


def make_net(kind="GRU", n_channels=10, hidden=64, n_flat=0, n_markets=5, n_horizons=3, **kw):
    return RecurrentForecaster(kind, n_channels, n_markets, n_horizons, n_flat=n_flat,
                              hidden=hidden, **kw)


class TestParamCount:
    def test_hidden_increases_param_count_monotonically(self):
        counts = [make_net(hidden=h).n_params() for h in (64, 128, 192, 256)]
        assert counts == sorted(counts)
        assert len(set(counts)) == 4

    def test_gru_has_more_params_than_rnn_at_equal_hidden(self):
        """GRU's three gates vs. the plain RNN's one is the whole reason a
        difference in results is supposed to be attributable to gating,
        per this project's own design note -- confirm the params actually
        differ the way that story requires."""
        gru = make_net(kind="GRU", hidden=96).n_params()
        rnn = make_net(kind="RNN", hidden=96).n_params()
        assert gru > rnn

    def test_adding_channels_costs_exactly_the_expected_amount(self):
        """Regression guard for exactly the kind of change made today
        (D-53's two new macro channels): adding channels should cost a
        known, derivable number of parameters, not an accidental amount."""
        base = make_net(kind="GRU", n_channels=10, hidden=96).n_params()
        plus_two = make_net(kind="GRU", n_channels=12, hidden=96).n_params()
        # a GRU's input-to-hidden weight matrix is 3 x hidden x n_channels
        # (one per gate); two more channels costs exactly 3 x hidden x 2.
        assert plus_two - base == 3 * 96 * 2

        base_rnn = make_net(kind="RNN", n_channels=10, hidden=96).n_params()
        plus_two_rnn = make_net(kind="RNN", n_channels=12, hidden=96).n_params()
        # a plain RNN has one gate, so the same channel addition costs a
        # third as much.
        assert plus_two_rnn - base_rnn == 96 * 2

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            make_net(kind="LSTM")

    @pytest.mark.skipif(not HAS_NUM_LAYERS, reason="num_layers not present on this branch's model.py")
    def test_num_layers_default_matches_pre_num_layers_behaviour(self):
        """Regression guard for D-29's own stated goal: adding num_layers
        must not change any existing config's parameter count."""
        with_default = make_net(kind="GRU", hidden=128).n_params()
        explicit_one = make_net(kind="GRU", hidden=128, num_layers=1).n_params()
        assert with_default == explicit_one

    @pytest.mark.skipif(not HAS_NUM_LAYERS, reason="num_layers not present on this branch's model.py")
    def test_two_layers_costs_more_than_one(self):
        one_layer = make_net(kind="GRU", hidden=128, num_layers=1).n_params()
        two_layer = make_net(kind="GRU", hidden=128, num_layers=2).n_params()
        assert two_layer > one_layer


@pytest.fixture
def toy_training_data():
    rng = np.random.default_rng(0)
    n_tr, n_va, lookback, n_channels, n_flat, n_horizons = 64, 16, 10, 5, 0, 2
    Xtr = torch.tensor(rng.normal(size=(n_tr, lookback, n_channels)), dtype=torch.float32)
    Ftr = torch.zeros(n_tr, n_flat)
    Itr = torch.randint(0, 3, (n_tr,))
    ytr = torch.tensor(rng.normal(size=(n_tr, n_horizons)), dtype=torch.float32)
    Wtr = torch.ones(n_tr)
    Xva = torch.tensor(rng.normal(size=(n_va, lookback, n_channels)), dtype=torch.float32)
    Fva = torch.zeros(n_va, n_flat)
    Iva = torch.randint(0, 3, (n_va,))
    yva = torch.tensor(rng.normal(size=(n_va, n_horizons)), dtype=torch.float32)
    return Xtr, Ftr, Itr, ytr, Wtr, Xva, Fva, Iva, yva


class TestTrainOne:

    def test_runs_and_produces_finite_predictions(self, toy_training_data):
        Xtr, Ftr, Itr, ytr, Wtr, Xva, Fva, Iva, yva = toy_training_data
        torch.manual_seed(0)
        net = RecurrentForecaster("GRU", n_channels=5, n_markets=3, n_horizons=2, hidden=8)
        net, info = train_one(net, Xtr, Ftr, Itr, ytr, Wtr, Xva, Fva, Iva, yva, epochs=3)
        assert info["epochs_run"] <= 3
        assert np.isfinite(info["best_val"])
        preds = predict(net, Xva, Fva, Iva)
        assert np.isfinite(preds).all()
        assert preds.shape == (16, 2)

    def test_raises_loudly_on_nan_input_instead_of_training_through_it(self, toy_training_data):
        Xtr, Ftr, Itr, ytr, Wtr, Xva, Fva, Iva, yva = toy_training_data
        Xtr = Xtr.clone()
        Xtr[0, 0, 0] = float("nan")
        torch.manual_seed(0)
        net = RecurrentForecaster("RNN", n_channels=5, n_markets=3, n_horizons=2, hidden=8)
        with pytest.raises(RuntimeError, match="non-finite"):
            train_one(net, Xtr, Ftr, Itr, ytr, Wtr, Xva, Fva, Iva, yva,
                     epochs=3, run_label="unit test")


class TestCheckpointRoundTrip:
    def test_save_and_load_reproduces_identical_predictions(self, toy_training_data, tmp_path):
        """The whole point of persisting a model (D-63 on main, mirrored
        here): loading it back must reproduce exactly what the original
        in-memory model would have predicted, not an approximation."""
        from data import Scaler

        Xtr, Ftr, Itr, ytr, Wtr, Xva, Fva, Iva, yva = toy_training_data
        torch.manual_seed(0)
        net = RecurrentForecaster("GRU", n_channels=5, n_markets=3, n_horizons=2, hidden=8)
        net, _ = train_one(net, Xtr, Ftr, Itr, ytr, Wtr, Xva, Fva, Iva, yva, epochs=3)

        scaler = Scaler()
        scaler.seq_mu = np.zeros(5)
        scaler.seq_sd = np.ones(5)
        scaler.flat_mu = scaler.flat_sd = None
        scaler.y_mu = np.zeros(2)
        scaler.y_sd = np.ones(2)

        original_preds = predict(net, Xva, Fva, Iva)

        ckpt_path = tmp_path / "checkpoint.pt"
        save_checkpoint(net, scaler, ckpt_path,
                        meta={"kind": "GRU", "hidden": 8, "n_channels": 5,
                              "n_markets": 3, "n_horizons": 2})

        state_dict, scaler_state, meta = load_checkpoint(ckpt_path)
        assert meta["hidden"] == 8
        assert scaler_state["seq_mu"].shape == (5,)

        reloaded_net = RecurrentForecaster(meta["kind"], meta["n_channels"],
                                          meta["n_markets"], meta["n_horizons"],
                                          hidden=meta["hidden"])
        reloaded_net.load_state_dict(state_dict)
        reloaded_preds = predict(reloaded_net, Xva, Fva, Iva)

        np.testing.assert_array_equal(original_preds, reloaded_preds)
