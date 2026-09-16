import copy
import inspect

import numpy as np
import pytest

from data import (
    SEQ_BASE_CHANNELS,
    _interpolate_within_span,
    _safe_log_ratio,
    build_flat,
    build_sequence,
)

# The macro (FX/inflation) channels exist only on the afex-multicommodity
# branch as of D-53; these names simply aren't defined on main yet.
try:
    from data import MACRO_CHANNELS, sequence_channel_names
    HAS_MACRO = True
except ImportError:
    MACRO_CHANNELS = []
    sequence_channel_names = None
    HAS_MACRO = False

HAS_USE_MACRO_PARAM = "use_macro" in inspect.signature(build_sequence).parameters


class TestSafeLogRatio:
    def test_normal_values(self):
        out = _safe_log_ratio(np.array([110.0]), np.array([100.0]))
        assert out[0] == pytest.approx(np.log(110.0) - np.log(100.0))

    def test_zero_numerator_or_denominator_returns_zero(self):
        out = _safe_log_ratio(np.array([0.0, 100.0]), np.array([100.0, 0.0]))
        assert (out == 0.0).all()

    def test_negative_values_return_zero(self):
        out = _safe_log_ratio(np.array([-5.0]), np.array([100.0]))
        assert out[0] == 0.0

    def test_nan_returns_zero(self):
        out = _safe_log_ratio(np.array([np.nan]), np.array([100.0]))
        assert out[0] == 0.0


class TestInterpolateWithinSpan:
    def test_fills_interior_gap_up_to_limit(self):
        col = np.array([10.0, np.nan, np.nan, 40.0, 50.0])
        out, filled = _interpolate_within_span(col, limit=13)
        assert np.isfinite(out).all()
        assert filled[1] and filled[2]
        assert out[1] == pytest.approx(20.0)
        assert out[2] == pytest.approx(30.0)

    def test_never_extrapolates_past_first_or_last_observation(self):
        col = np.array([np.nan, np.nan, 10.0, 20.0, np.nan])
        out, filled = _interpolate_within_span(col, limit=13)
        assert np.isnan(out[0]) and np.isnan(out[1])
        assert not filled[0] and not filled[1]
        assert np.isnan(out[4])
        assert not filled[4]

    def test_gap_longer_than_limit_fills_only_up_to_the_limit(self):
        """pandas' own `limit` semantics: fill the first N consecutive NaNs
        in a run, leave the rest of that run as NaN. Not "refuse to fill
        anything if the run exceeds the limit" -- confirmed against the
        actual behaviour, not assumed."""
        col = np.array([10.0] + [np.nan] * 5 + [20.0])
        out, filled = _interpolate_within_span(col, limit=2)
        assert filled[1:3].all()
        assert np.isfinite(out[1:3]).all()
        assert np.isnan(out[3:6]).all()
        assert not filled[3:6].any()

    def test_all_nan_column_is_left_untouched(self):
        col = np.full(5, np.nan)
        out, filled = _interpolate_within_span(col, limit=13)
        assert np.isnan(out).all()
        assert not filled.any()


class TestBuildSequence:
    def test_shape_matches_lookback_and_base_channel_count(self, tiny_panel):
        X, ok = build_sequence(tiny_panel, i=60, j=0, lookback=52)
        assert ok
        assert X.shape == (52, len(SEQ_BASE_CHANNELS))

    @pytest.mark.skipif(not HAS_USE_MACRO_PARAM, reason="use_macro not present on this branch's build_sequence")
    def test_use_macro_adds_exactly_two_channels(self, tiny_panel):
        p = copy.deepcopy(tiny_panel)
        p.fx_rate = np.full_like(p.price, 1000.0)
        p.inflation_yoy = np.full_like(p.price, 15.0)
        X, ok = build_sequence(p, i=60, j=0, lookback=52, use_macro=True)
        assert ok
        assert X.shape == (52, len(SEQ_BASE_CHANNELS) + len(MACRO_CHANNELS))

    @pytest.mark.skipif(not HAS_USE_MACRO_PARAM, reason="use_macro not present on this branch's build_sequence")
    def test_use_macro_without_fx_rate_raises(self, tiny_panel):
        with pytest.raises(ValueError):
            build_sequence(tiny_panel, i=60, j=0, lookback=52, use_macro=True)

    def test_upstream_mask_reflects_has_upstream(self, tiny_panel):
        # market 2 (index 2) has no upstream in the fixture
        X, ok = build_sequence(tiny_panel, i=60, j=2, lookback=52)
        assert ok
        mask_col = SEQ_BASE_CHANNELS.index("upstream_mask")
        assert (X[:, mask_col] == 0.0).all()
        X, ok = build_sequence(tiny_panel, i=60, j=0, lookback=52)
        mask_col = SEQ_BASE_CHANNELS.index("upstream_mask")
        assert (X[:, mask_col] == 1.0).all()

    def test_no_look_ahead(self, tiny_panel):
        """The single most important guarantee in this project (D-01/D-17):
        nothing after the origin index may change the returned window."""
        i, j, lookback = 60, 0, 52
        X_before, ok = build_sequence(tiny_panel, i, j, lookback)
        assert ok

        p_mutated = copy.deepcopy(tiny_panel)
        p_mutated.price[i + 1:, j] = 999999.0
        p_mutated.diesel[i + 1:, j] = 999999.0
        p_mutated.upstream[i + 1:, j] = 999999.0
        X_after, ok = build_sequence(p_mutated, i, j, lookback)
        assert ok
        np.testing.assert_array_equal(X_before, X_after)

    def test_insufficient_history_returns_not_ok(self, tiny_panel):
        _, ok = build_sequence(tiny_panel, i=5, j=0, lookback=52)
        assert not ok


class TestBuildFlat:
    def test_lag52_needs_full_history(self, tiny_panel):
        # i=60 has 60 weeks of history; h=26 needs i + h - 52 >= 0, i.e. i >= 26
        F, ok = build_flat(tiny_panel, i=60, j=0, horizons=[4, 13, 26],
                           use_lag52=True, use_realised_drivers=False)
        assert ok
        # anchor + window_return, one of each per horizon: 2 x 3 = 6
        assert F.shape == (6,)

    def test_lag52_fails_loudly_when_history_too_short(self, tiny_panel):
        _, ok = build_flat(tiny_panel, i=10, j=0, horizons=[4, 13, 26],
                           use_lag52=True, use_realised_drivers=False)
        assert not ok

    def test_no_flat_features_when_both_switches_off(self, tiny_panel):
        F, ok = build_flat(tiny_panel, i=60, j=0, horizons=[4, 13, 26],
                           use_lag52=False, use_realised_drivers=False)
        assert ok
        assert F.shape == (0,)


@pytest.mark.skipif(not HAS_MACRO, reason="sequence_channel_names not present on this branch's data.py")
class TestSequenceChannelNames:
    def test_default_is_ten_base_channels(self):
        assert sequence_channel_names() == SEQ_BASE_CHANNELS
        assert len(sequence_channel_names()) == 10

    def test_use_macro_appends_two_more_in_order(self):
        names = sequence_channel_names(use_macro=True)
        assert names == SEQ_BASE_CHANNELS + MACRO_CHANNELS
        assert len(names) == 12
