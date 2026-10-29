"""Joint forecast combination (src/joint_forecast.py)."""
import numpy as np
import pandas as pd

from joint_forecast import combine


def test_combine_is_geometric_mean_when_origin_prices_agree():
    p0 = pd.Series([300.0, 300.0])
    out = combine(pd.Series([330.0, 270.0]), p0, pd.Series([300.0, 300.0]), p0)
    assert np.allclose(out, np.sqrt([330.0 * 300.0, 270.0 * 300.0]))


def test_combine_keeps_each_models_change_and_starts_from_our_price():
    # ours: 300 -> 330 (+10%); his: 320 -> 352 (+10%) from a different origin price; joint = 300 x 1.10
    out = combine(pd.Series([330.0]), pd.Series([300.0]), pd.Series([352.0]), pd.Series([320.0]))
    assert np.isclose(out.iloc[0], 330.0)
