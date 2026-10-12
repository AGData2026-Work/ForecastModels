"""D-75 monthly production path: the gap-filled panel builder, the monthly model panel and the
monthly FX cutoff. Synthetic and small, like the rest of the suite; no real data files."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import build_gapfilled_panel as B
from build_monthly_gapfilled import build_grid, build_panel
from data import _fx_daily_features


def _inputs(months=None):
    """Two markets. 'Dandume' has a 2-month gap (filled from AFEX after >= 6 overlap months) and a
    later value recovered from a value_one_year_ago row; 'Biu' has a 1-month and a 3-month gap."""
    months = months or [str(p) for p in pd.period_range("2020-01", "2021-12", freq="M")]
    F = pd.DataFrame(index=months, columns=["Dandume", "Biu"], dtype=float)
    F["Dandume"] = np.linspace(100, 200, len(months))
    F["Biu"] = np.linspace(50, 80, len(months))
    F.loc[["2020-10", "2020-11"], "Dandume"] = np.nan          # 2-month gap, AFEX covers it
    F.loc["2021-03", "Dandume"] = np.nan                       # recovered from 2022-03's row
    F.loc["2020-05", "Biu"] = np.nan                           # 1-month gap -> interpolated
    F.loc[["2021-02", "2021-03", "2021-04"], "Biu"] = np.nan   # 3-month gap -> left empty
    afex = pd.DataFrame({"Dandume": F["Dandume"].fillna(0) * 0 + 1.0}, index=months)
    afex["Dandume"] = np.linspace(100, 200, len(months)) / 0.9   # AFEX runs 1/0.9 above FEWS NET
    X = pd.DataFrame({"market": ["Dandume"], "mp": [pd.Period("2022-03", "M")],
                      "value_one_year_ago": [16000.0], "value_one_month_ago": [np.nan]})
    empty = pd.DataFrame(index=months)
    return {"F": F, "X": X, "src": {"AFEX": afex, "NAERLS (old crop)": empty, "Other source Kaduna/Kano": empty}}


class TestGapFilledPanel:
    def test_observed_values_are_never_changed(self):
        inp = _inputs()
        P = B.build_long(inp).set_index(["market", "month"])
        obs = inp["F"].stack()
        for (m, mk), v in obs.items():
            assert P.loc[(mk, m), "price_NGN_per_kg"] == pytest.approx(round(v, 2))
            assert P.loc[(mk, m), "flag"] == "observed"

    def test_source_fill_uses_prior_overlap_ratio(self):
        P = B.build_long(_inputs()).set_index(["market", "month"])
        r = P.loc[("Dandume", "2020-10")]
        assert r.flag == "filled_direct" and r.source == "AFEX"
        assert r.ratio_applied == pytest.approx(0.9, abs=1e-4)
        assert r.ratio_overlap_months == 9            # Jan to Sep 2020, all before the gap

    def test_interpolation_only_for_gaps_of_two_months_or_less(self):
        P = B.build_long(_inputs()).set_index(["market", "month"])
        assert P.loc[("Biu", "2020-05"), "flag"] == "filled_interpolated"
        for m in ["2021-02", "2021-03", "2021-04"]:
            assert P.loc[("Biu", m), "flag"] == "missing"

    def test_recovered_value_is_observed(self):
        P = B.build_long(_inputs()).set_index(["market", "month"])
        assert P.loc[("Dandume", "2021-03"), "source"] == "FEWS NET API, value_one_year_ago"
        assert P.loc[("Dandume", "2021-03"), "price_NGN_per_kg"] == pytest.approx(160.0)

    def test_cutoff_drops_later_months_recovered_rows_and_late_anchors(self):
        P = B.build_long(_inputs(), cutoff="2021-06").set_index(["market", "month"])
        assert P.loc[("Dandume", "2021-07"), "flag"] == "missing"            # after the cutoff
        assert P.loc[("Dandume", "2021-03"), "source"] != "FEWS NET API, value_one_year_ago"
        # 2-month gap whose right anchor is after the cutoff cannot be interpolated
        Q = B.build_long(_inputs(), cutoff="2020-05").set_index(["market", "month"])
        assert Q.loc[("Biu", "2020-05"), "flag"] == "missing"


def _weekly():
    dates = pd.date_range("2012-05-02", "2026-09-30", freq="7D")
    rows = []
    for j, m in enumerate(["Dandume", "Biu"]):
        for i, d in enumerate(dates):
            rows.append(dict(date=d, market=m, diesel=100 + d.month, rainfall_simple_average=d.month,
                             rainfall_inverse_distance=d.month, ndvi_simple_average=0.1 * d.month,
                             ndvi_inverse_distance=0.1 * d.month, upstream_market=None if m == "Dandume" else "Dandume",
                             upstream_lag_weeks=1, market_kind="producing", state="X", market_id=j))
    return pd.DataFrame(rows)


class TestMonthlyPanel:
    def _long(self):
        months = [str(p) for p in pd.period_range("2012-05", "2026-08", freq="M")]
        rows = [dict(market=mk, month=m, price_NGN_per_kg=100.0 + i, flag="observed", fill_kind="")
                for mk in ["Dandume", "Biu"] for i, m in enumerate(months)]
        L = pd.DataFrame(rows)
        L.loc[(L.market == "Biu") & (L.month == "2020-06"), ["flag", "fill_kind"]] = ["filled_interpolated", "1-month gap between 2020-05 and 2020-07"]
        return L

    def test_published_lags_drivers_one_month(self):
        P, *_ = build_panel(self._long(), _weekly(), published=True)
        P0, *_ = build_panel(self._long(), _weekly(), published=False)
        a = P[(P.market == "Biu") & (P.date == "2020-07-01")].diesel.iloc[0]
        b = P0[(P0.market == "Biu") & (P0.date == "2020-06-01")].diesel.iloc[0]
        assert a == pytest.approx(b)

    def test_targets_are_observed_only_and_known_from_their_month(self):
        P, *_ = build_panel(self._long(), _weekly(), published=True)
        r = P[(P.market == "Biu") & (P.date == "2020-06-01")].iloc[0]
        assert np.isnan(r.target_price) and r.is_proxy_price
        r = P[(P.market == "Biu") & (P.date == "2020-07-01")].iloc[0]
        assert pd.Timestamp(r.target_known_date) == pd.Timestamp("2020-07-01")

    def test_observed_only_removes_fills(self):
        P, *_ = build_panel(self._long(), _weekly(), published=True, observed_only=True)
        assert np.isnan(P[(P.market == "Biu") & (P.date == "2020-06-01")].price.iloc[0])

    def test_grid_drops_origins_with_interpolation_anchored_later(self):
        _, _, price, flag, fill_kind, markets = build_panel(self._long(), _weekly(), published=True)
        G, drops = build_grid(price, flag, fill_kind, markets)
        g = G[G.market == "Biu"]
        assert pd.Timestamp("2020-06-01") not in set(g.origin)       # anchor 2020-07 is after it
        assert pd.Timestamp("2020-07-01") in set(g.origin)
        assert drops["interpolated_input_from_after_origin"] >= 1


def test_monthly_fx_cutoff_reads_day_before_forecast(tmp_path):
    days = pd.bdate_range("2001-12-03", "2026-09-30")
    pd.DataFrame({"date": days, "buyingrate": 100.0, "centralrate": np.linspace(100, 1500, len(days)),
                  "sellingrate": 100.0}).to_csv(tmp_path / "fx.csv", index=False)
    month = pd.DatetimeIndex(["2024-03-01"])
    cut = (month + pd.DateOffset(months=1)) + pd.Timedelta(days=13)
    X, log = _fx_daily_features(tmp_path / "fx.csv", cut)
    Xe, _ = _fx_daily_features(tmp_path / "fx.csv", pd.DatetimeIndex(["2024-04-14"]))
    np.testing.assert_allclose(X, Xe)
    assert log["cutoffs_past_last_day"] == 0
