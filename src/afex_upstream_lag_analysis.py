"""
Upstream lead-lag analysis: which AFEX maize market's price movements lead
which others, and by how many weeks -- sizing a future "upstream market,
lagged by N weeks" exogenous variable the way the original FEWSNET/NADIH
panel already carries one (data/panel_weekly.parquet's upstream_price /
upstream_market / upstream_lag_weeks columns).

    python src/afex_upstream_lag_analysis.py

Method
------
For every ordered pair of maize markets (A, B), cross-correlate A's weekly
log-return series against B's, at lags k = -13 .. +13 weeks. Returns, not
price levels, because price levels share a common inflation trend across
every market in this panel and would correlate strongly regardless of any
real lead-lag relationship -- the same reasoning behind using returns for
the sibling-commodity screen (DECISIONS D-31).

corr(r_A(t), r_B(t+k)) at k > 0 means A's move at t and B's move k weeks
later are correlated: A leads B by k weeks. For each market B, the "best
upstream candidate" is the market A with the highest correlation at a
strictly positive lag (A leads B, not the reverse and not contemporaneous).

This does not change, retrain or touch any existing model. It is scoped
purely to identify which lag would be worth building into a future
exogenous variable -- analogous to the existing project's own
upstream_market / upstream_lag_weeks design, but discovered from this
panel's own data rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAX_LAG = 13
MIN_OVERLAP = 20


def load_maize_wide(path: str) -> pd.DataFrame:
    pl = pd.read_excel(path, sheet_name="Panel_Long", parse_dates=["date"])
    maize = pl[pl["commodity"] == "Maize"]
    wide = maize.pivot_table(index="date", columns="market", values="price_NGN_per_kg", aggfunc="first")
    full_idx = pd.date_range(wide.index.min(), wide.index.max(), freq="7D")
    return wide.reindex(full_idx)


def cross_corr_at_lag(ra: pd.Series, rb: pd.Series, k: int) -> tuple[float, int]:
    """corr(ra(t), rb(t+k)). k>0: A leads B. k<0: B leads A."""
    if k >= 0:
        a, b = ra.iloc[: len(ra) - k if k else len(ra)], rb.shift(-k)
    else:
        a, b = ra, rb.shift(-k)
    both = pd.concat([a, b], axis=1).dropna()
    if len(both) < MIN_OVERLAP:
        return float("nan"), len(both)
    return float(both.iloc[:, 0].corr(both.iloc[:, 1])), len(both)


def main() -> None:
    path = "/Users/augmentumadvisory/Downloads/20260824_AFEX_MultiCommodity_Weekly_Panel_v1.xlsx"
    wide = load_maize_wide(path)
    markets = sorted(wide.columns)
    returns = np.log(wide).diff()

    detail_rows = []
    for a in markets:
        for b in markets:
            if a == b:
                continue
            for k in range(-MAX_LAG, MAX_LAG + 1):
                corr, n = cross_corr_at_lag(returns[a], returns[b], k)
                if np.isnan(corr):
                    continue
                detail_rows.append(dict(market_a=a, market_b=b, lag_weeks=k, n=n, correlation=corr))
    detail = pd.DataFrame(detail_rows)
    Path("outputs").mkdir(exist_ok=True)
    detail.to_csv("outputs/afex_upstream_lag_full_detail.csv", index=False)

    # Best lag per ordered pair (by |correlation|, but keep the sign)
    idx = detail.groupby(["market_a", "market_b"])["correlation"].apply(
        lambda s: s.abs().idxmax())
    best_pair = detail.loc[idx].reset_index(drop=True)
    best_pair.to_csv("outputs/afex_upstream_lag_best_per_pair.csv", index=False)

    # For each market B, best upstream candidate A: strictly positive lag
    # (A leads B), highest correlation among those.
    leads = detail[detail["lag_weeks"] > 0]
    tree_rows = []
    for b in markets:
        cand = leads[leads["market_b"] == b]
        if not len(cand):
            tree_rows.append(dict(market=b, best_upstream_market=None,
                                  lag_weeks=None, correlation=None, n=None))
            continue
        best = cand.loc[cand["correlation"].idxmax()]
        tree_rows.append(dict(market=b, best_upstream_market=best["market_a"],
                              lag_weeks=int(best["lag_weeks"]),
                              correlation=float(best["correlation"]), n=int(best["n"])))
    tree = pd.DataFrame(tree_rows).sort_values("correlation", ascending=False)
    tree.to_csv("outputs/afex_upstream_lag_tree.csv", index=False)

    pd.set_option("display.width", 200)
    print("BEST UPSTREAM CANDIDATE PER MARKET (strictly positive lag, i.e. a real lead, not contemporaneous)")
    print(tree.to_string(index=False))

    # Flag any 2-cycles: A's best upstream is B and B's best upstream is A
    cycles = []
    tmap = tree.set_index("market")["best_upstream_market"].to_dict()
    for b, a in tmap.items():
        if a and tmap.get(a) == b:
            pair = tuple(sorted([a, b]))
            if pair not in cycles:
                cycles.append(pair)
    if cycles:
        print(f"\n{len(cycles)} mutual (2-cycle) pairs, not a clean tree there: {cycles}")
    else:
        print("\nNo mutual 2-cycles: the best-upstream assignment forms a clean forest (no pair leads each other).")

    print("\nwrote outputs/afex_upstream_lag_full_detail.csv")
    print("wrote outputs/afex_upstream_lag_best_per_pair.csv")
    print("wrote outputs/afex_upstream_lag_tree.csv")


if __name__ == "__main__":
    main()
