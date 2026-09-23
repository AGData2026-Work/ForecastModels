"""
Walk-forward runner for the AFEX multi-commodity farmgate panel.

    python src/run_afex.py --config configs/afex_operational.yaml --kind RNN --smoke
    python src/run_afex.py --config configs/afex_operational_sorghum.yaml --kind GRU --device mps

No --convention: this panel has no forecast-window driver data, so there is
no "conditional"/foreknowledge arm. `data.sibling_commodity` in the config
(optional) repurposes the upstream channel to carry a sibling commodity's
own price history at the same market -- origin-time information only, never
the forecast window, so it is deployable, not foreknowledge (see
afex_data.py's docstring and DECISIONS D-31/D-32). No incumbent either, so
metrics are challenger vs a carry-forward "naive" benchmark, not vs a
panel-FE incumbent.

Writes to outputs/<build>/<kind>/:
    forecasts.csv            per-seed, per-origin, per-horizon predictions
    predictions_paired.csv   seed-aggregated
    metrics_by_horizon.csv   challenger and naive MAE/MAPE per horizon
    metrics_by_period.csv    the same, broken out by calendar quarter of the target week
    metrics_by_market.csv    the same, broken out by maize series
    epoch_log.csv, training_log.csv, split_log.csv
    run_metadata.json, cleaning_log.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from afex_data import build_afex_scored_windows, generate_afex_grid, load_afex_panel, maize_series_ids
from check_regression import append_run_history
from data import Scaler, build_training_windows, flat_feature_names, sequence_channel_names
from metrics import diebold_mariano
from model import RecurrentForecaster, predict, save_checkpoint, train_one
from walkforward import assign_to_cuts, chronological_split, recency_weights, retrain_cuts


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def mae(a, p) -> float:
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(p, float))))


def mape(a, p) -> float:
    a, p = np.asarray(a, float), np.asarray(p, float)
    ok = np.abs(a) > 1e-9
    return float(np.mean(np.abs((a[ok] - p[ok]) / a[ok])) * 100) if ok.any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--kind", required=True, choices=["RNN", "GRU"])
    ap.add_argument("--panel", default=None, help="override data.panel from the config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="3 cuts, 1 seed, 4 epochs: pipeline check, not a result")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--history", default="outputs/run_history.csv",
                    help="append-only log of MAE per (build, model, h), one row "
                         "per real (non-smoke) run, read by check_regression.py")
    ap.add_argument("--save-latest-cut-models", action="store_true",
                    help="persist each seed's trained model + scaler for the most "
                         "recent retrain cut only; off by default, zero effect on "
                         "any existing invocation")
    a = ap.parse_args()
    run_start = time.monotonic()

    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device)

    H = cfg["protocol"]["horizons"]
    L = cfg["model"]["lookback"]
    use_lag52 = bool(cfg["features"]["explicit_lag52"])
    use_macro = bool(cfg["features"].get("use_macro", False))
    seeds = a.seeds or cfg["training"]["seeds"]
    hidden = cfg["model"]["hidden"]
    if a.smoke:
        seeds = seeds[:1]

    default_out = f"outputs/{cfg['build']['name']}/{a.kind}"
    if a.smoke:
        default_out += "_smoke"
    out = Path(a.out or default_out)
    out.mkdir(parents=True, exist_ok=True)

    raw_upstream_map = cfg["data"].get("upstream_lag_map")
    upstream_lag_map = ({k: tuple(v) for k, v in raw_upstream_map.items()}
                        if raw_upstream_map else None)
    panel = load_afex_panel(a.panel or cfg["data"]["panel"],
                            sibling_commodity=cfg["data"].get("sibling_commodity"),
                            ndvi_path=cfg["data"].get("ndvi_path"),
                            rainfall_path=cfg["data"].get("rainfall_path"),
                            climate_state_map_path=cfg["data"].get("climate_state_map_path"),
                            upstream_lag_map=upstream_lag_map,
                            diesel_source_path=cfg["data"].get("diesel_source_path"),
                            fx_rate_path=cfg["data"].get("fx_rate_path"),
                            inflation_path=cfg["data"].get("inflation_path"))
    grid = generate_afex_grid(panel, H, L)
    train_ids = list(range(len(panel.markets)))

    cuts = retrain_cuts(grid["origin"], cfg["training"]["retrain_every_weeks"])
    grid = grid.assign(cut=assign_to_cuts(grid, cuts))
    if a.smoke:
        # This panel only spans 277 weeks; lag-52 anchors need 52 weeks of
        # history and h=26 needs room ahead, so no training window exists
        # before roughly week 79. The FIRST few retrain cuts are legitimately
        # empty (properly skipped and logged in a full run) -- unlike the
        # main pipeline's 12-year panel, where the first cuts already have
        # enough history. Smoke picks the LAST 3 cuts instead, which are
        # guaranteed to have data, so the smoke test actually proves the
        # pipeline runs rather than just exercising the skip path.
        keep_cuts = sorted(grid["cut"].unique())[-3:]
        grid = grid[grid["cut"].isin(keep_cuts)]
        cuts = list(keep_cuts)

    print(f"build={cfg['build']['name']} kind={a.kind}")
    print(f"panel {panel.log['n_weeks']}w x {panel.log['n_series']} series "
          f"({panel.log['n_maize_series']} maize, commodities: {panel.log['commodities']}) "
          f"({panel.log['first_week']} -> {panel.log['last_week']})")
    print(f"grid {len(grid)} maize-series-origin rows, {grid['origin'].nunique()} distinct origins")
    print(f"train on {len(train_ids)} series (all commodities pooled); {len(cuts)} retrain cuts "
          f"every {cfg['training']['retrain_every_weeks']}w")

    rows, tlog, skips, splits, epoch_rows = [], [], [], [], []
    T = lambda x: torch.tensor(np.asarray(x, dtype=np.float32), device=dev)
    I = lambda x: torch.tensor(np.asarray(x), dtype=torch.long, device=dev)
    net = None

    for ci, cut in enumerate(cuts):
        served = grid[grid["cut"] == cut]
        if not len(served):
            continue
        cut_i = panel.pos.get(pd.Timestamp(cut))
        if cut_i is None:
            continue

        Xtr_all, Ftr_all, ytr_all, mtr_all = build_training_windows(
            panel, H, L, origin_max_idx=cut_i - 1, train_market_ids=train_ids,
            use_lag52=use_lag52, use_realised_drivers=False, use_macro=use_macro,
        )
        if len(Xtr_all) < cfg["training"]["min_train_windows"]:
            skips.append(dict(cut=cut, reason="too_few_training_windows", n=len(Xtr_all)))
            continue

        tr_i, va_i, slog = chronological_split(
            mtr_all, cfg["training"]["val_fraction"],
            cfg["training"]["purge_weeks"], cfg["training"]["min_train_windows"])
        slog.update(cut=cut)
        splits.append(slog)

        Xtr, Ftr, ytr = Xtr_all[tr_i], Ftr_all[tr_i], ytr_all[tr_i]
        Xva, Fva, yva = Xtr_all[va_i], Ftr_all[va_i], ytr_all[va_i]
        idtr = mtr_all["market_id"].values[tr_i]
        idva = mtr_all["market_id"].values[va_i]

        sc = Scaler().fit(Xtr, Ftr, ytr if cfg["training"]["scale_targets"] else None)
        Xtr_s, Ftr_s = sc.transform(Xtr, Ftr)
        Xva_s, Fva_s = sc.transform(Xva, Fva)
        ytr_s, yva_s = sc.y_forward(ytr), sc.y_forward(yva)

        w = recency_weights(mtr_all["origin"].values[tr_i],
                            cfg["training"]["recency_half_life_weeks"])

        Xte, Fte, yte, mte, sk = build_afex_scored_windows(panel, served, H, L, use_lag52,
                                                           use_macro=use_macro)
        if len(sk):
            sk = sk.assign(cut=cut)
            skips.extend(sk.to_dict("records"))
        if not len(mte):
            continue
        Xte_s, Fte_s = sc.transform(Xte, Fte)

        if a.save_latest_cut_models:
            # Only the most recent cut's models are ever kept on disk (D-63
            # on main, mirrored here).
            models_dir = out / "models"
            if models_dir.exists():
                shutil.rmtree(models_dir)
            cut_dir = models_dir / f"cut_{pd.Timestamp(cut).date()}"
            cut_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = RecurrentForecaster(
                a.kind, Xtr.shape[-1], len(panel.markets), len(H),
                n_flat=Ftr.shape[1], hidden=hidden,
                emb=cfg["model"]["market_embedding_dim"],
                dropout=cfg["model"]["head_dropout"],
                input_dropout=cfg["model"]["input_dropout"],
            ).to(dev)
            net, info = train_one(
                net, T(Xtr_s), T(Ftr_s), I(idtr), T(ytr_s), T(w),
                T(Xva_s), T(Fva_s), I(idva), T(yva_s),
                epochs=4 if a.smoke else cfg["training"]["epochs"],
                batch=cfg["training"]["batch_size"],
                lr=cfg["training"]["lr"],
                patience=cfg["training"]["patience"],
                clip=cfg["training"]["grad_clip"],
                weight_decay=cfg["training"]["weight_decay"],
                lr_schedule=cfg["training"]["lr_schedule"],
                huber_delta=cfg["training"]["huber_delta"],
            )
            if a.save_latest_cut_models:
                save_checkpoint(net, sc, cut_dir / f"seed_{seed}.pt", meta={
                    "build": cfg["build"]["name"], "kind": a.kind,
                    "cut": str(pd.Timestamp(cut).date()), "seed": seed,
                    "hidden": hidden, "n_channels": Xtr.shape[-1],
                    "n_flat": Ftr.shape[1], "n_markets": len(panel.markets),
                    "n_horizons": len(H), "markets": panel.markets,
                    "market_embedding_dim": cfg["model"]["market_embedding_dim"],
                })
            yhat = sc.y_inverse(predict(net, T(Xte_s), T(Fte_s),
                                        I(mte["market_id"].values)))
            hist = info.pop("history", [])
            epoch_rows.extend(dict(cut=cut, seed=seed, **h) for h in hist)
            info.update(cut=cut, seed=seed, n_params=net.n_params(),
                        n_served=len(mte), purge_applied=slog["purge_weeks_applied"])
            tlog.append(info)
            for r in range(len(mte)):
                p0 = float(mte["origin_price"].iloc[r])
                for hi, h in enumerate(H):
                    rows.append(dict(
                        model=a.kind, build=cfg["build"]["name"], seed=seed,
                        origin=mte["origin"].iloc[r], market=mte["market"].iloc[r],
                        h=h, origin_price=p0,
                        actual=float(mte[f"actual_h{h}"].iloc[r]),
                        pred=float(p0 * np.exp(yhat[r, hi])),
                        naive=float(mte[f"naive_h{h}"].iloc[r])))
        print(f"  cut {ci+1}/{len(cuts)} {pd.Timestamp(cut).date()} "
              f"train={len(Xtr)} val={len(Xva)} purge={slog['purge_weeks_applied']}w "
              f"serves={len(mte)} params={net.n_params()}")

    if not rows:
        raise SystemExit("no forecasts produced; check the panel path and min_train_windows")

    fc = pd.DataFrame(rows)
    fc.to_csv(out / "forecasts.csv", index=False)
    pd.DataFrame(tlog).to_csv(out / "training_log.csv", index=False)
    pd.DataFrame(splits).to_csv(out / "split_log.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(out / "epoch_log.csv", index=False)

    agg = cfg["training"]["seed_aggregation"]
    g = fc.groupby(["origin", "market", "h"], as_index=False).agg(
        actual=("actual", "first"), origin_price=("origin_price", "first"),
        naive=("naive", "first"),
        pred=("pred", "median" if agg == "median" else "mean"),
        pred_sd=("pred", "std"))
    g = g.rename(columns={"pred": a.kind})
    g.to_csv(out / "predictions_paired.csv", index=False)

    dm = []
    for h in H:
        # sorted by (market, origin) so each market's rows stay in chronological
        # order together -- the DM overlap correction below is only valid within
        # one time-ordered series, not across the panel's pooled markets (D-50).
        s = g[g.h == h].sort_values(["market", "origin"])
        d = diebold_mariano(s["actual"], s[a.kind], s["naive"], h, groups=s["market"].values)
        d.update(h=h, vs="naive")
        dm.append(d)
    pd.DataFrame(dm).to_csv(out / "diebold_mariano.csv", index=False)

    metric_rows = []
    for h, gh in g.groupby("h"):
        ch_mae, na_mae = mae(gh["actual"], gh[a.kind]), mae(gh["actual"], gh["naive"])
        metric_rows.append(dict(
            h=int(h), n=len(gh),
            challenger_MAE=ch_mae, challenger_MAPE=mape(gh["actual"], gh[a.kind]),
            naive_MAE=na_mae, naive_MAPE=mape(gh["actual"], gh["naive"]),
            vs_naive_pct=(na_mae - ch_mae) / na_mae * 100 if na_mae else float("nan")))
    metrics_by_horizon = pd.DataFrame(metric_rows)
    metrics_by_horizon.to_csv(out / "metrics_by_horizon.csv", index=False)
    if not a.smoke:
        append_run_history(Path(a.history), cfg["build"]["name"], a.kind,
                           "n/a", metrics_by_horizon)

    g["target"] = g["origin"] + pd.to_timedelta(g["h"], unit="W")
    g["period"] = g["target"].dt.to_period("Q").astype(str)
    period_rows = []
    for (h, period), gp in g.groupby(["h", "period"]):
        ch_mae, na_mae = mae(gp["actual"], gp[a.kind]), mae(gp["actual"], gp["naive"])
        period_rows.append(dict(h=int(h), period=period, n=len(gp),
                                challenger_MAE=ch_mae, challenger_MAPE=mape(gp["actual"], gp[a.kind]),
                                naive_MAE=na_mae, naive_MAPE=mape(gp["actual"], gp["naive"])))
    pd.DataFrame(period_rows).sort_values(["h", "period"]).to_csv(out / "metrics_by_period.csv", index=False)

    market_rows = []
    for (h, market), gm in g.groupby(["h", "market"]):
        ch_mae, na_mae = mae(gm["actual"], gm[a.kind]), mae(gm["actual"], gm["naive"])
        market_rows.append(dict(h=int(h), market=market, n=len(gm),
                                challenger_MAE=ch_mae, challenger_MAPE=mape(gm["actual"], gm[a.kind]),
                                naive_MAE=na_mae, naive_MAPE=mape(gm["actual"], gm["naive"])))
    pd.DataFrame(market_rows).sort_values(["h", "market"]).to_csv(out / "metrics_by_market.csv", index=False)

    n_flat = int(Ftr.shape[1])
    wall_seconds = time.monotonic() - run_start
    (out / "run_metadata.json").write_text(json.dumps(jsonable(dict(
        build=cfg["build"]["name"], build_note=cfg["build"]["note"],
        kind=a.kind, smoke=a.smoke, wall_seconds=round(wall_seconds, 1),
        horizons=H, lookback=L, hidden=hidden, seeds=seeds, n_params=int(net.n_params()),
        param_breakdown=net.param_breakdown(),
        sequence_channels=sequence_channel_names(use_macro),
        n_flat_features=n_flat,
        flat_feature_names=flat_feature_names(H, use_lag52, False),
        train_series=panel.markets,
        scored_series=[panel.markets[j] for j in maize_series_ids(panel)],
        n_retrain_cuts=len(cuts), retrain_every_weeks=cfg["training"]["retrain_every_weeks"],
        config=cfg, panel_log=panel.log,
    )), indent=2))

    if not a.smoke:
        time_log_path = Path("outputs/training_time_log.csv")
        time_row = pd.DataFrame([dict(
            timestamp=pd.Timestamp.now().isoformat(), build=cfg["build"]["name"],
            kind=a.kind, convention="n/a", n_retrain_cuts=len(cuts),
            n_seeds=len(seeds), hidden=hidden, n_params=int(net.n_params()),
            device=a.device, wall_seconds=round(wall_seconds, 1))])
        if time_log_path.exists():
            time_row = pd.concat([pd.read_csv(time_log_path), time_row], ignore_index=True)
        time_log_path.parent.mkdir(parents=True, exist_ok=True)
        time_row.to_csv(time_log_path, index=False)

    (out / "cleaning_log.json").write_text(json.dumps(jsonable(dict(
        panel_log=panel.log,
        windows_skipped=skips,
        n_windows_skipped=len(skips),
        cuts_skipped_too_few_training_windows=[
            s for s in skips if isinstance(s, dict) and s.get("reason") == "too_few_training_windows"],
        scored_pairs_per_horizon=int(len(g) / len(H)),
    )), indent=2))

    print("\n" + metrics_by_horizon.round(2).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
