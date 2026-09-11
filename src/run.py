"""
Walk-forward runner.

    python src/run.py --config configs/build1.yaml --kind RNN --convention unconditional
    python src/run.py --config configs/build2.yaml --kind GRU --convention conditional --smoke

Writes to outputs/<build>/<kind>_<convention>/:
    forecasts.csv          per-seed, per-origin, per-horizon predictions
    predictions_paired.csv  seed-aggregated, joined to panel_fe and naive
    paired_metrics.csv      headline table with vs-incumbent percentages
    metrics_by_market.csv
    seed_variance.csv
    change_control.json
    training_log.csv
    run_metadata.json
    cleaning_log.json       every drop, fill and skip, with counts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data import (Scaler, SEQ_BASE_CHANNELS, build_grid_windows,
                  build_training_windows, flat_feature_names, load_grid, load_panel)
from metrics import (change_control, diebold_mariano, paired_table, score_long)
from model import RecurrentForecaster, predict, train_one
from walkforward import (assign_to_cuts, chronological_split, recency_weights,
                         retrain_cuts)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--kind", required=True, choices=["RNN", "GRU"])
    ap.add_argument("--convention", required=True,
                    choices=["unconditional", "conditional", "conditional_exog",
                            "forecast_drivers"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="3 cuts, 1 seed, 4 epochs: pipeline check, not a result")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--panel", default=None, help="override data.panel from the config")
    ap.add_argument("--baseline", default=None,
                    help="override data.baseline_forecasts from the config")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.panel:
        cfg["data"]["panel"] = a.panel
    if a.baseline:
        cfg["data"]["baseline_forecasts"] = a.baseline
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device)

    H = cfg["protocol"]["horizons"]
    max_h = max(H)
    L = cfg["model"]["lookback"]
    use_lag52 = bool(cfg["features"]["explicit_lag52"])
    use_realised = a.convention in ("conditional", "conditional_exog", "forecast_drivers")
    realised_upstream = a.convention in ("conditional", "forecast_drivers")
    driver_source = "forecast" if a.convention == "forecast_drivers" else "realised"
    seeds = a.seeds or cfg["training"]["seeds"]
    hidden = a.hidden or cfg["model"]["hidden"]
    if a.smoke:
        seeds = seeds[:1]

    default_out = f"outputs/{cfg['build']['name']}/{a.kind}_{a.convention}"
    if a.smoke:
        default_out += "_smoke"
    out = Path(a.out or default_out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    panel = load_panel(cfg["data"]["panel"],
                       rainfall_scheme=cfg["features"]["climate_scheme"],
                       fourier_k=cfg["features"]["fourier_k"],
                       interp_limit=cfg["data"]["price_interp_limit_weeks"])
    grid, glog = load_grid(cfg["data"]["baseline_forecasts"], H,
                           drop_targets=cfg["data"]["drop_target_dates"])

    scored = glog["markets"]
    if cfg["data"]["train_markets"] == "all_in_panel":
        train_mkts = panel.markets
    else:
        train_mkts = scored
    train_ids = [panel.midx[m] for m in train_mkts]

    cuts = retrain_cuts(grid["origin"], cfg["training"]["retrain_every_weeks"])
    grid = grid.assign(cut=assign_to_cuts(grid, cuts))
    if a.smoke:
        keep_cuts = sorted(grid["cut"].unique())[:3]
        grid = grid[grid["cut"].isin(keep_cuts)]
        cuts = list(keep_cuts)

    print(f"build={cfg['build']['name']} kind={a.kind} convention={a.convention}")
    print(f"panel {panel.log['n_weeks']}w x {panel.log['n_markets']}m "
          f"({panel.log['first_week']} -> {panel.log['last_week']})")
    print(f"grid  {len(grid)} market-origin pairs, {grid['origin'].nunique()} distinct origins, "
          f"{len(scored)} scored markets")
    print(f"train on {len(train_mkts)} markets; {len(cuts)} retrain cuts "
          f"every {cfg['training']['retrain_every_weeks']}w")

    rows, tlog, skips, splits, epoch_rows = [], [], [], [], []
    T = lambda x: torch.tensor(np.asarray(x, dtype=np.float32), device=dev)
    I = lambda x: torch.tensor(np.asarray(x), dtype=torch.long, device=dev)

    for ci, cut in enumerate(cuts):
        served = grid[grid["cut"] == cut]
        if not len(served):
            continue
        cut_i = panel.pos.get(pd.Timestamp(cut))
        if cut_i is None:
            continue

        Xtr_all, Ftr_all, ytr_all, mtr_all = build_training_windows(
            panel, H, L, origin_max_idx=cut_i - 1, train_market_ids=train_ids,
            use_lag52=use_lag52, use_realised_drivers=use_realised,
            realised_upstream=realised_upstream, driver_source=driver_source,
        )
        if len(Xtr_all) < cfg["training"]["min_train_windows"]:
            skips.append(dict(cut=cut, reason="too_few_training_windows",
                              n=len(Xtr_all)))
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

        Xte, Fte, yte, mte, sk = build_grid_windows(
            panel, served, H, L, use_lag52=use_lag52, use_realised_drivers=use_realised,
            realised_upstream=realised_upstream, driver_source=driver_source)
        if len(sk):
            sk = sk.assign(cut=cut)
            skips.extend(sk.to_dict("records"))
        if not len(mte):
            continue
        Xte_s, Fte_s = sc.transform(Xte, Fte)

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = RecurrentForecaster(
                a.kind, Xtr.shape[-1], len(panel.markets), len(H),
                n_flat=Ftr.shape[1], hidden=hidden,
                emb=cfg["model"]["market_embedding_dim"],
                dropout=cfg["model"]["head_dropout"],
                input_dropout=cfg["model"]["input_dropout"],
                num_layers=cfg["model"].get("num_layers", 1),
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
                        model=a.kind, build=cfg["build"]["name"],
                        convention=a.convention, seed=seed,
                        origin=mte["origin"].iloc[r], market=mte["market"].iloc[r],
                        h=h, origin_price=p0,
                        actual=float(mte[f"actual_h{h}"].iloc[r]),
                        pred=float(p0 * np.exp(yhat[r, hi])),
                        panel_fe=float(mte[f"panel_fe_h{h}"].iloc[r]),
                        naive=float(mte[f"naive_h{h}"].iloc[r])))
        print(f"  cut {ci+1}/{len(cuts)} {pd.Timestamp(cut).date()} "
              f"train={len(Xtr)} val={len(Xva)} purge={slog['purge_weeks_applied']}w "
              f"serves={len(mte)} params={net.n_params()}")

    if not rows:
        raise SystemExit("no forecasts produced; check the config paths")

    fc = pd.DataFrame(rows)
    fc.to_csv(out / "forecasts.csv", index=False)
    pd.DataFrame(tlog).to_csv(out / "training_log.csv", index=False)
    pd.DataFrame(splits).to_csv(out / "split_log.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(out / "epoch_log.csv", index=False)

    agg = cfg["training"]["seed_aggregation"]
    g = fc.groupby(["origin", "market", "h"], as_index=False).agg(
        actual=("actual", "first"), origin_price=("origin_price", "first"),
        panel_fe=("panel_fe", "first"), naive=("naive", "first"),
        pred=("pred", "median" if agg == "median" else "mean"),
        pred_sd=("pred", "std"))
    g = g.rename(columns={"pred": a.kind})
    g.to_csv(out / "predictions_paired.csv", index=False)

    paired = paired_table(g, a.kind)
    paired.to_csv(out / "paired_metrics.csv", index=False)

    dm = []
    for h in H:
        s = g[g.h == h]
        for ref in ("panel_fe", "naive"):
            d = diebold_mariano(s["actual"], s[a.kind], s[ref], h)
            d.update(h=h, vs=ref)
            dm.append(d)
    pd.DataFrame(dm).to_csv(out / "diebold_mariano.csv", index=False)

    score_long(g, [a.kind, "panel_fe", "naive"], ["market", "h"]).to_csv(
        out / "metrics_by_market.csv", index=False)
    score_long(fc.rename(columns={"pred": a.kind}), [a.kind], ["seed", "h"]).to_csv(
        out / "seed_variance.csv", index=False)

    cc = change_control(paired, cfg["protocol"]["gate_horizons"],
                        cfg["protocol"]["gate_threshold_pct"])
    (out / "change_control.json").write_text(json.dumps(jsonable(cc), indent=2))

    n_flat = int(Ftr.shape[1])
    (out / "run_metadata.json").write_text(json.dumps(jsonable(dict(
        build=cfg["build"]["name"], build_note=cfg["build"]["note"],
        kind=a.kind, convention=a.convention, smoke=a.smoke,
        horizons=H, gate_horizons=cfg["protocol"]["gate_horizons"],
        lookback=L, hidden=hidden, seeds=seeds, n_params=int(net.n_params()),
        param_breakdown=net.param_breakdown(),
        sequence_channels=SEQ_BASE_CHANNELS,
        n_flat_features=n_flat,
        flat_feature_names=flat_feature_names(H, use_lag52, use_realised, realised_upstream,
                                              driver_source),
        train_markets=train_mkts, scored_markets=scored,
        n_retrain_cuts=len(cuts), retrain_every_weeks=cfg["training"]["retrain_every_weeks"],
        config=cfg, panel_log=panel.log, grid_log=glog,
    )), indent=2))

    (out / "cleaning_log.json").write_text(json.dumps(jsonable(dict(
        price_cells_interpolated=panel.log["price_cells_interpolated"],
        price_interp_limit_weeks=panel.log["interp_limit_weeks"],
        full_panel_outage_weeks=panel.log["full_panel_outage_weeks"],
        grid_rows_dropped_by_target_filter=glog["dropped_by_date_filter"],
        grid_dropped_target_dates=glog["dropped_target_dates"],
        grid_rows_dropped_incomplete_horizons=glog["dropped_incomplete_horizon_set"],
        windows_skipped=skips,
        n_windows_skipped=len(skips),
        scored_pairs_per_horizon=int(len(g) / len(H)),
        expected_pairs_per_horizon=glog["n_market_origin_pairs"],
    )), indent=2))

    print("\n" + paired.round(2).to_string(index=False))
    print("\nchange control:", cc["verdict"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
