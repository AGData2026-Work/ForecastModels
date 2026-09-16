"""
The actual inference path (workplan 3.1/3.2/3.3): given a run directory
that already has a saved checkpoint (D-43's `--save-latest-cut-models`),
produce a real forecast from the most recent available data, not a
historical walk-forward evaluation.

Everything in this project up to now answers "how would this model have
done historically." Nothing before this answered "what does it predict
for next week, right now." This reuses `build_sequence`/`build_flat`/
`Scaler` exactly as training does, so there is only one code path for
featurization, not two that could silently drift apart (Monitoring test
#3, train/serve skew, by construction rather than by a separate check).

Decided as a weekly batch job (D-47's 3.1), not an on-demand service:
matches this project's own weekly data cadence and 13-week retrain
cycle, and needs no request-latency engineering a live service would.

    python src/predict.py --run-dir outputs/build3_underfit_corrected/GRU_unconditional

Writes outputs/<run>/predictions/<today>.csv and, if a previous
prediction file exists, prints both side by side (D-47's 3.3 canary
convention: a newly promoted model's forecasts are never trusted alone
until there is at least one prior forecast to compare against).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data import Scaler, build_flat, build_sequence, load_panel
from model import RecurrentForecaster, load_checkpoint, predict as net_predict


def load_current_checkpoints(run_dir: Path):
    """Loads every seed's checkpoint from the one cut directory D-43's
    save convention keeps (older cuts are deleted as each new one is
    saved, so there is always at most one). Raises a clear error if none
    exists, rather than a confusing downstream failure."""
    models_dir = run_dir / "models"
    if not models_dir.exists():
        raise SystemExit(f"{run_dir} has no models/ directory -- re-run its training "
                         "with --save-latest-cut-models first.")
    cut_dirs = sorted(models_dir.glob("cut_*"))
    if not cut_dirs:
        raise SystemExit(f"{models_dir} exists but has no cut_* subdirectory.")
    cut_dir = cut_dirs[-1]
    checkpoints = sorted(cut_dir.glob("seed_*.pt"))
    if not checkpoints:
        raise SystemExit(f"{cut_dir} has no seed_*.pt checkpoints.")
    return cut_dir, checkpoints


def rebuild_scaler(scaler_state: dict) -> Scaler:
    sc = Scaler()
    for k, v in scaler_state.items():
        setattr(sc, k, v)
    return sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-staleness-weeks", type=int, default=2,
                    help="refuse to forecast if the panel's last date is older "
                         "than this many weeks -- forecasting from stale input "
                         "silently is worse than refusing loudly")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    cfg = meta["config"]
    cut_dir, checkpoint_paths = load_current_checkpoints(run_dir)
    print(f"using checkpoint cut: {cut_dir.name} ({len(checkpoint_paths)} seeds)")

    panel = load_panel(cfg["data"]["panel"], rainfall_scheme=cfg["features"]["climate_scheme"],
                       fourier_k=cfg["features"]["fourier_k"],
                       interp_limit=cfg["data"]["price_interp_limit_weeks"])
    last_date = panel.dates[-1]
    staleness_weeks = (pd.Timestamp.now() - last_date).days / 7
    if staleness_weeks > a.max_staleness_weeks:
        raise SystemExit(f"panel's last date ({last_date.date()}) is {staleness_weeks:.1f} "
                         f"weeks old, over the {a.max_staleness_weeks}-week limit -- "
                         "refusing to forecast from stale input. Refresh the panel data "
                         "or raise --max-staleness-weeks if this is expected.")

    L = cfg["model"]["lookback"]
    use_lag52 = bool(cfg["features"]["explicit_lag52"])
    H = meta["horizons"]
    i = len(panel.dates) - 1  # the most recent available week

    nets, scaler = [], None
    for ckpt_path in checkpoint_paths:
        state_dict, scaler_state, ckpt_meta = load_checkpoint(ckpt_path)
        if scaler is None:
            scaler = rebuild_scaler(scaler_state)
        net = RecurrentForecaster(
            ckpt_meta["kind"], ckpt_meta["n_channels"], ckpt_meta["n_markets"],
            ckpt_meta["n_horizons"], n_flat=ckpt_meta["n_flat"],
            hidden=ckpt_meta["hidden"], emb=ckpt_meta["market_embedding_dim"],
            **({"num_layers": ckpt_meta["num_layers"]} if "num_layers" in ckpt_meta else {}),
        )
        net.load_state_dict(state_dict)
        net.eval()
        nets.append(net)
    market_list = load_checkpoint(checkpoint_paths[0])[2]["markets"]

    rows = []
    for j, market in enumerate(market_list):
        p0 = panel.price[i, j]
        if not np.isfinite(p0) or p0 <= 0:
            continue
        X, ok = build_sequence(panel, i, j, L)
        if not ok:
            continue
        F, ok = build_flat(panel, i, j, H, use_lag52, False)
        if not ok:
            continue
        Xs, Fs = scaler.transform(X[None], F[None])
        preds_this_market = []
        for net in nets:
            yhat = net_predict(net, torch.tensor(Xs, dtype=torch.float32),
                              torch.tensor(Fs, dtype=torch.float32),
                              torch.tensor([j], dtype=torch.long))
            preds_this_market.append(scaler.y_inverse(yhat)[0])
        median_yhat = np.median(np.stack(preds_this_market), axis=0)
        for hi, h in enumerate(H):
            rows.append(dict(market=market, origin=str(last_date.date()), h=h,
                             forecast_price=float(p0 * np.exp(median_yhat[hi]))))

    out = pd.DataFrame(rows)
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now().date().isoformat()
    out_path = predictions_dir / f"{today}.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} rows)")

    prior_files = sorted(f for f in predictions_dir.glob("*.csv") if f != out_path)
    if prior_files:
        prior = pd.read_csv(prior_files[-1])
        merged = out.merge(prior, on=["market", "h"], suffixes=("_new", "_prior"))
        print(f"\ncanary comparison against {prior_files[-1].name} "
             f"(D-47's 3.3 convention -- do not trust the new forecast alone yet):")
        print(merged.to_string(index=False))
    else:
        print("\nno prior prediction file to compare against yet -- this is the first "
             "forecast from this run directory.")


if __name__ == "__main__":
    main()
