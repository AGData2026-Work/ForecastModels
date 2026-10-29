"""
How much each input pushes the production GRU's forecasts (2026-10-26), in the same terms as the analyst's
"typical push" tables, so the two models can be compared input by input.

    python src/gru_attribution.py --run-dir outputs/production_monthly_parity_serving --out outputs/gru_attribution

Method: integrated gradients (Sundararajan et al. 2017) on the serving checkpoints, one per seed. For each forecast the
scaled inputs move in 32 steps from a baseline of all zeros (the training average of every input, after the model's own
scaling) to their actual values; gradients along the path, times the input change, split the forecast's log change
exactly into a part per input (the parts add up to forecast minus baseline, up to the step error, reported). The
baseline itself is the forecast this model would make for this market in average conditions; it is reported as its own
"starting point" line, like the analyst's "usual creep". Seeds are averaged (the production forecast takes their
median; the mean of seeds is reported next to it).

Inputs are grouped into the themes a reader would use: recent price path; upstream market price (with its availability
flag); diesel; rainfall and crop greenness; time of year; year-earlier prices; exchange rate; gap to import cost.
Typical push = median absolute push over all forecasts, as in the analyst's deck. Reads the serving model trained to
June 2026, so for past months it describes what TODAY's model relies on, not what each past forecast used.

Panel set-up mirrors src/predict.py (same functions, same order) so inputs match what the live model sees.
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
from data import attach_checks, attach_fx_summary, attach_nowcast, attach_parity, build_flat, build_sequence, load_panel  # noqa: E402
from model import RecurrentForecaster, load_checkpoint  # noqa: E402
from predict import load_current_checkpoints, rebuild_scaler  # noqa: E402

SEQ_GROUP = {"rel_log_price": "Recent price path", "rel_log_diesel": "Diesel", "rel_log_upstream": "Upstream market price",
             "upstream_mask": "Upstream market price", "rainfall": "Rainfall and crop greenness",
             "ndvi": "Rainfall and crop greenness", "sin1": "Time of year", "cos1": "Time of year", "sin2": "Time of year",
             "cos2": "Time of year"}
FLAT_GROUP = {"lag52": "Year-earlier prices", "fx_": "Exchange rate", "months_since_deval": "Exchange rate",
              "parity_gap": "Gap to import cost"}
STEPS = 32


def flat_group(name: str) -> str:
    return next(g for k, g in FLAT_GROUP.items() if name.startswith(k))


def panel_like_predict(cfg: dict):
    freq = cfg["data"].get("frequency", "W")
    p = load_panel(cfg["data"]["panel"], rainfall_scheme=cfg["features"]["climate_scheme"],
                   fourier_k=cfg["features"]["fourier_k"], interp_limit=cfg["data"]["price_interp_limit_weeks"],
                   frequency=freq)
    if cfg["features"].get("fx_summary", False):
        attach_fx_summary(p, cfg["data"]["fx_rate_path"], cfg["data"]["inflation_path"],
                          fx_lag_months=cfg["data"].get("fx_lag_months", 1),
                          inflation_lag_months=cfg["data"].get("inflation_lag_months", 2),
                          keep=cfg["features"].get("fx_summary_subset"), fx_daily_path=cfg["data"].get("fx_daily_path"),
                          origin_is_month_end=(freq == "MS"), monthly_cutoff_day=cfg["data"].get("fx_monthly_cutoff_day"),
                          monthly_cutoff_months=cfg["data"].get("fx_monthly_cutoff_months", 1),
                          fx_column=cfg["data"].get("fx_rate_column", "Official Exchange Rate"))
    if cfg["features"].get("parity_path"):
        attach_parity(p, cfg["features"]["parity_path"])
    if cfg["features"].get("nowcast_path"):
        attach_nowcast(p, cfg["features"]["nowcast_path"])
    attach_checks(p, cfg["features"])
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/production_monthly_parity_serving")
    ap.add_argument("--out", default="outputs/gru_attribution")
    ap.add_argument("--from-month", default="2017-01-01")
    a = ap.parse_args()
    run = Path(a.run_dir)
    meta = json.loads((run / "run_metadata.json").read_text())
    cfg, H = meta["config"], meta["horizons"]
    seq_names, flat_names = meta["sequence_channels"], meta["flat_feature_names"]
    for unsupported in ("use_macro", "real_target"):
        if cfg["features"].get(unsupported) or cfg["training"].get(unsupported):
            raise SystemExit(f"{unsupported} runs are not supported here")
    p = panel_like_predict(cfg)
    _, ckpts = load_current_checkpoints(run)
    nets, scaler = [], None
    for c in ckpts:
        sd, sc_state, m = load_checkpoint(c)
        scaler = scaler or rebuild_scaler(sc_state)
        net = RecurrentForecaster(m["kind"], m["n_channels"], m["n_markets"], m["n_horizons"], n_flat=m["n_flat"],
                                  hidden=m["hidden"], emb=m["market_embedding_dim"])
        net.load_state_dict(sd); net.eval(); nets.append(net)
    markets = load_checkpoint(ckpts[0])[2]["markets"]
    L, use_lag52 = cfg["model"]["lookback"], bool(cfg["features"]["explicit_lag52"])
    y_sd = np.asarray(scaler.y_sd); y_mu = np.asarray(scaler.y_mu)
    groups = sorted(set(SEQ_GROUP.values()) | set(FLAT_GROUP.values()))
    rows = []
    for i, date in enumerate(p.dates):
        if date < pd.Timestamp(a.from_month):
            continue
        for j, mk in enumerate(markets):
            X, ok = build_sequence(p, i, j, L, False)
            if not ok:
                continue
            F, ok = build_flat(p, i, j, H, use_lag52, False, use_fx_summary=True)
            if not ok:
                continue
            Xs, Fs = scaler.transform(X[None], F[None])
            Xs, Fs = torch.tensor(Xs, dtype=torch.float32), torch.tensor(Fs, dtype=torch.float32)
            alphas = torch.linspace(0, 1, STEPS + 1)[1:].view(-1, 1, 1)
            xp = (alphas * Xs).requires_grad_(True); fp = (alphas[:, :, 0] * Fs).requires_grad_(True)
            mid = torch.full((STEPS,), j, dtype=torch.long)
            att_seq = np.zeros((len(H),) + Xs.shape[1:]); att_flat = np.zeros((len(H), Fs.shape[1]))
            full, base, med = [], [], []
            for net in nets:
                for hi in range(len(H)):
                    out = net(xp, fp, mid)[:, hi].sum()
                    gx, gf = torch.autograd.grad(out, [xp, fp])
                    att_seq[hi] += (gx.mean(0) * Xs[0]).detach().numpy() * y_sd[hi] / len(nets)
                    att_flat[hi] += (gf.mean(0) * Fs[0]).detach().numpy() * y_sd[hi] / len(nets)
                with torch.no_grad():
                    full.append(net(Xs, Fs, mid[:1]).numpy()[0] * y_sd + y_mu)
                    base.append(net(torch.zeros_like(Xs), torch.zeros_like(Fs), mid[:1]).numpy()[0] * y_sd + y_mu)
            full, base = np.mean(full, 0), np.mean(base, 0)
            for hi, h in enumerate(H):
                r = dict(market=mk, origin=date, h=h, origin_price=float(p.price[i, j]), log_change_mean_of_seeds=float(full[hi]),
                         baseline=float(base[hi]))
                for g in groups:
                    r[g] = 0.0
                for c, nm in enumerate(seq_names):
                    r[SEQ_GROUP[nm]] += float(att_seq[hi][:, c].sum())
                for c, nm in enumerate(flat_names):
                    r[flat_group(nm)] += float(att_flat[hi][c])
                r["completeness_gap"] = r["log_change_mean_of_seeds"] - r["baseline"] - sum(r[g] for g in groups)
                rows.append(r)
    d = pd.DataFrame(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "pushes_by_forecast.csv", index=False)
    typ = d.groupby("h")[groups + ["baseline"]].agg(lambda s: 100 * s.abs().median()).T.round(2)
    typ.columns = [f"typical_push_{h}m_pct" for h in typ.columns]
    biggest = d.assign(top=d[groups].abs().idxmax(axis=1)).groupby("h").top.value_counts(normalize=True).unstack(0).fillna(0)
    typ = typ.join((100 * biggest).round(1).add_prefix("biggest_driver_share_").rename(columns=lambda c: c + "m_pct"))
    typ.to_csv(out / "typical_push.csv")
    print(typ.to_string())
    print("forecasts:", len(d) // len(H), "| median |completeness gap| (log):", round(d.completeness_gap.abs().median(), 5))


if __name__ == "__main__":
    main()
