"""
Recurrent forecaster: one pooled model across all markets, with a learnable
per-market embedding, emitting all horizons from a single forward pass.

The ONLY structural difference between the RNN and GRU variants is the
recurrent cell. Everything else -- embedding width, head, loss, optimiser,
scaling, splitting -- is shared, so a difference in results is attributable to
gating and nothing else.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RecurrentForecaster(nn.Module):
    def __init__(
        self,
        kind: str,
        n_channels: int,
        n_markets: int,
        n_horizons: int,
        n_flat: int = 0,
        hidden: int = 64,
        emb: int = 8,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if kind not in ("RNN", "GRU"):
            raise ValueError(kind)
        self.kind = kind
        cell = nn.RNN if kind == "RNN" else nn.GRU
        # forward()'s h[-1] already takes the TOP layer's final state
        # regardless of num_layers, so no change needed there. Inter-layer
        # dropout is left at PyTorch's default (0) rather than reusing
        # head_dropout, to keep this a single, isolated capacity lever.
        self.rnn = cell(n_channels, hidden, num_layers=num_layers, batch_first=True,
                        nonlinearity="tanh") if kind == "RNN" else \
            cell(n_channels, hidden, num_layers=num_layers, batch_first=True)
        self.emb = nn.Embedding(n_markets, emb)
        self.in_drop = nn.Dropout(input_dropout)
        head_in = hidden + emb + n_flat
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons),
        )

    def forward(self, x: torch.Tensor, flat: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
        x = self.in_drop(x)
        _, h = self.rnn(x)
        h = h[-1]
        parts = [h, self.emb(mid)]
        if flat is not None and flat.numel():
            parts.append(flat)
        return self.head(torch.cat(parts, dim=-1))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict:
        return {n: int(p.numel()) for n, p in self.named_parameters()}


def train_one(
    net: RecurrentForecaster,
    Xtr, Ftr, Itr, ytr, Wtr,
    Xva, Fva, Iva, yva,
    epochs: int = 80,
    batch: int = 256,
    lr: float = 5e-3,
    patience: int = 15,
    clip: float = 1.0,
    weight_decay: float = 0.0,
    lr_schedule: str = "none",
    huber_delta: float = 1.0,
    run_label: str = "",
) -> tuple[RecurrentForecaster, dict]:
    """Chronologically split, early-stopped fit. Returns the best-validation state.

    `run_label` is only used to make a NaN/Inf error message identify which
    cut/seed produced it; passing nothing still fails loudly, just without
    that context."""
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = None
    if lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.HuberLoss(delta=huber_delta, reduction="none")

    n = len(Xtr)
    train_loss_subsampled = n > 8000
    if train_loss_subsampled:
        sub = torch.from_numpy(
            np.random.choice(n, size=4000, replace=False)).long().to(Xtr.device)

    history = []
    best, best_state, bad, ran = float("inf"), None, 0, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            opt.zero_grad()
            pred = net(Xtr[idx], Ftr[idx], Itr[idx])
            per = loss_fn(pred, ytr[idx]).mean(dim=1)
            loss = (per * Wtr[idx]).sum() / Wtr[idx].sum()
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{run_label}: training loss is {loss.item()} (non-finite) at "
                    f"epoch {ep + 1}, batch starting at {k}. Stopping rather than "
                    f"continuing to train on a diverged model.")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
        if sched is not None:
            sched.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(Xva, Fva, Iva), yva).mean().item()
            if train_loss_subsampled:
                t = loss_fn(net(Xtr[sub], Ftr[sub], Itr[sub]), ytr[sub]).mean().item()
            else:
                t = loss_fn(net(Xtr, Ftr, Itr), ytr).mean().item()
        if not (np.isfinite(v) and np.isfinite(t)):
            raise RuntimeError(
                f"{run_label}: validation loss {v} or train-checkpoint loss {t} is "
                f"non-finite after epoch {ep + 1}. Stopping rather than early-stopping "
                f"on a diverged model.")
        history.append({"epoch": ep + 1, "train_loss": t, "val_loss": v,
                        "train_loss_subsampled": train_loss_subsampled})
        ran = ep + 1
        if v < best - 1e-7:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, {"best_val": best, "epochs_run": ran, "n_train": n, "n_val": len(Xva),
                "history": history}


@torch.no_grad()
def predict(net: RecurrentForecaster, X, F, I) -> np.ndarray:
    net.eval()
    return net(X, F, I).cpu().numpy()
