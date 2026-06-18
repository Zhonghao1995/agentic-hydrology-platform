"""Training and prediction loops for the runoff LSTM.

Coming from LLM fine-tuning, this is the part you never saw: with OpenAI you
POST a dataset and poll a job. Here *you own the loop* -- forward pass, loss,
backprop, optimizer step. It is worth reading every line once.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def get_device(prefer: str = "auto") -> torch.device:
    """Pick a device. 'auto' uses Apple-Silicon MPS if present, else CPU."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit(
    model,
    train_ds,
    val_ds,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device | None = None,
    verbose: bool = True,
):
    """Train with MSE on standardized discharge; keep the best-val weights."""
    device = device or get_device()
    model.to(device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine decay to ~0 over training: large steps early to escape the
    # "predict the mean" basin, small steps late to settle into a good minimum.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        val_loss /= len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:  # keep the checkpoint that generalises best
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"epoch {epoch:3d}/{epochs}  train {train_loss:.4f}  val {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict(model, ds, *, batch_size: int = 256, device: torch.device | None = None):
    """Return (predictions, targets) as standardized numpy arrays."""
    device = device or get_device()
    model.to(device).eval()
    loader = DataLoader(ds, batch_size=batch_size)
    preds, obs = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
        obs.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(obs)
