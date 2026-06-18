"""Turn a rainfall-runoff DataFrame into supervised LSTM training windows.

Two ideas matter here and are worth internalising:

1. **Many-to-one windows.** Sample *i* feeds the LSTM a window of `seq_len`
   days of forcings and asks it to predict discharge on the *last* day of that
   window. The look-back is how the network "sees" antecedent conditions
   (snowpack, wet soil) that a same-day model would miss.

2. **No leakage.** Scalers are fit on the training period only, and the
   train/val/test split is by *time*, never shuffled. Evaluating a forecasting
   model on randomly shuffled days would let it peek at the future.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Scaler:
    """Standardize using statistics from the training period only."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


def make_windows(features: np.ndarray, target: np.ndarray, seq_len: int):
    """Many-to-one sliding windows.

    Window *i* = features[i : i+seq_len] predicts target[i+seq_len-1].
    Returns float32 arrays X (n, seq_len, n_features) and y (n,).
    """
    n_samples = len(features) - seq_len + 1
    X = np.stack([features[i : i + seq_len] for i in range(n_samples)])
    y = target[seq_len - 1 :]
    return X.astype(np.float32), y.astype(np.float32)


class WindowDataset(Dataset):
    """Thin torch Dataset over pre-built (X, y) window arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_splits(
    df,
    feature_cols,
    target_col,
    seq_len: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
):
    """Temporal split -> fit scalers on train -> window each split.

    Returns ``(datasets, target_scaler)`` where ``datasets`` has keys
    ``train`` / ``val`` / ``test`` mapping to WindowDataset objects, and
    ``target_scaler`` lets you convert predictions back to mm/day.
    """
    features = df[feature_cols].to_numpy(np.float32)
    target = df[target_col].to_numpy(np.float32)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    # Fit scalers on the training period only -> no leakage from the future.
    f_scaler = Scaler(features[:n_train].mean(0), features[:n_train].std(0) + 1e-8)
    t_scaler = Scaler(target[:n_train].mean(), target[:n_train].std() + 1e-8)
    feats = f_scaler.transform(features)
    targ = t_scaler.transform(target)

    # Each split's windows may reach back into the previous split for their
    # look-back (that's forcing data, not the label), so every evaluation day
    # gets a full-length input window.
    bounds = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }
    datasets = {}
    for name, (lo, hi) in bounds.items():
        start = max(0, lo - (seq_len - 1))
        Xw, yw = make_windows(feats[start:hi], targ[start:hi], seq_len)
        label_days = np.arange(start + seq_len - 1, hi)
        keep = label_days >= lo  # only label days that fall inside this split
        datasets[name] = WindowDataset(Xw[keep], yw[keep])
    return datasets, t_scaler
