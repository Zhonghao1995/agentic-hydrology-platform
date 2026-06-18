"""Stage 1, step 3: train an LSTM to predict discharge, then evaluate it.

    python scripts/03_train_lstm.py                  # sensible defaults
    python scripts/03_train_lstm.py --features precip # weaker: rain only, no temp
    python scripts/03_train_lstm.py --seq-len 30      # too short to "remember" snow

Watch how the test NSE moves as you change the look-back and the features --
that is the whole experiment.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lstm_hydrology import RunoffLSTM, build_splits, fit, get_device, predict
from lstm_hydrology import metrics as M

# Optional target transforms. Discharge is right-skewed (many low-flow days, a
# few big peaks); modelling log/sqrt flow stops the network from hedging toward
# the mean and shaving the peaks. The inverse maps predictions back to mm/day.
TRANSFORMS = {
    "none": (lambda x: x, lambda x: np.clip(x, 0.0, None)),
    "sqrt": (np.sqrt, lambda x: np.clip(x, 0.0, None) ** 2),
    "log": (np.log1p, lambda x: np.expm1(np.clip(x, 0.0, None))),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("data/synthetic/basin.csv"))
    ap.add_argument("--features", nargs="+", default=["precip", "temp"])
    ap.add_argument("--target-transform", choices=list(TRANSFORMS), default="log")
    ap.add_argument("--seq-len", type=int, default=90)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.csv, index_col="date", parse_dates=True)
    forward, inverse = TRANSFORMS[args.target_transform]
    df["_target"] = forward(df["discharge"].to_numpy())
    datasets, y_scaler = build_splits(
        df, feature_cols=args.features, target_col="_target", seq_len=args.seq_len
    )
    print("windows per split:", {k: len(v) for k, v in datasets.items()})

    device = get_device(args.device)
    print(f"device: {device}")
    model = RunoffLSTM(
        n_features=len(args.features),
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    )
    history = fit(
        model, datasets["train"], datasets["val"],
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device,
    )

    # Evaluate on the held-out test period, back in physical units (mm/day):
    # undo standardization, then undo the target transform.
    pred_std, obs_std = predict(model, datasets["test"], device=device)
    sim = inverse(y_scaler.inverse_transform(pred_std))
    obs = inverse(y_scaler.inverse_transform(obs_std))
    scores = M.summary(obs, sim)
    print("\nTest-period skill:")
    for name, value in scores.items():
        print(f"  {name:8s} {value: .4f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    _plot_results(history, obs, sim, scores, args.outdir)
    (args.outdir / "test_scores.json").write_text(json.dumps(scores, indent=2))
    print(f"\nSaved figures + scores to {args.outdir}/")


def _plot_results(history, obs, sim, scores, outdir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (standardized)")
    ax.set_title("Training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "02_training_curve.png", dpi=120)

    n = min(730, len(obs))  # show up to the first 2 years of the test period
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(obs[:n], color="black", lw=1.0, label="observed")
    ax.plot(sim[:n], color="#C44E52", lw=1.0, alpha=0.8, label="LSTM")
    ax.set_xlabel("days into test period")
    ax.set_ylabel("Discharge [mm/day]")
    ax.set_title(f"Hydrograph (test)   NSE={scores['NSE']:.3f}   KGE={scores['KGE']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "03_hydrograph_test.png", dpi=120)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(obs, sim, s=6, alpha=0.3, color="#4C72B0")
    hi = float(max(obs.max(), sim.max()))
    ax.plot([0, hi], [0, hi], color="black", lw=0.8)
    ax.set_xlabel("observed [mm/day]")
    ax.set_ylabel("simulated [mm/day]")
    ax.set_title("Observed vs simulated (test)")
    fig.tight_layout()
    fig.savefig(outdir / "04_scatter.png", dpi=120)


if __name__ == "__main__":
    main()
