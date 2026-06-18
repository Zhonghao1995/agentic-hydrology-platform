"""Stage 1, step 2: visualise the data so you know what you are modelling.

Always look at the hydrograph before training. You should be able to *see* the
seasonal cycle, storm responses, and slow recessions the LSTM will learn.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file, no display needed
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("data/synthetic/basin.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/01_data_overview.png"))
    ap.add_argument("--years", type=int, default=4, help="how many years to plot")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, index_col="date", parse_dates=True)
    window = df.iloc[: args.years * 365]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].bar(window.index, window["precip"], width=1.0, color="#4C72B0")
    axes[0].set_ylabel("Precip [mm/day]")
    axes[0].invert_yaxis()  # rainfall hangs from the top, hydrology convention
    axes[0].set_title(f"Synthetic basin -- first {args.years} years")
    temp_ax = axes[0].twinx()
    temp_ax.plot(window.index, window["temp"], color="#C44E52", lw=0.7, alpha=0.7)
    temp_ax.set_ylabel("Temp [degC]", color="#C44E52")

    axes[1].plot(window.index, window["discharge"], color="#55A868", lw=0.9)
    axes[1].set_ylabel("Discharge [mm/day]")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
