"""Summarize the neuralhydrology smoke-test run: per-basin metrics + hydrographs.

Run (from repo root, after evaluating the test period):
    .venv-nh/bin/python experiments/real_smoketest/summarize.py
"""
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

run = sorted(Path("experiments/real_smoketest/runs").glob("*/"))[-1]
results_dir = Path("experiments/real_smoketest/results")
results_dir.mkdir(parents=True, exist_ok=True)

test_dir = sorted((run / "test").glob("model_epoch*"))[-1]
df = pd.read_csv(next(test_dir.glob("*metrics*.csv")), dtype={"basin": str})
df.to_csv(results_dir / "test_metrics.csv", index=False)
print(df.round(3).to_string(index=False))
print(f"\nmedian test NSE: {df['NSE'].median():.3f}   median KGE: {df['KGE'].median():.3f}")

with open(next(test_dir.glob("*results*.p")), "rb") as fh:
    res = pickle.load(fh)

basins = list(res.keys())
fig, axes = plt.subplots(len(basins), 1, figsize=(11, 2.3 * len(basins)))
for ax, basin in zip(axes, basins):
    xr = res[basin][list(res[basin].keys())[0]]["xr"]
    obs_var = [v for v in xr.data_vars if v.endswith("_obs")][0]
    sim_var = [v for v in xr.data_vars if v.endswith("_sim")][0]
    nse = float(df.loc[df.basin.astype(str) == str(basin), "NSE"].iloc[0])
    ax.plot(xr["date"], xr[obs_var].values.flatten(), color="black", lw=1.0, label="observed")
    ax.plot(xr["date"], xr[sim_var].values.flatten(), color="#C44E52", lw=1.0, alpha=0.8, label="LSTM")
    ax.set_title(f"basin {basin}   test NSE = {nse:.2f}", fontsize=10)
    ax.set_ylabel("Q [mm/d]")
    ax.legend(fontsize=8, loc="upper right")
fig.suptitle("Regional LSTM on 4 CAMELS-US basins — test period (neuralhydrology)")
fig.tight_layout()
fig.savefig(results_dir / "hydrographs_test.png", dpi=120, bbox_inches="tight")
print("saved", results_dir / "hydrographs_test.png")
