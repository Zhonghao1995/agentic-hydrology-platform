# Learning roadmap

A staged path from "what is an LSTM" to "regional flood/runoff models on real
data." Each stage has a **goal**, **do**, **success check**, and **exercises**.
Stages 0–2 run entirely in this repo on synthetic data; 3–5 move to real data.

---

## Stage 0 — Setup & orientation

**Goal:** run the whole pipeline once and understand what you are predicting.

**Do:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/01_generate_data.py     # synthetic 40-year basin
python scripts/02_explore_data.py       # look at results/01_data_overview.png
```
Read [`docs/01_lstm_intuition.md`](01_lstm_intuition.md) and, if you came from
LLMs, [`docs/02_from_llm_to_lstm.md`](02_from_llm_to_lstm.md).

**Success check:** you can point at the overview plot and say *why* discharge
spikes in spring even when it is not raining (snowmelt — stored winter precip).

---

## Stage 1 — LSTM mechanics on synthetic rainfall-runoff  ← core of this repo

**Goal:** train an LSTM to predict discharge from `[precip, temp]` and read
every line of the training loop.

**Do:**
```bash
python scripts/03_train_lstm.py         # NSE ~0.87 on the held-out test years
```
Open [`src/lstm_hydrology/model.py`](../src/lstm_hydrology/model.py) and
[`train.py`](../src/lstm_hydrology/train.py). Make sure you can explain the
many-to-one windowing, standardization fit on train only, and best-checkpoint.

**Success check:** test NSE > 0.8 and you can explain what NSE means
([`docs/03_hydrology_metrics.md`](03_hydrology_metrics.md)).

---

## Stage 2 — Experiment like a hydrologist

**Goal:** build intuition for what drives skill by running controlled ablations.

**Exercises** (each is one CLI flag — predict the result *before* you run it):

| Experiment | Command | What it teaches |
|---|---|---|
| Too-short memory | `--seq-len 10` | snowmelt needs a long look-back; NSE drops |
| Rain only, no temp | `--features precip` | without temperature the model cannot tell snow from rain |
| No target transform | `--target-transform none` | skewed targets → the model shaves peaks (worse bias) |
| Tiny / huge capacity | `--hidden 8` / `--hidden 256` | under- vs over-fitting |
| Seed variance | `--seed 1`, `--seed 2`, … | single runs vary; this motivates ensembles |

**Bigger exercise — ensembles:** train 5 seeds, average their predictions, and
compare NSE to any single run. (Kratzert et al. ensemble LSTMs for exactly this
reason.) This is the natural next script to write yourself.

**Success check:** you can predict the *direction* each flag moves NSE and explain why.

---

## Stage 3 — Real single-basin data (CAMELS)

**Goal:** repeat Stage 1 on one real catchment.

**Do:** download [CAMELS-US](https://ral.ucar.edu/solutions/products/camels),
pick one basin, map its columns (precip, temperature, PET, discharge) to the
same pipeline. Reuse `build_splits`, `RunoffLSTM`, `fit`, `metrics` unchanged —
only the data loader changes.

**Watch out for:** units (mm/day vs m³/s — normalize by catchment area), missing
data, and a realistic train/test split by water years.

**Success check:** an honest test-period NSE on real data (0.6–0.8 is good for a
single basin with few inputs) and a hydrograph you can critique.

---

## Stage 4 — Regional, multi-basin LSTM (neuralhydrology)

**Goal:** train *one* LSTM on *many* basins at once — the approach that beats
calibrated conceptual models.

**Do:** use the [`neuralhydrology`](https://neuralhydrology.readthedocs.io)
library (PyTorch, same concepts as here, plus static catchment attributes,
the NSE loss, and proper CAMELS loaders). Work through its tutorials.

**Success check:** a regional model that generalizes to basins it was *not*
trained on (the "prediction in ungauged basins", PUB, test).

---

## Stage 5 — Flood-focused extensions

**Goal:** specialize toward your interest — flood peaks and forecasting.

**Directions:**
- **Peak-flow skill:** add flood metrics (FHV — high-flow bias; peak timing error).
- **Forecast lead time:** predict discharge `h` days ahead from forcings up to today
  (and, later, from weather *forecasts* as inputs).
- **Extremes:** evaluate only on the largest annual events; LSTMs can struggle to
  extrapolate beyond the training range — quantify it.
- **Uncertainty:** predict a distribution, not a point (e.g. quantile or mixture-density
  heads), so a flood warning comes with a confidence band.

**Success check:** a model and an evaluation that a flood forecaster would trust —
judged on peaks and timing, not just average NSE.

---

### References
- Kratzert et al. (2018), *Rainfall–runoff modelling using LSTM networks*, HESS — [doi:10.5194/hess-22-6005-2018](https://doi.org/10.5194/hess-22-6005-2018)
- Kratzert et al. (2019), *Towards learning universal, regional, and local hydrological behaviors via ML*, HESS — [doi:10.5194/hess-23-5089-2019](https://doi.org/10.5194/hess-23-5089-2019)
- neuralhydrology — https://neuralhydrology.readthedocs.io
- CAMELS — https://ral.ucar.edu/solutions/products/camels
