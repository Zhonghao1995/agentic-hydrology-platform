# Data

## `synthetic/` (start here)

Regenerate any time (it is git-ignored):

```bash
python scripts/01_generate_data.py
```

This writes `synthetic/basin.csv` with daily columns:

| column      | units   | meaning                          |
|-------------|---------|----------------------------------|
| `date`      | —       | daily index                      |
| `precip`    | mm/day  | precipitation (model input)      |
| `temp`      | °C      | air temperature (model input)    |
| `discharge` | mm/day  | streamflow (the target to predict)|

The series comes from a small conceptual model (`src/lstm_hydrology/synthetic.py`)
with snow, soil-moisture, and groundwater stores — so today's discharge depends
on *weeks to months* of past weather. See `docs/01_lstm_intuition.md`.

## Real data (when you graduate)

- **CAMELS-US** — 671 US catchments, daily forcings + streamflow + static
  attributes. The benchmark dataset for LSTM rainfall-runoff work.
  https://ral.ucar.edu/solutions/products/camels
- **CAMELS variants** — `CAMELS-GB`, `CAMELS-BR`, `CAMELS-AUS`, `CAMELS-CL`, etc.
- **neuralhydrology** — the reference library; its tutorials wrap CAMELS loading,
  training, and evaluation. https://neuralhydrology.readthedocs.io

Large raw datasets do **not** belong in git — keep them outside the repo (or in
this folder, which is git-ignored) and commit only code, configs, and results.
