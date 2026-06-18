# Evaluating runoff models: NSE, KGE & friends

In hydrology a model is judged on how well it reproduces the **shape of the
hydrograph** — the timing and size of peaks, the recession tails, the baseflow —
not just average error. These are the metrics that matter, all implemented in
[`src/lstm_hydrology/metrics.py`](../src/lstm_hydrology/metrics.py).

## Why not just RMSE?

RMSE is in physical units (mm/day) and is fine, but it has no built-in sense of
scale: an RMSE of 0.7 is excellent for a small creek and terrible for a great
river. You cannot compare basins, and "is this good?" has no answer. The field
uses *skill scores* that compare the model to a trivial baseline.

## Nash–Sutcliffe Efficiency (NSE)

The standard. It compares your model's squared error to the error of simply
predicting the **long-term mean flow**:

```
NSE = 1 − Σ(sim − obs)²  /  Σ(obs − mean(obs))²
```

| NSE | Meaning |
|---|---|
| **1.0** | perfect |
| **0.0** | no better than predicting the mean every day |
| **< 0** | *worse* than the mean — the model is actively bad |

NSE is mathematically R² between simulated and observed flow. It is dominated by
the **largest flows** (errors are squared), so it is effectively a peak-focused
score — convenient for floods, but it can hide poor low-flow behavior. This repo's
default model reaches **NSE ≈ 0.87** on the held-out test years.

## Kling–Gupta Efficiency (KGE)

NSE blends three different ways of being wrong into one number. KGE pulls them
apart, which makes it far more *diagnostic*:

```
KGE = 1 − √[ (r − 1)² + (α − 1)² + (β − 1)² ]
```

- **r** — correlation: is the *timing* right?
- **α = σ_sim / σ_obs** — variability ratio: is the model too flashy or too damped?
- **β = μ_sim / μ_obs** — bias ratio: is the total water volume right?

When KGE is low, the components tell you *why*. Our model: r = 0.93 (good timing),
with the small remaining gap coming mostly from a slight low bias (β < 1).

## A few more you will meet

- **PBIAS** — percent bias, `100 · Σ(sim − obs) / Σ obs`. Positive = overestimates.
  Ours is ≈ −6%, i.e. it slightly underestimates total volume.
- **NSE on log-flow (log-NSE)** — recompute NSE on `log(flow)` to judge **low-flow
  and drought** behavior, which plain NSE ignores.
- **FHV (high-flow volume bias)** — error over just the top ~2% of flows; a
  **flood-relevant** score. Add it when you reach Stage 5.
- **Peak timing error** — is the flood peak on the right *day*? Critical for warnings.

## The NSE *loss* (for training)

You can train *directly* for NSE instead of plain MSE. Kratzert's basin-averaged
NSE loss weights each basin's squared error by `1/σ_obs²`, so that high-variance
and low-variance basins contribute comparably when training one model on many
basins. For a single basin it reduces to scaled MSE, which is why this repo
trains on MSE (of transformed flow) — but it is the natural upgrade at
[Stage 4](00_roadmap.md).
