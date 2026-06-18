"""Hydrology evaluation metrics on observed/simulated discharge.

Why not just RMSE? In hydrology a model is judged on how well it reproduces the
*shape* of the hydrograph, not just average error. See docs/03_hydrology_metrics.md.
"""
import numpy as np


def nse(obs, sim) -> float:
    """Nash-Sutcliffe Efficiency. 1 = perfect, 0 = no better than the mean, <0 = worse."""
    obs, sim = np.asarray(obs), np.asarray(sim)
    denom = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - np.sum((sim - obs) ** 2) / denom)


def kge(obs, sim):
    """Kling-Gupta Efficiency, returned with its (r, alpha, beta) components."""
    obs, sim = np.asarray(obs), np.asarray(sim)
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim.std() / obs.std()    # variability ratio
    beta = sim.mean() / obs.mean()   # bias ratio
    value = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return float(value), float(r), float(alpha), float(beta)


def rmse(obs, sim) -> float:
    obs, sim = np.asarray(obs), np.asarray(sim)
    return float(np.sqrt(np.mean((sim - obs) ** 2)))


def pbias(obs, sim) -> float:
    """Percent bias. Positive => the model overestimates on average."""
    obs, sim = np.asarray(obs), np.asarray(sim)
    return float(100.0 * np.sum(sim - obs) / np.sum(obs))


def summary(obs, sim) -> dict:
    """All metrics at once, as a printable/serialisable dict."""
    value, r, _alpha, _beta = kge(obs, sim)
    return {
        "NSE": nse(obs, sim),
        "KGE": value,
        "RMSE": rmse(obs, sim),
        "PBIAS_%": pbias(obs, sim),
        "r": r,
    }
