"""LSTM for hydrological runoff prediction -- a learning sandbox.

Public API is intentionally small; import what you need:

    from lstm_hydrology import make_dataset, build_splits, RunoffLSTM, fit, predict
    from lstm_hydrology import metrics
"""
from . import metrics
from .dataset import Scaler, WindowDataset, build_splits, make_windows
from .model import RunoffLSTM
from .synthetic import ModelParams, generate_weather, make_dataset, simulate_discharge
from .train import fit, get_device, predict

__all__ = [
    "ModelParams",
    "generate_weather",
    "simulate_discharge",
    "make_dataset",
    "build_splits",
    "make_windows",
    "WindowDataset",
    "Scaler",
    "RunoffLSTM",
    "metrics",
    "fit",
    "predict",
    "get_device",
]
