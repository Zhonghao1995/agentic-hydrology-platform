"""Synthetic rainfall-runoff data generator.

We drive a small conceptual hydrological model (an HBV-style "snow + soil +
two linear reservoirs" structure) with synthetic weather. The goal is *not*
hydrological realism -- it is to produce a discharge series whose value on any
day depends on *weeks to months* of past weather:

    * snowpack      -> stores winter precipitation, releases it in spring (long memory)
    * soil moisture -> modulates how much rain becomes runoff       (medium memory)
    * groundwater   -> drains slowly, producing baseflow recession  (medium memory)

That multi-timescale memory is exactly what an LSTM's cell state is good at
capturing, which makes this an honest sandbox for rainfall-runoff modelling
before you move on to real data (see docs/00_roadmap.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ModelParams:
    """Conceptual model parameters (HBV-style)."""

    tt: float = 0.0          # rain/snow temperature threshold [degC]
    ddf: float = 3.0         # degree-day snowmelt factor [mm/degC/day]
    fc: float = 250.0        # soil field capacity [mm]
    beta: float = 2.0        # runoff-generation nonlinearity [-]
    lp: float = 0.7          # soil fraction above which ET is unstressed [-]
    pet_factor: float = 0.5  # potential ET per +1 degC [mm/degC/day]
    k_fast: float = 0.30     # fast reservoir recession constant [1/day]
    k_slow: float = 0.03     # slow reservoir recession constant [1/day]
    k_perc: float = 0.05     # percolation, fast -> slow reservoir [1/day]


def generate_weather(n_days: int, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic daily temperature [degC] and precipitation [mm]."""
    rng = np.random.default_rng(seed)
    doy = np.arange(n_days) % 365

    # Temperature: seasonal sinusoid (coldest near new year) + persistent noise.
    seasonal = 10.0 - 12.0 * np.cos(2 * np.pi * doy / 365.0)
    noise = rng.normal(0.0, 2.5, n_days)
    for i in range(1, n_days):  # AR(1) smoothing so warm/cold spells last days
        noise[i] = 0.6 * noise[i - 1] + 0.8 * noise[i]
    temp = seasonal + noise

    # Precipitation: seasonal wet-day probability * gamma-distributed intensity.
    p_wet = 0.25 + 0.10 * np.sin(2 * np.pi * (doy - 200) / 365.0)
    wet = rng.random(n_days) < p_wet
    intensity = rng.gamma(shape=0.8, scale=13.0, size=n_days)
    precip = np.where(wet, intensity, 0.0)

    return pd.DataFrame({"temp": temp, "precip": precip})


def simulate_discharge(
    weather: pd.DataFrame,
    params: ModelParams | None = None,
    noise_frac: float = 0.04,
    seed: int = 0,
) -> np.ndarray:
    """Run the conceptual model on a weather frame -> daily discharge [mm/day]."""
    params = params or ModelParams()
    temp = weather["temp"].to_numpy()
    precip = weather["precip"].to_numpy()
    n = len(temp)

    snow, soil, fast, slow = 0.0, 0.5 * params.fc, 0.0, 10.0
    q = np.empty(n)

    for t in range(n):
        # 1. Split precipitation into snow / rain by temperature.
        if temp[t] <= params.tt:
            snow += precip[t]
            rain = 0.0
        else:
            rain = precip[t]
        # 2. Snowmelt (degree-day method).
        melt = min(snow, max(0.0, params.ddf * (temp[t] - params.tt)))
        snow -= melt
        liquid = rain + melt

        # 3. Soil moisture accounting + runoff generation.
        recharge = liquid * (soil / params.fc) ** params.beta
        soil += liquid - recharge
        pet = max(0.0, params.pet_factor * temp[t])
        soil -= pet * min(1.0, soil / (params.lp * params.fc))  # actual ET
        if soil > params.fc:        # spill excess soil water to runoff
            recharge += soil - params.fc
            soil = params.fc
        soil = max(0.0, soil)

        # 4. Route recharge through fast + slow linear reservoirs.
        fast += recharge
        perc = params.k_perc * fast
        fast -= perc
        slow += perc
        q_fast = params.k_fast * fast
        fast -= q_fast
        q_slow = params.k_slow * slow
        slow -= q_slow
        q[t] = q_fast + q_slow

    # Multiplicative observation noise (gauges are noisier at high flow).
    if noise_frac > 0:
        rng = np.random.default_rng(seed + 1)
        q = np.clip(q * (1.0 + rng.normal(0.0, noise_frac, n)), 0.0, None)
    return q


def make_dataset(
    n_years: int = 20,
    seed: int = 0,
    start: str = "2000-01-01",
    params: ModelParams | None = None,
) -> pd.DataFrame:
    """Build a tidy daily rainfall-runoff DataFrame indexed by date."""
    n_days = n_years * 365
    weather = generate_weather(n_days, seed=seed)
    discharge = simulate_discharge(weather, params=params, seed=seed)
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    df = pd.DataFrame(
        {
            "precip": weather["precip"].to_numpy(),
            "temp": weather["temp"].to_numpy(),
            "discharge": discharge,
        },
        index=dates,
    )
    df.index.name = "date"
    return df
