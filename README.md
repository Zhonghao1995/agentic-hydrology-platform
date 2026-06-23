# Agentic Hydrology Platform

> Ask in plain language. Get the right hydrological simulation.

A unified, agentic, **natural-language-driven** platform for modelling water across
**river basins, cities, and countryside**. Hydrological modelling today is fragmented and
expert-gated — separate tools for catchments and for cities, each with its own data
wrangling, setup, and interpretation. This platform puts one conversational interface in
front of all of it: describe the question, and an agentic runtime assembles the data,
selects and runs the right model, and explains the result — uniting data-driven and
physically based approaches in a single workflow.

![The Agentic Hydrology Platform](docs/overview.png)

## How it fits together

The platform is one **agentic runtime** conducting a set of specialised engines, designed
to interoperate, over a shared open-data foundation. You speak to the runtime; it speaks to
the models — and keeps every step as an inspectable, reproducible artifact.

### aiswmm — the agentic runtime
The conductor. You describe a modelling goal in plain language; aiswmm coordinates the
workflow while execution stays deterministic, inspectable, and artifact-based — with QA
verification, provenance tracking, and modelling memory, so results are reproducible and
auditable. It is exposed as MCP servers and Skills, so it also runs under Codex, Claude,
Hermes, or OpenClaw.
→ [agentic-swmm-workflow](https://github.com/Zhonghao1995/agentic-swmm-workflow)

### LSTM — catchment streamflow & flood forecasting
A data-driven engine that learns rainfall–runoff behaviour from records and predicts river
discharge and floods at catchment to regional scale. Best where gauged or large-sample data
exist and you need fast streamflow and flood forecasts.
→ *this repository (details below)*

### agentic SWMM — reproducible stormwater modelling
The SWMM framework the **aiswmm** runtime drives (same repository as above): a
verification-first, auditable workflow for EPA SWMM — GIS preprocessing, model generation,
deterministic SWMM runs, QA checks, calibration support, and provenance, natural-language
driven but artifact-based, with the modeller in control. For urban & rural drainage, runoff,
and Low-Impact-Development (LID) design.
→ [agentic-swmm-workflow](https://github.com/Zhonghao1995/agentic-swmm-workflow)

### agentic MIKE+ — headless MIKE+ automation
A headless, natural-language automation layer for DHI **MIKE+**: an MCP server + Skills that
let an agent inspect, edit, run, read, and plot a MIKE+ model end to end — no GUI. For
detailed urban-water and flood modelling and batch/scenario automation; reading results and
plotting need no MIKE+ license.
→ [Agentic-MIKE-Plus](https://github.com/Zhonghao1995/Agentic-MIKE-Plus)

### CIS — data preprocessing & model building
Draw an area (today, anywhere in Canada) and it pulls the open data for that spot — rainfall,
terrain, land cover, soil, and the city's storm network — and assembles a complete,
ready-to-run model. It uses real municipal networks where published (Victoria, Ottawa) and
synthesises one elsewhere. This is the upstream data-preprocessing stage that feeds the
engines above.
→ [SWMMCanada](https://github.com/Zhonghao1995/SWMMCanada)

## This repository — the LSTM engine

This repo is the platform's **deep-learning rainfall–runoff engine**: it trains LSTM networks
to predict river discharge and floods at catchment scale. It is built to be read and learned
from — you start on a zero-download synthetic basin and move to real catchment data, reusing
the same model and training code throughout.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/01_generate_data.py     # synthetic rainfall–runoff
python scripts/03_train_lstm.py         # train + evaluate  (NSE ≈ 0.87 on held-out years)
```

**Why LSTMs?** A river's flow today depends on weeks to months of past weather — snowpack,
soil moisture, groundwater. An LSTM's memory is built to capture exactly that kind of
long-range dependence, which is what makes it strong at streamflow and flood forecasting.

Learn the ideas: [LSTM intuition](docs/01_lstm_intuition.md) · [from LLMs to LSTMs](docs/02_from_llm_to_lstm.md) · [hydrology metrics](docs/03_hydrology_metrics.md).

## License

MIT
