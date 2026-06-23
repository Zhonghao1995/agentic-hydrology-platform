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

The platform is one **agentic runtime** conducting a set of specialised modelling engines
over a shared, open-data foundation. You speak to the runtime; it speaks to the models.

### aiswmm — the agentic runtime
The conductor. It interprets a natural-language request, decides which engines and data are
needed, runs them, and records every decision as an audit trail — so going from a question
("what is the flood risk for this town?") to a defensible answer takes no manual tool-wiring.
→ [agentic-swmm-workflow](https://github.com/Zhonghao1995/agentic-swmm-workflow)

### LSTM — catchment streamflow & flood forecasting
A data-driven engine that learns rainfall–runoff behaviour from records and predicts river
discharge and floods at catchment to regional scale. Best where gauged or large-sample data
exist and you need fast, accurate streamflow and flood forecasts.
→ *this repository (details below)*

### agentic SWMM — urban & rural drainage
Physically based stormwater modelling of pipes, channels, sub-catchments, and
Low-Impact-Development (LID) controls, across both city and countryside. Use it for drainage
design, runoff control, and green-infrastructure scenarios.
→ [agentic-swmm-workflow](https://github.com/Zhonghao1995/agentic-swmm-workflow)

### agentic MIKE+ — advanced urban & flood modelling
Integrated urban-water and flood modelling on MIKE+, for richer hydraulic detail — coupled
1D/2D flooding and complex networks — than a lumped model can provide. Use it when a study
needs detailed, coupled flood and drainage simulation.
→ [Agentic-MIKE-Plus](https://github.com/Zhonghao1995/Agentic-MIKE-Plus)

### CIS — data preprocessing
Turns raw open datasets — meteorology, terrain, catchment attributes, network data — into
clean, model-ready inputs for every engine above. It is the step that makes "set up a new
study area anywhere" actually work.
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
