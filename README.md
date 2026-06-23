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

The **[aiswmm](https://github.com/Zhonghao1995/agentic-swmm-workflow)** runtime is the
conductor: it reads intent from natural language, runs the right engine over open data,
and keeps an audit trail of every decision, so results stay reproducible and accountable.
It orchestrates a set of specialised modules:

- **[aiswmm](https://github.com/Zhonghao1995/agentic-swmm-workflow)** — the agentic runtime: natural-language orchestration, model selection, and step-by-step audit.
- **LSTM** — large-scale catchment streamflow & flood forecasting, learned from data. *(This repository.)*
- **[agentic SWMM](https://github.com/Zhonghao1995/agentic-swmm-workflow)** — urban & rural drainage, runoff, and Low-Impact-Development (LID) design.
- **[agentic MIKE+](https://github.com/Zhonghao1995/Agentic-MIKE-Plus)** — urban & flood modelling with MIKE+.
- **[CIS](https://github.com/Zhonghao1995/SWMMCanada)** — data preprocessing: turning raw open datasets into model-ready inputs.

Everything runs on **open data**, so a new study area can be set up anywhere, without proprietary inputs.

## This repository — the LSTM engine

This repo is the platform's **deep-learning rainfall–runoff engine**: it trains LSTM
networks to predict river discharge and floods at catchment scale. It is built to be read
and learned from — you start on a zero-download synthetic basin and move to real catchment
data, reusing the same model and training code throughout.

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
