# Agentic Hydrology Platform

> Ask in plain language. Get the right hydrological simulation.
>
> A unified, agentic, **natural-language-driven** platform for modelling water across
> **river basins, cities, and countryside** — uniting data-driven and physically based
> models behind a single conversational interface.

![The Agentic Hydrology Platform](docs/overview.png)

The runtime is **[aiswmm](https://github.com/Zhonghao1995/agentic-swmm-workflow)**: it
reads intent from natural language, selects and runs the right models over open data,
and audits every step. It drives two complementary engines:

- **LSTM** — large-scale catchment streamflow & flood forecasting · *this repository*
- **SWMM** — urban & rural drainage, runoff & LID, via the [agentic SWMM workflow](https://github.com/Zhonghao1995/agentic-swmm-workflow)

## This repository — the LSTM engine

From a zero-download synthetic sandbox to real catchment data:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/01_generate_data.py     # synthetic rainfall–runoff
python scripts/03_train_lstm.py         # train + evaluate
```

Concepts: [LSTM intuition](docs/01_lstm_intuition.md) · [hydrology metrics](docs/03_hydrology_metrics.md).

## License

MIT
