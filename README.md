# Agentic Hydrology Platform

> Ask in plain language. Get the right hydrological simulation.
>
> A unified, agentic, **natural-language-driven** platform for modelling water across
> **river basins, cities, and countryside** — uniting data-driven and physically based
> models behind a single conversational interface.

![The Agentic Hydrology Platform](docs/overview.png)

A natural-language **[aiswmm](https://github.com/Zhonghao1995/agentic-swmm-workflow)**
runtime reads intent in plain language, runs the right engine over open data, and audits
every step. The platform's modules:

- **[aiswmm](https://github.com/Zhonghao1995/agentic-swmm-workflow)** — agentic runtime · natural-language orchestration & audit
- **LSTM** — large-scale catchment streamflow & flood · *this repository*
- **[agentic SWMM](https://github.com/Zhonghao1995/agentic-swmm-workflow)** — urban & rural drainage, runoff & LID
- **[agentic MIKE+](https://github.com/Zhonghao1995/Agentic-MIKE-Plus)** — urban & flood modelling with MIKE+
- **[CIS](https://github.com/Zhonghao1995/SWMMCanada)** — data preprocessing

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
