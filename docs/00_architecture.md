# Architecture & vision

> The big picture for the **Agentic Hydrology Platform**. For the working LSTM
> component, see the [README](../README.md) or the [LSTM roadmap](00_roadmap.md).

![Platform architecture](architecture.svg)

## The problem

Answering a hydrological question today — *"what is the flood risk in this
catchment?"*, *"how should we size LID to cut this city's runoff?"* — means an
expert manually picking a model, finding and cleaning data, configuring, running,
and interpreting it. That is slow, hard to reproduce, and hard to audit.

## The idea

Put an **agentic runtime** in the expert's orchestration seat, over two
complementary models and a shared open-data layer.

## Layers

### Open data (foundation)
- **Natural catchments:** [Caravan](https://www.nature.com/articles/s41597-023-01975-w)
  — ERA5-Land forcing + HydroATLAS attributes + streamflow, standardized globally
  (6830 gauges) and **extensible to any region** via Google Earth Engine.
- **Urban:** OpenStreetMap, DEM, building footprints — the data stack
  [SWMManywhere](https://joss.theoj.org/papers/10.21105/joss.07729) uses to
  synthesize drainage networks from just a bounding box.

### Agentic runtime (orchestrator)
Five responsibilities (the heart of the platform):
1. **Data processing** — assemble forcing/attributes for the target region
   (Caravan extension for catchments; OSM/DEM for cities).
2. **Model selection** — choose LSTM vs SWMM (vs both) for the question + the data.
3. **Training** — fit/fine-tune with sane defaults; record the config.
4. **Running** — inference, scenarios, forecasts.
5. **Auditing** — a decision log + data/version provenance + metrics, so every
   result is **reproducible and accountable**.

### Models
- **LSTM — large-scale catchment streamflow & flood.** Data-driven, lumped, fast;
  trained **regionally** on many basins (the
  [*"never train an LSTM on a single basin"*](https://hess.copernicus.org/articles/28/4187/2024/)
  approach) with static catchment attributes, so one model serves a whole region
  and even ungauged basins. ✅ A working synthetic sandbox lives in
  `src/lstm_hydrology` (NSE 0.87).
- **SWMM — urban drainage & LID.** Physics-based, spatially distributed; models
  pipes, sub-catchments, and low-impact-development controls inside cities.
  Auto-built from open data in the **SWMManywhere** style. 🔭 Planned.

### Why two models
| | LSTM | SWMM |
|---|---|---|
| Type | data-driven (ML) | physics-based |
| Scale | catchment / regional, lumped | urban, distributed |
| Strength | gauged & large-scale flow, fast, ungauged transfer | pipes, LID, urban detail |
| Needs | streamflow records to train | network/terrain data (SWMManywhere builds it) |

They compose: an LSTM can supply **boundary inflows** to a SWMM city model; the
agent decides when to use which, and records why.

## Status (honest)
- ✅ **Component 1 (LSTM), sandbox:** synthetic data, full train/eval, NSE 0.87.
- 🔭 **Next:** real-data **regional LSTM on Caravan** → **agentic runtime**
  (declarative, audited runs) → **SWMM urban layer** → integration.

## Target structure (planned vs current)
```
src/
  lstm_hydrology/     ✅ Component 1 — LSTM runoff
  agentic_runtime/    🔭 Component 2 — data · select · train · run · audit
  swmm_urban/         🔭 Component 3 — SWMManywhere-style urban + LID
```
Only what exists is in the tree today — we add components as they become real.

## Component roadmap
- **C1 LSTM:** synthetic ✅ → Caravan regional → region fine-tune → flood metrics
  (FHV, peak timing). See [00_roadmap.md](00_roadmap.md).
- **C2 Agentic runtime:** define a `Run` spec (inputs → model → training →
  evaluation → **audit artifact**); first deliverable = wrap C1 training as a
  declarative, logged, reproducible job.
- **C3 SWMM urban:** wrap SWMManywhere to synthesize a SWMM model from a bbox; add
  LID scenario runs.
- **Integration:** the agent routes/chains C1 ↔ C3.

## References
- Caravan — [Kratzert et al. 2023, *Sci. Data*](https://www.nature.com/articles/s41597-023-01975-w) · [github.com/kratzert/Caravan](https://github.com/kratzert/Caravan)
- Regional training — [*HESS Opinions: Never train an LSTM on a single basin* (2024)](https://hess.copernicus.org/articles/28/4187/2024/)
- SWMManywhere — [Dobson et al. 2025, *Env. Modelling & Software*](https://www.sciencedirect.com/science/article/pii/S1364815225000428) · [JOSS](https://joss.theoj.org/papers/10.21105/joss.07729)
- neuralhydrology — [docs](https://neuralhydrology.readthedocs.io) (Caravan dataset support)
- Global LSTM flood forecasting on Caravan — [arXiv:2212.00719](https://arxiv.org/pdf/2212.00719)
