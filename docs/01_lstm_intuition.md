# LSTM intuition for rainfall-runoff

## Why memory matters in hydrology

Today's streamflow is **not** a function of today's rain alone. It depends on the
*state* of the catchment, which is the accumulated result of weeks to months of
past weather:

- **Snowpack** — winter precipitation falls as snow and is stored, sometimes for
  months, then released as a melt-driven flood in spring. (You can see this in
  `results/01_data_overview.png`: discharge peaks in late winter even on dry days.)
- **Soil moisture** — the same storm produces a trickle on dry soil but a flood on
  saturated soil. The catchment "remembers" whether it has been wet.
- **Groundwater** — drains slowly, producing the gentle *recession* tail after a
  storm and the baseflow that keeps rivers running in droughts.

A model that maps **today's inputs → today's output** is blind to all of this.
We need a model with **memory** of the input history. That is what an LSTM gives us.

## From RNN to LSTM in one breath

A plain Recurrent Neural Network (RNN) reads a sequence one step at a time,
carrying a hidden state forward. In principle it remembers the past — in practice
the gradient signal *vanishes* over long sequences, so it forgets things that
happened 100 days ago, which is exactly the snowpack timescale we care about.

The **LSTM (Long Short-Term Memory)** fixes this with an explicit memory channel.

## The LSTM cell

Two things flow from one timestep to the next:

- **cell state `c`** — a long-term memory "conveyor belt" that information can ride
  along almost unchanged for many steps.
- **hidden state `h`** — the short-term, output-facing state.

Three **gates** (each a small neural layer squashed to 0–1) decide how memory is
updated each day:

| Gate | Question it answers | Hydrology analogy |
|---|---|---|
| **forget** | what to erase from `c`? | snow melting away; soil drying out |
| **input** | what new info to write into `c`? | a storm recharging soil / building snowpack |
| **output** | what part of `c` to expose as `h` → discharge? | how much stored water is released to the stream right now |

So the **cell state behaves like catchment storage**, and the **gates behave like
the physical processes that charge and discharge that storage**. The network is
never told these rules — it *learns* them from data. That correspondence is why
LSTMs are so well-suited to rainfall-runoff modelling, and why researchers have
even read hydrological meaning out of trained LSTM cells.

## How we frame the prediction (many-to-one)

We feed the LSTM a **window** of the last `seq_len` days of forcings
(`precip`, `temp`) and ask it to predict discharge on the **last** day of the
window:

```
[precip,temp] day t-89 ─┐
[precip,temp] day t-88  │   ┌──────┐
        ...             ├──▶│ LSTM │──▶ discharge on day t
[precip,temp] day t-1   │   └──────┘
[precip,temp] day t   ──┘
```

The window length is a real modelling choice: too short and the model cannot
"see" the snowpack that has been building all winter; too long and the
optimization gets harder and slower. (Try `--seq-len 10` vs the default 90.)

The whole cell is ~30 lines in
[`src/lstm_hydrology/model.py`](../src/lstm_hydrology/model.py) — read it next.
