# From LLM fine-tuning to training an LSTM

You have fine-tuned LLMs with OpenAI. A lot of that intuition transfers — but
some key things are different, and the differences are exactly where the
learning is. This page maps one world onto the other.

## What carries over

Both are **sequence models trained by gradient descent**. An LLM reads a sequence
of *tokens*; an LSTM here reads a sequence of *days*. Both minimize a loss by
backpropagation, both care about train/validation/test splits, overfitting,
learning rate, and batch size.

## The mapping

| LLM fine-tuning (OpenAI) | LSTM runoff model (this repo) |
|---|---|
| Token sequence | Daily time series (`precip`, `temp`) |
| Predict the **next token** (classification over a vocab) | Predict **discharge** (regression, one number in mm/day) |
| Start from a **pretrained** model, adapt it | Train a small model **from scratch** |
| Billions of parameters | ~tens of thousands ([`model.py`](../src/lstm_hydrology/model.py)) |
| Loss = cross-entropy (hidden from you) | Loss = MSE, **you write it** ([`train.py`](../src/lstm_hydrology/train.py)) |
| Upload JSONL, start a job, poll | You own data prep, the training loop, evaluation |
| "Does the text read well?" | NSE / KGE on a held-out hydrograph ([metrics](03_hydrology_metrics.md)) |
| Runs on OpenAI's servers | Runs on your laptop (CPU or Apple-Silicon MPS) |

## The biggest mental shift: you own the loop

With the API you never saw the forward pass, the loss, the optimizer step. Here
they are explicit and worth reading once, slowly:

```python
for xb, yb in train_loader:        # a batch of windows + their target discharge
    opt.zero_grad()                # reset gradients
    loss = loss_fn(model(xb), yb)  # forward pass + MSE loss
    loss.backward()                # backprop: gradients of loss w.r.t. weights
    opt.step()                     # nudge weights downhill
```

That is the entire engine of deep learning. Everything else — schedulers,
dropout, checkpoints — is refinement around these four lines.

## "Fine-tuning" *does* exist in hydrology

The transfer-learning instinct you built with LLMs is exactly right for the
research frontier:

- **Pretrain on many basins → fine-tune on one.** Train a single LSTM on hundreds
  of catchments so it learns general hydrology, then adapt it to a specific basin
  with little local data.
- **Prediction in Ungauged Basins (PUB).** Apply the pretrained regional model to
  a catchment with *no* streamflow record — the headline result of the LSTM
  hydrology literature, and the analogue of zero-shot transfer.

That frontier — pretrain on many basins, then adapt to one — is where this field
is headed. First, build the from-scratch muscle here; it is the part the API
never showed you.
