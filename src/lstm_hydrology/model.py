"""A minimal LSTM regressor for rainfall-runoff modelling."""
import torch.nn as nn


class RunoffLSTM(nn.Module):
    """Many-to-one LSTM: a window of daily forcings -> one discharge value.

    Mirrors the architecture of Kratzert et al. (2018): an LSTM reads the
    forcing sequence and a linear head maps the final hidden state to runoff.
    Deliberately small -- the whole point is that you can read and modify it.
    """

    def __init__(self, n_features, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # hidden state at the final timestep
        return self.head(self.dropout(last)).squeeze(-1)
