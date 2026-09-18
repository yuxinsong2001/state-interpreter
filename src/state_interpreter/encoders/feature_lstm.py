"""PyTorch port of the compact XJTU-SY Feature LSTM baseline.

The architecture follows the MIT-licensed ``thfmn/xjtu-sy-bearing`` Feature
LSTM at commit ``7d7231c582961741bde629da6731e6c169d88785``. The module only
implements the network contract; it does not reproduce the upstream labels or
evaluation protocol. See THIRD_PARTY_NOTICES.md.
"""

from typing import NamedTuple

import torch
from torch import nn


class FeatureLSTMOutput(NamedTuple):
    """Scalar faithful-baseline score and temporal hidden representation."""

    score: torch.Tensor
    hidden: torch.Tensor


class FeatureLSTM(nn.Module):
    """BiLSTM(16) -> Dropout -> Dense(16) -> scalar baseline head."""

    def __init__(
        self,
        n_features: int = 65,
        hidden_size: int = 16,
        dense_size: int = 16,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if n_features <= 0 or hidden_size <= 0 or dense_size <= 0:
            raise ValueError("feature and hidden dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        representation_size = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(representation_size, dense_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(dense_size, 1)

    def forward(self, windows: torch.Tensor) -> FeatureLSTMOutput:
        if windows.ndim != 3 or windows.shape[-1] != self.n_features:
            raise ValueError(
                f"expected [batch, steps, {self.n_features}], "
                f"got {tuple(windows.shape)}"
            )
        if windows.shape[1] < 1:
            raise ValueError("feature windows must contain at least one step")
        _, (hidden_state, _) = self.lstm(windows)
        if self.bidirectional:
            temporal = torch.cat((hidden_state[-2], hidden_state[-1]), dim=-1)
        else:
            temporal = hidden_state[-1]
        hidden = self.activation(self.dense(self.dropout(temporal)))
        score = self.output(hidden)
        return FeatureLSTMOutput(score=score, hidden=hidden)
