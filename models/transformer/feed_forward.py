"""Position-wise Feed-Forward Network của Transformer."""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = Linear(d_ff -> d_model)(Dropout(ReLU(Linear(d_model -> d_ff)(x))))."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)."""
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
