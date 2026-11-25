"""Sinusoidal Positional Encoding (có cờ bật/tắt cho Ablation Study)."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    use_pe=False: bỏ qua PE (chỉ áp dụng dropout) để chạy Ablation.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512,
                 use_pe: bool = True):
        super().__init__()
        self.use_pe = use_pe
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)."""
        if self.use_pe:
            x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
