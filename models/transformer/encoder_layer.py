"""Encoder của Transformer (Post-LN, theo bài báo gốc "Attention Is All You Need")."""

import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """Self-Attention -> Add&Norm -> FFN -> Add&Norm."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, src_len, d_model)
        src_mask: BoolTensor broadcast được tới (B, 1, 1, src_len), True = vị trí <pad> cần che.
        """
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class TransformerEncoder(nn.Module):
    """Xếp chồng N TransformerEncoderLayer."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
