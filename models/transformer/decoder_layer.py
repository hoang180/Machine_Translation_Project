"""Decoder của Transformer (Post-LN, theo bài báo gốc)."""

import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class TransformerDecoderLayer(nn.Module):
    """Masked Self-Attn -> Add&Norm -> Cross-Attn -> Add&Norm -> FFN -> Add&Norm."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
        """
        x: (B, tgt_len, d_model); memory: (B, src_len, d_model)
        tgt_mask: BoolTensor (causal + pad), True = che; broadcast được tới (B, 1, tgt_len, tgt_len).
        memory_mask: BoolTensor (src pad), True = che; broadcast được tới (B, 1, 1, src_len).
        Trả về: (x, cross_attn_weights) với cross_attn_weights shape (B, H, tgt_len, src_len).
        """
        sa_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(sa_out))
        ca_out, cross_attn = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(ca_out))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x, cross_attn


class TransformerDecoder(nn.Module):
    """Xếp chồng N TransformerDecoderLayer; trả về attention của layer cuối."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
             for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                tgt_mask: torch.Tensor = None, memory_mask: torch.Tensor = None):
        cross_attn = None
        for layer in self.layers:
            x, cross_attn = layer(x, memory, tgt_mask, memory_mask)
        return x, cross_attn
