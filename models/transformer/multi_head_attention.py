
import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: torch.Tensor = None, dropout: nn.Module = None):
    """
    q: (B, H, Lq, d_k); k, v: (B, H, Lk, d_k)
    mask: BoolTensor broadcast được tới (B, H, Lq, Lk); True = vị trí cần che (gán -inf).
    Trả về: (output (B, H, Lq, d_k), attn_weights (B, H, Lq, Lk)).
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    # Hàng bị che hoàn toàn (toàn -inf) cho NaN -> đưa về 0
    attn = torch.nan_to_num(attn, nan=0.0)
    attn_out = dropout(attn) if dropout is not None else attn
    return torch.matmul(attn_out, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) -> (B, H, L, d_k)."""
        B, L, _ = x.size()
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None):
        """
        query: (B, Lq, d_model); key, value: (B, Lk, d_model)
        mask: BoolTensor broadcast được tới (B, 1, Lq, Lk) hoặc (B, H, Lq, Lk); True = che.
        Trả về: (output (B, Lq, d_model), attn_weights (B, H, Lq, Lk)).
        """
        B, Lq, _ = query.size()
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        out, attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        # (B, H, Lq, d_k) -> (B, Lq, d_model)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.w_o(out), attn
