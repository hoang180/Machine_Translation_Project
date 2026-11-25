"""TransformerModel hoàn chỉnh (from scratch, batch_first=True)."""

import math

import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding
from .encoder_layer import TransformerEncoder
from .decoder_layer import TransformerDecoder

PAD_IDX = 0


class TransformerModel(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_idx: int = PAD_IDX,
        use_pe: bool = True,
        tie_weights: bool = False,
        num_layers: int = None,
    ):
        super().__init__()
        if num_layers is not None:  # tương thích eval.py: dùng chung số layer cho encoder/decoder
            num_encoder_layers = num_decoder_layers = num_layers
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.num_heads = num_heads

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.src_pos = PositionalEncoding(d_model, dropout, max_len, use_pe)
        self.tgt_pos = PositionalEncoding(d_model, dropout, max_len, use_pe)

        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()
        if tie_weights:
            # Chia sẻ trọng số tgt_embedding <-> generator (đặt sau khi init)
            self.generator.weight = self.tgt_embedding.weight

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Giữ vector <pad> bằng 0
        with torch.no_grad():
            self.src_embedding.weight[self.pad_idx].zero_()
            self.tgt_embedding.weight[self.pad_idx].zero_()

    # ------------------------------------------------------------------ masks
    @staticmethod
    def make_src_mask(src_mask: torch.Tensor) -> torch.Tensor:
        """(B, src_len) bool -> (B, 1, 1, src_len); True = <pad> cần che."""
        return src_mask.unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_causal_mask(size: int, device) -> torch.Tensor:
        """(1, 1, size, size) bool; True = token tương lai cần che."""
        mask = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    # ------------------------------------------------------------- components
    def _embed_src(self, src: torch.Tensor) -> torch.Tensor:
        return self.src_pos(self.src_embedding(src) * math.sqrt(self.d_model))

    def _embed_tgt(self, tgt: torch.Tensor) -> torch.Tensor:
        return self.tgt_pos(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

    def _default_src_mask(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        if src_mask is None:
            src_mask = src == self.pad_idx
        return src_mask

    # ---------------------------------------------------------------- forward
    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor,
                src_mask: torch.Tensor = None, return_attention: bool = False):
        """
        src: (B, src_len); tgt_input: (B, tgt_len - 1); src_mask: (B, src_len) bool, True = <pad>.
        Trả về logits (B, tgt_len - 1, tgt_vocab_size), hoặc (logits, attn) với
        attn: (B, H, tgt_len - 1, src_len) khi return_attention=True.
        """
        src_mask = self._default_src_mask(src, src_mask)
        mem_mask = self.make_src_mask(src_mask)
        memory = self.encoder(self._embed_src(src), mem_mask)

        tgt_mask = self.make_causal_mask(tgt_input.size(1), tgt_input.device)
        out, attn = self.decoder(self._embed_tgt(tgt_input), memory, tgt_mask, mem_mask)
        logits = self.generator(out)
        if return_attention:
            return logits, attn
        return logits

    # ------------------------------------------------- encode / decode_step
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> dict:
        """Chạy Encoder 1 lần; trả về encoder_state dict."""
        src_mask = self._default_src_mask(src, src_mask)
        mem_mask = self.make_src_mask(src_mask)
        memory = self.encoder(self._embed_src(src), mem_mask)
        return {"memory": memory, "src_mask": src_mask}

    def decode_step(self, tgt_seq: torch.Tensor, encoder_state: dict,
                    return_attention: bool = False):
        """
        tgt_seq: (B_beam, cur_tgt_len). encoder_state: dict từ encode() với batch size B_beam
        (beam_search tự expand/reorder memory và src_mask).
        Trả về (next_token_logits (B_beam, V), encoder_state, step_attn (B_beam, H, src_len) | None).
        """
        memory = encoder_state["memory"]
        mem_mask = self.make_src_mask(encoder_state["src_mask"])
        tgt_mask = self.make_causal_mask(tgt_seq.size(1), tgt_seq.device)
        out, attn = self.decoder(self._embed_tgt(tgt_seq), memory, tgt_mask, mem_mask)
        next_logits = self.generator(out[:, -1])
        step_attn = attn[:, :, -1, :] if return_attention else None
        return next_logits, encoder_state, step_attn
