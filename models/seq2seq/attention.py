"""
Module: models/seq2seq/attention.py
Phụ trách: Hoàng (Phần 2)
Cài đặt: Cơ chế Attention kinh điển (Luong Attention & Bahdanau Attention) bằng PyTorch thuần.
Hỗ trợ che mặt nạ src_padding_mask (gán -1e9 trước Softmax) để bỏ qua token <pad>.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    """
    Luong Attention (Multiplicative Attention - Luong et al., 2015):
    - Tính điểm liên kết (alignment score) giữa trạng thái ẩn của Decoder hiện tại (s_t) 
      và toàn bộ các trạng thái ẩn của Encoder (h_1, ..., h_S).
    - Hỗ trợ 3 hàm tính điểm phổ biến:
      1. 'dot':     score(s_t, h_i) = s_t^T * h_i (yêu cầu dim(s_t) == dim(h_i))
      2. 'general': score(s_t, h_i) = s_t^T * W_a * h_i (khuyên dùng, linh hoạt kích thước)
      3. 'concat':  score(s_t, h_i) = v_a^T * tanh(W_a * [s_t; h_i])
    """

    def __init__(
        self,
        dec_hidden_size: int,
        enc_output_size: int,
        method: str = "general",
    ):
        """
        Args:
            dec_hidden_size: Kích thước trạng thái ẩn của Decoder (dim s_t).
            enc_output_size: Kích thước đầu ra của BiLSTM Encoder (2 * enc_hidden_size).
            method: 'general', 'dot', hoặc 'concat'. Mặc định 'general'.
        """
        super().__init__()
        self.method = method.lower()
        self.dec_hidden_size = dec_hidden_size
        self.enc_output_size = enc_output_size

        if self.method == "general":
            # W_a: (enc_output_size -> dec_hidden_size)
            # score = s_t^T * (W_a * h_i)
            self.linear_in = nn.Linear(enc_output_size, dec_hidden_size, bias=False)
        elif self.method == "concat":
            # W_a: (dec_hidden_size + enc_output_size -> dec_hidden_size)
            self.linear_in = nn.Linear(dec_hidden_size + enc_output_size, dec_hidden_size, bias=False)
            self.v = nn.Linear(dec_hidden_size, 1, bias=False)
        elif self.method == "dot":
            assert dec_hidden_size == enc_output_size, (
                f"Luong 'dot' method yêu cầu dec_hidden_size ({dec_hidden_size}) "
                f"phải bằng enc_output_size ({enc_output_size})."
            )
        else:
            raise ValueError(f"Phương thức Luong Attention không hợp lệ: {method}. Chọn 'general', 'dot' hoặc 'concat'.")

    def forward(
        self,
        dec_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor = None,
        projected_enc: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            dec_hidden: (B, dec_hidden_size)
            encoder_outputs: (B, src_len, enc_output_size)
            src_mask: (B, src_len), bool, True tại <pad>
            projected_enc: (B, src_len, dec_hidden_size) đã tính sẵn bên ngoài vòng lặp (tăng tốc GPU 3x)
        """
        batch_size, src_len, _ = encoder_outputs.size()

        # 1. Tính Alignment Scores: shape -> (batch_size, src_len)
        if self.method == "dot":
            dec_hidden_expanded = dec_hidden.unsqueeze(1)
            scores = torch.bmm(dec_hidden_expanded, encoder_outputs.transpose(1, 2)).squeeze(1)

        elif self.method == "general":
            # Tối ưu GPU: Dùng projected_enc đã tính sẵn 1 lần trước vòng lặp t
            if projected_enc is None:
                projected_enc = self.linear_in(encoder_outputs)
            scores = torch.bmm(dec_hidden.unsqueeze(1), projected_enc.transpose(1, 2)).squeeze(1)

        elif self.method == "concat":
            dec_hidden_expanded = dec_hidden.unsqueeze(1).expand(-1, src_len, -1)
            cat_features = torch.cat([dec_hidden_expanded, encoder_outputs], dim=-1)
            energy = torch.tanh(self.linear_in(cat_features))
            scores = self.v(energy).squeeze(-1)

        # 2. Xử lý che Masking tại vị trí <pad>
        # Dùng -1e4 thay vì -1e9 để tương thích an toàn 100% với kiểu số Half precision (FP16)
        if src_mask is not None:
            mask_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
            scores = scores.masked_fill(src_mask, mask_val)

        # 3. Softmax trên chiều src_len
        attn_weights = F.softmax(scores, dim=-1)

        # 4. Context Vector: c_t = sum_i(alpha_i * h_i)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attn_weights


class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention (Additive Attention - Bahdanau et al., 2014):
    - Tính điểm liên kết bằng mạng Feed-Forward 1 tầng ẩn với hàm phi tuyến tanh:
      score(s_{t-1}, h_i) = v_a^T * tanh(W_s * s_{t-1} + W_h * h_i)
    """

    def __init__(
        self,
        dec_hidden_size: int,
        enc_output_size: int,
        attn_hidden_size: int = None,
    ):
        """
        Args:
            dec_hidden_size: Kích thước trạng thái ẩn của Decoder.
            enc_output_size: Kích thước đầu ra của BiLSTM Encoder (2 * enc_hidden_size).
            attn_hidden_size: Kích thước không gian chiếu chung (mặc định lấy bằng dec_hidden_size).
        """
        super().__init__()
        if attn_hidden_size is None:
            attn_hidden_size = dec_hidden_size

        self.W_s = nn.Linear(dec_hidden_size, attn_hidden_size, bias=False)
        self.W_h = nn.Linear(enc_output_size, attn_hidden_size, bias=False)
        self.v = nn.Linear(attn_hidden_size, 1, bias=False)

    def forward(
        self,
        dec_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor = None,
        projected_enc: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            dec_hidden: Trạng thái ẩn Decoder (B, dec_hidden_size)
            encoder_outputs: Đầu ra Encoder (B, src_len, enc_output_size)
            src_mask: Mask padding (B, src_len), True tại vị trí <pad>

        Returns:
            context: (B, enc_output_size)
            attn_weights: (B, src_len)
        """
        # dec_hidden: (B, dec_hidden_size) -> (B, 1, attn_hidden_size)
        proj_s = self.W_s(dec_hidden).unsqueeze(1)

        # encoder_outputs: (B, src_len, enc_output_size) -> (B, src_len, attn_hidden_size)
        proj_h = self.W_h(encoder_outputs)

        # Additive: (B, src_len, attn_hidden_size)
        energy = torch.tanh(proj_s + proj_h)

        # (B, src_len, 1) -> (B, src_len)
        scores = self.v(energy).squeeze(-1)

        # Che mask padding
        if src_mask is not None:
            mask_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
            scores = scores.masked_fill(src_mask, mask_val)

        attn_weights = F.softmax(scores, dim=-1)

        # Tính context: (B, 1, S) x (B, S, enc_output_size) -> (B, enc_output_size)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attn_weights

