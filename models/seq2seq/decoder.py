"""
Module: models/seq2seq/decoder.py
Phụ trách: Hoàng (Phần 2)
Cài đặt: LSTM Decoder kết hợp cơ chế Attention (PyTorch thuần).
Cung cấp forward_step để phục vụ giải mã từng bước trong Beam Search & Greedy Search.
"""

import torch
import torch.nn as nn
from models.seq2seq.attention import LuongAttention, BahdanauAttention


class LSTMDecoder(nn.Module):
    """
    Unidirectional LSTM Decoder kết hợp Attention:
    - Nhận token đích tại bước t-1 (y_{t-1}) và trạng thái ẩn trước đó (hidden_{t-1}, cell_{t-1}).
    - Tính toán qua các tầng LSTM để cập nhật trạng thái ẩn hiện tại s_t.
    - Gọi tầng Attention để tính toán trọng số chú ý alpha_t và vector ngữ cảnh c_t từ encoder_outputs.
    - Kết hợp trạng thái ẩn s_t và vector ngữ cảnh c_t qua tầng chiếu ngữ cảnh:
        s_tilde_t = tanh(W_c * [s_t; c_t])
    - Dự đoán phân phối xác suất từ vựng kế tiếp qua tầng Generator:
        logits_t = W_v * s_tilde_t
    """

    def __init__(
        self,
        tgt_vocab_size: int,
        embed_size: int = 256,
        dec_hidden_size: int = 512,
        enc_output_size: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        attention_type: str = "luong",
        luong_method: str = "general",
    ):
        """
        Args:
            tgt_vocab_size: Kích thước từ vựng câu đích tiếng Việt.
            embed_size: Chiều không gian nhúng từ đích.
            dec_hidden_size: Kích thước trạng thái ẩn của Decoder.
            enc_output_size: Kích thước đầu ra của BiLSTM Encoder (2 * enc_hidden_size).
            num_layers: Số tầng LSTM (mặc định 2).
            dropout: Tỷ lệ Dropout.
            pad_idx: Index của token <pad> (mặc định 0).
            attention_type: 'luong' hoặc 'bahdanau'.
            luong_method: 'general', 'dot', hoặc 'concat' (nếu dùng Luong).
        """
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_size = embed_size
        self.dec_hidden_size = dec_hidden_size
        self.enc_output_size = enc_output_size
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        # Tầng nhúng từ đích
        self.embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size,
            embedding_dim=embed_size,
            padding_idx=pad_idx,
        )

        self.dropout = nn.Dropout(dropout)

        # Mạng LSTM 1 chiều cho Decoder
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=dec_hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Khởi tạo mô đun Attention tương ứng
        if attention_type.lower() == "luong":
            self.attention = LuongAttention(
                dec_hidden_size=dec_hidden_size,
                enc_output_size=enc_output_size,
                method=luong_method,
            )
        elif attention_type.lower() == "bahdanau":
            self.attention = BahdanauAttention(
                dec_hidden_size=dec_hidden_size,
                enc_output_size=enc_output_size,
            )
        else:
            raise ValueError(f"Không hỗ trợ attention_type: {attention_type}. Chọn 'luong' hoặc 'bahdanau'.")

        # Tầng kết hợp trạng thái ẩn của Decoder và Vector ngữ cảnh (Attentional Hidden State)
        # s_tilde = tanh(W_c * [s_t; c_t])
        self.fc_concat = nn.Linear(dec_hidden_size + enc_output_size, dec_hidden_size)

        # Tầng phân loại dự đoán phân phối từ vựng: (dec_hidden_size -> tgt_vocab_size)
        self.classifier = nn.Linear(dec_hidden_size, tgt_vocab_size)

    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor = None,
        projected_enc: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Thực hiện một bước tính toán giải mã tại bước thời gian t:
        Args:
            input_token: Tensor token ID của bước trước đó, shape: (batch_size,) hoặc (batch_size, 1)
            hidden: Trạng thái ẩn của Decoder, shape: (num_layers, batch_size, dec_hidden_size)
            cell: Trạng thái ô nhớ của Decoder, shape: (num_layers, batch_size, dec_hidden_size)
            encoder_outputs: Toàn bộ đầu ra Encoder, shape: (batch_size, src_len, enc_output_size)
            src_mask: Mặt nạ che padding câu nguồn, shape: (batch_size, src_len), True tại <pad>

        Returns:
            step_logits: Logits phân phối từ vựng tại bước t, shape: (batch_size, tgt_vocab_size)
            hidden: Trạng thái ẩn mới, shape: (num_layers, batch_size, dec_hidden_size)
            cell: Trạng thái ô nhớ mới, shape: (num_layers, batch_size, dec_hidden_size)
            step_attn: Trọng số chú ý tại bước t, shape: (batch_size, src_len)
        """
        # Đảm bảo input_token có shape (batch_size, 1)
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)

        # 1. Embedding + Dropout: (B, 1) -> (B, 1, embed_size)
        embedded = self.dropout(self.embedding(input_token))

        # 2. Bước LSTM: cập nhật hidden và cell
        # rnn_out shape: (B, 1, dec_hidden_size)
        rnn_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # Trạng thái ẩn tầng trên cùng tại bước hiện tại: (B, dec_hidden_size)
        cur_hidden = hidden[-1]

        # 3. Tính Attention: lấy context vector và ma trận trọng số chú ý
        context, step_attn = self.attention(
            dec_hidden=cur_hidden,
            encoder_outputs=encoder_outputs,
            src_mask=src_mask,
            projected_enc=projected_enc,
        )

        # 4. Ghép nối [cur_hidden; context]: shape (B, dec_hidden_size + enc_output_size)
        cat_features = torch.cat([cur_hidden, context], dim=1)

        # Ánh xạ qua tầng Linear kết hợp tanh: (B, dec_hidden_size)
        attentional_hidden = torch.tanh(self.fc_concat(cat_features))
        attentional_hidden = self.dropout(attentional_hidden)

        # 5. Chiếu ra không gian từ điển đích: (B, tgt_vocab_size)
        step_logits = self.classifier(attentional_hidden)

        return step_logits, hidden, cell, step_attn

