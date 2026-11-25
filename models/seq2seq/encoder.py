"""
Module: models/seq2seq/encoder.py
Phụ trách: Hoàng (Phần 2)
Cài đặt: Bi-directional LSTM Encoder (PyTorch thuần) cho bài toán Dịch máy Anh - Việt.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoder(nn.Module):
    """
    Bi-directional LSTM Encoder:
    - Nhận vào chuỗi token câu nguồn tiếng Anh (IDs).
    - Biến đổi qua tầng nhúng từ (Embedding Layer) và Dropout.
    - Xử lý qua mạng LSTM 2 chiều (Bidirectional LSTM) để trích xuất ngữ cảnh 
      từ trái sang phải (forward) và từ phải sang trái (backward).
    - Cung cấp tầng Linear (Bridge) để chuyển đổi trạng thái ẩn 2 chiều của Encoder 
      thành trạng thái khởi tạo phù hợp cho Unidirectional LSTM Decoder.
    """

    def __init__(
        self,
        src_vocab_size: int,
        embed_size: int = 256,
        enc_hidden_size: int = 512,
        dec_hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        """
        Args:
            src_vocab_size: Kích thước từ vựng câu nguồn tiếng Anh.
            embed_size: Chiều không gian nhúng từ (embedding dimension).
            enc_hidden_size: Kích thước vector trạng thái ẩn của mỗi chiều LSTM.
            dec_hidden_size: Kích thước vector trạng thái ẩn của Decoder (để chiếu sang).
            num_layers: Số tầng LSTM (mặc định 2 layers theo yêu cầu đề bài).
            dropout: Tỷ lệ Dropout chống quá khớp (overfitting).
            pad_idx: Index của token <pad> (mặc định 0 theo Contract).
        """
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.embed_size = embed_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        # Tầng nhúng từ vựng câu nguồn (bỏ qua cập nhật gradient tại vị trí <pad>)
        self.embedding = nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=embed_size,
            padding_idx=pad_idx,
        )

        self.dropout = nn.Dropout(dropout)

        # Mạng LSTM 2 chiều: forward và backward
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=enc_hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Tầng Linear chiếu từ 2 chiều Encoder (2 * enc_hidden_size) sang Decoder (dec_hidden_size)
        # cho cả hidden state (h) và cell state (c) của từng layer
        self.bridge_hidden = nn.Linear(enc_hidden_size * 2, dec_hidden_size)
        self.bridge_cell = nn.Linear(enc_hidden_size * 2, dec_hidden_size)

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass của Encoder:
        Args:
            src: Tensor câu nguồn tiếng Anh, shape: (batch_size, src_len)
            src_lengths: Tensor độ dài thực tế của từng câu (chưa tính pad), shape: (batch_size,)
                         Dùng để đóng gói pack_padded_sequence nhằm tăng tốc tính toán.

        Returns:
            encoder_outputs: Tensor chứa chuỗi trạng thái ẩn tại mọi bước thời gian,
                             shape: (batch_size, src_len, enc_hidden_size * 2)
            (init_dec_hidden, init_dec_cell): Trạng thái ẩn khởi tạo cho Decoder,
                             mỗi tensor có shape: (num_layers, batch_size, dec_hidden_size)
        """
        batch_size, src_len = src.size()

        # 1. Embedding + Dropout: (B, src_len) -> (B, src_len, embed_size)
        embedded = self.dropout(self.embedding(src))

        # 2. Xử lý qua BiLSTM (có hỗ trợ pack_padded_sequence nếu có src_lengths)
        if src_lengths is not None:
            # Chuyển độ dài về CPU (bắt buộc với pack_padded_sequence trong PyTorch)
            lengths_cpu = src_lengths.cpu()
            # Đóng gói chuỗi
            packed_embedded = pack_padded_sequence(
                embedded,
                lengths=lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
            # Mở gói chuỗi trở lại tensor đầy đủ: (B, src_len, 2 * enc_hidden_size)
            encoder_outputs, _ = pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                total_length=src_len,
            )
        else:
            # Chạy trực tiếp nếu không truyền độ dài
            encoder_outputs, (hidden, cell) = self.lstm(embedded)

        # hidden có shape: (num_layers * 2, batch_size, enc_hidden_size)
        # Trong đó: hidden[0:2*num_layers:2] là forward, hidden[1:2*num_layers:2] là backward

        # 3. Chuyển đổi trạng thái ẩn để khởi tạo Decoder:
        # Với mỗi layer l trong [0..num_layers-1]:
        # Nối forward state và backward state của layer đó:
        # forward_l = hidden[2 * l, :, :]      shape: (B, enc_hidden_size)
        # backward_l = hidden[2 * l + 1, :, :]  shape: (B, enc_hidden_size)
        # cat_l = [forward_l; backward_l]      shape: (B, enc_hidden_size * 2)
        # -> chiếu qua bridge_hidden + tanh -> (B, dec_hidden_size)
        init_h_list = []
        init_c_list = []

        for l in range(self.num_layers):
            h_fwd = hidden[2 * l]
            h_bwd = hidden[2 * l + 1]
            h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # (B, enc_hidden_size * 2)
            init_h = torch.tanh(self.bridge_hidden(h_cat))  # (B, dec_hidden_size)
            init_h_list.append(init_h)

            c_fwd = cell[2 * l]
            c_bwd = cell[2 * l + 1]
            c_cat = torch.cat([c_fwd, c_bwd], dim=1)  # (B, enc_hidden_size * 2)
            init_c = torch.tanh(self.bridge_cell(c_cat))  # (B, dec_hidden_size)
            init_c_list.append(init_c)

        # Xếp chồng các layer: (num_layers, batch_size, dec_hidden_size)
        init_dec_hidden = torch.stack(init_h_list, dim=0)
        init_dec_cell = torch.stack(init_c_list, dim=0)

        return encoder_outputs, (init_dec_hidden, init_dec_cell)

