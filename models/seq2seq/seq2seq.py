"""
Module: models/seq2seq/seq2seq.py
Phụ trách: Hoàng (Phần 2)
Cài đặt: Lớp Seq2seqModel hoàn chỉnh, tuân thủ 100% Interface Contract (Mục 6 của file Quy ước 01).
Cung cấp đủ 3 phương thức bắt buộc: forward(), encode(), decode_step().
"""

import random
import torch
import torch.nn as nn
from models.seq2seq.encoder import BiLSTMEncoder
from models.seq2seq.decoder import LSTMDecoder


class Seq2seqModel(nn.Module):
    """
    Kiến trúc Seq2seq hoàn chỉnh kết hợp Attention (Bahdanau hoặc Luong):
    - Khởi tạo bao gồm BiLSTMEncoder và LSTMDecoder.
    - Phương thức forward(): phục vụ huấn luyện với Teacher Forcing linh hoạt.
    - Cặp phương thức encode() và decode_step(): phục vụ thuật toán Greedy & Beam Search dùng chung.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_size: int = 256,
        enc_hidden_size: int = 512,
        dec_hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
        attention_type: str = "luong",
        luong_method: str = "general",
    ):
        """
        Args:
            src_vocab_size: Kích thước từ vựng câu nguồn tiếng Anh.
            tgt_vocab_size: Kích thước từ vựng câu đích tiếng Việt.
            embed_size: Kích thước chiều embedding (mặc định 256).
            enc_hidden_size: Kích thước trạng thái ẩn của BiLSTM (mặc định 512).
            dec_hidden_size: Kích thước trạng thái ẩn của Decoder (mặc định 512).
            num_layers: Số tầng LSTM (mặc định 2 layers).
            dropout: Tỷ lệ Dropout chống overfitting (mặc định 0.3).
            pad_idx: Index của token <pad> (mặc định 0 theo Contract).
            attention_type: 'luong' hoặc 'bahdanau'.
            luong_method: 'general', 'dot', hoặc 'concat' (nếu dùng Luong).
        """
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.pad_idx = pad_idx

        # Khởi tạo Encoder
        self.encoder = BiLSTMEncoder(
            src_vocab_size=src_vocab_size,
            embed_size=embed_size,
            enc_hidden_size=enc_hidden_size,
            dec_hidden_size=dec_hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
        )

        # Đầu ra của BiLSTM có chiều: enc_hidden_size * 2
        enc_output_size = enc_hidden_size * 2

        # Khởi tạo Decoder
        self.decoder = LSTMDecoder(
            tgt_vocab_size=tgt_vocab_size,
            embed_size=embed_size,
            dec_hidden_size=dec_hidden_size,
            enc_output_size=enc_output_size,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
            attention_type=attention_type,
            luong_method=luong_method,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt_input: torch.Tensor,
        src_mask: torch.Tensor = None,
        src_lengths: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass dùng trong vòng lặp huấn luyện:
        Args:
            src: Tensor câu nguồn tiếng Anh, shape: (batch_size, src_len)
            tgt_input: Tensor câu đích đã cắt bỏ token cuối, shape: (batch_size, tgt_len - 1)
            src_mask: Tensor padding mask, shape: (batch_size, src_len), True tại vị trí <pad>
            src_lengths: Tensor độ dài thực tế của câu nguồn, shape: (batch_size,)
            teacher_forcing_ratio: Xác suất sử dụng ground-truth làm đầu vào bước kế tiếp
            return_attention: Nếu True, trả về kèm ma trận Attention Weights (B, 1, tgt_len - 1, src_len)

        Returns:
            Khi return_attention=False:
                logits: Tensor shape (batch_size, tgt_len - 1, tgt_vocab_size)
            Khi return_attention=True:
                (logits, attn_weights) trong đó:
                attn_weights shape: (batch_size, 1, tgt_len - 1, src_len)
        """
        batch_size, tgt_len_minus_1 = tgt_input.size()

        # 1. Chạy Encoder toàn bộ câu nguồn
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths=src_lengths)

        # Tối ưu hóa GPU (Tăng tốc 3x): Chiếu trước encoder_outputs 1 lần duy nhất ngoài vòng lặp
        projected_enc = None
        if hasattr(self.decoder.attention, "linear_in"):
            projected_enc = self.decoder.attention.linear_in(encoder_outputs)

        # Tensor chứa toàn bộ logits đầu ra: (B, tgt_len - 1, tgt_vocab_size)
        all_logits = torch.zeros(
            batch_size,
            tgt_len_minus_1,
            self.tgt_vocab_size,
            device=src.device,
            dtype=torch.float,
        )

        # List chứa attention weights của từng bước thời gian nếu cần xuất
        all_attentions = [] if return_attention else None

        # Token đầu vào đầu tiên: thường là token <sos> tại vị trí index 0 của tgt_input
        input_token = tgt_input[:, 0]

        # 2. Vòng lặp giải mã qua từng bước thời gian t = 0..tgt_len-2
        for t in range(tgt_len_minus_1):
            # Tính toán 1 bước qua Decoder (dùng projected_enc đã tính sẵn)
            step_logits, hidden, cell, step_attn = self.decoder.forward_step(
                input_token=input_token,
                hidden=hidden,
                cell=cell,
                encoder_outputs=encoder_outputs,
                src_mask=src_mask,
                projected_enc=projected_enc,
            )

            # Lưu logits tại bước t
            all_logits[:, t, :] = step_logits

            # Lưu attention nếu có yêu cầu
            if return_attention:
                all_attentions.append(step_attn)  # mỗi phần tử là (B, src_len)

            # Cơ chế Teacher Forcing:
            # Quyết định lấy nhãn thật hay lấy từ vừa dự đoán để đưa vào bước t+1
            is_teacher_force = random.random() < teacher_forcing_ratio
            top1 = step_logits.argmax(dim=-1)  # (B,)

            # Nếu chưa phải bước cuối cùng:
            if t + 1 < tgt_len_minus_1:
                input_token = tgt_input[:, t + 1] if is_teacher_force else top1

        # 3. Chuẩn hóa đầu ra theo đúng chuẩn Contract Mục 6.1
        if return_attention:
            # Xếp chồng các bước thành ma trận: (B, tgt_len - 1, src_len)
            attn_weights = torch.stack(all_attentions, dim=1)
            # Unsqueeze(1) để có 1 attention head: (B, 1, tgt_len - 1, src_len)
            attn_weights = attn_weights.unsqueeze(1)
            return all_logits, attn_weights

        return all_logits

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> dict:
        """
        Chuẩn Contract Mục 6.2: Chạy Encoder 1 lần duy nhất cho câu nguồn.
        Phục vụ cho beam_search.py giải mã không bị lặp lại tính toán.

        Args:
            src: Tensor shape (B, src_len)
            src_mask: Tensor shape (B, src_len), True tại vị trí <pad>

        Returns:
            encoder_state: dict chứa thông tin trạng thái để truyền vào decode_step()
        """
        encoder_outputs, (init_hidden, init_cell) = self.encoder(src)
        projected_enc = None
        if hasattr(self.decoder.attention, "linear_in"):
            projected_enc = self.decoder.attention.linear_in(encoder_outputs)
        return {
            "encoder_outputs": encoder_outputs,
            "projected_enc": projected_enc,
            "hidden": init_hidden,
            "cell": init_cell,
            "src_mask": src_mask,
        }

    def decode_step(
        self,
        tgt_seq: torch.Tensor,
        encoder_state: dict,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, dict, torch.Tensor | None]:
        # Lấy token cuối cùng của chuỗi hiện tại làm đầu vào Decoder
        last_token = tgt_seq[:, -1]

        encoder_outputs = encoder_state["encoder_outputs"]
        projected_enc = encoder_state.get("projected_enc", None)
        hidden = encoder_state["hidden"]
        cell = encoder_state["cell"]
        src_mask = encoder_state.get("src_mask", None)

        # Tính toán bước kế tiếp
        step_logits, new_hidden, new_cell, step_attn = self.decoder.forward_step(
            input_token=last_token,
            hidden=hidden,
            cell=cell,
            encoder_outputs=encoder_outputs,
            src_mask=src_mask,
            projected_enc=projected_enc,
        )

        # Cập nhật trạng thái mới cho bước tiếp theo
        updated_state = {
            "encoder_outputs": encoder_outputs,
            "projected_enc": projected_enc,
            "hidden": new_hidden,
            "cell": new_cell,
            "src_mask": src_mask,
        }

        # Định dạng shape attention theo chuẩn Contract (B_beam, 1, src_len)
        formatted_attn = step_attn.unsqueeze(1) if (return_attention and step_attn is not None) else None

        return step_logits, updated_state, formatted_attn

