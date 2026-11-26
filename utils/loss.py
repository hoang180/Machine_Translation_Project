"""
Module: utils/loss.py
Phụ trách: Sơn (Phần 3)
Cài đặt: Label Smoothing Cross-Entropy Loss bỏ qua PAD_IDX = 0 (tuân thủ Contract Mục 6.3).
Hỗ trợ smoothing = 0.0 để dùng cho thí nghiệm Ablation (tắt Label Smoothing).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.common import PAD_IDX


class LabelSmoothingLoss(nn.Module):
    """
    Cross-Entropy Loss kết hợp Label Smoothing (epsilon = smoothing).
    - Phân bổ (1 - smoothing) cho nhãn đúng.
    - Phân bổ đều `smoothing / (vocab_size - 2)` cho các token hợp lệ khác (không gán xác suất cho <pad>).
    - Bỏ qua hoàn toàn các vị trí target == padding_idx khi tính trung bình loss.
    """

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = PAD_IDX,
        smoothing: float = 0.1,
    ):
        super().__init__()
        if vocab_size <= 2:
            raise ValueError("vocab_size phải lớn hơn 2.")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing phải nằm trong khoảng [0.0, 1.0).")

        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor shape (N, vocab_size) hoặc (B, L, vocab_size)
            target: LongTensor shape (N,) hoặc (B, L)
        Returns:
            Scalar loss trung bình trên các token khác padding_idx.
        """
        if logits.dim() > 2:
            logits = logits.reshape(-1, self.vocab_size)
        if target.dim() > 1:
            target = target.reshape(-1)

        # Tính log_softmax ở float32 để ổn định số học khi bật FP16 AMP
        log_probs = F.log_softmax(logits.float(), dim=-1)

        non_pad_mask = target.ne(self.padding_idx)
        num_tokens = non_pad_mask.sum()
        if num_tokens == 0:
            return logits.sum() * 0.0

        # Thay tạm padding_idx bằng 0 để gather an toàn, sau đó loại bằng non_pad_mask
        safe_target = target.masked_fill(~non_pad_mask, 0)
        nll_loss = -log_probs.gather(dim=-1, index=safe_target.unsqueeze(-1)).squeeze(-1)

        if self.smoothing == 0.0:
            loss = nll_loss
        else:
            # Tổng -log_probs trên toàn bộ token khác padding_idx và khác nhãn đúng
            sum_non_pad = -log_probs.sum(dim=-1) + log_probs[:, self.padding_idx]
            smooth_loss = (sum_non_pad - nll_loss) / (self.vocab_size - 2)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        return loss.masked_select(non_pad_mask).sum() / num_tokens
