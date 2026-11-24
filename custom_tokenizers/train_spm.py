"""
Module: custom_tokenizers/train_spm.py
Phụ trách: Phần 1 (Lê)
Cài đặt: Huấn luyện và bọc bộ tách từ SentencePiece (Google) theo mô hình Unigram.
Đảm bảo ánh xạ chuẩn 4 special tokens: <pad>=0, <unk>=1, <sos>=2, <eos>=3.
"""

import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from base_tokenizer import BaseTokenizerWrapper
from utils.common import (
    PAD_IDX,
    UNK_IDX,
    SOS_IDX,
    EOS_IDX,
    PAD_TOKEN,
    UNK_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
)


class SPMTokenizer(BaseTokenizerWrapper):
    """
    Bộ tách từ SentencePiece (Unigram):
    - Độc lập với ngôn ngữ (không cần tiền xử lý tách khoảng trắng).
    - Cực kỳ mạnh cho tiếng Việt với các thanh điệu và âm tiết.
    """

    def __init__(self, model_file: str = None):
        super().__init__()
        self.sp = None
        if model_file and os.path.exists(model_file):
            self.load_model(model_file)

    def load_model(self, model_file: str):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if not text or self.sp is None:
            return [self.sos_idx, self.eos_idx] if add_special_tokens else []
        ids = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            ids = [self.sos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if self.sp is None:
            return ""
        if skip_special_tokens:
            ids = [i for i in ids if i not in (self.pad_idx, self.sos_idx, self.eos_idx)]
        return self.sp.decode(ids)

    def id_to_piece(self, idx: int) -> str:
        return self.sp.id_to_piece(idx) if self.sp else UNK_TOKEN

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size() if self.sp else 0


def train_spm_model(
    train_file: str,
    model_prefix: str,
    vocab_size: int = 16000,
):
    """Huấn luyện SentencePiece Unigram model với đúng 4 ID đặc biệt."""
    import sentencepiece as spm

    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    print(f"[INFO] Huấn luyện SentencePiece trên: {train_file} (vocab={vocab_size:,})...")

    # Bắt buộc gán đúng index:
    # pad_id=0, unk_id=1, bos_id=2 (<sos>), eos_id=3 (<eos>)
    spm.SentencePieceTrainer.train(
        input=train_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        pad_id=PAD_IDX,
        unk_id=UNK_IDX,
        bos_id=SOS_IDX,
        eos_id=EOS_IDX,
        pad_piece=PAD_TOKEN,
        unk_piece=UNK_TOKEN,
        bos_piece=SOS_TOKEN,
        eos_piece=EOS_TOKEN,
    )
    print(f"  ✓ Đã lưu mô hình SPM vào: {model_prefix}.model")


def train_spm_tokenizers(
    train_src="data/processed/train.en",
    train_tgt="data/processed/train.vi",
    save_dir="custom_tokenizers/spm",
    vocab_size=16000,
):
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 60)
    print("HUẤN LUYỆN SENTENCEPIECE TOKENIZER (ANH & VIỆT)")
    print("=" * 60)

    train_spm_model(
        train_file=train_src,
        model_prefix=os.path.join(save_dir, "spm.en"),
        vocab_size=vocab_size,
    )

    train_spm_model(
        train_file=train_tgt,
        model_prefix=os.path.join(save_dir, "spm.vi"),
        vocab_size=vocab_size,
    )

    print("=" * 60)
    print("HOÀN TẤT HUẤN LUYỆN SENTENCEPIECE TOKENIZER!")
    print("=" * 60)


if __name__ == "__main__":
    train_spm_tokenizers()
