"""
Module: custom_tokenizers/train_bpe.py
Phụ trách: Phần 1 (Lê)
Cài đặt: Huấn luyện và bọc bộ tách từ BPE (Byte Pair Encoding) qua thư viện tokenizers của HuggingFace.
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
    SPECIAL_TOKENS,
)


class BPETokenizer(BaseTokenizerWrapper):
    """
    Bộ tách từ Byte Pair Encoding (BPE):
    - Dùng thư viện tokenizers của HuggingFace.
    - Loại bỏ hoàn toàn lỗi Out-Of-Vocabulary (OOV) bằng cách phân mảnh từ lạ thành các subwords.
    """

    def __init__(self, model_file: str = None):
        super().__init__()
        self.tokenizer = None
        if model_file and os.path.exists(model_file):
            self.load_model(model_file)

    def load_model(self, model_file: str):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(model_file)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if not text:
            return [self.sos_idx, self.eos_idx] if add_special_tokens else []
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids
        if add_special_tokens:
            ids = [self.sos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i not in (self.pad_idx, self.sos_idx, self.eos_idx)]
        return self.tokenizer.decode(ids)

    def id_to_piece(self, idx: int) -> str:
        return self.tokenizer.id_to_token(idx) or UNK_TOKEN

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0


def train_bpe_model(
    train_file: str,
    save_path: str,
    vocab_size: int = 16000,
):
    """Huấn luyện mô hình BPE Tokenizer trên file văn bản."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    print(f"[INFO] Huấn luyện BPE trên: {train_file} (vocab={vocab_size:,})...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Khởi tạo mô hình BPE với unknown token là <unk>
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()

    # Bắt buộc khai báo danh sách special_tokens theo đúng thứ tự 0..3: <pad>, <unk>, <sos>, <eos>
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=[],
    )

    tokenizer.train([train_file], trainer)
    tokenizer.save(save_path)
    print(f"  ✓ Đã lưu mô hình BPE vào: {save_path} (vocab={tokenizer.get_vocab_size():,})")


def train_bpe_tokenizers(
    train_src="data/processed/train.en",
    train_tgt="data/processed/train.vi",
    save_dir="custom_tokenizers/bpe",
    vocab_size=16000,
):
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 60)
    print("HUẤN LUYỆN BPE TOKENIZER (ANH & VIỆT)")
    print("=" * 60)

    # Huấn luyện cho câu tiếng Anh
    train_bpe_model(
        train_file=train_src,
        save_path=os.path.join(save_dir, "bpe.en.json"),
        vocab_size=vocab_size,
    )

    # Huấn luyện cho câu tiếng Việt
    train_bpe_model(
        train_file=train_tgt,
        save_path=os.path.join(save_dir, "bpe.vi.json"),
        vocab_size=vocab_size,
    )

    print("=" * 60)
    print("HOÀN TẤT HUẤN LUYỆN BPE TOKENIZER!")
    print("=" * 60)


if __name__ == "__main__":
    train_bpe_tokenizers()
