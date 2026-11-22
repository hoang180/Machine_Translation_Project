"""
Module: tokenizers/train_word.py
Phụ trách: Phần 1 (Lê)
Cài đặt: Bộ tách từ mức từ (Word-level Tokenizer) và script xây dựng từ điển cho tiếng Anh & tiếng Việt.
Tuân thủ chuẩn Contract: <pad>=0, <unk>=1, <sos>=2, <eos>=3.
"""

import os
import sys
import json
from collections import Counter

# Đảm bảo đường dẫn import không bị lỗi namespace
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


class WordTokenizer(BaseTokenizerWrapper):
    """
    Bộ tách từ Word-level:
    - Tách câu dựa trên khoảng trắng và dấu câu cơ bản.
    - Duy trì từ điển ánh xạ: token <-> id.
    - Mọi từ không có trong từ điển được ánh xạ về <unk> (index 1).
    """

    def __init__(self, vocab_file: str = None):
        super().__init__()
        self.w2i = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
        }
        self.i2w = {idx: token for token, idx in self.w2i.items()}

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)

    def tokenize(self, text: str) -> list[str]:
        """Tách câu thành danh sách từ (lowercase, tách khoảng trắng)."""
        if not text:
            return []
        return text.strip().lower().split()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = self.tokenize(text)
        ids = [self.w2i.get(token, self.unk_idx) for token in tokens]
        if add_special_tokens:
            ids = [self.sos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        words = []
        special_ids = {self.pad_idx, self.sos_idx, self.eos_idx}
        for idx in ids:
            if skip_special_tokens and idx in special_ids:
                continue
            word = self.i2w.get(idx, UNK_TOKEN)
            words.append(word)
        return " ".join(words)

    def id_to_piece(self, idx: int) -> str:
        return self.i2w.get(idx, UNK_TOKEN)

    @property
    def vocab_size(self) -> int:
        return len(self.w2i)

    def build_vocab(self, file_path: str, max_vocab_size: int = 25000, min_freq: int = 2):
        """Xây dựng từ điển từ file văn bản."""
        print(f"[INFO] Đang xây dựng từ điển từ: {file_path} (max_vocab={max_vocab_size:,})...")
        counter = Counter()

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = self.tokenize(line)
                counter.update(tokens)

        # Lọc các từ phổ biến nhất
        most_common = counter.most_common(max_vocab_size - 4)

        for word, freq in most_common:
            if freq < min_freq:
                continue
            if word not in self.w2i:
                idx = len(self.w2i)
                self.w2i[word] = idx
                self.i2w[idx] = word

        print(f"  ✓ Đã xây dựng xong từ điển! Tổng số từ: {len(self.w2i):,}")

    def save_vocab(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"w2i": self.w2i}, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Đã lưu từ điển vào: {save_path}")

    def load_vocab(self, load_path: str):
        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.w2i = data["w2i"]
            self.i2w = {int(idx): word for word, idx in self.w2i.items()}
        print(f"[INFO] Đã load từ điển từ {load_path} ({len(self.w2i):,} từ)")


def train_word_tokenizers(
    train_src="data/processed/train.en",
    train_tgt="data/processed/train.vi",
    save_dir="tokenizers/word",
    vocab_size=25000,
):
    os.makedirs(save_dir, exist_ok=True)
    print("=" * 60)
    print("HUẤN LUYỆN WORD-LEVEL TOKENIZER (ANH & VIỆT)")
    print("=" * 60)

    # 1. Tiếng Anh
    en_tok = WordTokenizer()
    en_tok.build_vocab(train_src, max_vocab_size=vocab_size)
    en_tok.save_vocab(os.path.join(save_dir, "vocab.en.json"))

    # 2. Tiếng Việt
    vi_tok = WordTokenizer()
    vi_tok.build_vocab(train_tgt, max_vocab_size=vocab_size)
    vi_tok.save_vocab(os.path.join(save_dir, "vocab.vi.json"))

    print("=" * 60)
    print("HOÀN TẤT HUẤN LUYỆN WORD TOKENIZER!")
    print("=" * 60)


if __name__ == "__main__":
    train_word_tokenizers()
