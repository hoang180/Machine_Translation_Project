"""
Module: data/dataset.py
Phụ trách: Phần 1 (Lê)
Cài đặt: Lớp TranslationDataset, hàm collate_fn và hàm build_dataloader() cung cấp dữ liệu
cho cả Seq2seq (Hoàng) và Transformer (Sơn) theo chuẩn Contract Mục 4.2 của file Quy ước 01.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.common import PAD_IDX


class TranslationDataset(Dataset):
    """
    Dataset song ngữ Anh - Việt:
    - Đọc các cặp câu từ 2 file văn bản src_file và tgt_file.
    - Dùng tokenizer tương ứng (Word / BPE / SentencePiece) để encode thành danh sách Token ID.
    - Lọc bỏ các câu vượt quá max_seq_len (nếu is_train=True).
    """

    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        src_tokenizer,
        tgt_tokenizer,
        max_seq_len: int = 80,
        is_train: bool = True,
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len

        self.samples = []

        if not os.path.exists(src_file) or not os.path.exists(tgt_file):
            raise FileNotFoundError(f"Không tìm thấy file: {src_file} hoặc {tgt_file}")

        with open(src_file, "r", encoding="utf-8") as f_src, open(tgt_file, "r", encoding="utf-8") as f_tgt:
            for src_line, tgt_line in zip(f_src, f_tgt):
                src_text = src_line.strip()
                tgt_text = tgt_line.strip()

                if not src_text or not tgt_text:
                    continue

                # Encode kèm <sos> và <eos>
                src_ids = self.src_tokenizer.encode(src_text, add_special_tokens=True)
                tgt_ids = self.tgt_tokenizer.encode(tgt_text, add_special_tokens=True)

                # Giới hạn độ dài khi huấn luyện để tiết kiệm bộ nhớ
                if is_train:
                    if len(src_ids) > max_seq_len or len(tgt_ids) > max_seq_len:
                        continue

                self.samples.append(
                    {
                        "src_ids": src_ids,
                        "tgt_ids": tgt_ids,
                        "src_raw": src_text,
                        "tgt_raw": tgt_text,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch_samples: list[dict]) -> dict:
    """
    Hàm ghép batch động (Dynamic Padding):
    - Tìm độ dài lớn nhất trong batch cho src và tgt.
    - Đệm PAD_IDX = 0 vào các câu ngắn hơn.
    - Tạo mặt nạ src_padding_mask kiểu boolean (True tại vị trí <pad>).
    - Trả về đúng dictionary theo chuẩn Contract Mục 4.2.
    """
    batch_size = len(batch_samples)

    src_lengths = [len(s["src_ids"]) for s in batch_samples]
    tgt_lengths = [len(s["tgt_ids"]) for s in batch_samples]

    max_src_len = max(src_lengths)
    max_tgt_len = max(tgt_lengths)

    # Khởi tạo tensor chứa toàn PAD_IDX = 0
    src_tensor = torch.full((batch_size, max_src_len), fill_value=PAD_IDX, dtype=torch.long)
    tgt_tensor = torch.full((batch_size, max_tgt_len), fill_value=PAD_IDX, dtype=torch.long)

    src_raw = []
    tgt_raw = []

    for i, sample in enumerate(batch_samples):
        s_len = len(sample["src_ids"])
        t_len = len(sample["tgt_ids"])

        src_tensor[i, :s_len] = torch.tensor(sample["src_ids"], dtype=torch.long)
        tgt_tensor[i, :t_len] = torch.tensor(sample["tgt_ids"], dtype=torch.long)

        src_raw.append(sample["src_raw"])
        tgt_raw.append(sample["tgt_raw"])

    # src_padding_mask: True tại vị trí <pad> (Contract Mục 4.2)
    src_padding_mask = (src_tensor == PAD_IDX)
    src_lengths_tensor = torch.tensor(src_lengths, dtype=torch.long)

    return {
        "src": src_tensor,                # (B, src_len)
        "tgt": tgt_tensor,                # (B, tgt_len)
        "src_padding_mask": src_padding_mask,  # (B, src_len), bool
        "src_lengths": src_lengths_tensor,     # (B,), long
        "src_raw": src_raw,               # list[str]
        "tgt_raw": tgt_raw,               # list[str]
    }


def get_tokenizer_pair(tokenizer_type="word"):
    """Hàm helper khởi tạo và load cặp Tokenizer tương ứng."""
    tokenizer_type = tokenizer_type.lower()

    if tokenizer_type == "word":
        from custom_tokenizers.train_word import WordTokenizer
        src_tok = WordTokenizer(vocab_file="custom_tokenizers/word/vocab.en.json")
        tgt_tok = WordTokenizer(vocab_file="custom_tokenizers/word/vocab.vi.json")

    elif tokenizer_type == "bpe":
        from custom_tokenizers.train_bpe import BPETokenizer
        src_tok = BPETokenizer(model_file="custom_tokenizers/bpe/bpe.en.json")
        tgt_tok = BPETokenizer(model_file="custom_tokenizers/bpe/bpe.vi.json")

    elif tokenizer_type == "spm":
        from custom_tokenizers.train_spm import SPMTokenizer
        src_tok = SPMTokenizer(model_file="custom_tokenizers/spm/spm.en.model")
        tgt_tok = SPMTokenizer(model_file="custom_tokenizers/spm/spm.vi.model")

    else:
        raise ValueError(f"Loại tokenizer không hợp lệ: {tokenizer_type}. Chọn 'word', 'bpe' hoặc 'spm'.")

    return src_tok, tgt_tok


def build_dataloader(
    src_file: str,
    tgt_file: str,
    tokenizer_type: str = "word",
    batch_size: int = 64,
    max_seq_len: int = 80,
    is_train: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Hàm cung cấp DataLoader hoàn chỉnh dùng trong train_seq2seq.py và train_transformer.py:
    """
    src_tok, tgt_tok = get_tokenizer_pair(tokenizer_type)

    dataset = TranslationDataset(
        src_file=src_file,
        tgt_file=tgt_file,
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        max_seq_len=max_seq_len,
        is_train=is_train,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,  # Shuffle khi train, không shuffle khi val/test
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader
