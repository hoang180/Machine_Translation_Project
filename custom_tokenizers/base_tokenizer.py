"""
Module: custom_tokenizers/base_tokenizer.py
Phụ trách: Phần 1 (Lê)
Cài đặt: Lớp giao diện chuẩn BaseTokenizerWrapper theo đúng Contract Mục 4.1 của file Quy ước 01.
Giúp dataset.py, eval.py, beam_search.py và demo.py hoạt động thống nhất với mọi loại Tokenizer.
"""

from abc import ABC, abstractmethod
from utils.common import PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX


class BaseTokenizerWrapper(ABC):
    """
    Giao diện lớp cơ sở cho mọi Tokenizer (Word, BPE, SentencePiece):
    Mọi lớp con bắt buộc phải cài đặt các phương thức:
    - encode()
    - decode()
    - id_to_piece()
    - vocab_size
    """

    def __init__(self):
        self.pad_idx = PAD_IDX
        self.unk_idx = UNK_IDX
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Chuyển chuỗi văn bản thành list ID số nguyên.
        Nếu add_special_tokens=True: tự động thêm [SOS_IDX] ở đầu và [EOS_IDX] ở cuối.
        """
        pass

    @abstractmethod
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Chuyển list ID về lại câu văn bản hoàn chỉnh.
        Nếu skip_special_tokens=True: loại bỏ các token <pad>, <sos>, <eos>, <unk>.
        """
        pass

    @abstractmethod
    def id_to_piece(self, idx: int) -> str:
        """
        Trả về chuỗi hiển thị của 1 token ID (dùng gắn nhãn trục X/Y khi vẽ Attention Heatmap).
        """
        pass

    def encode_as_pieces(self, ids: list[int]) -> list[str]:
        """
        Chuyển danh sách ID thành danh sách từng token/subword rời để vẽ các ô trên Heatmap.
        """
        return [self.id_to_piece(i) for i in ids]

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        Tổng kích thước từ vựng (bao gồm cả 4 token đặc biệt 0..3).
        """
        pass
