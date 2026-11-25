"""
Module: utils/common.py
Chứa các hằng số quy ước chung, hàm cố định seed và hàm tạo batch dữ liệu giả lập (dummy batch)
để kiểm thử độc lập mô hình theo chuẩn Contract của dự án.
"""

import random
import os
import numpy as np
import torch

# ==========================================
# 1. QUY ƯỚC TOKEN ĐẶC BIỆT (SPECIAL TOKENS)
# ==========================================
PAD_IDX = 0  # <pad>: Đệm các câu ngắn trong batch cho bằng max_len
UNK_IDX = 1  # <unk>: Out-of-vocabulary (từ ngoài từ điển)
SOS_IDX = 2  # <sos>: Start of sequence (bắt đầu câu đích)
EOS_IDX = 3  # <eos>: End of sequence (kết thúc câu)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


# ==========================================
# 2. CỐ ĐỊNH SEED ĐỂ TÁI LẬP KẾT QUẢ (REPRODUCIBILITY)
# ==========================================
def set_seed(seed: int = 42) -> None:
    """
    Cố định random seed cho toàn bộ môi trường: Python, NumPy, PyTorch (CPU & GPU).
    Đảm bảo thí nghiệm có thể tái lập được kết quả 100%.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==========================================
# 3. HÀM TẠO DỮ LIỆU GIẢ LẬP (DUMMY BATCH)
# ==========================================
def get_dummy_batch(
    batch_size: int = 8,
    src_len: int = 16,
    tgt_len: int = 20,
    src_vocab_size: int = 5000,
    tgt_vocab_size: int = 5000,
    device: str = "cpu",
) -> dict:
    """
    Tạo một batch dữ liệu ngẫu nhiên chuẩn hóa đúng theo Interface Contract để
    Hoàng và Sơn có thể test mô hình ngay từ Ngày 1 mà không cần chờ dữ liệu thật từ Lê.

    Args:
        batch_size: Số lượng câu trong batch.
        src_len: Chiều dài câu nguồn tiếng Anh (bao gồm cả pad).
        tgt_len: Chiều dài câu đích tiếng Việt (bao gồm cả pad).
        src_vocab_size: Kích thước từ vựng nguồn giả lập.
        tgt_vocab_size: Kích thước từ vựng đích giả lập.
        device: 'cpu' hoặc 'cuda'.

    Returns:
        dict chứa:
            - 'src': torch.LongTensor shape (B, src_len)
            - 'tgt': torch.LongTensor shape (B, tgt_len)
            - 'src_padding_mask': torch.BoolTensor shape (B, src_len) - True tại vị trí <pad>
            - 'src_lengths': torch.LongTensor shape (B,) - độ dài thực của từng câu
            - 'src_raw': list[str] câu tiếng Anh gốc giả lập
            - 'tgt_raw': list[str] câu tiếng Việt gốc giả lập
    """
    # Khởi tạo tensor ngẫu nhiên từ index 4 trở lên (bỏ qua 4 token đặc biệt 0..3)
    src = torch.randint(low=4, high=src_vocab_size, size=(batch_size, src_len), dtype=torch.long, device=device)
    tgt = torch.randint(low=4, high=tgt_vocab_size, size=(batch_size, tgt_len), dtype=torch.long, device=device)

    # Đặt token bắt đầu <sos> ở vị trí đầu tiên
    src[:, 0] = SOS_IDX
    tgt[:, 0] = SOS_IDX

    # Tạo độ dài câu ngẫu nhiên để mô phỏng padding thực tế
    for b in range(batch_size):
        # Đảm bảo độ dài câu tối thiểu là 4 token (<sos> + ít nhất 2 từ + <eos>)
        actual_src_len = random.randint(4, src_len)
        actual_tgt_len = random.randint(4, tgt_len)

        # Gán <eos> tại vị trí kết thúc câu thực tế
        src[b, actual_src_len - 1] = EOS_IDX
        tgt[b, actual_tgt_len - 1] = EOS_IDX

        # Vùng sau <eos> là <pad>
        if actual_src_len < src_len:
            src[b, actual_src_len:] = PAD_IDX
        if actual_tgt_len < tgt_len:
            tgt[b, actual_tgt_len:] = PAD_IDX

    # Quy ước Contract: True là vị trí <pad> (cần che giấu, masked_fill=-inf)
    src_padding_mask = (src == PAD_IDX)
    src_lengths = (~src_padding_mask).sum(dim=1).to(device)

    return {
        "src": src,
        "tgt": tgt,
        "src_padding_mask": src_padding_mask,
        "src_lengths": src_lengths,
        "src_raw": ["dummy english sentence " + str(i) for i in range(batch_size)],
        "tgt_raw": ["câu tiếng việt giả lập " + str(i) for i in range(batch_size)],
    }

