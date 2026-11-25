"""
Script: data/preprocess.py
Phụ trách: Phần 1 (Dữ liệu)
Chức năng: Làm sạch và chuẩn hóa văn bản song ngữ Anh - Việt:
- Chuẩn hóa Unicode tiếng Việt về dựng sẵn (NFC).
- Giải mã HTML entities (&apos;, &quot;, &#91;, &amp;, v.v.).
- Làm sạch khoảng trắng thừa, xóa câu rỗng hoặc câu quá ngắn (< 2 từ).
- Lọc bỏ các cặp câu bị lệch độ dài bất thường trong tập train.
- Xuất dữ liệu sạch ra thư mục data/processed/.
"""

import os
import re
import html
import unicodedata
from tqdm import tqdm


def normalize_text(text: str, is_vietnamese: bool = False) -> str:
    """
    Chuẩn hóa 1 câu văn bản:
    1. Unescape HTML entities (&apos; -> ', &quot; -> ", v.v.)
    2. Chuẩn hóa Unicode NFC (bắt buộc cho tiếng Việt)
    3. Chuẩn hóa dấu nháy và dấu gạch nối
    4. Xóa khoảng trắng thừa
    """
    if not text:
        return ""

    # 1. Giải mã HTML entities
    text = html.unescape(text)

    # 2. Chuẩn hóa Unicode về dạng NFC (Canonical Composition)
    text = unicodedata.normalize("NFC", text)

    # 3. Chuẩn hóa các loại dấu ngoặc kép / dấu nháy đơn lạ
    text = re.sub(r"[\u2018\u2019\u201a\u201b\u2032]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e\u201f\u2033]", '"', text)
    text = re.sub(r"[\u2013\u2014]", "-", text)

    # 4. Gom nhiều khoảng trắng liên tiếp thành 1 khoảng trắng duy nhất
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_parallel_corpus(
    src_raw_path: str,
    tgt_raw_path: str,
    src_out_path: str,
    tgt_out_path: str,
    is_train: bool = True,
    max_len: int = 100,
    min_len: int = 1,
    max_ratio: float = 2.5,
):
    """
    Làm sạch một cặp file song ngữ (src và tgt):
    - Đọc từng dòng song song
    - Lọc bỏ dòng rỗng, dòng quá ngắn hoặc quá dài (khi train)
    - Lọc bỏ các cặp có độ dài lệch quá max_ratio (khi train)
    - Ghi ra file đích
    """
    print(f"\n[INFO] Đang xử lý: {os.path.basename(src_raw_path)} <-> {os.path.basename(tgt_raw_path)}...")

    if not os.path.exists(src_raw_path) or not os.path.exists(tgt_raw_path):
        raise FileNotFoundError(f"Không tìm thấy file raw: {src_raw_path} hoặc {tgt_raw_path}")

    os.makedirs(os.path.dirname(src_out_path), exist_ok=True)

    src_lines_cleaned = []
    tgt_lines_cleaned = []
    total_pairs = 0
    kept_pairs = 0

    with open(src_raw_path, "r", encoding="utf-8") as f_src, open(tgt_raw_path, "r", encoding="utf-8") as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            total_pairs += 1

            src_clean = normalize_text(src_line, is_vietnamese=False)
            tgt_clean = normalize_text(tgt_line, is_vietnamese=True)

            # Bỏ qua nếu 1 trong 2 câu rỗng
            if not src_clean or not tgt_clean:
                continue

            # Các bước lọc chặt chẽ chỉ áp dụng cho tập huấn luyện (train)
            # Tập val và test phải giữ nguyên mọi câu để kết quả BLEU so sánh khách quan!
            if is_train:
                src_words = src_clean.split()
                tgt_words = tgt_clean.split()
                len_src = len(src_words)
                len_tgt = len(tgt_words)

                # Lọc theo độ dài từ
                if len_src < min_len or len_tgt < min_len:
                    continue
                if len_src > max_len or len_tgt > max_len:
                    continue

                # Lọc theo tỷ lệ độ dài
                ratio = max(len_src, len_tgt) / max(1, min(len_src, len_tgt))
                if ratio > max_ratio:
                    continue

            src_lines_cleaned.append(src_clean)
            tgt_lines_cleaned.append(tgt_clean)
            kept_pairs += 1

    # Ghi ra file đã làm sạch
    with open(src_out_path, "w", encoding="utf-8") as f_out_src:
        f_out_src.write("\n".join(src_lines_cleaned) + "\n")

    with open(tgt_out_path, "w", encoding="utf-8") as f_out_tgt:
        f_out_tgt.write("\n".join(tgt_lines_cleaned) + "\n")

    drop_count = total_pairs - kept_pairs
    drop_pct = (drop_count / total_pairs) * 100 if total_pairs > 0 else 0
    print(f"  ✓ Tổng số cặp câu gốc: {total_pairs:,}")
    print(f"  ✓ Số cặp câu giữ lại: {kept_pairs:,} ({100 - drop_pct:.2f}%)")
    print(f"  ✓ Số cặp câu bị lọc bỏ: {drop_count:,} ({drop_pct:.2f}%)")


def run_preprocess(raw_dir="data/raw", out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 60)
    print("BẮT ĐẦU LÀM SẠCH VÀ CHUẨN HÓA DỮ LIỆU IWSLT'15")
    print("=" * 60)

    # 1. Tập Train (áp dụng lọc)
    clean_parallel_corpus(
        src_raw_path=os.path.join(raw_dir, "train.en"),
        tgt_raw_path=os.path.join(raw_dir, "train.vi"),
        src_out_path=os.path.join(out_dir, "train.en"),
        tgt_out_path=os.path.join(out_dir, "train.vi"),
        is_train=True,
        max_len=80,
    )

    # 2. Tập Validation (tst2012 - giữ nguyên toàn bộ)
    clean_parallel_corpus(
        src_raw_path=os.path.join(raw_dir, "tst2012.en"),
        tgt_raw_path=os.path.join(raw_dir, "tst2012.vi"),
        src_out_path=os.path.join(out_dir, "tst2012.en"),
        tgt_out_path=os.path.join(out_dir, "tst2012.vi"),
        is_train=False,
    )

    # 3. Tập Test (tst2013 - giữ nguyên toàn bộ để đo BLEU chuẩn)
    clean_parallel_corpus(
        src_raw_path=os.path.join(raw_dir, "tst2013.en"),
        tgt_raw_path=os.path.join(raw_dir, "tst2013.vi"),
        src_out_path=os.path.join(out_dir, "tst2013.en"),
        tgt_out_path=os.path.join(out_dir, "tst2013.vi"),
        is_train=False,
    )

    print("\n" + "=" * 60)
    print(f"TIỀN XỬ LÝ HOÀN TẤT! Dữ liệu đã lưu tại: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run_preprocess()
