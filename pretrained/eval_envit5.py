"""
Script: pretrained/eval_envit5.py
Phụ trách: Phần 1 (Lê)
Chức năng: Chạy mô hình SOTA Pretrained VietAI/envit5-translation (hoặc mBART) qua HuggingFace
trên tập test tst2013 để lấy mốc tham chiếu trần hiệu năng (Reference Upper-Bound) cho bài báo ACL.
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import compute_bleu


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình Pretrained EnViT5 trên tập tst2013")
    parser.add_argument(
        "--model_name",
        type=str,
        default="VietAI/envit5-translation",
        help="Tên mô hình trên HuggingFace (ví dụ VietAI/envit5-translation hoặc facebook/mbart-large-50-many-to-many-mmt)",
    )
    parser.add_argument(
        "--test_src",
        type=str,
        default="data/processed/tst2013.en",
        help="Đường dẫn file câu nguồn tiếng Anh",
    )
    parser.add_argument(
        "--test_tgt",
        type=str,
        default="data/processed/tst2013.vi",
        help="Đường dẫn file câu đích tiếng Việt chuẩn",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size khi inference",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Giới hạn số câu để test nhanh",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Thiết bị sử dụng: {device}")
    print(f"[INFO] Đang tải mô hình pretrained từ HuggingFace: {args.model_name}...")

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError:
        print("[ERROR] Cần cài đặt thư viện transformers: pip install transformers")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)
    model.eval()

    # Đọc dữ liệu test
    if not os.path.exists(args.test_src) or not os.path.exists(args.test_tgt):
        print(f"[ERROR] Không tìm thấy dữ liệu test tại: {args.test_src}")
        return

    with open(args.test_src, "r", encoding="utf-8") as f_s, open(args.test_tgt, "r", encoding="utf-8") as f_t:
        src_lines = [l.strip() for l in f_s if l.strip()]
        tgt_lines = [l.strip() for l in f_t if l.strip()]

    if args.max_samples:
        src_lines = src_lines[: args.max_samples]
        tgt_lines = tgt_lines[: args.max_samples]

    print(f"[INFO] Bắt đầu dịch {len(src_lines):,} câu bằng {args.model_name}...")

    hypotheses = []
    # EnViT5 yêu cầu prefix 'en: ' cho câu tiếng Anh nguồn
    prefix = "en: " if "envit5" in args.model_name.lower() else ""

    for i in tqdm(range(0, len(src_lines), args.batch_size), desc="Pretrained Translation"):
        batch_src = [prefix + s for s in src_lines[i : i + args.batch_size]]
        inputs = tokenizer(batch_src, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)

        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Bỏ tiền tố 'vi: ' nếu có ở đầu câu tiếng Việt sinh ra bởi EnViT5
        for pred in batch_preds:
            if pred.startswith("vi: "):
                pred = pred[4:]
            hypotheses.append(pred.strip())

    # Tính điểm SacreBLEU
    results = compute_bleu(hypotheses, tgt_lines)

    print("\n" + "=" * 60)
    print(f"KẾT QUẢ MỐC THAM CHIẾU PRETRAINED ({args.model_name})")
    print("=" * 60)
    print(f"  • Điểm SacreBLEU: {results['bleu']} / 100")
    print(f"  • Brevity Penalty: {results['bp']}")
    print(f"  • Signature: {results['signature']}")
    print("=" * 60)

    # Lưu kết quả dịch ra file để sau này chèn vào bảng so sánh
    out_file = "checkpoints/envit5_predictions_tst2013.txt"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(hypotheses) + "\n")
    print(f"[INFO] Đã lưu bản dịch của EnViT5 vào: {out_file}")


if __name__ == "__main__":
    main()
