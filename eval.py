"""
Script: eval.py
Phụ trách: Phần 1 (Lê)
Chức năng: Đánh giá mô hình thống nhất trên tập test tst2013 (SacreBLEU).
Tự động đọc file checkpoint best_model.pt (theo chuẩn Mục 8.1 của file Quy ước 01),
khôi phục mô hình (Seq2seq hoặc Transformer), dịch toàn bộ tập test và in bảng kết quả.
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.common import set_seed, PAD_IDX
from utils.metrics import compute_bleu
from utils.beam_search import greedy_decode, beam_search_decode
from data.dataset import get_tokenizer_pair


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình dịch máy trên tập test tst2013")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/seq2seq/best_model.pt",
        help="Đường dẫn file checkpoint .pt",
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
        help="Đường dẫn file câu đích tiếng Việt chuẩn (Ground truth)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=4,
        help="Độ rộng chùm tia Beam Search (1 = Greedy Search)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Hệ số Length Penalty",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Hệ số phạt lặp từ (1.0 = tắt, 1.2 = giảm hiện tượng lặp từ)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Giới hạn số câu test để kiểm tra nhanh",
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Tự động khôi phục mô hình từ dictionary checkpoint theo chuẩn Mục 8.1."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_type = ckpt.get("model_type", "seq2seq")
    config = ckpt.get("config", {})

    src_vocab_size = ckpt["src_vocab_size"]
    tgt_vocab_size = ckpt["tgt_vocab_size"]

    print(f"[INFO] Khôi phục mô hình loại: '{model_type}' (Epoch {ckpt.get('epoch', '?')})")

    if model_type == "seq2seq":
        from models.seq2seq import Seq2seqModel

        m_cfg = config.get("model", {})
        model = Seq2seqModel(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_size=m_cfg.get("embed_size", 256),
            enc_hidden_size=m_cfg.get("enc_hidden_size", 512),
            dec_hidden_size=m_cfg.get("dec_hidden_size", 512),
            num_layers=m_cfg.get("num_layers", 2),
            dropout=0.0,  # Không cần dropout khi evaluate
            pad_idx=PAD_IDX,
            attention_type=m_cfg.get("attention_type", "luong"),
            luong_method=m_cfg.get("luong_method", "general"),
        )
    elif model_type == "transformer":
        from models.transformer.transformer import TransformerModel

        m_cfg = config.get("model", {})
        model = TransformerModel(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=m_cfg.get("d_model", 256),
            num_heads=m_cfg.get("num_heads", 4),
            num_layers=m_cfg.get("num_layers", 4),
            d_ff=m_cfg.get("d_ff", 1024),
            dropout=0.0,
            pad_idx=PAD_IDX,
            use_pe=m_cfg.get("use_pe", True),
        )
    else:
        raise ValueError(f"model_type không được hỗ trợ: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer_type = ckpt.get("tokenizer_type", "word")
    return model, tokenizer_type


def main():
    args = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Thiết bị đánh giá: {device}")

    # 1. Khôi phục Model & Tokenizer
    model, tokenizer_type = load_model_from_checkpoint(args.checkpoint, device=device)
    src_tok, tgt_tok = get_tokenizer_pair(tokenizer_type)

    # 2. Đọc tập Test
    if not os.path.exists(args.test_src) or not os.path.exists(args.test_tgt):
        print(f"[WARN] Chưa có file test thật tại {args.test_src}. Tạo dữ liệu mẫu để test script!")
        test_src_lines = ["How are you ?", "This is a great book .", "Thank you very much ."]
        test_tgt_lines = ["Bạn khỏe không ?", "Đây là một cuốn sách hay .", "Cảm ơn bạn rất nhiều ."]
    else:
        with open(args.test_src, "r", encoding="utf-8") as f_s, open(args.test_tgt, "r", encoding="utf-8") as f_t:
            test_src_lines = [l.strip() for l in f_s if l.strip()]
            test_tgt_lines = [l.strip() for l in f_t if l.strip()]

    if args.max_samples:
        test_src_lines = test_src_lines[: args.max_samples]
        test_tgt_lines = test_tgt_lines[: args.max_samples]

    print(f"[INFO] Số câu kiểm tra: {len(test_src_lines):,} (Decoding: Beam size = {args.beam_size})")

    # 3. Dịch từng câu
    hypotheses = []
    references = test_tgt_lines

    for src_text in tqdm(test_src_lines, desc="Translating Test Set"):
        # Mã hóa câu nguồn
        src_ids = src_tok.encode(src_text, add_special_tokens=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src_tensor == PAD_IDX)

        # Giải mã
        if args.beam_size == 1:
            pred_ids, _ = greedy_decode(
                model,
                src_tensor,
                src_mask,
                max_len=60,
                repetition_penalty=args.repetition_penalty,
                device=device,
            )
        else:
            pred_ids, _ = beam_search_decode(
                model,
                src_tensor,
                src_mask,
                beam_size=args.beam_size,
                max_len=60,
                alpha=args.alpha,
                repetition_penalty=args.repetition_penalty,
                device=device,
            )

        # Giải mã IDs thành chuỗi văn bản
        pred_text = tgt_tok.decode(pred_ids, skip_special_tokens=True)
        hypotheses.append(pred_text)

    # 4. Tính điểm SacreBLEU
    results = compute_bleu(hypotheses, references)

    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (SACREBLEU)")
    print("=" * 60)
    print(f"  • Điểm BLEU: {results['bleu']} / 100")
    print(f"  • 1-gram Precision: {results['bleu_1']}%")
    print(f"  • 2-gram Precision: {results['bleu_2']}%")
    print(f"  • 3-gram Precision: {results['bleu_3']}%")
    print(f"  • 4-gram Precision: {results['bleu_4']}%")
    print(f"  • Brevity Penalty: {results['bp']}")
    print(f"  • Signature: {results['signature']}")
    print("=" * 60)

    # In 3 ví dụ mẫu
    print("\n[VÍ DỤ DỊCH THỰC TẾ]:")
    for i in range(min(3, len(test_src_lines))):
        print(f"[{i+1}] Nguồn (En): {test_src_lines[i]}")
        print(f"    Chuẩn (Ref): {references[i]}")
        print(f"    Dịch  (Hyp): {hypotheses[i]}")
        print("-" * 50)


if __name__ == "__main__":
    main()
