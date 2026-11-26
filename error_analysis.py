"""
Script: error_analysis.py
Phụ trách: Hoàng (Phần 2)
Chức năng:
1. Benchmark chiến lược giải mã trên tập test tst2013:
   - Greedy Search (k=1) vs Beam Search (k=3, k=5, có/không Length Penalty alpha=0.6).
   - Đo điểm SacreBLEU và tốc độ giải mã (ms/câu, sent/s).
2. Phân tích lỗi (Error Analysis) theo độ dài câu nguồn:
   - Ngắn (<15 từ), Trung bình (15-30 từ), Dài (>30 từ).
   - Tính BLEU riêng từng nhóm để đối chiếu hiện tượng suy giảm ngữ cảnh xa.
   - Xuất file CSV/JSON gồm các cặp câu dịch tốt nhất & tệ nhất (theo sentence BLEU)
     phục vụ trực tiếp cho mục Error Analysis trong báo cáo ACL.
"""

import os
import sys
import csv
import json
import time
import argparse
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.common import PAD_IDX, set_seed
from utils.metrics import compute_bleu
from utils.beam_search import greedy_decode, beam_search_decode
from data.dataset import get_tokenizer_pair
from eval import load_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Decoding & Error Analysis trên tập tst2013")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/seq2seq/best_model.pt",
        help="Đường dẫn checkpoint mô hình cần phân tích",
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
        "--out_dir",
        type=str,
        default="outputs/error_analysis",
        help="Thư mục lưu bảng CSV/JSON kết quả",
    )
    parser.add_argument(
        "--benchmark_decoding",
        action="store_true",
        help="Chạy benchmark đầy đủ k=1 (Greedy), k=3, k=4 (alpha=0.6), k=5 (alpha=0.0 & 0.6)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=4,
        help="Beam size mặc định cho phần phân tích lỗi theo độ dài",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Length Penalty mặc định",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Repetition Penalty (1.0 = tắt)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Giới hạn số câu để chạy thử nhanh",
    )
    return parser.parse_args()


def decode_corpus(
    model,
    src_tok,
    tgt_tok,
    src_lines: list[str],
    beam_size: int = 4,
    alpha: float = 0.6,
    repetition_penalty: float = 1.2,
    max_len: int = 60,
    device: str = "cpu",
) -> tuple[list[str], float]:
    """Dịch toàn bộ tập câu và trả về (hypotheses, elapsed_seconds)."""
    hypotheses = []
    t_start = time.perf_counter()

    for src_text in tqdm(src_lines, desc=f"Decoding (k={beam_size}, a={alpha})", leave=False):
        src_ids = src_tok.encode(src_text, add_special_tokens=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src_tensor == PAD_IDX)

        if beam_size <= 1:
            pred_ids, _ = greedy_decode(
                model,
                src_tensor,
                src_mask,
                max_len=max_len,
                repetition_penalty=repetition_penalty,
                device=device,
            )
        else:
            pred_ids, _ = beam_search_decode(
                model,
                src_tensor,
                src_mask,
                beam_size=beam_size,
                max_len=max_len,
                alpha=alpha,
                repetition_penalty=repetition_penalty,
                device=device,
            )
        hypotheses.append(tgt_tok.decode(pred_ids, skip_special_tokens=True))

    elapsed = time.perf_counter() - t_start
    return hypotheses, elapsed


def compute_sentence_bleu(hyp: str, ref: str) -> float:
    """Tính điểm BLEU ở mức câu để xếp hạng bản dịch tốt/xấu."""
    try:
        import sacrebleu
        return round(sacrebleu.sentence_bleu(hyp, [ref]).score, 2)
    except Exception:
        # Fallback đơn giản theo tỉ lệ trùng từ (Jaccard) nếu thiếu sacrebleu
        h_set = set(hyp.lower().split())
        r_set = set(ref.lower().split())
        if not h_set or not r_set:
            return 0.0
        return round(100.0 * len(h_set & r_set) / len(h_set | r_set), 2)


def analyze_by_length(
    src_lines: list[str],
    ref_lines: list[str],
    hyp_lines: list[str],
) -> tuple[dict, list[dict]]:
    """
    Phân loại câu theo độ dài câu nguồn:
    - short:  < 15 từ
    - medium: 15 - 30 từ
    - long:   > 30 từ
    """
    buckets = {
        "short (<15)": {"hyps": [], "refs": []},
        "medium (15-30)": {"hyps": [], "refs": []},
        "long (>30)": {"hyps": [], "refs": []},
    }
    detailed_samples = []

    for idx, (src, ref, hyp) in enumerate(zip(src_lines, ref_lines, hyp_lines)):
        src_len = len(src.split())
        if src_len < 15:
            bucket_name = "short (<15)"
        elif src_len <= 30:
            bucket_name = "medium (15-30)"
        else:
            bucket_name = "long (>30)"

        buckets[bucket_name]["hyps"].append(hyp)
        buckets[bucket_name]["refs"].append(ref)

        sent_bleu = compute_sentence_bleu(hyp, ref)
        detailed_samples.append({
            "id": idx,
            "src_len": src_len,
            "bucket": bucket_name,
            "sentence_bleu": sent_bleu,
            "src": src,
            "ref": ref,
            "hyp": hyp,
        })

    bucket_summary = {}
    for name, data in buckets.items():
        count = len(data["hyps"])
        if count > 0:
            scores = compute_bleu(data["hyps"], data["refs"])
            bucket_summary[name] = {
                "count": count,
                "bleu": scores["bleu"],
                "bleu_1": scores["bleu_1"],
                "bleu_4": scores["bleu_4"],
                "bp": scores["bp"],
            }
        else:
            bucket_summary[name] = {"count": 0, "bleu": 0.0}

    return bucket_summary, detailed_samples


def main():
    args = parse_args()
    set_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Thiết bị đánh giá: {device}")

    model, tokenizer_type = load_model_from_checkpoint(args.checkpoint, device=device)
    src_tok, tgt_tok = get_tokenizer_pair(tokenizer_type)

    with open(args.test_src, "r", encoding="utf-8") as fs, open(args.test_tgt, "r", encoding="utf-8") as ft:
        src_lines = [line.strip() for line in fs if line.strip()]
        ref_lines = [line.strip() for line in ft if line.strip()]

    if args.max_samples:
        src_lines = src_lines[: args.max_samples]
        ref_lines = ref_lines[: args.max_samples]

    n_samples = len(src_lines)
    print(f"[INFO] Tổng số câu kiểm tra: {n_samples:,} | Tokenizer: {tokenizer_type}")

    # 1. Benchmark chiến lược giải mã (nếu bật --benchmark_decoding)
    if args.benchmark_decoding:
        configs = [
            ("Greedy (k=1)", 1, 0.0),
            ("Beam (k=3, alpha=0.0)", 3, 0.0),
            ("Beam (k=3, alpha=0.6)", 3, 0.6),
            ("Beam (k=5, alpha=0.0)", 5, 0.0),
            ("Beam (k=5, alpha=0.6)", 5, 0.6),
        ]
        bench_rows = []
        print("\n" + "=" * 75)
        print("BENCHMARK CHIẾN LƯỢC GIẢI MÃ (GREEDY VS BEAM SEARCH)")
        print("=" * 75)

        for label, k, a in configs:
            hyps, elapsed = decode_corpus(
                model, src_tok, tgt_tok, src_lines,
                beam_size=k, alpha=a,
                repetition_penalty=args.repetition_penalty,
                device=device,
            )
            bleu_res = compute_bleu(hyps, ref_lines)
            ms_per_sent = (elapsed * 1000.0) / max(1, n_samples)
            sent_per_sec = n_samples / max(1e-6, elapsed)

            row = {
                "strategy": label,
                "beam_size": k,
                "alpha": a,
                "bleu": bleu_res["bleu"],
                "bleu_1": bleu_res["bleu_1"],
                "bleu_4": bleu_res["bleu_4"],
                "bp": bleu_res["bp"],
                "ms_per_sent": round(ms_per_sent, 2),
                "sent_per_sec": round(sent_per_sec, 2),
            }
            bench_rows.append(row)
            print(
                f"  • {label:24s} | BLEU: {row['bleu']:5.2f} | "
                f"{row['ms_per_sent']:7.2f} ms/câu ({row['sent_per_sec']:5.2f} sent/s)"
            )

        bench_csv = os.path.join(args.out_dir, "decoding_benchmark.csv")
        with open(bench_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(bench_rows[0].keys()))
            writer.writeheader()
            writer.writerows(bench_rows)
        print(f"[SAVED] Bảng benchmark decoding -> {bench_csv}")

    # 2. Phân tích lỗi theo độ dài câu nguồn
    print(f"\n[INFO] Đang giải mã phân tích lỗi theo độ dài (k={args.beam_size}, alpha={args.alpha})...")
    hyps, elapsed = decode_corpus(
        model, src_tok, tgt_tok, src_lines,
        beam_size=args.beam_size,
        alpha=args.alpha,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )
    overall = compute_bleu(hyps, ref_lines)
    bucket_summary, detailed_samples = analyze_by_length(src_lines, ref_lines, hyps)

    print("\n" + "=" * 75)
    print(f"PHÂN TÍCH BLEU THEO ĐỘ DÀI CÂU (Overall BLEU = {overall['bleu']})")
    print("=" * 75)
    for b_name, b_stats in bucket_summary.items():
        print(f"  • {b_name:16s} | Số câu: {b_stats['count']:4d} | BLEU: {b_stats['bleu']:5.2f}")

    # Lưu kết quả tổng hợp độ dài ra JSON
    summary_json = os.path.join(args.out_dir, "length_analysis.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "tokenizer_type": tokenizer_type,
                "overall": overall,
                "by_length": bucket_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Sắp xếp câu để lọc Top 20 dịch tốt nhất & Top 20 dịch kém nhất (Error Analysis)
    sorted_samples = sorted(detailed_samples, key=lambda x: x["sentence_bleu"])
    worst_20 = sorted_samples[:20]
    best_20 = sorted_samples[-20:][::-1]

    cases_json = os.path.join(args.out_dir, "error_cases.json")
    with open(cases_json, "w", encoding="utf-8") as f:
        json.dump(
            {"worst_20": worst_20, "best_20": best_20},
            f,
            ensure_ascii=False,
            indent=2,
        )

    all_csv = os.path.join(args.out_dir, "all_predictions.csv")
    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "src_len", "bucket", "sentence_bleu", "src", "ref", "hyp"])
        writer.writeheader()
        writer.writerows(detailed_samples)

    print(f"\n[SAVED] Tổng hợp theo độ dài -> {summary_json}")
    print(f"[SAVED] Top 20 câu tốt/xấu   -> {cases_json}")
    print(f"[SAVED] Toàn bộ câu dịch     -> {all_csv}")


if __name__ == "__main__":
    main()
