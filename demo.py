"""
Script: demo.py
Phụ trách: Hoàng (Phần 2)
Chức năng: Demo dịch máy Anh - Việt tương tác (CLI hoặc Gradio Web UI).
- Hỗ trợ nạp đồng thời mô hình Seq2seq (Hoàng) và Transformer (Sơn) từ checkpoint.
- Giải mã bằng Greedy Search hoặc Beam Search (có Length Penalty & Repetition Penalty).
- Xuất và lưu biểu đồ Attention Heatmap (300 DPI) cho câu vừa dịch.
"""

import os
import sys
import time
import argparse
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.common import PAD_IDX, set_seed
from utils.beam_search import greedy_decode, beam_search_decode
from utils.visualize import plot_attention_heatmap
from data.dataset import get_tokenizer_pair
from data.preprocess import normalize_text
from eval import load_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Demo dịch máy Anh - Việt tương tác (Seq2seq & Transformer)")
    parser.add_argument(
        "--seq2seq_ckpt",
        type=str,
        default="checkpoints/seq2seq/best_model.pt",
        help="Đường dẫn checkpoint mô hình Seq2seq",
    )
    parser.add_argument(
        "--transformer_ckpt",
        type=str,
        default=None,
        help="Đường dẫn checkpoint mô hình Transformer (tùy chọn để so sánh song song)",
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
        default=1.2,
        help="Hệ số phạt lặp từ (1.0 = tắt)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=60,
        help="Độ dài tối đa câu dịch",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Dịch nhanh 1 câu tiếng Anh rồi thoát (nếu bỏ trống sẽ chạy chế độ tương tác)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/demo_attention",
        help="Thư mục lưu ảnh Attention Heatmap",
    )
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Khởi chạy giao diện web Gradio (cần cài đặt thư viện gradio)",
    )
    return parser.parse_args()


def translate_sentence(
    text: str,
    model,
    src_tok,
    tgt_tok,
    beam_size: int = 4,
    alpha: float = 0.6,
    repetition_penalty: float = 1.2,
    max_len: int = 60,
    device: str = "cpu",
    save_attn_path: str = None,
    title: str = "Attention Alignment Heatmap",
) -> dict:
    """
    Dịch một câu tiếng Anh sang tiếng Việt, đo thời gian và vẽ Attention Heatmap (nếu có save_attn_path).
    """
    cleaned = normalize_text(text, is_vietnamese=False)
    if not cleaned:
        return {"translation": "", "time_ms": 0.0, "attn_path": None}

    src_ids = src_tok.encode(cleaned, add_special_tokens=True)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src_tensor == PAD_IDX)

    t0 = time.perf_counter()
    if beam_size <= 1:
        pred_ids, attn_matrix = greedy_decode(
            model,
            src_tensor,
            src_mask,
            max_len=max_len,
            repetition_penalty=repetition_penalty,
            device=device,
        )
    else:
        pred_ids, attn_matrix = beam_search_decode(
            model,
            src_tensor,
            src_mask,
            beam_size=beam_size,
            max_len=max_len,
            alpha=alpha,
            repetition_penalty=repetition_penalty,
            device=device,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    translation = tgt_tok.decode(pred_ids, skip_special_tokens=True)

    saved_path = None
    if save_attn_path and attn_matrix is not None and len(pred_ids) > 0:
        src_pieces = src_tok.encode_as_pieces(src_ids)
        tgt_pieces = tgt_tok.encode_as_pieces(pred_ids)
        plot_attention_heatmap(
            attn_matrix=attn_matrix,
            src_tokens=src_pieces,
            tgt_tokens=tgt_pieces,
            save_path=save_attn_path,
            head_idx="mean",
            title=title,
        )
        saved_path = save_attn_path

    return {
        "translation": translation,
        "time_ms": round(elapsed_ms, 2),
        "attn_path": saved_path,
    }


def load_available_models(args, device: str) -> list[dict]:
    """Nạp các mô hình có checkpoint tồn tại trên đĩa."""
    loaded = []
    candidates = [("Seq2seq", args.seq2seq_ckpt), ("Transformer", args.transformer_ckpt)]

    for name, ckpt_path in candidates:
        if ckpt_path and os.path.exists(ckpt_path):
            model, tok_type = load_model_from_checkpoint(ckpt_path, device=device)
            src_tok, tgt_tok = get_tokenizer_pair(tok_type)
            loaded.append({
                "name": name,
                "model": model,
                "src_tok": src_tok,
                "tgt_tok": tgt_tok,
                "tok_type": tok_type,
            })
        elif ckpt_path:
            print(f"[WARN] Không tìm thấy checkpoint {name} tại '{ckpt_path}', bỏ qua.")

    if not loaded:
        raise FileNotFoundError(
            "Không tìm thấy checkpoint hợp lệ nào! "
            "Hãy kiểm tra đường dẫn --seq2seq_ckpt hoặc --transformer_ckpt."
        )
    return loaded


def run_cli(models_info: list[dict], args, device: str):
    """Chế độ chạy dòng lệnh tương tác (Interactive CLI)."""
    print("\n" + "=" * 65)
    print("DEMO DỊCH MÁY ANH - VIỆT TƯƠNG TÁC (Gõ 'exit' hoặc 'quit' để thoát)")
    print(f"Mô hình đang bật: {', '.join(m['name'] for m in models_info)} | Beam={args.beam_size}")
    print("=" * 65)

    os.makedirs(args.save_dir, exist_ok=True)
    step = 0

    while True:
        try:
            text = input("\n[En] Nhập câu tiếng Anh: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã thoát Demo.")
            break

        if not text:
            continue
        if text.lower() in {"exit", "quit", "q"}:
            print("Tạm biệt!")
            break

        step += 1
        print("-" * 65)
        for m in models_info:
            attn_file = os.path.join(args.save_dir, f"sample_{step}_{m['name'].lower()}.png")
            res = translate_sentence(
                text=text,
                model=m["model"],
                src_tok=m["src_tok"],
                tgt_tok=m["tgt_tok"],
                beam_size=args.beam_size,
                alpha=args.alpha,
                repetition_penalty=args.repetition_penalty,
                max_len=args.max_len,
                device=device,
                save_attn_path=attn_file,
                title=f"{m['name']} ({m['tok_type']}) Attention",
            )
            print(f"[{m['name']:11s}] ({res['time_ms']:6.1f} ms): {res['translation']}")
            if res["attn_path"]:
                print(f"              Attention Map: {res['attn_path']}")


def run_gradio(models_info: list[dict], args, device: str):
    """Chế độ giao diện Web Gradio (nếu có cài gradio)."""
    try:
        import gradio as gr
    except ImportError:
        print("[ERROR] Chưa cài đặt gradio. Hãy chạy: pip install gradio")
        print("[INFO] Chuyển sang chế độ CLI mặc định...")
        run_cli(models_info, args, device)
        return

    os.makedirs(args.save_dir, exist_ok=True)

    def _predict(english_text: str, beam_k: int):
        if not english_text or not english_text.strip():
            return "", None
        lines = []
        first_img = None
        ts = int(time.time())
        for m in models_info:
            attn_file = os.path.join(args.save_dir, f"gradio_{ts}_{m['name'].lower()}.png")
            res = translate_sentence(
                text=english_text,
                model=m["model"],
                src_tok=m["src_tok"],
                tgt_tok=m["tgt_tok"],
                beam_size=int(beam_k),
                alpha=args.alpha,
                repetition_penalty=args.repetition_penalty,
                max_len=args.max_len,
                device=device,
                save_attn_path=attn_file,
                title=f"{m['name']} ({m['tok_type']}) Attention",
            )
            lines.append(f"### {m['name']} ({res['time_ms']} ms)\n{res['translation']}")
            if first_img is None and res["attn_path"]:
                first_img = res["attn_path"]
        return "\n\n".join(lines), first_img

    demo = gr.Interface(
        fn=_predict,
        inputs=[
            gr.Textbox(lines=3, label="Câu tiếng Anh (English Input)", placeholder="Enter an English sentence..."),
            gr.Slider(minimum=1, maximum=8, step=1, value=args.beam_size, label="Beam Size (1 = Greedy)"),
        ],
        outputs=[
            gr.Markdown(label="Kết quả dịch tiếng Việt"),
            gr.Image(type="filepath", label="Attention Alignment Heatmap"),
        ],
        title="🇻🇳🇬🇧 Demo Dịch Máy Anh - Việt (Seq2seq Attention & Transformer)",
        description="Đề tài 8 - Môn Học Sâu (Deep Learning)",
    )
    demo.launch(share=False)


def main():
    args = parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Thiết bị suy luận: {device}")

    models_info = load_available_models(args, device=device)

    if args.text:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"\n[En] {args.text}")
        print("-" * 65)
        for m in models_info:
            attn_file = os.path.join(args.save_dir, f"single_{m['name'].lower()}.png")
            res = translate_sentence(
                text=args.text,
                model=m["model"],
                src_tok=m["src_tok"],
                tgt_tok=m["tgt_tok"],
                beam_size=args.beam_size,
                alpha=args.alpha,
                repetition_penalty=args.repetition_penalty,
                max_len=args.max_len,
                device=device,
                save_attn_path=attn_file,
                title=f"{m['name']} ({m['tok_type']}) Attention",
            )
            print(f"[{m['name']:11s}] ({res['time_ms']:6.1f} ms): {res['translation']}")
            if res["attn_path"]:
                print(f"              Attention Map: {res['attn_path']}")
    elif args.gradio:
        run_gradio(models_info, args, device=device)
    else:
        run_cli(models_info, args, device=device)


if __name__ == "__main__":
    main()
