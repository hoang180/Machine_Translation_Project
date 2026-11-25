"""
Module: utils/metrics.py
Phụ trách: Phần 1 (Lê)
Cài đặt: Hàm tính điểm SacreBLEU chuẩn học thuật cho bài báo ACL theo Contract Mục 7.2.
"""

def compute_bleu(hypotheses: list[str], references: list[str]) -> dict:
    """
    Tính điểm SacreBLEU chuẩn quốc tế (chuẩn của hội nghị ACL / WMT):
    Args:
        hypotheses: Danh sách câu tiếng Việt dự đoán của mô hình.
        references: Danh sách câu tiếng Việt chuẩn từ tập nhãn.

    Returns:
        dict:
            - 'bleu': Điểm BLEU tổng (thang 0 - 100)
            - 'bleu_1'..'bleu_4': Precision của từng n-gram
            - 'bp': Brevity Penalty (phạt câu quá ngắn)
            - 'signature': Chuỗi hash thông số đánh giá để trích dẫn trong bài báo ACL
    """
    try:
        import sacrebleu

        # Thêm force=True để bỏ qua cảnh báo tokenized period
        bleu = sacrebleu.corpus_bleu(hypotheses, [references], force=True)

        # Lấy signature an toàn tương thích mọi phiên bản sacrebleu
        if hasattr(bleu, "signature"):
            sig = str(bleu.signature)
        elif hasattr(bleu, "get_signature"):
            sig = str(bleu.get_signature())
        else:
            sig = str(bleu)

        return {
            "bleu": round(bleu.score, 2),
            "bleu_1": round(bleu.precisions[0], 2) if len(bleu.precisions) > 0 else 0.0,
            "bleu_2": round(bleu.precisions[1], 2) if len(bleu.precisions) > 1 else 0.0,
            "bleu_3": round(bleu.precisions[2], 2) if len(bleu.precisions) > 2 else 0.0,
            "bleu_4": round(bleu.precisions[3], 2) if len(bleu.precisions) > 3 else 0.0,
            "bp": round(bleu.bp, 4),
            "signature": sig,
        }
    except ImportError:
        # Fallback tính BLEU bằng NLTK nếu máy local chưa cài sacrebleu
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

        hyp_tokens = [h.strip().split() for h in hypotheses]
        ref_tokens = [[r.strip().split()] for r in references]
        smooth = SmoothingFunction().method1

        # Tính BLEU tổng (tương đương 4-gram)
        score = corpus_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth) * 100
        p1 = corpus_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth) * 100
        p2 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0, 1, 0, 0), smoothing_function=smooth) * 100
        p3 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0, 0, 1, 0), smoothing_function=smooth) * 100
        p4 = corpus_bleu(ref_tokens, hyp_tokens, weights=(0, 0, 0, 1), smoothing_function=smooth) * 100

        # Brevity penalty xấp xỉ
        hyp_len = sum(len(h) for h in hyp_tokens)
        ref_len = sum(len(r[0]) for r in ref_tokens)
        bp = 1.0 if hyp_len >= ref_len else round(pow(2.71828, 1.0 - ref_len / max(1, hyp_len)), 4)

        return {
            "bleu": round(score, 2),
            "bleu_1": round(p1, 2),
            "bleu_2": round(p2, 2),
            "bleu_3": round(p3, 2),
            "bleu_4": round(p4, 2),
            "bp": bp,
            "signature": "nltk-fallback",
        }
