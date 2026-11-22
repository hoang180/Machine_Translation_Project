"""
Module: utils/beam_search.py
Phụ trách: Hoàng (Phần 2)
Cài đặt: Thuật toán Greedy Decode và Beam Search kèm Length Penalty (alpha) dùng chung cho cả nhóm.
Tuân thủ chuẩn Contract Mục 7.1 của file Quy ước 01.
Giao tiếp hoàn toàn qua 2 phương thức: model.encode() và model.decode_step().
"""

import torch
import torch.nn.functional as F
from utils.common import SOS_IDX, EOS_IDX


def greedy_decode(
    model,
    src_tensor: torch.Tensor,
    src_mask: torch.Tensor = None,
    max_len: int = 60,
    sos_idx: int = SOS_IDX,
    eos_idx: int = EOS_IDX,
    repetition_penalty: float = 1.0,
    device: str = "cpu",
) -> tuple[list[int], torch.Tensor | None]:
    """
    Giải mã tham lam (Greedy Search):
    Tại mỗi bước thời gian, chọn token có xác suất cao nhất argmax P(w | y_{<t}, x).

    Args:
        model: Mô hình Seq2seq hoặc Transformer tuân thủ Contract Mục 6.
        src_tensor: Tensor câu nguồn shape (1, src_len).
        src_mask: Tensor padding mask shape (1, src_len).
        max_len: Chiều dài tối đa sinh ra.
        sos_idx: Index token bắt đầu câu (mặc định 2).
        eos_idx: Index token kết thúc câu (mặc định 3).
        device: 'cpu' hoặc 'cuda'.

    Returns:
        pred_tokens: list[int] chứa các token ID dự đoán (đã lược bỏ <sos> và <eos>).
        attn_matrix: Tensor shape (num_heads, pred_len, src_len) trên CPU để vẽ Heatmap,
                     hoặc None nếu mô hình không trả về attention.
    """
    model.eval()
    src_tensor = src_tensor.to(device)
    if src_mask is not None:
        src_mask = src_mask.to(device)

    with torch.no_grad():
        # 1. Chạy Encoder 1 lần duy nhất cho câu nguồn
        encoder_state = model.encode(src_tensor, src_mask=src_mask)

        # 2. Bắt đầu với token <sos>
        cur_seq = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
        pred_tokens = []
        attentions = []

        for _ in range(max_len):
            # Gọi decode_step
            next_logits, encoder_state, step_attn = model.decode_step(
                tgt_seq=cur_seq,
                encoder_state=encoder_state,
                return_attention=True,
            )

            # Áp dụng Repetition Penalty (nếu có)
            if repetition_penalty > 1.0:
                for prev_tok in set(pred_tokens):
                    if next_logits[0, prev_tok] > 0:
                        next_logits[0, prev_tok] /= repetition_penalty
                    else:
                        next_logits[0, prev_tok] *= repetition_penalty

            # Lấy token có logit cao nhất: (1, vocab_size) -> scalar
            next_token = next_logits.argmax(dim=-1).item()

            if step_attn is not None:
                # step_attn shape: (1, num_heads, src_len) -> lưu CPU
                attentions.append(step_attn.squeeze(0).cpu())

            # Nếu gặp token kết thúc <eos> thì dừng
            if next_token == eos_idx:
                break

            pred_tokens.append(next_token)

            # Ghép token mới vào chuỗi đã sinh: (1, cur_len + 1)
            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            cur_seq = torch.cat([cur_seq, next_token_tensor], dim=1)

        # Định dạng ma trận attention: (num_heads, pred_len, src_len)
        if len(attentions) > 0 and len(pred_tokens) > 0:
            # attentions có len = số bước sinh (kể cả bước sinh eos)
            # Chỉ lấy tương ứng số token trong pred_tokens
            attn_list = attentions[: len(pred_tokens)]
            # stack theo chiều thời gian (pred_len):
            # Mỗi phần tử là (num_heads, src_len) -> stack dim=1 thành (num_heads, pred_len, src_len)
            attn_matrix = torch.stack(attn_list, dim=1)
        else:
            attn_matrix = None

    return pred_tokens, attn_matrix


def beam_search_decode(
    model,
    src_tensor: torch.Tensor,
    src_mask: torch.Tensor = None,
    beam_size: int = 4,
    max_len: int = 60,
    alpha: float = 0.6,
    repetition_penalty: float = 1.0,
    sos_idx: int = SOS_IDX,
    eos_idx: int = EOS_IDX,
    device: str = "cpu",
) -> tuple[list[int], torch.Tensor | None]:
    """
    Giải mã chùm tia (Beam Search) có áp dụng Phạt độ dài (Length Penalty):
    - Duy trì k giả thuyết có điểm log probability tích lũy cao nhất.
    - Công thức chuẩn hóa độ dài của Google GNMT (Wu et al., 2016):
        LP(Y) = ((5 + len(Y)) / 6) ^ alpha
        Score(Y) = sum(log_prob) / LP(Y)

    Args:
        model: Mô hình Seq2seq hoặc Transformer tuân thủ Contract Mục 6.
        src_tensor: Tensor câu nguồn shape (1, src_len).
        src_mask: Tensor padding mask shape (1, src_len).
        beam_size: Độ rộng chùm tia k (mặc định 4).
        max_len: Chiều dài tối đa sinh ra (mặc định 60).
        alpha: Hệ số Length Penalty (mặc định 0.6 theo Contract).
        sos_idx: Index token <sos> (mặc định 2).
        eos_idx: Index token <eos> (mặc định 3).
        device: 'cpu' hoặc 'cuda'.

    Returns:
        best_hypothesis: list[int] chứa các token ID tốt nhất (đã lược bỏ <sos> và <eos>).
        attn_matrix: Tensor shape (num_heads, pred_len, src_len) trên CPU của câu tốt nhất.
    """
    model.eval()
    src_tensor = src_tensor.to(device)
    if src_mask is not None:
        src_mask = src_mask.to(device)

    with torch.no_grad():
        # 1. Chạy Encoder 1 lần duy nhất cho câu nguồn
        encoder_state = model.encode(src_tensor, src_mask=src_mask)

        # Cấu trúc 1 nhánh Beam (Hypothesis):
        # (cumulative_log_prob, [token_ids], encoder_state, [attn_tensors])
        beams = [
            (
                0.0,
                [sos_idx],
                encoder_state,
                [],
            )
        ]

        completed_beams = []

        for step in range(max_len):
            new_candidates = []

            for cum_log_prob, tokens, state, attns in beams:
                # Nếu nhánh này đã sinh ra <eos> từ bước trước thì cho vào completed
                if tokens[-1] == eos_idx:
                    completed_beams.append((cum_log_prob, tokens, attns))
                    continue

                # Chuỗi token đưa vào decode_step: shape (1, cur_len)
                tgt_seq = torch.tensor([tokens], dtype=torch.long, device=device)

                # Chạy decode_step
                step_logits, new_state, step_attn = model.decode_step(
                    tgt_seq=tgt_seq,
                    encoder_state=state,
                    return_attention=True,
                )

                # Áp dụng Repetition Penalty (nếu có)
                if repetition_penalty > 1.0:
                    for prev_tok in set(tokens[1:]):
                        if step_logits[0, prev_tok] > 0:
                            step_logits[0, prev_tok] /= repetition_penalty
                        else:
                            step_logits[0, prev_tok] *= repetition_penalty

                # Log softmax để lấy log probabilities: (1, vocab_size)
                log_probs = F.log_softmax(step_logits, dim=-1).squeeze(0)

                # Lấy top-k token có xác suất cao nhất tại nhánh này
                topk_log_probs, topk_indices = torch.topk(log_probs, k=beam_size)

                for k in range(beam_size):
                    token_id = topk_indices[k].item()
                    token_log_prob = topk_log_probs[k].item()

                    new_cum_log_prob = cum_log_prob + token_log_prob
                    new_tokens = tokens + [token_id]

                    new_attns = list(attns)
                    if step_attn is not None:
                        new_attns.append(step_attn.squeeze(0).cpu())

                    new_candidates.append(
                        (new_cum_log_prob, new_tokens, new_state, new_attns)
                    )

            if not new_candidates:
                break

            # Sắp xếp các ứng viên mới theo cumulative_log_prob giảm dần
            # và giữ lại beam_size ứng viên tốt nhất cho bước tiếp theo
            new_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = new_candidates[:beam_size]

            # Kiểm tra điều kiện dừng sớm (nếu đã đủ số nhánh hoàn thành tốt)
            if len(completed_beams) >= beam_size:
                break

        # Nếu chưa có nhánh nào gặp <eos>, lấy luôn các nhánh hiện tại vào completed
        if len(completed_beams) == 0:
            for cum_log_prob, tokens, _, attns in beams:
                completed_beams.append((cum_log_prob, tokens, attns))

        # 3. Chuẩn hóa điểm số bằng Length Penalty:
        # LP(Y) = ((5 + len(Y)) / 6) ^ alpha
        best_score = -float("inf")
        best_hypothesis = []
        best_attns = []

        for cum_log_prob, tokens, attns in completed_beams:
            # Độ dài câu dịch thực tế (bỏ <sos>)
            hyp_len = max(1, len(tokens) - 1)
            length_penalty = ((5.0 + hyp_len) / 6.0) ** alpha
            normalized_score = cum_log_prob / length_penalty

            if normalized_score > best_score:
                best_score = normalized_score
                best_hypothesis = tokens
                best_attns = attns

        # 4. Làm sạch đầu ra: bỏ token <sos> và <eos>
        cleaned_tokens = [t for t in best_hypothesis if t not in (sos_idx, eos_idx)]

        # Định dạng ma trận Attention của câu tốt nhất: (num_heads, pred_len, src_len)
        if len(best_attns) > 0 and len(cleaned_tokens) > 0:
            attn_list = best_attns[: len(cleaned_tokens)]
            attn_matrix = torch.stack(attn_list, dim=1)
        else:
            attn_matrix = None

    return cleaned_tokens, attn_matrix

