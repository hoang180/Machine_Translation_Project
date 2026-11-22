"""
Unit Test: Kiểm thử tính đúng đắn và chuẩn Contract của mô hình Seq2seq (Hoàng - Phần 2).
Kiểm tra:
1. get_dummy_batch() từ utils/common.py
2. Forward pass với Luong Attention (return_attention=False và True)
3. Forward pass với Bahdanau Attention
4. Tính CrossEntropyLoss và Backward pass (gradient flow)
5. Chuẩn Contract encode() và decode_step() phục vụ Beam Search
"""

import sys
import os

# Thêm đường dẫn gốc vào sys.path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from utils.common import get_dummy_batch, set_seed, PAD_IDX
from models.seq2seq import Seq2seqModel


def test_seq2seq_pipeline():
    print("=" * 60)
    print("BẮT ĐẦU KIỂM THỬ MÔ HÌNH SEQ2SEQ (HOÀNG - PHẦN 2)")
    print("=" * 60)

    set_seed(42)

    # 1. Tạo Dummy Batch
    batch_size = 4
    src_len = 10
    tgt_len = 12
    src_vocab = 1000
    tgt_vocab = 1200

    batch = get_dummy_batch(
        batch_size=batch_size,
        src_len=src_len,
        tgt_len=tgt_len,
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
    )

    src = batch["src"]
    tgt = batch["tgt"]
    src_mask = batch["src_padding_mask"]
    src_lengths = batch["src_lengths"]

    tgt_input = tgt[:, :-1]  # shape: (B, tgt_len - 1) = (4, 11)
    tgt_label = tgt[:, 1:]   # shape: (B, tgt_len - 1) = (4, 11)

    print(f"✓ Dummy batch tạo thành công:")
    print(f"  - src shape: {src.shape}")
    print(f"  - tgt_input shape: {tgt_input.shape}")
    print(f"  - tgt_label shape: {tgt_label.shape}")
    print(f"  - src_mask shape: {src_mask.shape}")

    # 2. Khởi tạo Seq2seq với Luong Attention
    model_luong = Seq2seqModel(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        embed_size=64,
        enc_hidden_size=128,
        dec_hidden_size=128,
        num_layers=2,
        dropout=0.1,
        pad_idx=PAD_IDX,
        attention_type="luong",
        luong_method="general",
    )

    # 3. Test forward pass bình thường (return_attention=False)
    logits = model_luong(src, tgt_input, src_mask=src_mask, src_lengths=src_lengths, return_attention=False)
    expected_logits_shape = (batch_size, tgt_len - 1, tgt_vocab)
    assert logits.shape == expected_logits_shape, f"Lỗi shape logits: {logits.shape} != {expected_logits_shape}"
    print(f"✓ Test 1: Forward Luong (return_attention=False) PASSED! Shape: {logits.shape}")

    # 4. Test forward pass với return_attention=True
    logits_attn, attn_weights = model_luong(src, tgt_input, src_mask=src_mask, return_attention=True)
    expected_attn_shape = (batch_size, 1, tgt_len - 1, src_len)
    assert attn_weights.shape == expected_attn_shape, f"Lỗi shape attention: {attn_weights.shape} != {expected_attn_shape}"
    print(f"✓ Test 2: Forward Luong (return_attention=True) PASSED! Attention Shape: {attn_weights.shape}")

    # Kiểm tra Attention weight tại các vị trí <pad> có bị triệt tiêu về 0 không
    for b in range(batch_size):
        pad_indices = (src_mask[b] == True).nonzero(as_tuple=True)[0]
        if len(pad_indices) > 0:
            pad_attn_sum = attn_weights[b, 0, :, pad_indices].sum().item()
            assert pad_attn_sum < 1e-4, f"Lỗi: Trọng số attention tại token <pad> chưa được triệt tiêu về 0! Sum: {pad_attn_sum}"
    print("✓ Test 3: Attention Masking PASSED (Trọng số tại vị trí <pad> = 0 tuyệt đối)!")

    # 5. Test Loss & Backward Pass
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    loss = criterion(logits.reshape(-1, tgt_vocab), tgt_label.reshape(-1))
    loss.backward()
    print(f"✓ Test 4: Loss & Backward Pass PASSED! Loss = {loss.item():.4f}")

    # 6. Test Chuẩn Contract encode() và decode_step() cho Beam Search
    encoder_state = model_luong.encode(src, src_mask=src_mask)
    assert "encoder_outputs" in encoder_state
    assert "hidden" in encoder_state
    assert "cell" in encoder_state
    print(f"✓ Test 5: Hàm encode() PASSED! encoder_outputs shape: {encoder_state['encoder_outputs'].shape}")

    # Thử gọi 1 bước decode_step
    cur_tgt_seq = tgt[:, :1]  # Bắt đầu chỉ có token <sos>, shape (B, 1)
    next_logits, updated_state, step_attn = model_luong.decode_step(
        tgt_seq=cur_tgt_seq,
        encoder_state=encoder_state,
        return_attention=True,
    )
    assert next_logits.shape == (batch_size, tgt_vocab), f"Lỗi decode_step logits: {next_logits.shape}"
    assert step_attn.shape == (batch_size, 1, src_len), f"Lỗi decode_step attn: {step_attn.shape}"
    print(f"✓ Test 6: Hàm decode_step() PASSED! next_logits shape: {next_logits.shape}, step_attn shape: {step_attn.shape}")

    # 7. Test với Bahdanau Attention
    model_bahdanau = Seq2seqModel(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        embed_size=64,
        enc_hidden_size=128,
        dec_hidden_size=128,
        num_layers=2,
        dropout=0.1,
        pad_idx=PAD_IDX,
        attention_type="bahdanau",
    )
    logits_b, attn_b = model_bahdanau(src, tgt_input, src_mask=src_mask, return_attention=True)
    assert logits_b.shape == expected_logits_shape
    assert attn_b.shape == expected_attn_shape
    print(f"✓ Test 7: Bahdanau Attention PASSED! Shape: {logits_b.shape}")

    print("=" * 60)
    print("TẤT CẢ 7 BÀI TEST ĐỀU ĐÃ VƯỢT QUA XUẤT SẮC 100%!")
    print("=" * 60)


if __name__ == "__main__":
    test_seq2seq_pipeline()

