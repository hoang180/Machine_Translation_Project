"""
Module: utils/visualize.py
Phụ trách: Phần 1 (Lê)
Cài đặt: Hàm vẽ và lưu Attention Heatmap gióng hàng từ-từ giữa tiếng Anh và tiếng Việt.
Hỗ trợ cả Seq2seq (1 head) và Transformer (nhiều heads / mean các heads).
Lưu ảnh chất lượng cao 300 DPI theo đúng Contract Mục 7.3.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_attention_heatmap(
    attn_matrix,
    src_tokens: list[str],
    tgt_tokens: list[str],
    save_path: str = None,
    head_idx: int | str = "mean",
    title: str = "Attention Alignment Heatmap (En -> Vi)",
):
    """
    Vẽ ma trận Attention Heatmap:
    Args:
        attn_matrix: Tensor hoặc numpy array:
                     - Shape: (tgt_len, src_len) hoặc (num_heads, tgt_len, src_len)
        src_tokens: Danh sách token câu nguồn tiếng Anh (trục hoành X)
        tgt_tokens: Danh sách token câu đích tiếng Việt (trục tung Y)
        save_path: Đường dẫn file ảnh (.png hoặc .pdf) để lưu
        head_idx: 'mean' (trung bình các head) hoặc số nguyên 0..H-1 (chọn head cụ thể)
        title: Tiêu đề biểu đồ
    """
    # 1. Chuyển tensor sang numpy
    if isinstance(attn_matrix, torch.Tensor):
        attn_matrix = attn_matrix.detach().cpu().numpy()

    # 2. Xử lý số chiều
    if attn_matrix.ndim == 3:
        # shape: (num_heads, tgt_len, src_len)
        num_heads, t_len, s_len = attn_matrix.shape
        if head_idx == "mean":
            attn_2d = attn_matrix.mean(axis=0)
            plot_title = f"{title} (Mean of {num_heads} Heads)"
        else:
            idx = int(head_idx)
            attn_2d = attn_matrix[idx]
            plot_title = f"{title} (Head #{idx + 1}/{num_heads})"
    elif attn_matrix.ndim == 2:
        # shape: (tgt_len, src_len)
        attn_2d = attn_matrix
        plot_title = title
    else:
        raise ValueError(f"Shape của attn_matrix không hợp lệ: {attn_matrix.shape}")

    # Cắt ma trận khớp đúng với số lượng tokens truyền vào
    attn_2d = attn_2d[: len(tgt_tokens), : len(src_tokens)]

    # 3. Vẽ biểu đồ heatmap
    plt.figure(figsize=(max(8, len(src_tokens) * 0.5), max(6, len(tgt_tokens) * 0.4)))
    sns.set_theme(style="white")

    ax = sns.heatmap(
        attn_2d,
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        cmap="YlGnBu",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="lightgray",
    )

    # Đưa trục X lên phía trên đầu biểu đồ cho giống bài báo gốc Attention Is All You Need
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.xticks(rotation=45, ha="left", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.xlabel("Câu nguồn (Tiếng Anh)", fontsize=12, labelpad=10)
    plt.ylabel("Câu đích (Tiếng Việt)", fontsize=12, labelpad=10)
    plt.title(plot_title, fontsize=13, pad=20, weight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Đã lưu biểu đồ Attention Heatmap vào: {save_path}")

    plt.close()
