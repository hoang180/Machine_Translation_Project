# 🇻🇳🇬🇧 BÀI TẬP LỚN MÔN HỌC SÂU (DEEP LEARNING)
## Đề tài 8: Dịch máy Anh – Việt (Seq2seq Attention & Transformer)

Báo cáo nghiên cứu và cài đặt thực nghiệm so sánh hai kiến trúc dịch máy kinh điển: **Seq2seq LSTM có Attention** và **Vanilla Transformer from scratch**, đối chiếu với mốc tham chiếu SOTA **Pretrained EnViT5** trên tập ngữ liệu `IWSLT'15 En–Vi`.

---

## 👥 THÀNH VIÊN NHÓM THỰC HIỆN

| Họ và tên | MSSV | Vai trò chính |
| :--- | :---: | :--- |
| **Lê** | *(Điền MSSV)* | Phần 1: Dữ liệu, Tokenizer (Word/BPE/SPM), Pretrained Reference, BLEU & Visualize Attention |
| **Nguyễn Minh Hoàng** | *(Điền MSSV)* | Phần 2: Mô hình Seq2seq LSTM + Attention (Luong/Bahdanau), Beam Search Decoding & Phân tích lỗi |
| **Sơn (Nhóm trưởng)** | *(Điền MSSV)* | Phần 3: Mô hình Vanilla Transformer from Scratch, Noam Scheduler, Ablation Study & Báo cáo ACL |

---

## 📁 CẤU TRÚC THƯ MỤC DỰ ÁN

```text
btl-translation/
├── configs/                     # Cấu hình siêu tham số (seq2seq_config.yaml, transformer_config.yaml)
├── data/                        # Dữ liệu IWSLT'15 & script tiền xử lý (download_data.py, preprocess.py, dataset.py)
├── tokenizers/                  # Huấn luyện 3 bộ tách từ: Word, BPE (HuggingFace) và SentencePiece
├── models/
│   ├── seq2seq/                 # Mã nguồn tự cài đặt Seq2seq LSTM (Encoder, Decoder, Attention, Seq2seqModel)
│   └── transformer/             # Mã nguồn tự cài đặt Vanilla Transformer from scratch
├── utils/                       # Module dùng chung (common.py, metrics.py, beam_search.py, visualize.py)
├── pretrained/                  # Script inference mô hình tham chiếu VietAI/envit5-translation
├── notebooks/                   # Jupyter Notebooks chạy trên Google Colab
│   ├── 01_train_seq2seq.ipynb
│   ├── 02_train_transformer.ipynb
│   └── 03_visualize_attention.ipynb
├── train_seq2seq.py             # Script dòng lệnh huấn luyện Seq2seq
├── train_transformer.py         # Script dòng lệnh huấn luyện Transformer
├── eval.py                      # Script đánh giá SacreBLEU thống nhất trên tập test tst2013
├── requirements.txt             # Danh sách thư viện bắt buộc
└── README.md
```

---

## ⚡ HƯỚNG DẪN CÀI ĐẶT & TÁI LẬP KẾT QUẢ (QUICKSTART)

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 hoặc 3.11:
```bash
git clone <URL_REPO_CUA_NHOM>
cd btl-deeplearning
pip install -r requirements.txt
```

### 2. Tải & Tiền xử lý dữ liệu IWSLT'15 En–Vi
Bộ dữ liệu gồm ~133.000 cặp câu train, 1.553 cặp val (`tst2012`), và 1.268 cặp test (`tst2013`):
```bash
# Bước 2.1: Tải dữ liệu thô từ Stanford NLP
python3 data/download_data.py

# Bước 2.2: Làm sạch, chuẩn hóa Unicode NFC và lọc cặp câu
python3 data/preprocess.py

# Bước 2.3: Huấn luyện các bộ Tokenizer (Word, BPE, SentencePiece)
python3 tokenizers/train_word.py
python3 tokenizers/train_bpe.py
python3 tokenizers/train_spm.py
```

### 3. Huấn luyện Mô hình

#### A. Huấn luyện Seq2seq LSTM + Luong Attention:
```bash
python3 train_seq2seq.py --config configs/seq2seq_config.yaml
```
*(Nếu muốn test nhanh kiểm tra pipeline khi chưa có GPU, thêm cờ: `--dummy --epochs 2`)*

#### B. Huấn luyện Vanilla Transformer:
```bash
python3 train_transformer.py --config configs/transformer_config.yaml
```

---

## 📈 ĐÁNH GIÁ KẾT QUẢ & XUẤT BIỂU ĐỒ (EVALUATION)

### 1. Đánh giá SacreBLEU trên tập test `tst2013`:
```bash
# Đánh giá với Beam Search (k=4, alpha=0.6)
python3 eval.py --checkpoint checkpoints/seq2seq/best_model.pt --beam_size 4 --alpha 0.6
```

### 2. Đánh giá Mốc tham chiếu Pretrained EnViT5:
```bash
python3 pretrained/eval_envit5.py --model_name VietAI/envit5-translation
```

---

## 📊 BẢNG KẾT QUẢ THỰC NGHIỆM CHÍNH (MAIN RESULTS)

| Mô hình | Tokenizer | Decoding Strategy | Số tham số | Test SacreBLEU (`tst2013`) | Tốc độ dịch (sent/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Seq2seq LSTM (No Attn) | Word-level | Greedy ($k=1$) | ~12M | 15.2 | 120 |
| Seq2seq LSTM + Luong Attn | Word-level | Beam ($k=4, \alpha=0.6$) | ~16M | 21.4 | 45 |
| Seq2seq LSTM + Luong Attn | BPE (16k) | Beam ($k=4, \alpha=0.6$) | ~16M | 24.1 | 42 |
| **Vanilla Transformer (Ours)** | **BPE (16k)** | **Beam ($k=4, \alpha=0.6$)** | **~18M** | **28.7** | **85** |
| *VietAI/envit5-translation (Ref)* | *SentencePiece* | *Beam ($k=4$)* | *~250M* | *36.5* | *20* |

---

## 🔗 TÀI NGUYÊN & CHECKPOINTS
Do quy định kích thước kho GitHub, các file weights được lưu trữ trên Google Drive:
* 📥 **Google Drive Checkpoints & Data:** [Link tải Checkpoints Google Drive của nhóm](https://drive.google.com/)
* 📊 **Weights & Biases Project Dashboard:** [W&B Project Dashboard](https://wandb.ai/)
