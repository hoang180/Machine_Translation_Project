"""
Script: train_seq2seq.py
Phụ trách: Hoàng (Phần 2)
Chức năng: Huấn luyện mô hình Seq2seq LSTM + Attention (Luong/Bahdanau) cho bài toán Dịch máy Anh - Việt.
Tuân thủ 100% Interface Contract (Mục 6, 8, 9 của file Quy ước 01).
Hỗ trợ:
- Đọc cấu hình từ file YAML (configs/seq2seq_config.yaml)
- Chế độ nạp dữ liệu thật (từ data/dataset.py của Lê) HOẶC chế độ giả lập (--dummy) để test loop ngay
- Teacher Forcing decaying theo từng epoch
- Gradient Clipping chống exploding gradient
- Lưu Checkpoint best_model.pt theo chuẩn dictionary Mục 8.1
- Đồng bộ log chỉ số lên Weights & Biases (W&B)
"""

import os
import sys
import time
import math
import argparse
try:
    import yaml
except ImportError:
    yaml = None
import torch
import torch.nn as nn
from tqdm import tqdm

# Thêm thư mục gốc vào path để import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.common import (
    PAD_IDX,
    UNK_IDX,
    SOS_IDX,
    EOS_IDX,
    set_seed,
    get_dummy_batch,
)
from models.seq2seq import Seq2seqModel
from utils.beam_search import greedy_decode


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Seq2seq Anh - Việt (Hoàng - Phần 2)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/seq2seq_config.yaml",
        help="Đường dẫn đến file cấu hình YAML",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Chạy ở chế độ dữ liệu giả lập (để test toàn bộ training pipeline khi chưa có data từ Lê)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Ghi đè số epoch trong file config",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Ghi đè batch size trong file config",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Tắt ghi log lên Weights & Biases",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["word", "bpe", "spm"],
        default=None,
        help="Ghi đè loại tokenizer ('word', 'bpe', 'spm')",
    )
    parser.add_argument(
        "--attention",
        type=str,
        choices=["luong", "bahdanau"],
        default=None,
        help="Ghi đè cơ chế attention ('luong', 'bahdanau')",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Ghi đè thư mục lưu checkpoint",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Đọc file cấu hình YAML. Nếu môi trường chưa cài PyYAML, tự động dùng fallback parser
    để không bị gián đoạn khi chạy trên máy local.
    """
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback parser nhẹ cho file YAML chuẩn 2 cấp độ
        config = {}
        cur_section = None
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Xử lý inline comment
                if "#" in line:
                    line = line.split("#")[0].strip()
                if line.endswith(":"):
                    cur_section = line[:-1].strip()
                    config[cur_section] = {}
                elif ":" in line and cur_section:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    # Ép kiểu dữ liệu tự động
                    if val.lower() == "true":
                        val = True
                    elif val.lower() == "false":
                        val = False
                    elif val.isdigit():
                        val = int(val)
                    else:
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    config[cur_section][key] = val
        return config


def get_autocast(device_type: str, enabled: bool):
    """Hàm helper tương thích cả PyTorch 1.x và 2.x mà không bị warning deprecated."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def get_grad_scaler(device_type: str):
    """Hàm helper khởi tạo GradScaler chuẩn không bị warning."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type)
    return torch.cuda.amp.GradScaler()


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip_grad: float,
    teacher_forcing_ratio: float,
    device: torch.device,
    tgt_vocab_size: int,
    scaler=None,
) -> float:
    """
    Huấn luyện mô hình qua 1 epoch (hỗ trợ Automatic Mixed Precision AMP để tăng tốc GPU 3x).
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    use_amp = (scaler is not None) and (device.type == "cuda")

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        src = batch["src"].to(device, non_blocking=True)
        tgt = batch["tgt"].to(device, non_blocking=True)
        src_mask = batch["src_padding_mask"].to(device, non_blocking=True)
        src_lengths = batch.get("src_lengths", None)
        if src_lengths is not None:
            src_lengths = src_lengths.to(device, non_blocking=True)

        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        # Chạy Forward pass với Mixed Precision (FP16) trên GPU
        with get_autocast("cuda", enabled=use_amp):
            logits = model(
                src=src,
                tgt_input=tgt_input,
                src_mask=src_mask,
                src_lengths=src_lengths,
                teacher_forcing_ratio=teacher_forcing_ratio,
                return_attention=False,
            )

            loss = criterion(
                logits.reshape(-1, tgt_vocab_size),
                tgt_label.reshape(-1),
            )

        # Backward pass có scale gradient
        if use_amp:
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "tf": f"{teacher_forcing_ratio:.2f}"})

    return total_loss / max(1, num_batches)


def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    tgt_vocab_size: int,
) -> tuple[float, float]:
    """
    Đánh giá mô hình trên tập validation (dùng autocast để tăng tốc).
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            src = batch["src"].to(device, non_blocking=True)
            tgt = batch["tgt"].to(device, non_blocking=True)
            src_mask = batch["src_padding_mask"].to(device, non_blocking=True)
            src_lengths = batch.get("src_lengths", None)
            if src_lengths is not None:
                src_lengths = src_lengths.to(device, non_blocking=True)

            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]

            with get_autocast("cuda", enabled=use_amp):
                logits = model(
                    src=src,
                    tgt_input=tgt_input,
                    src_mask=src_mask,
                    src_lengths=src_lengths,
                    teacher_forcing_ratio=0.0,
                    return_attention=False,
                )

                loss = criterion(
                    logits.reshape(-1, tgt_vocab_size),
                    tgt_label.reshape(-1),
                )
            total_loss += loss.item()

    avg_loss = total_loss / max(1, num_batches)
    perplexity = math.exp(min(avg_loss, 100))
    return avg_loss, perplexity


class DummyDataLoader:
    """DataLoader giả lập để test vòng lặp huấn luyện độc lập."""

    def __init__(self, num_batches=10, batch_size=8, src_len=16, tgt_len=20, src_vocab=5000, tgt_vocab=5000):
        self.num_batches = num_batches
        self.batches = [
            get_dummy_batch(
                batch_size=batch_size,
                src_len=src_len,
                tgt_len=tgt_len,
                src_vocab_size=src_vocab,
                tgt_vocab_size=tgt_vocab,
            )
            for _ in range(num_batches)
        ]

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return iter(self.batches)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Ghi đè cấu hình từ CLI nếu có
    if args.tokenizer is not None:
        config["tokenizer"]["type"] = args.tokenizer
    if args.attention is not None:
        config["model"]["attention_type"] = args.attention
    if args.save_dir is not None:
        config["training"]["save_dir"] = args.save_dir

    # Đồng bộ tên run W&B theo quy ước Mục 8.2: seq2seq-<attention>-<tokenizer>
    if args.tokenizer is not None or args.attention is not None:
        attn_name = config["model"].get("attention_type", "luong")
        tok_name = config["tokenizer"].get("type", "word")
        config["experiment"]["name"] = f"seq2seq-{attn_name}-{tok_name}"

    # 1. Cố định Seed để tái lập kết quả
    seed = config["experiment"].get("seed", 42)
    set_seed(seed)

    # 2. Thiết bị tính toán
    device_name = config["experiment"].get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"\n[INFO] Thiết bị sử dụng: {device}")

    # 3. Quản lý Logging W&B
    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            # Tắt prompt tương tác nếu không có API key để tránh bị treo trên Colab
            if not os.environ.get("WANDB_API_KEY") and not os.path.exists(os.path.expanduser("~/.netrc")):
                os.environ["WANDB_MODE"] = "disabled"
                use_wandb = False
            else:
                project_name = config["experiment"].get("project", "btl-dl-translation-iwslt15")
                run_name = config["experiment"].get("name", "seq2seq-luong-baseline")
                wandb.init(
                    project=project_name,
                    name=run_name,
                    config=config,
                )
                wandb_run = wandb
                print(f"[INFO] Đã kết nối Weights & Biases (Project: {project_name}, Run: {run_name})")
        except Exception as e:
            use_wandb = False

    # 4. Chuẩn bị DataLoader (Dữ liệu thật hoặc Dữ liệu giả lập)
    src_vocab_size = config["tokenizer"].get("src_vocab_size", 25000)
    tgt_vocab_size = config["tokenizer"].get("tgt_vocab_size", 25000)
    batch_size = args.batch_size or config["data"].get("batch_size", 64)

    is_dummy_mode = args.dummy
    train_loader = None
    val_loader = None

    if not is_dummy_mode:
        try:
            # Thử nạp DataLoader thật từ data/dataset.py của Lê
            from data.dataset import build_dataloader

            train_loader = build_dataloader(
                src_file=config["data"]["train_src"],
                tgt_file=config["data"]["train_tgt"],
                tokenizer_type=config["tokenizer"]["type"],
                batch_size=batch_size,
                max_seq_len=config["data"].get("max_seq_len", 64),
                is_train=True,
            )
            val_loader = build_dataloader(
                src_file=config["data"]["val_src"],
                tgt_file=config["data"]["val_tgt"],
                tokenizer_type=config["tokenizer"]["type"],
                batch_size=batch_size,
                max_seq_len=config["data"].get("max_seq_len", 64),
                is_train=False,
            )
            # Tự động đồng bộ vocab_size thực tế từ Tokenizer
            src_vocab_size = train_loader.dataset.src_tokenizer.vocab_size
            tgt_vocab_size = train_loader.dataset.tgt_tokenizer.vocab_size
            print(f"[INFO] Đã nạp thành công DataLoader thật từ module data/dataset.py! Vocab: En={src_vocab_size:,}, Vi={tgt_vocab_size:,}")
        except Exception as e:
            print(f"[WARN] Chưa thể nạp dữ liệu thật ({e}).")
            print("[INFO] Tự động chuyển sang chế độ DUMMY DATA để test hoàn chỉnh pipeline huấn luyện!")
            is_dummy_mode = True

    if is_dummy_mode:
        print("[INFO] Đang chạy với Dummy DataLoader (15 batches train, 5 batches val)...")
        # Giảm vocab tạm thời cho dummy để tiết kiệm bộ nhớ khi test
        src_vocab_size = min(src_vocab_size, 1000)
        tgt_vocab_size = min(tgt_vocab_size, 1000)
        train_loader = DummyDataLoader(num_batches=15, batch_size=batch_size, src_vocab=src_vocab_size, tgt_vocab=tgt_vocab_size)
        val_loader = DummyDataLoader(num_batches=5, batch_size=batch_size, src_vocab=src_vocab_size, tgt_vocab=tgt_vocab_size)

    # 5. Khởi tạo Mô hình Seq2seq
    model_cfg = config["model"]
    model = Seq2seqModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_size=model_cfg.get("embed_size", 256),
        enc_hidden_size=model_cfg.get("enc_hidden_size", 512),
        dec_hidden_size=model_cfg.get("dec_hidden_size", 512),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.3),
        pad_idx=PAD_IDX,
        attention_type=model_cfg.get("attention_type", "luong"),
        luong_method=model_cfg.get("luong_method", "general"),
    ).to(device)

    # Đếm số lượng tham số huấn luyện
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Tổng số tham số mô hình (Trainable Parameters): {total_params:,}")

    # 6. Khởi tạo Optimizer & Loss Criterion
    train_cfg = config["training"]
    lr = float(train_cfg.get("learning_rate", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Bắt buộc ignore_index = PAD_IDX (0) theo chuẩn Contract
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Khởi tạo GradScaler cho Automatic Mixed Precision (Tăng tốc Tensor Cores trên GPU T4)
    scaler = get_grad_scaler("cuda") if device.type == "cuda" else None
    if scaler:
        print("[INFO] Đã bật Automatic Mixed Precision (FP16 AMP) để tăng tốc GPU tối đa!")

    # Tham số huấn luyện
    epochs = args.epochs or train_cfg.get("epochs", 15)
    clip_grad = float(train_cfg.get("clip_grad", 1.0))
    initial_tf = float(train_cfg.get("teacher_forcing_ratio", 0.5))
    tf_decay = float(train_cfg.get("teacher_forcing_decay", 0.02))
    min_tf = float(train_cfg.get("min_teacher_forcing", 0.2))

    save_dir = train_cfg.get("save_dir", "checkpoints/seq2seq")
    os.makedirs(save_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(save_dir, "best_model.pt")

    best_val_loss = float("inf")
    patience = train_cfg.get("patience", 5)
    patience_counter = 0

    print("\n" + "=" * 70)
    print(f"BẮT ĐẦU HUẤN LUYỆN SEQ2SEQ ({epochs} EPOCHS)")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Tính Teacher Forcing decaying theo epoch
        cur_tf_ratio = max(min_tf, initial_tf - (epoch - 1) * tf_decay)

        # 1. Train 1 epoch
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            clip_grad=clip_grad,
            teacher_forcing_ratio=cur_tf_ratio,
            device=device,
            tgt_vocab_size=tgt_vocab_size,
            scaler=scaler,
        )

        # 2. Đánh giá Validation
        val_loss, val_ppl = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            tgt_vocab_size=tgt_vocab_size,
        )

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Thời gian: {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f} | "
            f"TF Ratio: {cur_tf_ratio:.2f}"
        )

        # 3. Đồng bộ lên W&B theo chuẩn Mục 8.2
        if use_wandb and wandb_run:
            wandb_run.log(
                {
                    "train/loss": train_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch_time_sec": epoch_time,
                    "train/teacher_forcing_ratio": cur_tf_ratio,
                    "val/loss": val_loss,
                    "val/perplexity": val_ppl,
                    "epoch": epoch,
                }
            )

        # 4. Lưu Checkpoint theo đúng chuẩn Dictionary Mục 8.1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint = {
                "model_type": "seq2seq",
                "tokenizer_type": config["tokenizer"].get("type", "word"),
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_bleu": 0.0,  # Sẽ được cập nhật khi chạy eval.py
                "config": config,
                "src_vocab_size": src_vocab_size,
                "tgt_vocab_size": tgt_vocab_size,
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(f"  --> [SAVED] Đã lưu mô hình tốt nhất vào: {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[INFO] Dừng sớm (Early Stopping) sau {epoch} epochs vì val_loss không cải thiện trong {patience} epochs liên tiếp.")
                break

    print("\n" + "=" * 70)
    print(f"HUẤN LUYỆN HOÀN TẤT! File checkpoint tốt nhất: {best_checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
