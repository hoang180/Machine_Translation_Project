"""
Script: train_transformer.py
Phụ trách: Sơn (Phần 3 - Nhóm trưởng)
Chức năng: Huấn luyện Vanilla Transformer from Scratch cho bài toán Dịch máy Anh - Việt (IWSLT'15).
Tuân thủ 100% Interface Contract (Mục 6, 8, 9 của file Quy ước 01):
- Optimizer: Adam (beta1=0.9, beta2=0.98, eps=1e-9) + Noam Learning Rate Scheduler
- Loss: LabelSmoothingLoss (epsilon=0.1, padding_idx=0)
- Hỗ trợ chế độ --dummy để test pipeline độc lập
- Hỗ trợ ghi đè CLI (--tokenizer, --num_layers, --num_heads, --use_pe, --smoothing, ...) cho Ablation Study
- Log đầy đủ lên Weights & Biases (project: btl-dl-translation-iwslt15)
- Lưu checkpoint best_model.pt đúng chuẩn dictionary Mục 8.1 để eval.py đọc tự động
"""

import os
import sys
import json
import time
import math
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.common import PAD_IDX, SOS_IDX, EOS_IDX, set_seed, get_dummy_batch
from utils.lr_scheduler import NoamScheduler
from utils.loss import LabelSmoothingLoss
from utils.beam_search import greedy_decode
from utils.metrics import compute_bleu
from models.transformer import TransformerModel


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in ("yes", "true", "t", "1", "y"):
        return True
    if val in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Giá trị boolean không hợp lệ: {v}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện Vanilla Transformer Anh - Việt & Ablation Study (Sơn - Phần 3)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/transformer_config.yaml",
        help="Đường dẫn file cấu hình YAML",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["word", "bpe", "spm"],
        default=None,
        help="Ghi đè loại tokenizer (word | bpe | spm)",
    )
    parser.add_argument("--d_model", type=int, default=None, help="Ghi đè d_model")
    parser.add_argument("--num_layers", type=int, default=None, help="Ghi đè số layer Encoder/Decoder")
    parser.add_argument("--num_heads", type=int, default=None, help="Ghi đè số attention heads")
    parser.add_argument("--d_ff", type=int, default=None, help="Ghi đè kích thước FFN")
    parser.add_argument("--dropout", type=float, default=None, help="Ghi đè dropout")
    parser.add_argument(
        "--use_pe",
        type=str2bool,
        default=None,
        help="Bật/tắt Positional Encoding (true/false) cho Ablation Study",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=None,
        help="Ghi đè hệ số Label Smoothing (0.1 = mặc định, 0.0 = tắt cho Ablation)",
    )
    parser.add_argument("--warmup_steps", type=int, default=None, help="Ghi đè số bước warmup Noam")
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè số epoch")
    parser.add_argument("--batch_size", type=int, default=None, help="Ghi đè batch size")
    parser.add_argument("--grad_accum", type=int, default=None, help="Số bước tích lũy gradient")
    parser.add_argument(
        "--eval_bleu_samples",
        type=int,
        default=None,
        help="Số câu val dùng tính Greedy BLEU mỗi epoch (0 = tắt)",
    )
    parser.add_argument("--save_dir", type=str, default=None, help="Thư mục lưu checkpoint")
    parser.add_argument("--run_name", type=str, default=None, help="Tên run trên W&B")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Chạy chế độ dữ liệu giả lập (get_dummy_batch) để test pipeline",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Tắt ghi log lên Weights & Biases",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Đọc file cấu hình YAML (có fallback parser nhẹ nếu môi trường chưa cài PyYAML)."""
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        config = {}
        cur_section = None
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "#" in line:
                    line = line.split("#")[0].strip()
                if line.endswith(":"):
                    cur_section = line[:-1].strip()
                    config[cur_section] = {}
                elif ":" in line and cur_section:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
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


def apply_cli_overrides(config: dict, args) -> dict:
    """Áp dụng các tham số CLI vào config để checkpoint lưu chính xác cấu hình đã train."""
    if args.tokenizer is not None:
        config["tokenizer"]["type"] = args.tokenizer
    if args.d_model is not None:
        config["model"]["d_model"] = args.d_model
    if args.num_layers is not None:
        config["model"]["num_layers"] = args.num_layers
    if args.num_heads is not None:
        config["model"]["num_heads"] = args.num_heads
    if args.d_ff is not None:
        config["model"]["d_ff"] = args.d_ff
    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout
    if args.use_pe is not None:
        config["model"]["use_pe"] = args.use_pe
    if args.smoothing is not None:
        config["training"]["label_smoothing"] = args.smoothing
    if args.warmup_steps is not None:
        config["training"]["warmup_steps"] = args.warmup_steps
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        config["training"]["gradient_accumulation_steps"] = args.grad_accum
    if args.eval_bleu_samples is not None:
        config["training"]["eval_bleu_samples"] = args.eval_bleu_samples
    if args.save_dir is not None:
        config["training"]["save_dir"] = args.save_dir

    # Tự động đặt tên run theo quy ước Mục 8.2 nếu người dùng không truyền --run_name
    if args.run_name is not None:
        config["experiment"]["name"] = args.run_name
    else:
        tok = config["tokenizer"]["type"]
        m = config["model"]
        t = config["training"]
        tags = []
        if m["num_layers"] != 4 or m["num_heads"] != 8:
            tags.append(f"L{m['num_layers']}H{m['num_heads']}")
        if not m.get("use_pe", True):
            tags.append("noPE")
        if float(t.get("label_smoothing", 0.1)) == 0.0:
            tags.append("noLS")
        suffix = "ablation-" + "-".join(tags) if tags else "base"
        config["experiment"]["name"] = f"transformer-{tok}-{suffix}"

    return config


def get_autocast(device_type: str, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def get_grad_scaler(device_type: str):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type)
    return torch.cuda.amp.GradScaler()


class DummyDataLoader:
    """DataLoader giả lập để test vòng lặp huấn luyện Transformer độc lập."""

    def __init__(self, num_batches=10, batch_size=8, src_len=16, tgt_len=20, src_vocab=1000, tgt_vocab=1000):
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


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    criterion: nn.Module,
    clip_grad: float,
    grad_accum_steps: int,
    device: torch.device,
    scaler=None,
) -> float:
    """Huấn luyện Transformer qua 1 epoch (hỗ trợ FP16 AMP & Gradient Accumulation)."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    use_amp = (scaler is not None) and (device.type == "cuda")

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc="Training", leave=False)

    for step_idx, batch in enumerate(pbar, start=1):
        src = batch["src"].to(device, non_blocking=True)
        tgt = batch["tgt"].to(device, non_blocking=True)
        src_mask = batch["src_padding_mask"].to(device, non_blocking=True)

        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        with get_autocast("cuda", enabled=use_amp):
            logits = model(src, tgt_input, src_mask, return_attention=False)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_label.reshape(-1))
            scaled_loss = loss / grad_accum_steps

        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        is_update_step = (step_idx % grad_accum_steps == 0) or (step_idx == num_batches)
        if is_update_step:
            if use_amp:
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # Chỉ tăng bước Noam khi optimizer thực sự cập nhật (không bị skip do inf/nan)
                if scaler.get_scale() >= prev_scale:
                    scheduler.step()
            else:
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    ce_criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Đánh giá validation loss (theo criterion) và perplexity chuẩn (theo Cross-Entropy)."""
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    num_batches = len(dataloader)
    use_amp = device.type == "cuda"

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        src = batch["src"].to(device, non_blocking=True)
        tgt = batch["tgt"].to(device, non_blocking=True)
        src_mask = batch["src_padding_mask"].to(device, non_blocking=True)

        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        with get_autocast("cuda", enabled=use_amp):
            logits = model(src, tgt_input, src_mask, return_attention=False)
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_label = tgt_label.reshape(-1)
            loss = criterion(flat_logits, flat_label)
            ce_loss = ce_criterion(flat_logits.float(), flat_label)

        total_loss += loss.item()
        total_ce += ce_loss.item()

    avg_loss = total_loss / max(1, num_batches)
    avg_ce = total_ce / max(1, num_batches)
    perplexity = math.exp(min(avg_ce, 100.0))
    return avg_loss, perplexity


@torch.no_grad()
def evaluate_val_bleu(
    model: nn.Module,
    val_loader,
    tgt_tokenizer,
    device: torch.device,
    max_samples: int = 200,
    max_len: int = 60,
) -> float:
    """Tính SacreBLEU nhanh bằng Greedy Decode trên tối đa `max_samples` câu validation."""
    if max_samples <= 0 or tgt_tokenizer is None:
        return 0.0

    model.eval()
    hypotheses = []
    references = []

    for batch in val_loader:
        src = batch["src"].to(device)
        src_mask = batch["src_padding_mask"].to(device)
        tgt_raw = batch["tgt_raw"]

        for i in range(src.size(0)):
            pred_ids, _ = greedy_decode(
                model=model,
                src_tensor=src[i : i + 1],
                src_mask=src_mask[i : i + 1],
                max_len=max_len,
                sos_idx=SOS_IDX,
                eos_idx=EOS_IDX,
                device=device,
            )
            hypotheses.append(tgt_tokenizer.decode(pred_ids, skip_special_tokens=True))
            references.append(tgt_raw[i])
            if len(hypotheses) >= max_samples:
                break
        if len(hypotheses) >= max_samples:
            break

    if not hypotheses:
        return 0.0
    return compute_bleu(hypotheses, references)["bleu"]


def main():
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)

    # 1. Cố định Seed
    seed = int(config["experiment"].get("seed", 42))
    set_seed(seed)

    # 2. Thiết bị
    device_name = config["experiment"].get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"\n[INFO] Thiết bị sử dụng: {device}")
    print(f"[INFO] Run name: {config['experiment']['name']}")

    # 3. Weights & Biases
    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb

            if not os.environ.get("WANDB_API_KEY") and not os.path.exists(os.path.expanduser("~/.netrc")):
                os.environ["WANDB_MODE"] = "disabled"
                use_wandb = False
            else:
                project_name = config["experiment"].get("project", "btl-dl-translation-iwslt15")
                run_name = config["experiment"]["name"]
                wandb.init(project=project_name, name=run_name, config=config)
                wandb_run = wandb
                print(f"[INFO] Đã kết nối W&B (Project: {project_name}, Run: {run_name})")
        except Exception as e:
            print(f"[WARN] Không thể kết nối W&B ({e}). Tắt logging W&B.")
            use_wandb = False

    # 4. DataLoader (dữ liệu thật hoặc --dummy)
    src_vocab_size = int(config["tokenizer"].get("src_vocab_size", 16000))
    tgt_vocab_size = int(config["tokenizer"].get("tgt_vocab_size", 16000))
    batch_size = int(config["data"].get("batch_size", 128))
    max_seq_len = int(config["data"].get("max_seq_len", 64))
    tokenizer_type = config["tokenizer"].get("type", "bpe")

    is_dummy_mode = args.dummy
    tgt_tokenizer = None

    if not is_dummy_mode:
        try:
            from data.dataset import build_dataloader

            train_loader = build_dataloader(
                src_file=config["data"]["train_src"],
                tgt_file=config["data"]["train_tgt"],
                tokenizer_type=tokenizer_type,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                is_train=True,
                num_workers=int(config["data"].get("num_workers", 2)),
            )
            val_loader = build_dataloader(
                src_file=config["data"]["val_src"],
                tgt_file=config["data"]["val_tgt"],
                tokenizer_type=tokenizer_type,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                is_train=False,
                num_workers=int(config["data"].get("num_workers", 2)),
            )
            src_vocab_size = train_loader.dataset.src_tokenizer.vocab_size
            tgt_vocab_size = train_loader.dataset.tgt_tokenizer.vocab_size
            tgt_tokenizer = train_loader.dataset.tgt_tokenizer
            config["tokenizer"]["src_vocab_size"] = src_vocab_size
            config["tokenizer"]["tgt_vocab_size"] = tgt_vocab_size
            print(
                f"[INFO] Đã nạp DataLoader thật (tokenizer={tokenizer_type})! "
                f"Vocab: En={src_vocab_size:,}, Vi={tgt_vocab_size:,}"
            )
        except Exception as e:
            print(f"[WARN] Chưa thể nạp dữ liệu thật ({e}). Tự động chuyển sang DUMMY DATA!")
            is_dummy_mode = True

    if is_dummy_mode:
        print("[INFO] Đang chạy với Dummy DataLoader (12 batches train, 4 batches val)...")
        src_vocab_size = min(src_vocab_size, 1000)
        tgt_vocab_size = min(tgt_vocab_size, 1000)
        train_loader = DummyDataLoader(
            num_batches=12, batch_size=min(batch_size, 16), src_vocab=src_vocab_size, tgt_vocab=tgt_vocab_size
        )
        val_loader = DummyDataLoader(
            num_batches=4, batch_size=min(batch_size, 16), src_vocab=src_vocab_size, tgt_vocab=tgt_vocab_size
        )

    # 5. Khởi tạo TransformerModel
    m_cfg = config["model"]
    d_model = int(m_cfg.get("d_model", 256))
    num_heads = int(m_cfg.get("num_heads", 8))
    num_layers = int(m_cfg.get("num_layers", 4))
    d_ff = int(m_cfg.get("d_ff", 1024))
    dropout = float(m_cfg.get("dropout", 0.1))
    max_len = int(m_cfg.get("max_len", 512))
    use_pe = bool(m_cfg.get("use_pe", True))
    tie_weights = bool(m_cfg.get("tie_weights", False))

    model = TransformerModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
        pad_idx=PAD_IDX,
        use_pe=use_pe,
        tie_weights=tie_weights,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[INFO] Transformer (N={num_layers}, H={num_heads}, d_model={d_model}, d_ff={d_ff}, "
        f"use_pe={use_pe}) | Trainable params: {total_params:,}"
    )

    # 6. Optimizer (Adam chuẩn Vaswani), Noam Scheduler & Label Smoothing Loss
    t_cfg = config["training"]
    beta1 = float(t_cfg.get("beta1", 0.9))
    beta2 = float(t_cfg.get("beta2", 0.98))
    eps = float(t_cfg.get("eps", 1e-9))
    warmup_steps = int(t_cfg.get("warmup_steps", 2000))
    noam_factor = float(t_cfg.get("noam_factor", 1.0))
    smoothing = float(t_cfg.get("label_smoothing", 0.1))

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(beta1, beta2), eps=eps)
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, factor=noam_factor)
    criterion = LabelSmoothingLoss(vocab_size=tgt_vocab_size, padding_idx=PAD_IDX, smoothing=smoothing)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    scaler = get_grad_scaler("cuda") if device.type == "cuda" else None
    if scaler:
        print("[INFO] Đã bật Automatic Mixed Precision (FP16 AMP) trên GPU.")

    epochs = int(t_cfg.get("epochs", 20))
    clip_grad = float(t_cfg.get("clip_grad", 1.0))
    grad_accum = max(1, int(t_cfg.get("gradient_accumulation_steps", 1)))
    eval_bleu_samples = int(t_cfg.get("eval_bleu_samples", 200))
    decode_max_len = int(config.get("decoding", {}).get("max_len", 60))
    patience = int(t_cfg.get("patience", 5))

    save_dir = t_cfg.get("save_dir", "checkpoints/transformer")
    os.makedirs(save_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(save_dir, "best_model.pt")

    best_val_loss = float("inf")
    best_val_bleu = 0.0
    patience_counter = 0
    history = []

    print("\n" + "=" * 75)
    print(f"BẮT ĐẦU HUẤN LUYỆN TRANSFORMER ({epochs} EPOCHS | smoothing={smoothing})")
    print("=" * 75)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            clip_grad=clip_grad,
            grad_accum_steps=grad_accum,
            device=device,
            scaler=scaler,
        )

        val_loss, val_ppl = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            ce_criterion=ce_criterion,
            device=device,
        )

        val_bleu = (
            evaluate_val_bleu(
                model=model,
                val_loader=val_loader,
                tgt_tokenizer=tgt_tokenizer,
                device=device,
                max_samples=eval_bleu_samples,
                max_len=decode_max_len,
            )
            if (not is_dummy_mode and eval_bleu_samples > 0)
            else 0.0
        )

        epoch_time = time.time() - start_time
        cur_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Time: {epoch_time:.1f}s | "
            f"LR: {cur_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f} | "
            f"Val BLEU: {val_bleu:.2f}"
        )

        record = {
            "epoch": epoch,
            "train/loss": round(train_loss, 4),
            "train/lr": cur_lr,
            "train/epoch_time_sec": round(epoch_time, 2),
            "val/loss": round(val_loss, 4),
            "val/bleu": round(val_bleu, 2),
            "val/perplexity": round(val_ppl, 2),
        }
        history.append(record)

        if use_wandb and wandb_run:
            wandb_run.log(record)

        # Lưu checkpoint tốt nhất theo val_loss (chuẩn Dictionary Mục 8.1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_bleu = max(best_val_bleu, val_bleu)
            patience_counter = 0

            checkpoint = {
                "model_type": "transformer",
                "tokenizer_type": tokenizer_type,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_bleu": val_bleu,
                "config": config,
                "src_vocab_size": src_vocab_size,
                "tgt_vocab_size": tgt_vocab_size,
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(f"  --> [SAVED] Đã lưu best_model.pt vào: {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"\n[INFO] Early Stopping sau {epoch} epochs "
                    f"(val_loss không cải thiện trong {patience} epochs liên tiếp)."
                )
                break

    # Xuất tóm tắt kết quả ra JSON để phục vụ tổng hợp bảng Ablation Study (GĐ3)
    summary = {
        "run_name": config["experiment"]["name"],
        "tokenizer_type": tokenizer_type,
        "trainable_params": total_params,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_bleu": round(best_val_bleu, 2),
        "avg_epoch_time_sec": round(sum(r["train/epoch_time_sec"] for r in history) / max(1, len(history)), 2),
        "config": config,
        "history": history,
    }
    summary_path = os.path.join(save_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if use_wandb and wandb_run:
        wandb_run.finish()

    print("\n" + "=" * 75)
    print(f"HUẤN LUYỆN HOÀN TẤT! Checkpoint: {best_checkpoint_path}")
    print(f"Tóm tắt kết quả đã lưu tại: {summary_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
