"""
Module: utils/lr_scheduler.py
Phụ trách: Sơn (Phần 3)
Cài đặt: Noam Learning Rate Scheduler theo bài báo "Attention Is All You Need" (Vaswani et al., 2017):
    lr(step) = factor * d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
"""

import torch


class NoamScheduler:
    """
    Noam Learning Rate Scheduler với giai đoạn warmup tuyến tính và suy giảm theo step^(-0.5).
    Gọi scheduler.step() sau mỗi bước cập nhật trọng số của optimizer.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ):
        if d_model <= 0:
            raise ValueError("d_model phải lớn hơn 0.")
        if warmup_steps <= 0:
            raise ValueError("warmup_steps phải lớn hơn 0.")

        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0
        self._last_lr = 0.0
        # Khởi tạo ngay LR của bước 1 để optimizer.step() đầu tiên không dùng lr mặc định
        self.step()

    def compute_lr(self, step: int) -> float:
        """Tính giá trị learning rate tại bước `step` (>= 1)."""
        step = max(1, step)
        scale = self.d_model ** (-0.5)
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        return self.factor * scale * min(arg1, arg2)

    def step(self) -> float:
        """Tăng bộ đếm bước và cập nhật learning rate cho toàn bộ param_groups của optimizer."""
        self.step_num += 1
        lr = self.compute_lr(self.step_num)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self._last_lr = lr
        return lr

    def get_last_lr(self) -> list[float]:
        """Trả về learning rate hiện tại (tương thích giao diện PyTorch LRScheduler)."""
        return [self._last_lr]

    def state_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "factor": self.factor,
            "step_num": self.step_num,
            "_last_lr": self._last_lr,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.d_model = state_dict["d_model"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.factor = state_dict["factor"]
        self.step_num = state_dict["step_num"]
        self._last_lr = state_dict.get("_last_lr", self.compute_lr(self.step_num))
