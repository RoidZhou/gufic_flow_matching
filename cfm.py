import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# Path definition
# x_t = (1-t)x0 + t x1 + alpha sin(pi t) n(x1-x0)
# u_t = dx_t/dt = (x1-x0) + alpha pi cos(pi t) n(x1-x0)
# In 2D, n(v) = R90(v) / ||v||
# ============================================================

class CurvedPathCFM:
    def __init__(self, alpha: float = 0.35, eps: float = 1e-8):
        self.alpha = alpha
        self.eps = eps

    def normal_unit(self, delta: torch.Tensor) -> torch.Tensor:
        """
        delta: [B, 2]
        返回与 delta 正交的单位向量。
        2D 中用 90 度旋转实现: (dx, dy) -> (-dy, dx)
        """
        rotated = torch.stack([-delta[:, 1], delta[:, 0]], dim=1)
        norm = torch.norm(rotated, dim=1, keepdim=True).clamp_min(self.eps)
        return rotated / norm

    def sample_training_tuple(self, x1: torch.Tensor):
        """
        输入目标点 x1，在线采样 x0, t，并返回 x_t 和真实速度 u_t。
        x1: [B, 2]
        """
        device = x1.device
        batch = x1.shape[0]
        seq_len = x1.shape[1]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch, 1, 1, device=device) # FM 的 flow time t 应该是“每个样本一个标量”，而不是“每个时间步一个不同的标量”。

        delta = x1 - x0

        xt = (1.0 - t) * x0 + t * x1
        ut = delta
        return x0, x1, t, xt, ut

    def teacher_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        用于可视化教师路径。
        x0, x1: [B, 2]
        t: [B, 1]
        """
        delta = x1 - x0
        n = self.normal_unit(delta)
        xt = (1.0 - t) * x0 + t * x1 + self.alpha * torch.sin(math.pi * t) * n
        return xt
