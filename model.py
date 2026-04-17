
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

# ============================================================
# Model
# 输入: x_t, t, x1
# 输出: 速度 v_theta(x_t, t, x1)
# ============================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half,
                device=t.device,
                dtype=t.dtype,
            )
        )
        angles = t * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb


class VelocityRegressiveMLP(nn.Module):
    """
    学习连续速度场:
      dx/dt = v_theta(x, t, goal)
    """
    def __init__(self, x_dim=6, cond_dim=6, time_dim=64, hidden_dim=256, num_layers=4, use_cond=True):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.use_cond = use_cond
        if self.use_cond:
            in_dim = x_dim + cond_dim + time_dim
        else:
            in_dim = x_dim + time_dim
        
        layers = []

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [x_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t, fe=None):
        t_emb = self.time_emb(t)
        if self.use_cond:
            h = torch.cat([x_t, fe, t_emb], dim=-1)
        else:
            h = torch.cat([x_t, t_emb], dim=-1)
        return self.net(h)

# ============================================================
# Flow Matching MLP version
# 输入:
#   x_t: [B, T, x_dim]
#   t:   [B, 1] 或 [B, T, 1]
#   fe:  [B, T, cond_dim] 或 [B, cond_dim] 或 None
# 输出:
#   v:   [B, T, x_dim]
# ============================================================
class VelocityFMMLP(nn.Module):
    """
    学习连续速度场:
      dx/dt = v_theta(x, t, goal)
    """
    def __init__(self, x_dim=6, cond_dim=6, time_dim=64, hidden_dim=256, num_layers=4, use_cond=True):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.use_cond = use_cond
        if self.use_cond:
            in_dim = x_dim + cond_dim + time_dim
        else:
            in_dim = x_dim + time_dim
        
        layers = []

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [x_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t, fe=None):
        B, T, _ = x_t.shape
        # 处理 t
        if t.dim() == 2:
            # [B,1] -> [B,T,1]
            t = t.unsqueeze(1) if t.shape[-1] != 1 else t[:, None, :]
            t = t.expand(B, T, 1)
        elif t.dim() == 3:
            if t.shape[1] == 1:
                t = t.expand(B, T, 1)
        else:
            raise ValueError(f"Unexpected t shape: {t.shape}")

        t_emb = self.time_emb(t)
        if self.use_cond:
            h = torch.cat([x_t, fe, t_emb], dim=-1)
        else:
            h = torch.cat([x_t, t_emb], dim=-1)
        return self.net(h)
    
# ============================================================
# Flow Matching Transformer version
# 输入:
#   x_t: [B, T, x_dim]
#   t:   [B, 1] 或 [B, T, 1]
#   fe:  [B, T, cond_dim] 或 [B, cond_dim] 或 None
# 输出:
#   v:   [B, T, x_dim]
# ============================================================
class VelocityFMTransformer(nn.Module):
    """
    为了最小改动，类名仍然保留 BoltVelocityMLP
    但内部已经改成 Transformer Encoder 版本
    """
    def __init__(
        self,
        x_dim=6,
        cond_dim=6,
        time_dim=64,
        hidden_dim=256,
        num_layers=4,
        use_cond=True,
        nhead=8,
        dropout=0.1,
        max_seq_len=12000,
    ):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        self.use_cond = use_cond
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        if self.use_cond:
            in_dim = x_dim + cond_dim + time_dim
        else:
            in_dim = x_dim + time_dim

        # 输入投影
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # 输入输出都是 [B, T, C]
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 输出投影回速度维度
        self.output_proj = nn.Linear(hidden_dim, x_dim)

    def forward(self, x_t, t, fe=None):
        """
        x_t:
          [B, T, x_dim]
          也兼容 [B, x_dim]，会自动扩成 T=1

        t:
          [B, 1]      -> 自动 broadcast 到 [B, T, 1]
          或 [B, T, 1]

        fe:
          None
          或 [B, cond_dim]
          或 [B, T, cond_dim]
        """
        squeeze_back = False

        # 兼容单步输入 [B, x_dim]
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)   # [B, 1, x_dim]
            squeeze_back = True

        B, T, _ = x_t.shape

        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}")

        # 处理 t
        if t.dim() == 2:
            # [B,1] -> [B,T,1]
            t = t.unsqueeze(1) if t.shape[-1] != 1 else t[:, None, :]
            t = t.expand(B, T, 1)
        elif t.dim() == 3:
            if t.shape[1] == 1:
                t = t.expand(B, T, 1)
        else:
            raise ValueError(f"Unexpected t shape: {t.shape}")

        t_emb = self.time_emb(t)   # [B, T, time_dim]

        # 处理条件 fe
        if self.use_cond:
            if fe is None:
                raise ValueError("use_cond=True 时，fe 不能为 None")

            if fe.dim() == 2:
                fe = fe.unsqueeze(1).expand(B, T, fe.shape[-1])   # [B, T, cond_dim]
            elif fe.dim() == 3:
                if fe.shape[1] == 1:
                    fe = fe.expand(B, T, fe.shape[-1])
            else:
                raise ValueError(f"Unexpected fe shape: {fe.shape}")

            h = torch.cat([x_t, fe, t_emb], dim=-1)   # [B, T, in_dim]
        else:
            h = torch.cat([x_t, t_emb], dim=-1)

        # 输入投影 + 位置编码
        h = self.input_proj(h)                        # [B, T, hidden_dim]
        h = h + self.pos_embed[:, :T, :]             # [B, T, hidden_dim]

        # Transformer
        h = self.transformer(h)                       # [B, T, hidden_dim]

        # 输出速度场
        out = self.output_proj(h)                     # [B, T, x_dim]

        if squeeze_back:
            out = out.squeeze(1)                      # 回到 [B, x_dim]

        return out