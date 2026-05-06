
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from gufic_env.flow_matching.diffusion_model.diffusion.conditional_unet1d import ConditionalUnet1D
from gufic_env.flow_matching.diffusion_model.vision.pointnet import PointNetBackbone
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
    def __init__(
        self,
        x_dim=6,
        cond_dim=6,          # 这里只放 cond_main 维度，不再包含 guide_dim
        guide_dim=16,        # 新增
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

        # x_t + t_emb 先单独投影
        self.input_proj = nn.Linear(x_dim + time_dim, hidden_dim)

        # 主条件分支：p/R/Fe 历史
        if self.use_cond:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.cond_proj = None

        # guide_feat 单独控制 hidden
        self.guide_scale = nn.Sequential(
            nn.Linear(guide_dim, hidden_dim),
            nn.Tanh(),   # 控制 scale 幅度
        )
        self.guide_shift = nn.Linear(guide_dim, hidden_dim)

        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_proj = nn.Linear(hidden_dim, x_dim)

    def forward(self, x_t, t, cond_main=None, guide=None):
        """
        x_t: [B,T,6] or [B,6]
        t:   [B,1] or [B,T,1]
        cond_main: [B,cond_dim] or [B,T,cond_dim]
        guide:     [B,guide_dim] or [B,T,guide_dim]
        """
        squeeze_back = False

        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
            squeeze_back = True

        B, T, _ = x_t.shape
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}")

        # t -> [B,T,1]
        if t.dim() == 2:
            t = t[:, None, :]
            t = t.expand(B, T, 1)
        elif t.dim() == 3 and t.shape[1] == 1:
            t = t.expand(B, T, 1)

        t_emb = self.time_emb(t)                       # [B,T,time_dim]
        h = self.input_proj(torch.cat([x_t, t_emb], dim=-1))   # [B,T,H]

        # 主条件注入
        if self.use_cond:
            if cond_main is None:
                raise ValueError("use_cond=True 时，cond_main 不能为 None")
            if cond_main.dim() == 2:
                cond_main = cond_main.unsqueeze(1).expand(B, T, cond_main.shape[-1])
            elif cond_main.dim() == 3 and cond_main.shape[1] == 1:
                cond_main = cond_main.expand(B, T, cond_main.shape[-1])

            h = h + self.cond_proj(cond_main)

        # guide_feat 直接做 FiLM
        if guide is not None:
            if guide.dim() == 2:
                guide = guide.unsqueeze(1).expand(B, T, guide.shape[-1])
            elif guide.dim() == 3 and guide.shape[1] == 1:
                guide = guide.expand(B, T, guide.shape[-1])

            scale = self.guide_scale(guide)   # [B,T,H]
            shift = self.guide_shift(guide)   # [B,T,H]
            h = h * (1.0 + scale) + shift

        h = h + self.pos_embed[:, :T, :]
        h = self.transformer(h)
        out = self.output_proj(h)

        if squeeze_back:
            out = out.squeeze(1)

        return out

class VisionDeltaPoseNet(nn.Module):
    def __init__(self, state_dim=9, guide_dim=16, embed_dim=256, input_channels=3, input_transform=False):
        super().__init__()
        self.pointnet = PointNetBackbone(
            embed_dim= embed_dim,
            input_channels= input_channels,
            input_transform= input_transform,
        )

        self.backbone = nn.Sequential(
            nn.Linear(embed_dim + state_dim, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
        )

        self.delta_head = nn.Linear(128, 9)   # Δp(3) + ΔR6d(6)

        self.guide_proj = nn.Sequential(
            nn.Linear(128 + 9, 64),
            nn.Mish(),
            nn.Linear(64, guide_dim),
        )

    def forward(self, pc_now, x_now):
        """
        pc_now: [B, P, 3]
        x_now:  [B, 9]
        """
        pc_feat = self.pointnet(pc_now)   # [B,P,C]
        h = self.backbone(torch.cat([pc_feat, x_now], dim=-1))           # [B,128]

        delta_pose_pred = self.delta_head(h)                             # [B,9]
        guide_feat = self.guide_proj(torch.cat([h, delta_pose_pred], dim=-1))  # [B,guide_dim]

        return guide_feat, delta_pose_pred
# ============================================================
# Flow Matching Conditional Unet1D version
# 输入:
#   x_t: [B, T, x_dim]
#   t:   [B, 1] 或 [B, T, 1]
#   fe:  [B, T, cond_dim] 或 [B, cond_dim] 或 None
# 输出:
#   v:   [B, T, x_dim]
# ============================================================
class VelocityFMCondUnet1D(nn.Module):
    def __init__(self, x_dim=6, cond_dim=6, time_dim=64, kernel_size=5, use_cond=True):
        super().__init__()
        self.x_dim = x_dim
        self.use_cond = use_cond
        if self.use_cond:
            in_dim = x_dim + cond_dim
        else:
            in_dim = x_dim

        self.unet = ConditionalUnet1D(
            input_dim=in_dim,
            global_cond_dim=x_dim,
            diffusion_step_embed_dim=time_dim,
            kernel_size=kernel_size,
            use_down_condition=False,
            use_mid_condition=False,
            use_up_condition=False,
        )

    def forward(self, x_t, t, fe=None):
        """
        x_t: [B, T, x_dim]
        t:   [B,1,1] 或 [B,1] 或 [B]
        fe:  [B, T, cond_dim] 或 None
        """
        if t.dim() == 3:
            t_scalar = t[:, 0, 0]   # [B]
        elif t.dim() == 2:
            t_scalar = t[:, 0]
        elif t.dim() == 1:
            t_scalar = t
        else:
            raise ValueError(f"Unexpected t shape: {t.shape}")

        if self.use_cond:
            if fe is None:
                raise ValueError("use_cond=True 时 fe 不能为 None")
            out = self.unet(
                sample=x_t,
                timestep=t_scalar,
                local_cond=fe,
                global_cond=None,
            )
        else:
            out = self.unet(
                sample=x_t,
                timestep=t_scalar,
                local_cond=None,
                global_cond=None,
            )
        return out