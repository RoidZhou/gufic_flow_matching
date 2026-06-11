
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
 
class VisionDeltaPoseFMTransformer(nn.Module):
    """
    用 PointNet + 条件编码器生成 guide_feat；
    用 Flow Matching Transformer 生成 delta_pose_pred = [Δp(3), ΔR6d(6)]。

    训练时:
        forward(..., x_t=xt_pose, t=t_pose)
        返回 guide_feat, delta_pose_flow_pred

    推理时:
        forward(..., x_t=None, t=None)
        内部从高斯噪声采样 delta_pose_pred
        返回 guide_feat, delta_pose_pred
    """
    def __init__(
        self,
        state_dim=9,
        cond_dim=105,
        guide_dim=16,
        embed_dim=256,
        input_channels=3,
        input_transform=False,
        time_dim=64,
        hidden_dim=256,
        num_layers=4,
        nhead=8,
        dropout=0.1,
        sample_steps=10,
    ):
        super().__init__()

        self.sample_steps = sample_steps
        self.guide_dim = guide_dim

        self.pointnet = PointNetBackbone(
            embed_dim=embed_dim,
            input_channels=input_channels,
            input_transform=input_transform,
        )

        self.pc_proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
        )

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Mish(),
            nn.Linear(128, 128),
            nn.Mish(),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
        )

        self.fuse = nn.Sequential(
            nn.Linear(128 + 128 + 128, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
        )

        self.guide_proj = nn.Sequential(
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, guide_dim),
        )

        # 用同一个 VelocityFMTransformer 结构生成 delta_pose
        # x_dim=9: Δp(3) + ΔR6d(6)
        self.delta_fm = VelocityFMTransformer(
            x_dim=9,
            cond_dim=128,
            guide_dim=guide_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_cond=True,
            nhead=nhead,
            dropout=dropout,
            max_seq_len=128,
        )

    def encode_condition(self, pc_now, x_now, cond_hist):
        """
        pc_now:    [B,P,3] 或 [B,H,P,3]
        x_now:     [B,9]
        cond_hist: [B,cond_dim]
        """
        pc_feat = self.pointnet(pc_now)     # [B,embed_dim]
        pc_feat = self.pc_proj(pc_feat)     # [B,128]

        x_feat = self.state_proj(x_now)     # [B,128]

        if cond_hist is None:
            cond_feat = torch.zeros_like(x_feat)
        else:
            cond_feat = self.cond_proj(cond_hist)

        h = self.fuse(torch.cat([pc_feat, x_feat, cond_feat], dim=-1))  # [B,128]
        guide_feat = self.guide_proj(h)                                 # [B,guide_dim]

        return h, guide_feat

    def forward(self, pc_now, x_now, cond_hist=None, x_t=None, t=None):
        """
        训练:
            x_t, t 不为 None，返回 delta_pose 的 flow velocity 预测。

        推理:
            x_t, t 为 None，从噪声积分生成 delta_pose_pred。
        """
        cond_feat, guide_feat = self.encode_condition(pc_now, x_now, cond_hist)

        # 训练模式：预测 flow velocity
        if x_t is not None and t is not None:
            delta_flow_pred = self.delta_fm(
                x_t=x_t,
                t=t,
                cond_main=cond_feat,
                guide=guide_feat,
            )
            return guide_feat, delta_flow_pred

        # 推理模式：从 N(0,I) 采样 delta_pose
        B = x_now.shape[0]
        device = x_now.device
        z = torch.randn(B, 9, device=device, dtype=x_now.dtype)

        dt = 1.0 / float(self.sample_steps)

        for i in range(self.sample_steps):
            tau = torch.full(
                (B, 1),
                i / float(self.sample_steps),
                device=device,
                dtype=x_now.dtype,
            )

            u = self.delta_fm(
                x_t=z,
                t=tau,
                cond_main=cond_feat,
                guide=guide_feat,
            )

            z = z + u * dt

        delta_pose_pred = z
        return guide_feat, delta_pose_pred

class VisionPoseObsEncoder(nn.Module):
    def __init__(
        self,
        state_dim=9,
        cond_dim=105,
        obs_dim=128,
        guide_dim=16,
        embed_dim=256,
        input_channels=3,
        input_transform=False,
    ):
        super().__init__()

        self.pointnet = PointNetBackbone(
            embed_dim=embed_dim,
            input_channels=input_channels,
            input_transform=input_transform,
        )

        self.pc_proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.Mish(),
            nn.Linear(256, obs_dim),
            nn.Mish(),
        )

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, obs_dim),
            nn.Mish(),
            nn.Linear(obs_dim, obs_dim),
            nn.Mish(),
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.Mish(),
            nn.Linear(256, obs_dim),
            nn.Mish(),
        )

        self.fuse = nn.Sequential(
            nn.Linear(obs_dim * 3, 256),
            nn.Mish(),
            nn.Linear(256, obs_dim),
            nn.Mish(),
        )

        # pose FM 的条件特征
        self.pose_cond_proj = nn.Sequential(
            nn.Linear(obs_dim, obs_dim),
            nn.Mish(),
            nn.Linear(obs_dim, obs_dim),
            nn.Mish(),
        )

        # velocity FM 的视觉 guide
        self.vel_guide_proj = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Mish(),
            nn.Linear(64, guide_dim),
        )

    def forward(self, pc_now, x_now, cond_hist=None):
        """
        pc_now:    [B,P,3] 或 [B,H,P,3]
        x_now:     [B,9]
        cond_hist: [B,cond_dim]

        return:
            nx_pose:   [B, obs_dim]    给 pose_model 用
            guide_vel: [B, guide_dim]  给 velocity_model 用
        """

        # 如果输入是历史点云 [B,H,P,C]，只取当前帧
        pc_feat = self.pointnet(pc_now)      # [B, embed_dim]
        pc_feat = self.pc_proj(pc_feat)      # [B, obs_dim]

        x_feat = self.state_proj(x_now)      # [B, obs_dim]

        if cond_hist is None:
            cond_feat = torch.zeros_like(x_feat)
        else:
            cond_feat = self.cond_proj(cond_hist)  # [B, obs_dim]

        h = self.fuse(
            torch.cat([pc_feat, x_feat, cond_feat], dim=-1)
        )

        nx_pose = h
        guide_vel = self.vel_guide_proj(h)

        return nx_pose, guide_vel

class VisionPoseObsEncoderPoseSimpleV2(nn.Module):
    def __init__(
        self,
        state_dim=9,
        cond_dim=105,
        obs_dim=128,
        guide_dim=16,
        embed_dim=256,
        input_channels=3,
        input_transform=False,
        dropout=0.05,
    ):
        super().__init__()

        self.pointnet = PointNetBackbone(
            embed_dim=embed_dim,
            input_channels=input_channels,
            input_transform=input_transform,
        )

        # =========================
        # pose branch
        # 只用于 pose_model 生成 pd, Rd
        # 不直接使用力历史
        # =========================
        self.pose_pc_proj = nn.Linear(embed_dim, obs_dim)
        self.pose_state_proj = nn.Linear(state_dim, obs_dim)
        self.pose_norm = nn.LayerNorm(obs_dim)

        # =========================
        # velocity branch
        # 用于生成 velocity guide
        # 可以使用点云 + 当前状态 + p/R/Fe 历史
        # =========================
        self.vel_pc_proj = nn.Linear(embed_dim, obs_dim)
        self.vel_state_proj = nn.Linear(state_dim, obs_dim)
        self.vel_cond_proj = nn.Linear(cond_dim, obs_dim)

        self.vel_guide_proj = nn.Sequential(
            nn.Linear(obs_dim * 3, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, guide_dim),
        )

        self.act = nn.Mish()

    def forward(self, pc_now, x_now, cond_hist=None):
        """
        pc_now:    [B,P,3] 或 [B,H,P,3]
        x_now:     [B,9]
        cond_hist: [B,cond_dim]

        return:
            nx_pose:   [B, obs_dim]    给 pose_model 用
            guide_vel: [B, guide_dim]  给 velocity_model 用
        """

        pc_global = self.pointnet(pc_now)  # [B, embed_dim]

        # =====================================================
        # 1. pose condition: only point cloud + current state
        # =====================================================
        pose_pc = self.pose_pc_proj(pc_global)
        pose_x = self.pose_state_proj(x_now)

        nx_pose = self.act(
            self.pose_norm(pose_pc + pose_x)
        )  # [B, obs_dim]

        # =====================================================
        # 2. velocity guide: point cloud + current state + history
        # =====================================================
        vel_pc = self.vel_pc_proj(pc_global)
        vel_x = self.vel_state_proj(x_now)

        if cond_hist is None:
            vel_cond = torch.zeros_like(vel_x)
        else:
            vel_cond = self.vel_cond_proj(cond_hist)

        h_vel = torch.cat([vel_pc, vel_x, vel_cond], dim=-1)  # [B, 3*obs_dim]
        guide_vel = self.vel_guide_proj(h_vel)                # [B, guide_dim]

        return nx_pose, guide_vel
    

class VisionPoseObsEncoderNoPCForVelocity(nn.Module):
    def __init__(
        self,
        state_dim=9,
        cond_dim=105,
        obs_dim=128,
        guide_dim=16,
        embed_dim=256,
        input_channels=3,
        input_transform=False,
        dropout=0.0,
    ):
        super().__init__()

        self.pointnet = PointNetBackbone(
            embed_dim=embed_dim,
            input_channels=input_channels,
            input_transform=input_transform,
        )

        # pose branch: PointNet + x_now
        self.pose_pc_proj = nn.Linear(embed_dim, obs_dim)
        self.pose_state_proj = nn.Linear(state_dim, obs_dim)
        self.pose_norm = nn.LayerNorm(obs_dim)

        # velocity guide branch: 不使用 PointNet，只用 x_now + cond_hist
        self.vel_state_proj = nn.Linear(state_dim, obs_dim)
        self.vel_cond_proj = nn.Linear(cond_dim, obs_dim)

        self.vel_guide_proj = nn.Sequential(
            nn.Linear(obs_dim * 2, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.Mish(),
            nn.Linear(64, guide_dim),
        )

        self.act = nn.Mish()

    def forward(self, pc_now, x_now, cond_hist=None):
        # ======================
        # pose branch uses PointNet
        # ======================
        pc_global = self.pointnet(pc_now)

        pose_pc = self.pose_pc_proj(pc_global)
        pose_x = self.pose_state_proj(x_now)

        nx_pose = self.act(self.pose_norm(pose_pc + pose_x))

        # ======================
        # velocity branch does NOT use PointNet
        # ======================
        vel_x = self.vel_state_proj(x_now)

        if cond_hist is None:
            vel_cond = torch.zeros_like(vel_x)
        else:
            vel_cond = self.vel_cond_proj(cond_hist)

        h_vel = torch.cat([vel_x, vel_cond], dim=-1)
        guide_vel = self.vel_guide_proj(h_vel)

        return nx_pose, guide_vel
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