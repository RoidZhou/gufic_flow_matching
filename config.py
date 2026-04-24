import torch
from dataclasses import dataclass, asdict

# ============================================================
# Config
# ============================================================
@dataclass
class TrainConfig:
    train_demo_dir: str = "/home/zhou/autolab/GUFIC_mujoco-main/bolt_demos"
    val_demo_dir: str = "/home/zhou/autolab/GUFIC_mujoco-main/bolt_demos"
    save_dir: str = "./checkpoints_fm"

    train_mode: str = "rolling_horizon"  # "fixed_length" or "rolling_horizon"
    model: str = "transformer"  # "mlp" or "transformer" or "unet"

    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 30

    add_state_noise: bool = True
    pos_noise_std: float = 0.001
    ori_noise_std: float = 0.01

    hidden_dim: int = 256
    num_layers: int = 4
    time_dim: int = 64

    # 条件 = 最近 K 步力序列，展平后维度 = 6*K
    x_hist_len: int = 1
    force_hist_len: int = 16
    # cond_dim: int = 6 * 16 + 9   # K=16 步 6 维 力历史，1 步 9 维状态
    cond_dim: int = 15 * 16        # K=16 步 6 维 力历史，16 步 9 维状态

    # 未来预测 horizon
    pred_horizon: int = 100

    # 采样步长
    stride: int = 1

    # 混合损失权重
    lambda_vel: float = 1.0
    lambda_fm: float = 0.2
    lambda_fm_ori: float = 0.05
    # 推理采样步长
    infer_steps: int = 10

    # path
    alpha: float = 0.35
    eps: float = 1e-8
    use_cond: bool = True
    cond_key: str = "fe"
