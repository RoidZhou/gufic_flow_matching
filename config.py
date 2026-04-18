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

    model: str = "transformer"  # "mlp" or "transformer"

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
    cond_dim: int = 6

    # 混合损失权重
    lambda_vel: float = 1.0
    lambda_fm: float = 0.2
    lambda_fm_ori: float = 0.05

    # path
    alpha: float = 0.35
    eps: float = 1e-8
    
    cond_key: str = "fe"
