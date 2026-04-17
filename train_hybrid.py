import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FlowMatchingHybridDataset
from model import BoltVelocityMLP
from config import TrainConfig

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def train_velocity_field_mixed(cfg: TrainConfig):
    set_seed(42)
    ensure_dir(cfg.save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = FlowMatchingHybridDataset(
        demo_dir=cfg.train_demo_dir,
        cond_key=cfg.cond_key,
    )
    val_dataset = FlowMatchingHybridDataset(
        demo_dir=cfg.val_demo_dir,
        cond_key=cfg.cond_key,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = BoltVelocityMLP(
        x_dim=6,
        cond_dim=cfg.cond_dim,
        time_dim=cfg.time_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val = float("inf")
    train_curve = []
    val_curve = []
    train_vel_curve = []
    train_fm_curve = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_sum, train_count = 0.0, 0
        train_vel_sum, train_fm_sum = 0.0, 0.0

        for x0, x1, v0, v1, fe0, fe1, t0, t1, dt in train_loader:
            x0 = x0.to(device).float()
            x1 = x1.to(device).float()
            v0 = v0.to(device).float()
            v1 = v1.to(device).float()
            fe0 = fe0.to(device).float()
            fe1 = fe1.to(device).float()
            t0 = t0.to(device).float()
            t1 = t1.to(device).float()
            dt = dt.to(device).float()                     # [B,1]

            B = x0.shape[0]

            # 采样 tau
            tau = torch.rand(B, 1, device=device)

            # 构造插值状态 / 条件 / 时间
            x_tau = (1.0 - tau) * x0 + tau * x1           # [B,6]
            fe_tau = (1.0 - tau) * fe0 + tau * fe1        # [B,C]
            t_tau = (1.0 - tau) * t0 + tau * t1           # [B,1]

            # 模型预测
            pred = model(x_tau, t_tau, fe_tau)            # [B,6]

            # ------------------------------------------------
            # L_vel: 拟合采集到的每一时刻速度场
            # 用相邻两帧的 teacher velocity 线性插值
            # ------------------------------------------------
            v_teacher_tau = (1.0 - tau) * v0 + tau * v1
            loss_vel = F.mse_loss(pred, v_teacher_tau)

            # ------------------------------------------------
            # L_fm: flow matching 结构损失
            # 最简单版本: 用相邻状态差分近似目标场
            # 注意: 对于 x=[p,euler]，姿态部分只是近似
            # ------------------------------------------------
            u_fm = (x1 - x0) / torch.clamp(dt, min=1e-8)  # [B,6]

            loss_fm_pos = F.mse_loss(pred[:, :3], u_fm[:, :3])
            loss_fm_ori = F.mse_loss(pred[:, 3:6], u_fm[:, 3:6])
            loss_fm = loss_fm_pos + cfg.lambda_fm_ori * loss_fm_ori

            # 总损失
            loss = cfg.lambda_vel * loss_vel + cfg.lambda_fm * loss_fm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x0.shape[0]
            train_sum += loss.item() * bs
            train_vel_sum += loss_vel.item() * bs
            train_fm_sum += loss_fm.item() * bs
            train_count += bs

        train_loss = train_sum / max(train_count, 1)
        train_vel_loss = train_vel_sum / max(train_count, 1)
        train_fm_loss = train_fm_sum / max(train_count, 1)

        train_curve.append(train_loss)
        train_vel_curve.append(train_vel_loss)
        train_fm_curve.append(train_fm_loss)

        # --------------------------
        # validation
        # --------------------------
        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for x0, x1, v0, v1, fe0, fe1, t0, t1, dt in val_loader:
                x0 = x0.to(device).float()
                x1 = x1.to(device).float()
                v0 = v0.to(device).float()
                v1 = v1.to(device).float()
                fe0 = fe0.to(device).float()
                fe1 = fe1.to(device).float()
                t0 = t0.to(device).float()
                t1 = t1.to(device).float()
                dt = dt.to(device).float()

                B = x0.shape[0]
                tau = torch.rand(B, 1, device=device)

                x_tau = (1.0 - tau) * x0 + tau * x1
                fe_tau = (1.0 - tau) * fe0 + tau * fe1
                t_tau = (1.0 - tau) * t0 + tau * t1

                pred = model(x_tau, t_tau, fe_tau)

                v_teacher_tau = (1.0 - tau) * v0 + tau * v1
                loss_vel = F.mse_loss(pred, v_teacher_tau)

                u_fm = (x1 - x0) / torch.clamp(dt, min=1e-8)
                loss_fm_pos = F.mse_loss(pred[:, :3], u_fm[:, :3])
                loss_fm_ori = F.mse_loss(pred[:, 3:6], u_fm[:, 3:6])
                loss_fm = loss_fm_pos + cfg.lambda_fm_ori * loss_fm_ori

                loss = cfg.lambda_vel * loss_vel + cfg.lambda_fm * loss_fm

                bs = x0.shape[0]
                val_sum += loss.item() * bs
                val_count += bs

        val_loss = val_sum / max(val_count, 1)
        val_curve.append(val_loss)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"(vel={train_vel_loss:.6f}, fm={train_fm_loss:.6f}) "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_val": best_val,
                    "train_cfg": cfg.__dict__,
                },
                os.path.join(cfg.save_dir, "bolt_velocity_field_best.pt"),
            )

    # 画 loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(train_curve, label="train_total")
    plt.plot(val_curve, label="val_total")
    plt.plot(train_vel_curve, label="train_vel")
    plt.plot(train_fm_curve, label="train_fm")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Mixed Loss Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "loss_curve_mixed.png"), dpi=180)
    plt.close()

    print(f"Training done. Best val loss = {best_val:.6f}")

if __name__ == "__main__":
    cfg = TrainConfig(
    train_demo_dir="/home/zhou/autolab/GUFIC_mujoco-main/bolt_demos",
    val_demo_dir="/home/zhou/autolab/GUFIC_mujoco-main/bolt_demos",
    save_dir="./checkpoints_bolt_real_hybrid",
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=100,
    hidden_dim=256,
    num_layers=4,
    time_dim=64,
    cond_dim=6,
    lambda_vel=1.0,
    lambda_fm=0.1,
    lambda_fm_ori=0.02,
    cond_key="fe",
    )

    train_velocity_field_mixed(cfg)