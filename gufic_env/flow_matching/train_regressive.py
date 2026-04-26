import math
import os
import random
from dataclasses import dataclass, asdict
from config import TrainConfig
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import VelocityFieldRegressiveDataset
from model import VelocityRegressiveMLP
from cfm import CurvedPathCFM

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ============================================================
# Train / Infer
# ============================================================
def train_velocity_field(cfg: TrainConfig):
    set_seed(42)
    ensure_dir(cfg.save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = VelocityFieldRegressiveDataset(
        demo_dir=cfg.train_demo_dir,
        add_state_noise=cfg.add_state_noise,
        pos_noise_std=cfg.pos_noise_std,
        ori_noise_std=cfg.ori_noise_std,
        use_dVd_star=False,
    )

    val_dataset = VelocityFieldRegressiveDataset(
        demo_dir=cfg.val_demo_dir,
        add_state_noise=False,
        use_dVd_star=False,
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

    model = VelocityRegressiveMLP(
        x_dim=6,
        cond_dim=6,
        time_dim=cfg.time_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    train_curve = []
    val_curve = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_sum, train_count = 0.0, 0

        for x_t, dx_t, fe_t, goal, t in train_loader:
            x_t = x_t.to(device).float()
            dx_t = dx_t.to(device).float()
            fe_t = fe_t.to(device).float()
            goal = goal.to(device).float()
            t = t.to(device).float()

            pred = model(x_t, t, fe_t)
            loss = F.mse_loss(pred, dx_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x_t.shape[0]
            train_sum += loss.item() * bs
            train_count += bs

        train_loss = train_sum / max(train_count, 1)
        train_curve.append(train_loss)

        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for x_t, dx_t, fe_t, goal, t in val_loader:
                x_t = x_t.to(device).float()
                dx_t = dx_t.to(device).float()
                dx_t = dx_t.to(device).float()
                fe_t = fe_t.to(device).float()
                goal = goal.to(device).float()
                t = t.to(device).float()

                pred = model(x_t, t, fe_t)
                loss = F.mse_loss(pred, dx_t)

                bs = x_t.shape[0]
                val_sum += loss.item() * bs
                val_count += bs

        val_loss = val_sum / max(val_count, 1)
        val_curve.append(val_loss)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_val": best_val,
                    "train_cfg": cfg.__dict__,
                },
                os.path.join(cfg.save_dir, "velocity_field_regressive_best.pt"),
            )

    plt.figure(figsize=(7, 4))
    plt.plot(train_curve, label="train")
    plt.plot(val_curve, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Velocity Field Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "loss_curve.png"), dpi=180)
    plt.close()

    print(f"Training done. Best val loss = {best_val:.6f}")

if __name__ == "__main__":
        cfg = TrainConfig(
        train_demo_dir="/home/zhou/autolab/GUFIC_mujoco-main/bolt_demos",
        val_demo_dir="/home/zhou/autolab/GUFIC_mujoco-main/bolt_demos",
        save_dir="/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_regressive",
        batch_size=1024,
        lr=1e-3,
        weight_decay=1e-5,
        epochs=100,
        add_state_noise=True,
        pos_noise_std=0.001,
        ori_noise_std=0.01,
        )
        train_velocity_field(cfg)
