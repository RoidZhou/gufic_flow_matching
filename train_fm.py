import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FlowMatchingDataset
from model import VelocityFMMLP, VelocityFMTransformer
from config import TrainConfig
from cfm import CurvedPathCFM

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def train_velocity_field_mixed(cfg: TrainConfig, path_sampler: CurvedPathCFM):
    set_seed(42)
    ensure_dir(cfg.save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = FlowMatchingDataset(
        demo_dir=cfg.train_demo_dir,
        cond_key=cfg.cond_key,
    )
    train_stats = train_dataset.get_stats()

    val_dataset = FlowMatchingDataset(
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
    if cfg.model == "transformer":
        model = VelocityFMTransformer(
            x_dim=6,
            cond_dim=cfg.cond_dim,
            time_dim=cfg.time_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            use_cond=False
        ).to(device)
    else:
        model = VelocityFMMLP(
            x_dim=6,
            cond_dim=cfg.cond_dim,
            time_dim=cfg.time_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            use_cond=False
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,   # 一个完整余弦周期 = 训练总 epoch
        eta_min=1e-6        # 最小学习率
    )

    best_loss = float("inf")
    train_curve = []
    val_curve = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_sum, train_count = 0.0, 0

        for x, v, fe, t in train_loader:
            # x1 = x.to(device).float()
            v_t = v.to(device).float()
            fe_t = fe.to(device).float()
            t = t.to(device).float()

            # 采样 tau
            _, x1, t, xt, ut = path_sampler.sample_training_tuple(v_t)

            # 模型预测
            pred = model(xt, t, x1)            # [B,6]

            loss_vel = F.mse_loss(pred, ut)

            optimizer.zero_grad()
            loss_vel.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = x1.shape[0]
            train_sum += loss_vel.item() * batch_size
            train_count += batch_size

        train_loss = train_sum / max(train_count, 1)
        train_curve.append(train_loss)
        scheduler.step()

        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for x, v, fe, t in val_loader:
                # x1 = x.to(device).float()
                v_t = v.to(device).float()
                fe_t = fe.to(device).float()
                t = t.to(device).float()

                _, x1, t, xt, ut = path_sampler.sample_training_tuple(v_t)

                pred = model(xt, t, x1)
                loss = F.mse_loss(pred, ut)

                batch_size = x1.shape[0]
                val_sum += loss.item() * batch_size
                val_count += batch_size

        val_loss = val_sum / max(val_count, 1)
        val_curve.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "train_cfg": cfg.__dict__,
                    "best_loss": best_loss,
                    "stats": train_stats,
                },
                os.path.join(cfg.save_dir, "fm_best.pt"),
            )
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )

    # 画 loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(train_curve, label="train_total")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Mixed Loss Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "loss_curve_mixed.png"), dpi=180)
    plt.close()


if __name__ == "__main__":
    cfg = TrainConfig(epochs=1000, batch_size=8, save_dir="/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_fm")
    path_sampler = CurvedPathCFM(alpha=cfg.alpha, eps=cfg.eps)

    train_velocity_field_mixed(cfg, path_sampler)