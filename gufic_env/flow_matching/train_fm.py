import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FlowMatchingDataset, RollingForceHistoryFMDataset
from model import VelocityFMMLP, VelocityFMTransformer,VelocityFMCondUnet1D, VisionDeltaPoseNet
from config import TrainConfig
from cfm import CurvedPathCFM
import csv
from datetime import datetime
from diffusion_model.vision.pointnet import PointNetBackbone

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def train_velocity_field_fixed_length(cfg: TrainConfig, path_sampler: CurvedPathCFM):
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
            use_cond=True
        ).to(device)
    elif cfg.model == "unet":
        model = VelocityFMCondUnet1D(
            x_dim=6,
            cond_dim=cfg.cond_dim,
            time_dim=cfg.time_dim,
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
            pred = model(xt, t, fe_t)            # [B,6]

            loss_vel = F.mse_loss(pred, ut)

            optimizer.zero_grad()
            loss_vel.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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

                pred = model(xt, t, fe_t)
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

def train_velocity_field_rolling_horizon(cfg: TrainConfig, path_sampler: CurvedPathCFM):
    set_seed(42)
    ensure_dir(cfg.save_dir)
    log_dir = os.path.join(cfg.save_dir, "csv_logs")
    ensure_dir(log_dir)

    csv_path = os.path.join(
        log_dir,
        f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr", "best_loss"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = RollingForceHistoryFMDataset(
        demo_dir=cfg.train_demo_dir,
        x_hist_len=cfg.x_hist_len,
        pc_hist_len=cfg.pc_hist_len,
        force_hist_len=cfg.force_hist_len,
        pred_horizon=cfg.pred_horizon,
        stride=cfg.stride,
        normalize_v=True,
        cond_stats=None,
        use_pc_color=cfg.use_pc_color
    )
    cond_stats = train_dataset.get_cond_stats()

    val_dataset = RollingForceHistoryFMDataset(
        demo_dir=cfg.val_demo_dir,
        x_hist_len=cfg.x_hist_len,
        pc_hist_len=cfg.pc_hist_len,
        force_hist_len=cfg.force_hist_len,
        pred_horizon=cfg.pred_horizon,
        stride=cfg.stride,
        normalize_v=True,
        cond_stats=cond_stats,
        use_pc_color=cfg.use_pc_color
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

    obs_encoder = VisionDeltaPoseNet(
        state_dim=cfg.state_dim,
        guide_dim=cfg.guide_dim,
        embed_dim= cfg.embed_dim,
        input_channels= cfg.input_channels,
        input_transform= cfg.input_transform,
        ).to(device)

    model = VelocityFMTransformer(
        x_dim=6,
        cond_dim=cfg.cond_dim,   # = 6*K
        time_dim=cfg.time_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        use_cond=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(obs_encoder.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=1e-6,
    )

    best_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_sum, train_count = 0.0, 0

        for cond_hist, pc_hist, delta_pose_target, v_future in train_loader:
            cond_hist_flat = cond_hist.to(device).float()           # [B, 6K]
            pc_hist = pc_hist.to(device).float()           # [B, T, P, C]
            # pc_hist = pc_hist.squeeze(1)          # [B, P, C]
            delta_pose_target = delta_pose_target.to(device).float()  # [B,9]
            v_future = v_future.to(device).float()           # [B, H, 6]

            # 当前状态 x_now 就是 cond_main 的前 9 维
            x_now = cond_hist_flat[:, :9]

            # FM tuple on future velocity trajectory
            _, x1, t, xt, ut = path_sampler.sample_training_tuple(v_future)
            guide_feat, delta_pose_pred = obs_encoder(pc_hist, x_now)   # [B,guide_dim], [B,9]
            cond_hist = torch.cat([cond_hist_flat, guide_feat], dim=-1)         # [B, cond_dim]

            pred = model(xt, t, cond_hist)                # 条件 = 最近 K 步力历史

            loss_fm = F.mse_loss(pred, ut)
            loss_delta = F.smooth_l1_loss(delta_pose_pred, delta_pose_target)
            loss = loss_fm + cfg.lambda_delta * loss_delta

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            bs = cond_hist.shape[0]
            train_sum += loss.item() * bs
            train_count += bs

        train_loss = train_sum / max(train_count, 1)
        scheduler.step()

        model.eval()
        val_sum, val_count = 0.0, 0
        # 让 val 固定，减少波动
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

        with torch.no_grad():
            for cond_hist_flat, pc_hist, delta_pose_target, v_future in val_loader:
                cond_hist_flat = cond_hist_flat.to(device).float()
                pc_hist = pc_hist.to(device).float()           # [B, 6K]
                delta_pose_target = delta_pose_target.to(device).float()  # [B,9]
                v_future = v_future.to(device).float()
                # 当前状态 x_now 就是 cond_main 的前 9 维
                x_now = cond_hist_flat[:, :9]

                _, x1, t, xt, ut = path_sampler.sample_training_tuple(v_future)
                guide_feat, delta_pose_pred = obs_encoder(pc_hist, x_now)   # [B,guide_dim], [B,9]
                cond_hist = torch.cat([cond_hist_flat, guide_feat], dim=-1)         # [B, cond_dim]
                
                pred = model(xt, t, cond_hist)
                loss_fm = F.mse_loss(pred, ut)
                loss_delta = F.smooth_l1_loss(delta_pose_pred, delta_pose_target)
                loss = loss_fm + cfg.lambda_delta * loss_delta

                bs = cond_hist.shape[0]
                val_sum += loss.item() * bs
                val_count += bs

        val_loss = val_sum / max(val_count, 1)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "obs_encoder": obs_encoder.state_dict(),
                    "train_cfg": cfg.__dict__,
                    "best_loss": best_loss,
                    "cond_stats": cond_stats,
                },
                os.path.join(cfg.save_dir, f"cfm_{cfg.model}_{cfg.type}_best.pt"),
            )
        current_lr = optimizer.param_groups[0]["lr"]

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, current_lr, best_loss])
    
        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

if __name__ == "__main__":
    # type = "fixed_start"
    type = "random_start"
    
    cfg = TrainConfig(train_demo_dir="/home/zhou/autolab/GUFIC_mujoco-main/bolt_vis_demo",
                        val_demo_dir="/home/zhou/autolab/GUFIC_mujoco-main/bolt_vis_demo",
                        type=type,
                        epochs=1000, 
                        batch_size=8, 
                        save_dir=f"/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_cfm_transformer_pRFe_{type}")
    path_sampler = CurvedPathCFM(alpha=cfg.alpha, eps=cfg.eps)

    if cfg.train_mode == "fixed_length":
        train_velocity_field_fixed_length(cfg, path_sampler)
    elif cfg.train_mode == "rolling_horizon":
        train_velocity_field_rolling_horizon(cfg, path_sampler)
    else:
        raise ValueError(f"Unknown train_mode: {cfg.train_mode}")