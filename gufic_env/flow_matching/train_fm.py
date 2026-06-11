import mujoco
import mujoco.viewer
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FlowMatchingDataset, RollingForceHistoryFMDataset
from model import VelocityFMMLP, VelocityFMTransformer,VelocityFMCondUnet1D, VisionDeltaPoseNet, VisionDeltaPoseFMTransformer, VisionPoseObsEncoder, VisionPoseObsEncoderPoseSimple, VisionPoseObsEncoderPoseSimpleV2,VisionPoseObsEncoderNoPCForVelocity
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

def save_train_checkpoint(
    save_path,
    epoch,
    model,
    obs_encoder,
    pose_model,
    optimizer,
    scheduler,
    best_loss,
    cond_stats,
    cfg,
):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "obs_encoder": obs_encoder.state_dict(),
        "pose_model": pose_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_loss": best_loss,
        "cond_stats": cond_stats,
        "train_cfg": cfg.__dict__,
    }
    torch.save(ckpt, save_path)


def try_resume_from_checkpoint(
    resume_path,
    model,
    obs_encoder,
    optimizer=None,
    scheduler=None,
    device="cuda",
):
    """
    兼容两种 checkpoint:
    1. 新版: 含 epoch / optimizer / scheduler，可严格断点续训
    2. 旧版: 只有 model / obs_encoder / best_loss / cond_stats，只能“加载后继续训”
    """
    if (resume_path is None) or (not os.path.exists(resume_path)):
        return 1, float("inf"), None, False

    ckpt = torch.load(resume_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"])
    if ("obs_encoder" in ckpt) and (ckpt["obs_encoder"] is not None):
        obs_encoder.load_state_dict(ckpt["obs_encoder"])

    has_opt_state = optimizer is not None and ("optimizer" in ckpt)
    has_sch_state = scheduler is not None and ("scheduler" in ckpt)

    if has_opt_state:
        optimizer.load_state_dict(ckpt["optimizer"])
    if has_sch_state:
        scheduler.load_state_dict(ckpt["scheduler"])

    # 老 checkpoint 没有 epoch 时，从 1 开始继续
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_loss = float(ckpt.get("best_loss", float("inf")))
    cond_stats = ckpt.get("cond_stats", None)

    print(f"[Resume] loaded from: {resume_path}")
    print(f"[Resume] start_epoch={start_epoch}, best_loss={best_loss:.6f}")

    if not has_opt_state:
        print("[Resume] optimizer state not found, optimizer will restart.")
    if not has_sch_state:
        print("[Resume] scheduler state not found, scheduler will restart.")

    return start_epoch, best_loss, cond_stats, True

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

    # 固定一个日志文件，方便 resume 后继续追加
    csv_path = os.path.join(log_dir, "train_log_resume.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "val_loss", "lr", "best_loss",
                "train_fm", "val_fm", "train_dp", "val_dp", "train_dR", "val_dR"
            ])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 先建模型/优化器，方便 resume =====
    # obs_encoder = VisionPoseObsEncoder(
    #     state_dim=cfg.state_dim,
    #     cond_dim=cfg.cond_dim,
    #     guide_dim=cfg.guide_dim,
    #     embed_dim=cfg.embed_dim,
    #     input_channels=cfg.input_channels,
    #     input_transform=cfg.input_transform,
    # ).to(device)
    obs_dim = getattr(cfg, "obs_dim", 128)

    obs_encoder = VisionPoseObsEncoderNoPCForVelocity(
        state_dim=cfg.state_dim,
        cond_dim=cfg.cond_dim,
        obs_dim=obs_dim,
        guide_dim=cfg.guide_dim,
        embed_dim=cfg.embed_dim,
        input_channels=cfg.input_channels,
        input_transform=cfg.input_transform,
    ).to(device)

    pose_model = VelocityFMTransformer(
        x_dim=9,                 # [pd(3), Rd6d(6)]
        cond_dim=obs_dim,            # nx 维度
        guide_dim=cfg.guide_dim,
        time_dim=cfg.time_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        use_cond=True,
    ).to(device)

    velocity_model = VelocityFMTransformer(
        x_dim=6,
        cond_dim=cfg.cond_dim,
        guide_dim=cfg.guide_dim,
        time_dim=cfg.time_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        use_cond=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(velocity_model.parameters()) + list(obs_encoder.parameters()) + list(pose_model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=1e-6,
    )

    # ===== resume =====
    resume_path = getattr(cfg, "resume_path", None)
    start_epoch = 1
    best_loss = float("inf")
    resume_cond_stats = None

    start_epoch, best_loss, resume_cond_stats, resumed = try_resume_from_checkpoint(
        resume_path=resume_path,
        model=velocity_model,
        obs_encoder=obs_encoder,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # ===== 数据集 =====
    # 如果 resume checkpoint 里有 cond_stats，就直接沿用，保证归一化一致
    train_dataset = RollingForceHistoryFMDataset(
        demo_dir=cfg.train_demo_dir,
        x_hist_len=cfg.x_hist_len,
        pc_hist_len=cfg.pc_hist_len,
        force_hist_len=cfg.force_hist_len,
        pred_horizon=cfg.pred_horizon,
        stride=cfg.stride,
        normalize_v=True,
        cond_stats=resume_cond_stats,
        use_pc_color=cfg.use_pc_color,
        robot_model="indy7",
        # robot_task=cfg.task
        robot_task="sphere",
        use_pose_augmentation=False,
        separation_vp=cfg.separation_vp
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
        use_pc_color=cfg.use_pc_color,
        robot_model="indy7",
        # robot_task=cfg.task
        robot_task="sphere",
        use_pose_augmentation=False,
        separation_vp=cfg.separation_vp
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

    # 如果 resume 的 epoch 已经超过 cfg.epochs，直接退出
    if start_epoch > cfg.epochs:
        print(f"[Resume] start_epoch={start_epoch} > cfg.epochs={cfg.epochs}, nothing to do.")
        return

    for epoch in range(start_epoch, cfg.epochs + 1):
        velocity_model.train()
        pose_model.train()
        obs_encoder.train()

        train_sum, train_count = 0.0, 0
        train_fm_sum, train_dp_sum, train_dR_sum = 0.0, 0.0, 0.0

        for cond_hist, x_now, pc_hist, desired_pose_target, v_future in train_loader:
            cond_hist_flat = cond_hist.to(device).float()
            pc_hist = pc_hist.to(device).float()
            x_now = x_now.to(device).float()
            desired_pose_target = desired_pose_target.to(device).float()  # [B,9]
            v_future = v_future.to(device).float()                        # [B,H,6]

            # obs encoder 只生成条件
            nx_pose, guide_vel = obs_encoder(
                pc_hist,
                x_now,
                cond_hist=cond_hist_flat,
            )

            # ========== 1) pose Flow Matching ==========
            desired_pose_seq = desired_pose_target.unsqueeze(1)  # [B,1,9]

            _, _, t_pose, zt_pose, ut_pose = path_sampler.sample_training_tuple(
                desired_pose_seq
            )

            pred_pose_flow = pose_model(
                x_t=zt_pose,
                t=t_pose,
                cond_main=nx_pose,
                guide=None,
            )

            train_loss_pose_p = F.mse_loss(pred_pose_flow[..., :3], ut_pose[..., :3])
            train_loss_pose_R = F.mse_loss(pred_pose_flow[..., 3:], ut_pose[..., 3:])
            train_loss_pose = 2.0 * train_loss_pose_p + 1.0 * train_loss_pose_R

            # ========== 2) velocity Flow Matching ==========
            _, _, t_v, zt_v, ut_v = path_sampler.sample_training_tuple(v_future)

            pred_v_flow = velocity_model(
                x_t=zt_v,
                t=t_v,
                cond_main=cond_hist_flat,
                guide=guide_vel,
            )

            train_loss_vel = F.mse_loss(pred_v_flow, ut_v)

            train_loss = train_loss_vel + cfg.lambda_delta * train_loss_pose

            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(pose_model.parameters()) + list(obs_encoder.parameters()) + list(velocity_model.parameters()),
                max_norm=0.5
            )
            optimizer.step()

            bs = cond_hist_flat.shape[0]
            train_sum += train_loss.item() * bs
            train_fm_sum += train_loss_vel.item() * bs
            train_dp_sum += train_loss_pose_p.item() * bs
            train_dR_sum += train_loss_pose_R.item() * bs
            train_count += bs

        train_loss = train_sum / max(train_count, 1)
        train_fm = train_fm_sum / max(train_count, 1)
        train_dp = train_dp_sum / max(train_count, 1)
        train_dR = train_dR_sum / max(train_count, 1)

        scheduler.step()

        pose_model.eval()
        obs_encoder.eval()
        velocity_model.eval()

        val_sum, val_count = 0.0, 0
        val_fm_sum, val_dp_sum, val_dR_sum = 0.0, 0.0, 0.0

        # 固定 val 随机性
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)

        with torch.no_grad():
            for cond_hist_flat, x_now, pc_hist, desired_pose_target, v_future in val_loader:
                cond_hist_flat = cond_hist.to(device).float()
                pc_hist = pc_hist.to(device).float()
                x_now = x_now.to(device).float()
                desired_pose_target = desired_pose_target.to(device).float()  # [B,9]
                v_future = v_future.to(device).float()                        # [B,H,6]

                # obs encoder 只生成条件
                nx, guide_feat = obs_encoder(
                    pc_hist,
                    x_now,
                    cond_hist=cond_hist_flat,
                )

                # ========== 1) pose Flow Matching ==========
                desired_pose_seq = desired_pose_target.unsqueeze(1)  # [B,1,9]

                _, _, t_pose, zt_pose, ut_pose = path_sampler.sample_training_tuple(
                    desired_pose_seq
                )

                pred_pose_flow = pose_model(
                    x_t=zt_pose,
                    t=t_pose,
                    cond_main=nx,
                    guide=None,
                )

                val_loss_pose_p = F.mse_loss(pred_pose_flow[..., :3], ut_pose[..., :3])
                val_loss_pose_R = F.mse_loss(pred_pose_flow[..., 3:], ut_pose[..., 3:])
                val_loss_pose = 2.0 * val_loss_pose_p + 1.0 * val_loss_pose_R

                # ========== 2) velocity Flow Matching ==========
                _, _, t_v, zt_v, ut_v = path_sampler.sample_training_tuple(v_future)

                pred_v_flow = velocity_model(
                    x_t=zt_v,
                    t=t_v,
                    cond_main=cond_hist_flat,
                    guide=guide_feat,
                )

                val_loss_vel = F.mse_loss(pred_v_flow, ut_v)

                val_loss = val_loss_vel + cfg.lambda_delta * val_loss_pose

                bs = cond_hist_flat.shape[0]
                val_sum += val_loss.item() * bs
                val_fm_sum += val_loss_vel.item() * bs
                val_dp_sum += val_loss_pose_p.item() * bs
                val_dR_sum += val_loss_pose_R.item() * bs
                val_count += bs

        val_loss = val_sum / max(val_count, 1)
        val_fm = val_fm_sum / max(val_count, 1)
        val_dp = val_dp_sum / max(val_count, 1)
        val_dR = val_dR_sum / max(val_count, 1)

        # ===== 每个 epoch 都保存 last，便于精确 resume =====
        save_train_checkpoint(
            save_path=os.path.join(cfg.save_dir, "last.pt"),
            epoch=epoch,
            model=velocity_model,
            obs_encoder=obs_encoder,
            pose_model=pose_model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_loss=best_loss,
            cond_stats=cond_stats,
            cfg=cfg,
        )

        # ===== best =====
        if val_loss < best_loss:
            best_loss = val_loss
            save_train_checkpoint(
                save_path=os.path.join(cfg.save_dir, f"cfm_{cfg.model}_{cfg.type}_best.pt"),
                epoch=epoch,
                model=velocity_model,
                obs_encoder=obs_encoder,
                pose_model=pose_model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_loss=best_loss,
                cond_stats=cond_stats,
                cfg=cfg,
            )

        current_lr = optimizer.param_groups[0]["lr"]

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, val_loss, current_lr, best_loss,
                train_fm, val_fm, train_dp, val_dp, train_dR, val_dR
            ])

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} "
            f"train_fm_loss={train_fm:.6f} "
            f"val_fm_loss={val_fm:.6f} "
            f"train_dp_loss={train_dp:.6f} "
            f"val_dp_loss={val_dp:.6f} "
            f"train_dR_loss={train_dR:.6f} "
            f"val_dR_loss={val_dR:.6f}"
        )

if __name__ == "__main__":
    type = "random_start"

    cfg = TrainConfig(
        train_demo_dir="/home/zhou/autolab/GUFIC_mujoco-main/boltnut3_vis_demo",
        val_demo_dir="/home/zhou/autolab/GUFIC_mujoco-main/boltnut3_vis_demo",
        type=type,
        epochs=1000,
        batch_size=8,
        save_dir=f"/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_cfm_transformer_pRFe_{type}"
    )
    resume = False

    if resume:
        cfg.resume_path = os.path.join(cfg.save_dir, "last.pt")
    else:
        cfg.resume_path = None

    path_sampler = CurvedPathCFM(alpha=cfg.alpha, eps=cfg.eps)

    if cfg.train_mode == "fixed_length":
        train_velocity_field_fixed_length(cfg, path_sampler)
    elif cfg.train_mode == "rolling_horizon":
        train_velocity_field_rolling_horizon(cfg, path_sampler)
    else:
        raise ValueError(f"Unknown train_mode: {cfg.train_mode}")