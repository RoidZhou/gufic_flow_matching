import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from gufic_env.flow_matching.model import VelocityFMMLP, VelocityFMTransformer, VelocityFMCondUnet1D
from  gufic_env.flow_matching.config import TrainConfig


# ============================================================
# Config / checkpoint loading
# ============================================================

def build_cfg_from_ckpt(ckpt_config: dict):
    """
    用 checkpoint 里的配置覆盖 TrainConfig 默认值
    """
    cfg = TrainConfig()
    if ckpt_config is not None:
        for k, v in ckpt_config.items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    return cfg


def load_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    ckpt_cfg = ckpt.get("config", {})
    cfg = build_cfg_from_ckpt(ckpt_cfg)

    model_name = getattr(cfg, "model", "mlp")

    if model_name == "transformer":
        model = VelocityFMTransformer(
            x_dim=6,
            cond_dim=cfg.cond_dim,
            time_dim=cfg.time_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            use_cond=True,
        ).to(device)
    elif model_name == "unet":
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
            use_cond=False,
        ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    stats = ckpt.get("v_stats", None)
    if stats is None:
        raise ValueError("Checkpoint 中没有找到 stats，请先在训练保存时把 v_mean / v_std 一起存进去。")

    if "v_mean" not in stats or "v_std" not in stats:
        raise ValueError("stats 里必须包含 'v_mean' 和 'v_std'。")

    return model, cfg, ckpt, stats


# ============================================================
# Normalization helpers
# 只对 v 做归一化 / 反归一化
# ============================================================

def normalize_v(v, stats):
    return (v - stats["v_mean"]) / stats["v_std"]


def denormalize_v(v, stats):
    return v * stats["v_std"] + stats["v_mean"]


# ============================================================
# Demo loading
# 主对象统一成 Vd_star
# ============================================================

def load_one_demo(npz_path):
    data = np.load(npz_path)
    demo = {
        "v": data["Vd_star"].astype(np.float32),      # [T, 6]
        "fe": data["Fe"].astype(np.float32),      # [T, 6]
        "t": data["t"].astype(np.float32),            # [T]
        "total_time": float(data["total_time"][0]),
    }
    return demo


# ============================================================
# Unconditional FM sampling
# 生成的是速度轨迹样本 v，而不是状态 x
# ============================================================

@torch.no_grad()
def sample_velocity_trajectory(
    model,
    traj_len,
    stats,
    device="cuda",
    steps=100,
    return_history=True,
    seed=None,
    cfg=None,
    cond=None,
):
    """
    条件 / 无条件 FM 采样

    无条件:
      v_t ~ N(0, I)
      u_pred = model(v_t, t)
      v_t <- v_t + u_pred * dt

    条件:
      v_t ~ N(0, I)
      u_pred = model(v_t, t, fe_cond)
      v_t <- v_t + u_pred * dt

    这里:
      - v_t 是当前生成中的“速度轨迹样本”（normalized space）
      - u_pred 是 FM 的流速度（normalized space）
      - 最终生成结果是 v_sample_final，而不是 u_final

    Args:
      traj_len:       轨迹长度 T
      stats:          {"v_mean": ..., "v_std": ...}
      steps:          ODE 采样步数
      return_history: 是否返回采样历史
      seed:           随机种子，可选
      cfg:            训练配置，要求含 use_cond / cond_dim
      cond:           条件输入
                      - 若使用最近 K 步力序列，可传 [K, 6]
                      - 或传已经 flatten 后的 [6*K]

    Returns:
      result = {
        "v_sample_final":       [T, 6]   # 反归一化后的最终生成速度轨迹
        "v_sample_final_norm":  [T, 6]   # 归一化空间里的最终轨迹
        "u_final_norm":         [T, 6]   # 最后一步模型输出（flow velocity）
        "v_sample_history":     [steps+1, T, 6] or None   # 反归一化后的历史
        "v_sample_history_norm":[steps+1, T, 6] or None   # 归一化空间里的历史
        "u_history_norm":       [steps, T, 6] or None
        "step_t":               [steps+1]
      }
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    dt = 1.0 / steps

    # 初始噪声：速度轨迹样本（normalized space）
    v_t = torch.randn(1, traj_len, 6, device=device)

    # --------------------------------------------------------
    # 处理条件：最近 K 步力序列
    # cond 支持:
    #   [K, 6]   -> 自动 flatten 成 [1, 6*K]
    #   [6*K]    -> 自动变成 [1, 6*K]
    # --------------------------------------------------------
    fe_cond = None
    use_cond = bool(getattr(cfg, "use_cond", False)) if cfg is not None else False

    if use_cond:
        if cond is None:
            raise ValueError("cfg.use_cond=True 时，cond 不能为 None。")

        if isinstance(cond, np.ndarray):
            cond_np = cond.astype(np.float32)
        else:
            cond_np = np.asarray(cond, dtype=np.float32)

        # cond: [K,6] -> [6K]
        if cond_np.ndim == 2:
            cond_np = cond_np.reshape(-1)

        # cond: [6K] -> [1,6K]
        if cond_np.ndim == 1:
            cond_np = cond_np[None, :]

        # 最终要求 [1, cond_dim]
        if cond_np.ndim != 2:
            raise ValueError(f"cond 期望形状为 [K,6] 或 [6*K]，当前是 {cond_np.shape}")

        fe_cond = torch.from_numpy(cond_np).to(device).float()   # [1, cond_dim]

        # 可选检查
        if hasattr(cfg, "cond_dim"):
            if fe_cond.shape[-1] != cfg.cond_dim:
                raise ValueError(
                    f"cond 维度不匹配: got {fe_cond.shape[-1]}, expected {cfg.cond_dim}"
                )

    v_sample_history_norm = []
    u_history_norm = []
    step_t = []

    if return_history:
        v_sample_history_norm.append(v_t.squeeze(0).detach().cpu().numpy().copy())
        step_t.append(0.0)

    for i in range(steps):
        # flow time，对整条轨迹共用一个标量
        t_value = torch.full(
            (1, 1, 1),
            i / steps,
            device=v_t.device,
            dtype=v_t.dtype
        )

        if use_cond:
            # Transformer forward 支持 fe: [B, cond_dim]
            # 内部会自动扩成 [B, T, cond_dim]
            u_pred = model(x_t=v_t, t=t_value, fe=fe_cond)   # [1, T, 6]
        else:
            u_pred = model(x_t=v_t, t=t_value)               # [1, T, 6]

        if return_history:
            u_history_norm.append(u_pred.squeeze(0).detach().cpu().numpy().copy())

        # Euler 更新
        v_t = v_t + u_pred * dt

        if return_history:
            v_sample_history_norm.append(v_t.squeeze(0).detach().cpu().numpy().copy())
            step_t.append((i + 1) / steps)

    v_sample_final_norm = v_t.squeeze(0).detach().cpu().numpy().astype(np.float32)   # [T,6]
    u_final_norm = u_pred.squeeze(0).detach().cpu().numpy().astype(np.float32)        # [T,6]

    # 只对 v 做反归一化
    v_sample_final = denormalize_v(v_sample_final_norm, stats).astype(np.float32)

    if return_history:
        v_sample_history_norm = np.stack(v_sample_history_norm, axis=0).astype(np.float32)  # [steps+1, T, 6]
        v_sample_history = denormalize_v(v_sample_history_norm, stats).astype(np.float32)
        u_history_norm = np.stack(u_history_norm, axis=0).astype(np.float32)                 # [steps, T, 6]
        step_t = np.array(step_t, dtype=np.float32)
    else:
        v_sample_history_norm = None
        v_sample_history = None
        u_history_norm = None
        step_t = None

    result = {
        "v_sample_final": v_sample_final,
        "v_sample_final_norm": v_sample_final_norm,
        "u_final_norm": u_final_norm,
        "v_sample_history": v_sample_history,
        "v_sample_history_norm": v_sample_history_norm,
        "u_history_norm": u_history_norm,
        "step_t": step_t,
    }
    return result


# ============================================================
# Visualization helpers
# 全部围绕 v / Vd_star
# ============================================================

def plot_generated_velocity_components(v_pred, v_gt=None, save_path=None):
    """
    横轴 = trajectory step index
    """
    step_idx = np.arange(len(v_pred))
    labels = ["vx", "vy", "vz", "wx", "wy", "wz"]

    fig, axes = plt.subplots(6, 1, figsize=(10, 13), sharex=True)

    for i in range(6):
        if v_gt is not None and len(v_gt) == len(v_pred):
            axes[i].plot(step_idx, v_gt[:, i], "--", linewidth=1.5, label="teacher")
        axes[i].plot(step_idx, v_pred[:, i], linewidth=1.5, label="generated")
        axes[i].set_ylabel(labels[i])
        axes[i].grid(alpha=0.3)
        if i == 0:
            axes[i].legend()

    axes[-1].set_xlabel("trajectory step")
    fig.suptitle("Generated Velocity Trajectory Components")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=180)
        plt.close()
    else:
        plt.show()


def plot_generated_velocity_error(v_pred, v_gt, save_path=None):
    """
    注意：
      这只是在“单条生成轨迹 vs 单条 demo 轨迹”上的逐点误差。
      对无条件 FM，这只是参考，不是最核心指标。
    """
    if v_gt is None or len(v_gt) != len(v_pred):
        return

    step_idx = np.arange(len(v_pred))
    err = np.linalg.norm(v_pred - v_gt, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(step_idx, err, linewidth=1.5)
    plt.xlabel("trajectory step")
    plt.ylabel("||v_pred - v_teacher||")
    plt.title("Velocity Trajectory Error")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=180)
        plt.close()
    else:
        plt.show()


def plot_velocity_norm_hist(v_pred, v_gt=None, save_path=None):
    """
    线速度 / 角速度模长分布
    """
    pred_lin = np.linalg.norm(v_pred[:, :3], axis=1)
    pred_ang = np.linalg.norm(v_pred[:, 3:6], axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    axes[0].hist(pred_lin, bins=50, alpha=0.7, label="generated")
    if v_gt is not None:
        gt_lin = np.linalg.norm(v_gt[:, :3], axis=1)
        axes[0].hist(gt_lin, bins=50, alpha=0.5, label="teacher")
    axes[0].set_title("Linear Velocity Norm Distribution")
    axes[0].set_xlabel("||v||")
    axes[0].set_ylabel("count")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(pred_ang, bins=50, alpha=0.7, label="generated")
    if v_gt is not None:
        gt_ang = np.linalg.norm(v_gt[:, 3:6], axis=1)
        axes[1].hist(gt_ang, bins=50, alpha=0.5, label="teacher")
    axes[1].set_title("Angular Velocity Norm Distribution")
    axes[1].set_xlabel("||w||")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=180)
        plt.close()
    else:
        plt.show()


def plot_generated_linear_velocity_scatter_3d(v_pred, v_gt=None, save_path=None):
    """
    3D 线速度点云：(vx, vy, vz)
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        v_pred[:, 0], v_pred[:, 1], v_pred[:, 2],
        s=3, alpha=0.7, label="generated"
    )

    if v_gt is not None:
        ax.scatter(
            v_gt[:, 0], v_gt[:, 1], v_gt[:, 2],
            s=2, alpha=0.3, label="teacher"
        )

    ax.set_xlabel("vx")
    ax.set_ylabel("vy")
    ax.set_zlabel("vz")
    ax.set_title("Generated Linear Velocity Samples")
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=180)
        plt.close()
    else:
        plt.show()


def plot_generation_progress(step_t, v_sample_history, save_path=None):
    """
    看整个生成过程中，生成出来的速度轨迹样本模长怎么变化
    这里用的是“生成样本本身”，不是 u_pred
    """
    if step_t is None or v_sample_history is None:
        return

    # v_sample_history: [steps+1, T, 6]
    lin_norm = np.linalg.norm(v_sample_history[:, :, :3], axis=2)   # [steps+1, T]
    ang_norm = np.linalg.norm(v_sample_history[:, :, 3:6], axis=2)  # [steps+1, T]

    mean_lin = lin_norm.mean(axis=1)
    std_lin = lin_norm.std(axis=1)

    mean_ang = ang_norm.mean(axis=1)
    std_ang = ang_norm.std(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(step_t, mean_lin, linewidth=2, label="mean linear norm")
    axes[0].fill_between(step_t, mean_lin - std_lin, mean_lin + std_lin, alpha=0.2)
    axes[0].set_ylabel("||v||")
    axes[0].set_title("Generation Progress: Linear Velocity Norm")
    axes[0].grid(alpha=0.3)

    axes[1].plot(step_t, mean_ang, linewidth=2, label="mean angular norm")
    axes[1].fill_between(step_t, mean_ang - std_ang, mean_ang + std_ang, alpha=0.2)
    axes[1].set_ylabel("||w||")
    axes[1].set_xlabel("generation time")
    axes[1].set_title("Generation Progress: Angular Velocity Norm")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=180)
        plt.close()
    else:
        plt.show()


# ============================================================
# Main infer entry
# ============================================================

def run_direct_field_inference(
    ckpt_path,
    demo_path,
    out_dir="./infer_fm",
    max_points=10000,
    steps=100,
    seed=None,
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, cfg, ckpt, stats = load_model(ckpt_path, device=device)

    demo = load_one_demo(demo_path)

    # ========================================================
    # 无条件/条件 FM 速度轨迹采样
    # ========================================================
    if cfg.train_mode == "fixed_length":
        v_gt = demo["v"]
        if cfg.use_cond:
            cond = demo["fe"]
        else:
            cond = None

        if len(v_gt) > max_points:
            idx = np.linspace(0, len(v_gt) - 1, max_points).astype(int)
            v_gt = v_gt[idx]
        traj_len = len(v_gt)

        result = sample_velocity_trajectory(
            model=model,
            traj_len=traj_len,
            stats=stats,
            device=device,
            steps=steps,
            return_history=True,
            seed=seed,
            cfg=cfg,
            cond=cond,
        )

        v_sample_pred = result["v_sample_final"]          # [T,6]，真正的生成结果（反归一化后）
        v_sample_pred_norm = result["v_sample_final_norm"]
        u_final_norm = result["u_final_norm"]             # 最后一步 flow velocity（normalized space）
        v_sample_history = result["v_sample_history"]     # [steps+1, T, 6]（反归一化后）
        v_sample_history_norm = result["v_sample_history_norm"]
        u_history_norm = result["u_history_norm"]
        step_t = result["step_t"]
    elif cfg.train_mode == "rolling_horizon":
        v_sample_final = []
        v_sample_final_norm = []
        u_final_norm = []
        v_sample_history = []
        v_sample_history_norm = []
        u_history_norm = []
        step_t = []

        traj_len = cfg.pred_horizon

        # 滚动 horizon 模式下 demo 轨迹太长了，直接用 max_points 定死生成长度
        for i in range(len(demo["v"])-1):
            left = max(0, i-cfg.force_hist_len+1)
            cond = demo["fe"][left : i + 1]      # [K,6]，滚动取最近 K 步力作为条件
            if cond.shape[0] < cfg.force_hist_len:
                # 如果不足 K 步历史，就在前面补零
                pad_len = cfg.force_hist_len - cond.shape[0]
                cond = np.pad(cond, ((pad_len, 0), (0, 0)), mode="constant")

            result = sample_velocity_trajectory(
                model=model,
                traj_len=traj_len,
                stats=stats,
                device=device,
                steps=steps,
                return_history=True,
                seed=seed,
                cfg=cfg,
                cond=cond,
            )
            
            v_sample_final.append(result["v_sample_final"][0:cfg.stride,:])
            v_sample_final_norm.append(result["v_sample_final_norm"][0:cfg.stride,:])
            u_final_norm.append(result["u_final_norm"][0:cfg.stride,:])
            v_sample_history.append(result["v_sample_history"][0:cfg.stride,:])
            v_sample_history_norm.append(result["v_sample_history_norm"][0:cfg.stride,:])
            u_history_norm.append(result["u_history_norm"][0:cfg.stride,:])
            step_t.append(result["step_t"][0:cfg.stride])
        
        v_sample_pred = np.stack(v_sample_final, axis=0).astype(np.float32) 
        v_sample_pred_norm = np.stack(v_sample_final_norm, axis=0).astype(np.float32)
        u_final_norm = np.stack(u_final_norm, axis=0).astype(np.float32)
        v_sample_history = np.stack(v_sample_history, axis=0).astype(np.float32)
        v_sample_history_norm = np.stack(v_sample_history_norm, axis=0).astype(np.float32)
        u_history_norm = np.stack(u_history_norm, axis=0).astype(np.float32)
        step_t = np.stack(step_t, axis=0).astype(np.float32)

        v_sample_pred = v_sample_pred[:, 0, :]
        v_sample_pred_norm = v_sample_pred_norm[:, 0, :]
        u_final_norm = u_final_norm[:, 0, :]
        v_sample_history = v_sample_history[:, 0, :]
        v_sample_history_norm = v_sample_history_norm[:, 0, :]
        u_history_norm = u_history_norm[:, 0, :]
        step_t = step_t[:, 0]
        v_gt = demo["v"][1:1+len(v_sample_pred)]

    else:
        raise ValueError(f"Unknown train_mode: {cfg.train_mode}")



    print("==== Unconditional FM Velocity Generation ====")
    print(f"generated traj len : {len(v_sample_pred)}")

    # 注意：这是单条生成轨迹 vs 单条 demo 的参考误差
    if v_gt is not None and len(v_gt) == len(v_sample_pred):
        mse = np.mean((v_sample_pred - v_gt) ** 2)
        mae = np.mean(np.abs(v_sample_pred - v_gt))
        err_norm = np.linalg.norm(v_sample_pred - v_gt, axis=1)

        print(f"velocity MSE   : {mse:.6f}")
        print(f"velocity MAE   : {mae:.6f}")
        print(f"mean ||error|| : {err_norm.mean():.6f}")
        print(f"max  ||error|| : {err_norm.max():.6f}")
    else:
        print("No aligned v_gt for point-wise comparison.")

    np.savez_compressed(
        os.path.join(out_dir, "generated_velocity_trajectory.npz"),
        v_pred=v_sample_pred,
        v_pred_norm=v_sample_pred_norm,
        v_gt=v_gt if v_gt is not None else np.zeros_like(v_sample_pred),
        step_t=step_t,
        v_mean=stats["v_mean"],
        v_std=stats["v_std"],
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    plot_generated_velocity_components(
        v_sample_pred,
        v_gt=v_gt if (v_gt is not None and len(v_gt) == len(v_sample_pred)) else None,
        save_path=os.path.join(out_dir, "generated_velocity_components.png"),
    )

    if v_gt is not None and len(v_gt) == len(v_sample_pred):
        plot_generated_velocity_error(
            v_sample_pred,
            v_gt,
            save_path=os.path.join(out_dir, "generated_velocity_error.png"),
        )

    plot_velocity_norm_hist(
        v_sample_pred,
        v_gt=v_gt if (v_gt is not None and len(v_gt) == len(v_sample_pred)) else None,
        save_path=os.path.join(out_dir, "velocity_norm_hist.png"),
    )

    plot_generated_linear_velocity_scatter_3d(
        v_sample_pred,
        v_gt=v_gt if (v_gt is not None and len(v_gt) == len(v_sample_pred)) else None,
        save_path=os.path.join(out_dir, "generated_linear_velocity_scatter_3d.png"),
    )

    plot_generation_progress(
        step_t,
        v_sample_history,
        save_path=os.path.join(out_dir, "generation_progress.png"),
    )

    print(f"Saved to: {out_dir}")
    return {
        "v_gt": v_gt,
        "v_sample_pred": v_sample_pred,
        "v_sample_pred_norm": v_sample_pred_norm,
        "u_final_norm": u_final_norm,
        "v_sample_history": v_sample_history,
        "v_sample_history_norm": v_sample_history_norm,
        "u_history_norm": u_history_norm,
        "step_t": step_t,
    }


if __name__ == "__main__":
    run_direct_field_inference(
        ckpt_path="/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_cfm_transformer_fixed_start/fm_transformer_best_4.22v2.pt",
        demo_path="/home/zhou/autolab/GUFIC_mujoco-main/bolt_demos/bolt_demo_0000.npz",
        out_dir="/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_cfm_transformer_fixed_start",
        max_points=10000,
        steps=10,
        seed=42,
    )