import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from model import VelocityFMMLP, TimeEmbedding, VelocityFMTransformer
from config import TrainConfig


# ============================================================
# Load checkpoint
# ============================================================

def load_model(ckpt_path, cfg, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


# ============================================================
# Demo loading
# ============================================================

def load_one_demo(npz_path):
    data = np.load(npz_path)
    demo = {
        "x": data["x"].astype(np.float32),                    # [T,6]
        "t": data["t"].astype(np.float32),                    # [T]
        "total_time": float(data["total_time"][0]),
    }
    if "Vd_star" in data:
        demo["Vd_star"] = data["Vd_star"].astype(np.float32)  # [T,6]
    return demo


# ============================================================
# Modified infer: generate 10000 samples from noise
# ============================================================

@torch.no_grad()
def sample_velocity_trajectory(
    model,
    x_query,
    t_query,
    x1,
    device="cuda",
    steps=100,
    return_history=True,
):
    """
    按你当前修改后的推理逻辑：
      - 初始 x_t 是高斯噪声
      - 迭代 steps 次
      - 每一步都用 model(x_t, t) 预测速度
      - Euler 更新 x_t = x_t + v_pred * dt

    注意：
      这里 x_query / t_query / x1 主要只用来确定样本个数和做后续对比；
      当前生成过程本身并没有真正使用 x_query / x1 条件。

    Returns:
      result = {
        "x_final":   [N,6],
        "v_final":   [N,6],
        "x_history": [steps, N, 6] or None,
        "v_history": [steps, N, 6] or None,
        "step_t":    [steps]
      }
    """
    x_query = np.asarray(x_query, dtype=np.float32)
    N = x_query.shape[0]

    dt = 1.0 / steps

    # 初始噪声
    x_t = torch.randn(1, N, 6, device=device)

    x_history = []
    v_history = []
    step_t = []

    for i in range(steps):
        t_value = torch.full(
            (x_t.shape[0], x_t.shape[1], 1),
            i / steps,
            device=x_t.device,
            dtype=x_t.dtype
        )

        v_pred = model(x_t=x_t, t=t_value)   # [1, N, 6]

        if return_history:
            x_history.append(x_t.squeeze(0).detach().cpu().numpy().copy())
            v_history.append(v_pred.squeeze(0).detach().cpu().numpy().copy())
            step_t.append(i / steps)

        # Euler 更新
        x_t = x_t + v_pred * dt

    x_final = x_t.squeeze(0).detach().cpu().numpy().astype(np.float32)   # [N,6]
    v_final = v_pred.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [N,6]

    result = {
        "x_final": x_final,
        "v_final": v_final,
        "x_history": np.stack(x_history, axis=0).astype(np.float32) if return_history else None,  # [steps,N,6]
        "v_history": np.stack(v_history, axis=0).astype(np.float32) if return_history else None,  # [steps,N,6]
        "step_t": np.array(step_t, dtype=np.float32) if return_history else None,
    }
    return result


# ============================================================
# Visualization helpers
# ============================================================

def plot_generated_state_components(x_pred, x_gt=None, save_path=None):
    """
    横轴 = sample index
    """
    sample_idx = np.arange(len(x_pred))
    labels = ["px", "py", "pz", "rx", "ry", "rz"]

    fig, axes = plt.subplots(6, 1, figsize=(10, 13), sharex=True)

    for i in range(6):
        if x_gt is not None and len(x_gt) == len(x_pred):
            axes[i].plot(sample_idx, x_gt[:, i], "--", linewidth=1.5, label="teacher")
        axes[i].plot(sample_idx, x_pred[:, i], linewidth=1.5, label="generated")
        axes[i].set_ylabel(labels[i])
        axes[i].grid(alpha=0.3)
        if i == 0:
            axes[i].legend()

    axes[-1].set_xlabel("sample index")
    fig.suptitle("Generated State Components")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=180)
        plt.close()
    else:
        plt.show()


def plot_generated_velocity_components(v_pred, v_gt=None, save_path=None):
    """
    横轴 = sample index
    注意：如果 v_gt 存在，这里只是“按索引粗对齐”的比较，
    不代表严格的一一对应。
    """
    sample_idx = np.arange(len(v_pred))
    labels = ["vx", "vy", "vz", "wx", "wy", "wz"]

    fig, axes = plt.subplots(6, 1, figsize=(10, 13), sharex=True)

    for i in range(6):
        if v_gt is not None and len(v_gt) == len(v_pred):
            axes[i].plot(sample_idx, v_gt[:, i], "--", linewidth=1.5, label="teacher")
        axes[i].plot(sample_idx, v_pred[:, i], linewidth=1.5, label="generated")
        axes[i].set_ylabel(labels[i])
        axes[i].grid(alpha=0.3)
        if i == 0:
            axes[i].legend()

    axes[-1].set_xlabel("sample index")
    fig.suptitle("Generated Velocity Components")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=180)
        plt.close()
    else:
        plt.show()


def plot_generated_velocity_error(v_pred, v_gt, save_path=None):
    """
    仅当样本数一致时做一个“粗对齐误差”
    """
    if v_gt is None or len(v_gt) != len(v_pred):
        return

    sample_idx = np.arange(len(v_pred))
    err = np.linalg.norm(v_pred - v_gt, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(sample_idx, err, linewidth=1.5)
    plt.xlabel("sample index")
    plt.ylabel("||v_pred - v_teacher||")
    plt.title("Generated Velocity Error (index-wise rough comparison)")
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


def plot_generated_state_scatter_3d(x_pred, x_gt=None, save_path=None):
    """
    生成状态点云（只看位置前三维）
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        x_pred[:, 0], x_pred[:, 1], x_pred[:, 2],
        s=3, alpha=0.7, label="generated"
    )

    if x_gt is not None:
        ax.scatter(
            x_gt[:, 0], x_gt[:, 1], x_gt[:, 2],
            s=2, alpha=0.3, label="teacher"
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Generated 3D State Samples")
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=180)
        plt.close()
    else:
        plt.show()


def plot_generation_progress(step_t, v_history, save_path=None):
    """
    看整个生成过程中速度模长怎么变化
    """
    if step_t is None or v_history is None:
        return

    # v_history: [steps, N, 6]
    lin_norm = np.linalg.norm(v_history[:, :, :3], axis=2)   # [steps, N]
    ang_norm = np.linalg.norm(v_history[:, :, 3:6], axis=2)  # [steps, N]

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
    axes[1].set_xlabel("generation step time")
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
    out_dir="./fm_field_query_vis",
    max_points=10000,
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainConfig()

    model, cfg, ckpt = load_model(ckpt_path, cfg, device=device)
    demo = load_one_demo(demo_path)

    x = demo["x"]
    t = demo["t"] / max(demo["total_time"], 1e-8)   # 仍然保留，方便统计/对照
    x1 = x[-1]

    if len(x) > max_points:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        x_query = x[idx]
        t_query = t[idx]
        v_gt = demo["Vd_star"][idx] if "Vd_star" in demo else None
    else:
        x_query = x
        t_query = t
        v_gt = demo["Vd_star"] if "Vd_star" in demo else None

    # ========================================================
    # 生成式推理
    # ========================================================
    result = sample_velocity_trajectory(
        model=model,
        x_query=x_query,
        t_query=t_query,
        x1=x1,
        device=device,
        steps=100,
        return_history=True,
    )

    x_pred = result["x_final"]          # [N,6]
    v_pred = result["v_final"]          # [N,6]
    x_history = result["x_history"]     # [steps,N,6]
    v_history = result["v_history"]     # [steps,N,6]
    step_t = result["step_t"]           # [steps]

    print("==== Direct Field Inference (Modified Form) ====")
    print(f"num generated samples : {len(x_pred)}")
    print(f"target x1             : {x1}")

    if v_gt is not None and len(v_gt) == len(v_pred):
        mse = np.mean((v_pred - v_gt) ** 2)
        mae = np.mean(np.abs(v_pred - v_gt))
        err_norm = np.linalg.norm(v_pred - v_gt, axis=1)

        print(f"velocity MSE (rough index match) : {mse:.6f}")
        print(f"velocity MAE (rough index match) : {mae:.6f}")
        print(f"mean ||error||                   : {err_norm.mean():.6f}")
        print(f"max  ||error||                   : {err_norm.max():.6f}")
    else:
        print("No aligned v_gt for strict point-wise comparison.")

    np.savez_compressed(
        os.path.join(out_dir, "predicted_velocity_field.npz"),
        x_query=x_query,
        t_query=t_query,
        x1=x1,
        x_pred=x_pred,
        v_pred=v_pred,
        v_gt=v_gt if v_gt is not None else np.zeros_like(v_pred),
        step_t=step_t,
    )

    # --------------------------------------------------------
    # 适配你当前修改推理形式的新可视化
    # --------------------------------------------------------
    plot_generated_state_components(
        x_pred,
        x_gt=x_query if len(x_query) == len(x_pred) else None,
        save_path=os.path.join(out_dir, "generated_state_components.png"),
    )

    plot_generated_velocity_components(
        v_pred,
        v_gt=v_gt if (v_gt is not None and len(v_gt) == len(v_pred)) else None,
        save_path=os.path.join(out_dir, "generated_velocity_components.png"),
    )

    if v_gt is not None and len(v_gt) == len(v_pred):
        plot_generated_velocity_error(
            v_pred,
            v_gt,
            save_path=os.path.join(out_dir, "generated_velocity_error.png"),
        )

    plot_velocity_norm_hist(
        v_pred,
        v_gt=v_gt if (v_gt is not None and len(v_gt) == len(v_pred)) else None,
        save_path=os.path.join(out_dir, "velocity_norm_hist.png"),
    )

    plot_generated_state_scatter_3d(
        x_pred,
        x_gt=x_query if len(x_query) == len(x_pred) else None,
        save_path=os.path.join(out_dir, "generated_state_scatter_3d.png"),
    )

    plot_generation_progress(
        step_t,
        v_history,
        save_path=os.path.join(out_dir, "generation_progress.png"),
    )

    print(f"Saved to: {out_dir}")
    return {
        "x_query": x_query,
        "t_query": t_query,
        "x1": x1,
        "x_pred": x_pred,
        "v_pred": v_pred,
        "v_gt": v_gt,
        "x_history": x_history,
        "v_history": v_history,
        "step_t": step_t,
    }


if __name__ == "__main__":
    run_direct_field_inference(
        ckpt_path="/home/zhou/autolab/GUFIC_mujoco-main/checkpoints_fm/fm_best.pt",
        demo_path="/home/zhou/autolab/GUFIC_mujoco-main/bolt_demos/bolt_demo_0000.npz",
        out_dir="/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_fm",
        max_points=10000,
    )