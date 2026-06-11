from gufic_env.flow_matching.dataset import (
    rotmat_batch_to_rot6d,
    uniform_sample_one_frame,
    get_hand_eye_from_xml,
    pointcloud_cam_to_world_batch,
)
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from gufic_env.flow_matching.model import (
    VelocityFMMLP,
    VelocityFMTransformer,
    VelocityFMCondUnet1D,
    VisionDeltaPoseNet,
    VisionPoseObsEncoder,
    VisionPoseObsEncoderPoseSimpleV2,
    VisionPoseObsEncoderNoPCForVelocity,
)
from gufic_env.flow_matching.config import TrainConfig


# ============================================================
# Config / checkpoint loading
# ============================================================

def build_cfg_from_ckpt(ckpt_config: dict):
    """用 checkpoint 里的配置覆盖 TrainConfig 默认值。"""
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

    ckpt_cfg = ckpt.get("train_cfg", {})
    cfg = build_cfg_from_ckpt(ckpt_cfg)

    model_name = getattr(cfg, "model", "mlp")

    if model_name == "transformer":
        model = VelocityFMTransformer(
            x_dim=6,
            cond_dim=cfg.cond_dim,
            guide_dim=getattr(cfg, "guide_dim", 16),
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
            use_cond=False,
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

    # 按照 PointNet/obs_encoder 提条件，pose_model 用 Flow Matching 生成期望位姿的结构加载。
    # obs_encoder.forward(pc, x_now, cond_hist) -> nx, guide_feat
    obs_dim = int(getattr(cfg, "obs_dim", 128))

    obs_encoder = VisionPoseObsEncoderNoPCForVelocity(
        state_dim=cfg.state_dim,
        cond_dim=cfg.cond_dim,
        obs_dim=obs_dim,
        guide_dim=cfg.guide_dim,
        embed_dim=cfg.embed_dim,
        input_channels=cfg.input_channels,
        input_transform=cfg.input_transform,
    ).to(device)
    obs_encoder.load_state_dict(ckpt["obs_encoder"])
    obs_encoder.eval()

    if "pose_model" not in ckpt:
        raise KeyError(
            "Checkpoint 中没有 pose_model。请确认训练脚本已经保存 "
            "'pose_model': pose_model.state_dict()。"
        )

    pose_model = VelocityFMTransformer(
        x_dim=9,                 # [pd(3), Rd6d(6)]
        cond_dim=obs_dim,        # nx 维度，必须和 VisionPoseObsEncoder 输出一致
        guide_dim=cfg.guide_dim,
        time_dim=cfg.time_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        use_cond=True,
    ).to(device)
    pose_model.load_state_dict(ckpt["pose_model"])
    pose_model.eval()

    # 兼容两种命名：新版 cond_stats，旧版 stats
    stats = ckpt.get("cond_stats", None)
    if stats is None:
        stats = ckpt.get("stats", None)

    if stats is None:
        raise ValueError("Checkpoint 中没有找到 cond_stats / stats。")


    return model, obs_encoder, pose_model, cfg, ckpt, stats


# ============================================================
# Normalization helpers
# ============================================================
def recover_desired_pose_from_abs(desired_pose_pred_norm, stats):
    desired_pose_pred_norm = np.asarray(
        desired_pose_pred_norm,
        dtype=np.float32,
    ).reshape(9)

    pd = denormalize_data(
        desired_pose_pred_norm[:3][None, :],
        stats,
        "desired_p",
    ).reshape(3)

    Rd6d = denormalize_data(
        desired_pose_pred_norm[3:][None, :],
        stats,
        "desired_R",
    ).reshape(6)

    Rd = rot6d_to_rotmat_np(Rd6d)
    return pd.astype(np.float32), Rd.astype(np.float32)

def normalize_data(data, stats, key="v"):
    return (data - stats[f"{key}_mean"]) / stats[f"{key}_std"]


def denormalize_data(data, stats, key="v"):
    return data * stats[f"{key}_std"] + stats[f"{key}_mean"]


def get_velocity_key(stats):
    """
    如果当前模型训练目标是 Vd_body_future，则 checkpoint 中应包含 vd_mean/vd_std。
    否则退回到 v_mean/v_std。
    """
    if "vd_mean" in stats and "vd_std" in stats:
        return "vd"
    return "v"


def pointcloud_cam_to_ee_batch(pc, R_ec, t_ec):
    """
    Eye-in-hand 点云从相机系直接变到末端系：
        x_e = R_ec @ x_c + t_ec

    这样和采样时的当前 p/R 无关，pc_hist_len > 1 时也不会把历史点云错用当前位姿变换。
    """
    pc = np.asarray(pc, dtype=np.float32)
    xyz_cam = pc[..., :3]
    R_ec = np.asarray(R_ec, dtype=np.float32).reshape(3, 3)
    t_ec = np.asarray(t_ec, dtype=np.float32).reshape(3)
    xyz_ee = np.einsum("ij,tpj->tpi", R_ec, xyz_cam) + t_ec.reshape(1, 1, 3)
    return xyz_ee.astype(np.float32)


# ============================================================
# Demo loading
# ============================================================

def load_one_demo(npz_path):
    data = np.load(npz_path)
    demo = {
        "v": data["Vd_star"].astype(np.float32),       # [T, 6]
        "p": data["p"].astype(np.float32),             # [T, 3]
        "R": data["R"].astype(np.float32),             # [T, 3, 3]
        "fe": data["Fe"].astype(np.float32),           # [T, 6]
        "pc": data["point_cloud"].astype(np.float32),  # [T, P, C]
        "t": data["t"].astype(np.float32),             # [T]
        "total_time": float(data["total_time"][0]),
    }

    # 真实期望位姿，用于画 VDP-Net 预测的 pd/Rd 对比曲线
    demo["pd"] = data["pd"].astype(np.float32) if "pd" in data else None
    demo["Rd"] = data["Rd"].astype(np.float32) if "Rd" in data else None

    # 可选：真实期望轨迹速度，如果你后续想对比 Vd_body
    if "dpd" in data and "dRd" in data and "Rd" in data:
        demo["dpd"] = data["dpd"].astype(np.float32)
        demo["dRd"] = data["dRd"].astype(np.float32)
    else:
        demo["dpd"] = None
        demo["dRd"] = None

    return demo


# ============================================================
# Lie / rotation helpers
# ============================================================

def vee_map(R):
    v3 = -R[0, 1]
    v1 = -R[1, 2]
    v2 = R[0, 2]
    return np.array([v1, v2, v3], dtype=np.float32).reshape((-1, 1))


def hat_map(w):
    wx, wy, wz = w.reshape(3)
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ],
        dtype=np.float32,
    )


def rot6d_to_rotmat_np(r6d: np.ndarray) -> np.ndarray:
    """
    r6d: [6]，格式与 rotmat_to_rot6d_one / rotmat_batch_to_rot6d 一致：
         [R[:,0], R[:,1]]
    return: [3,3]
    """
    r6d = np.asarray(r6d, dtype=np.float32).reshape(6)

    a1 = r6d[:3]
    a2 = r6d[3:6]

    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2_orth = a2 - np.dot(b1, a2) * b1
    b2 = a2_orth / (np.linalg.norm(a2_orth) + 1e-8)
    b3 = np.cross(b1, b2)

    R = np.stack([b1, b2, b3], axis=1)
    return R.astype(np.float32)


def rotation_geodesic_error_deg(R_pred, R_gt):
    """
    R_pred: [N,3,3]
    R_gt:   [N,3,3]
    return: [N], unit: degree
    """
    R_err = np.einsum("nij,njk->nik", np.transpose(R_pred, (0, 2, 1)), R_gt)
    trace = np.trace(R_err, axis1=1, axis2=2)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return theta * 180.0 / np.pi


def recover_pose_from_delta(
    p_now_raw: np.ndarray,
    R_now_raw: np.ndarray,
    delta_pose_pred_norm: np.ndarray,
    stats: dict,
):
    """
    根据 VDP-Net 输出的归一化局部 delta pose，恢复世界系预测期望位姿。

    Dataset 中 delta_pose_target 的定义是：
        delta_p_local = R_now.T @ (p_d - p_now)
        R_rel = R_now.T @ R_d

    因此推理恢复：
        p_d_pred = p_now + R_now @ delta_p_local
        R_d_pred = R_now @ R_rel_pred
    """
    delta_pose_pred_norm = np.asarray(delta_pose_pred_norm, dtype=np.float32).reshape(9)

    # 1. 反归一化 delta p / delta R6D
    delta_p_local = denormalize_data(
        delta_pose_pred_norm[:3][None, :], stats, "delta_p"
    ).reshape(3).astype(np.float32)

    delta_R6d = denormalize_data(
        delta_pose_pred_norm[3:9][None, :], stats, "delta_R"
    ).reshape(6).astype(np.float32)

    # 2. 6D rotation -> relative rotation
    R_rel_pred = rot6d_to_rotmat_np(delta_R6d)

    # 3. 局部增量恢复成世界系期望位姿
    p_des_pred = p_now_raw.reshape(3) + R_now_raw.reshape(3, 3) @ delta_p_local
    R_des_pred = R_now_raw.reshape(3, 3) @ R_rel_pred

    return (
        p_des_pred.astype(np.float32),
        R_des_pred.astype(np.float32),
        delta_p_local.astype(np.float32),
        R_rel_pred.astype(np.float32),
    )


def vd_body_to_dpd_dRd(Vd_body, Rd):
    """
    Vd_body: [6] = [vd_body, wd_body]
    Rd: [3,3]

    return:
        dpd: [3]
        dRd: [3,3]
    """
    vd_body = Vd_body[:3].reshape(3)
    wd_body = Vd_body[3:].reshape(3)

    dpd = Rd @ vd_body
    dRd = Rd @ hat_map(wd_body)

    return dpd.astype(np.float32), dRd.astype(np.float32)


def get_velocity_field(g, pd, Rd, dpd, dRd, zeta_v=50.0, zeta_w=10.0):
    """
    根据预测的 pd/Rd 和 Vd_body 转换得到的 dpd/dRd，
    使用解析几何速度场公式恢复最终 Vd_star。
    """
    p = g[:3, 3]
    R = g[:3, :3]

    Vd_star = np.zeros(6, dtype=np.float32)

    vd_star = (
        R.T @ dRd @ Rd.T @ (p - pd)
        + R.T @ dpd
        - zeta_v * R.T @ (p - pd)
    )

    wd_star = vee_map(
        R.T @ dRd @ Rd.T @ R
        - zeta_w * (Rd.T @ R - R.T @ Rd)
    ).reshape((-1,))

    Vd_star[:3] = vd_star
    Vd_star[3:] = wd_star

    return Vd_star.astype(np.float32)


# ============================================================
# FM sampling
# ============================================================

@torch.no_grad()
def sample_velocity_trajectory(
    model,
    obs_encoder,
    pose_model,
    traj_len,
    stats,
    device="cuda",
    steps=100,
    return_history=True,
    seed=None,
    cfg=None,
    cond=None,
    cond_pc_np=None,
    vel_key="v",
):
    """
    条件 / 无条件 FM 采样。
    若 stats 中存在 vd_mean/vd_std，则默认生成对象是 Vd_body。
    否则默认生成对象是 Vd_star。
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    flow_dt = 1.0 / steps

    # 初始噪声：速度轨迹样本，normalized space
    v_t = torch.randn(1, traj_len, 6, device=device)

    fe_cond = None
    guide_feat = None
    desired_pose_pred_norm = None

    use_cond = bool(getattr(cfg, "use_cond", False)) if cfg is not None else False

    if use_cond:
        if cond is None:
            raise ValueError("cfg.use_cond=True 时，cond 不能为 None。")
        if cond_pc_np is None:
            raise ValueError("cfg.use_cond=True 且使用 obs_encoder 时，cond_pc_np 不能为 None。")

        cond_np = cond.astype(np.float32) if isinstance(cond, np.ndarray) else np.asarray(cond, dtype=np.float32)

        if cond_np.ndim == 2:
            cond_np = cond_np.reshape(-1)
        if cond_np.ndim == 1:
            cond_np = cond_np[None, :]
        if cond_np.ndim != 2:
            raise ValueError(f"cond 期望形状为 [K,D] 或 [D]，当前是 {cond_np.shape}")

        fe_cond = torch.from_numpy(cond_np).to(device).float()

        if hasattr(cfg, "cond_dim") and fe_cond.shape[-1] != cfg.cond_dim:
            raise ValueError(
                f"cond 维度不匹配: got {fe_cond.shape[-1]}, expected {cfg.cond_dim}"
            )

        # x_now 是 cond_main 中最后一帧归一化状态 [p_norm, R6d_norm]
        now_left = 9 * (cfg.x_hist_len - 1)
        now_right = 9 * cfg.x_hist_len
        x_now = fe_cond[:, now_left:now_right]

        cond_pc = torch.from_numpy(cond_pc_np).to(device).float()
        cond_pc = cond_pc.unsqueeze(0)  # [1, H, P, C] 或 [1, P, C]

        # obs_encoder 只编码条件，不直接预测位姿：
        #   nx:         给 pose_model 作为 global condition
        #   guide_feat: 给 velocity model 作为 FiLM guide
        nx, guide_feat = obs_encoder(
            cond_pc,
            x_now,
            cond_hist=fe_cond,
        )

        if pose_model is None:
            raise ValueError("use_cond=True 时必须传入 pose_model。")

        # 生成绝对期望位姿 [pd, Rd6d]，normalized space。
        # 离线评估可以用随机初值；在线控制建议改成 zeros 或固定低通。
        z_pose = torch.randn(1, 1, 9, device=device)

        for i in range(steps):
            t_pose = torch.full(
                (1, 1, 1),
                i / steps,
                device=device,
                dtype=z_pose.dtype,
            )

            u_pose = pose_model(
                x_t=z_pose,
                t=t_pose,
                cond_main=nx,
                guide=None,
            )

            z_pose = z_pose + u_pose * (1.0 / steps)

        desired_pose_pred_norm = (
            z_pose.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        )  # [9]

    v_sample_history_norm = []
    u_history_norm = []
    step_t = []

    if return_history:
        v_sample_history_norm.append(v_t.squeeze(0).detach().cpu().numpy().copy())
        step_t.append(0.0)

    for i in range(steps):
        t_value = torch.full(
            (1, 1, 1),
            i / steps,
            device=v_t.device,
            dtype=v_t.dtype,
        )

        if use_cond:
            u_pred = model(
                x_t=v_t,
                t=t_value,
                cond_main=fe_cond,
                guide=guide_feat,
            )
        else:
            u_pred = model(x_t=v_t, t=t_value)

        if return_history:
            u_history_norm.append(u_pred.squeeze(0).detach().cpu().numpy().copy())

        v_t = v_t + u_pred * flow_dt

        if return_history:
            v_sample_history_norm.append(v_t.squeeze(0).detach().cpu().numpy().copy())
            step_t.append((i + 1) / steps)

    v_sample_final_norm = (
        v_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
    )  # [T,6]
    u_final_norm = (
        u_pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
    )  # [T,6]

    v_sample_final = denormalize_data(v_sample_final_norm, stats, vel_key).astype(np.float32)

    if return_history:
        v_sample_history_norm = np.stack(v_sample_history_norm, axis=0).astype(np.float32)
        v_sample_history = denormalize_data(v_sample_history_norm, stats, vel_key).astype(np.float32)
        u_history_norm = np.stack(u_history_norm, axis=0).astype(np.float32)
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
        "desired_pose_pred_norm": desired_pose_pred_norm,
        "v_sample_history": v_sample_history,
        "v_sample_history_norm": v_sample_history_norm,
        "u_history_norm": u_history_norm,
        "step_t": step_t,
        "vel_key": vel_key,
    }
    return result


# ============================================================
# Visualization helpers
# ============================================================

def plot_generated_velocity_components(v_pred, v_gt=None, save_path=None):
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
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(v_pred[:, 0], v_pred[:, 1], v_pred[:, 2], s=3, alpha=0.7, label="generated")

    if v_gt is not None:
        ax.scatter(v_gt[:, 0], v_gt[:, 1], v_gt[:, 2], s=2, alpha=0.3, label="teacher")

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
    if step_t is None or v_sample_history is None:
        return

    lin_norm = np.linalg.norm(v_sample_history[:, :, :3], axis=2)
    ang_norm = np.linalg.norm(v_sample_history[:, :, 3:6], axis=2)

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


def plot_desired_pose_comparison(pd_pred, Rd_pred, pd_gt, Rd_gt, save_dir):
    """
    pd_pred: [N,3]
    Rd_pred: [N,3,3]
    pd_gt:   [N,3]
    Rd_gt:   [N,3,3]
    """
    os.makedirs(save_dir, exist_ok=True)
    step_idx = np.arange(len(pd_pred))

    # 1. pd 分量对比
    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i in range(3):
        axes[i].plot(step_idx, pd_gt[:, i], "--", linewidth=1.5, label=f"pd_gt_{labels[i]}")
        axes[i].plot(step_idx, pd_pred[:, i], linewidth=1.5, label=f"pd_pred_{labels[i]}")
        axes[i].set_ylabel(f"p_d {labels[i]}")
        axes[i].grid(alpha=0.3)
        axes[i].legend()

    axes[-1].set_xlabel("trajectory step")
    fig.suptitle("Desired Position Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pd_comparison.png"), dpi=180)
    plt.close()

    # 2. pd 误差范数
    pd_err = np.linalg.norm(pd_pred - pd_gt, axis=1)

    plt.figure(figsize=(9, 4))
    plt.plot(step_idx, pd_err, linewidth=1.5)
    plt.xlabel("trajectory step")
    plt.ylabel(r"$||\hat p_d - p_d||$")
    plt.title("Desired Position Error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pd_error_norm.png"), dpi=180)
    plt.close()

    # 3. Rd 测地线误差
    rot_err_deg = rotation_geodesic_error_deg(Rd_pred, Rd_gt)

    plt.figure(figsize=(9, 4))
    plt.plot(step_idx, rot_err_deg, linewidth=1.5)
    plt.xlabel("trajectory step")
    plt.ylabel("rotation error [deg]")
    plt.title("Desired Orientation Geodesic Error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Rd_geodesic_error_deg.png"), dpi=180)
    plt.close()

    # 4. Rd 矩阵元素对比
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    for r in range(3):
        for c in range(3):
            ax = axes[r, c]
            ax.plot(step_idx, Rd_gt[:, r, c], "--", linewidth=1.2, label="gt")
            ax.plot(step_idx, Rd_pred[:, r, c], linewidth=1.2, label="pred")
            ax.set_title(f"R[{r},{c}]")
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend()

    fig.suptitle("Desired Rotation Matrix Components")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Rd_matrix_components.png"), dpi=180)
    plt.close()

    print("==== Desired Pose Prediction Error ====")
    print(f"pd mean error     : {pd_err.mean():.6f}")
    print(f"pd max error      : {pd_err.max():.6f}")
    print(f"Rd mean error deg : {rot_err_deg.mean():.6f}")
    print(f"Rd max error deg  : {rot_err_deg.max():.6f}")

    return {"pd_err": pd_err, "rot_err_deg": rot_err_deg}


# ============================================================
# Main infer entry
# ============================================================

def run_direct_field_inference(
    ckpt_path,
    demo_path,
    out_dir="./infer_fm",
    max_points=15000,
    steps=100,
    seed=None,
    robot_model=None,
    robot_task=None,
    separation_vp=True,
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, obs_encoder, pose_model, cfg, ckpt, stats = load_model(ckpt_path, device=device)
    demo = load_one_demo(demo_path)

    pd_pred_arr = None
    Rd_pred_arr = None
    pd_gt_arr = None
    Rd_gt_arr = None
    vel_key = get_velocity_key(stats)
    if separation_vp and vel_key != "vd":
        raise ValueError("separation_vp=True，但 checkpoint stats 中没有 vd_mean/vd_std。")

    if cfg.train_mode == "fixed_length":
        v_gt = demo["v"]
        cond = demo["fe"] if getattr(cfg, "use_cond", False) else None

        if len(v_gt) > max_points:
            idx = np.linspace(0, len(v_gt) - 1, max_points).astype(int)
            v_gt = v_gt[idx]

        traj_len = len(v_gt)

        result = sample_velocity_trajectory(
            model=model,
            obs_encoder=obs_encoder,
            pose_model=pose_model,
            traj_len=traj_len,
            stats=stats,
            device=device,
            steps=steps,
            return_history=True,
            seed=seed,
            cfg=cfg,
            cond=cond,
        )

        v_sample_pred = result["v_sample_final"]
        v_sample_pred_norm = result["v_sample_final_norm"]
        u_final_norm = result["u_final_norm"]
        v_sample_history = result["v_sample_history"]
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

        pd_pred_list = []
        Rd_pred_list = []
        pd_gt_list = []
        Rd_gt_list = []

        traj_len = cfg.pred_horizon
        R_ec, t_ec, _ = get_hand_eye_from_xml(robot_model, robot_task)

        loop_N = len(demo["v"]) - 1
        if max_points is not None:
            loop_N = min(loop_N, max_points)

        for i in range(loop_N):
            fe_left = max(0, i - cfg.force_hist_len + 1)
            x_left = max(0, i - cfg.x_hist_len + 1)
            pc_left = max(0, i - cfg.pc_hist_len + 1)

            # 1. 历史力
            cond_fe = demo["fe"][fe_left: i + 1]
            cond_fe = normalize_data(cond_fe, stats, "fe").astype(np.float32)

            # 2. 历史位姿
            p_raw = demo["p"][x_left: i + 1].astype(np.float32)
            R_raw = demo["R"][x_left: i + 1].astype(np.float32)
            R6d = rotmat_batch_to_rot6d(R_raw)

            p_norm = normalize_data(p_raw, stats, "p").astype(np.float32)
            R6d_norm = normalize_data(R6d, stats, "R").astype(np.float32)
            cond_x = np.concatenate([p_norm, R6d_norm], axis=-1)

            # 3. 当前/历史点云
            if cfg.use_pc_color:
                cond_pc = demo["pc"][pc_left: i + 1].astype(np.float32)
            else:
                cond_pc = demo["pc"][pc_left: i + 1, :, :3].astype(np.float32)

            # 4. padding
            if cond_fe.shape[0] < cfg.force_hist_len:
                pad_len = cfg.force_hist_len - cond_fe.shape[0]
                cond_fe = np.pad(cond_fe, ((pad_len, 0), (0, 0)), mode="constant")

            if cond_x.shape[0] < cfg.x_hist_len:
                pad_len = cfg.x_hist_len - cond_x.shape[0]
                pad = np.repeat(cond_x[0:1], pad_len, axis=0)
                cond_x = np.concatenate([pad, cond_x], axis=0)

            if cond_pc.shape[0] < cfg.pc_hist_len:
                pad_len = cfg.pc_hist_len - cond_pc.shape[0]
                pad_pc = np.repeat(cond_pc[0:1], pad_len, axis=0)
                cond_pc = np.concatenate([pad_pc, cond_pc], axis=0)

            # 5. 点云转到末端系。
            # Eye-in-hand 外参固定，camera -> ee 不需要依赖当前 p/R；
            # 这样 pc_hist_len > 1 时历史点云也不会被当前位姿错误变换。
            pc_ee = pointcloud_cam_to_ee_batch(cond_pc, R_ec, t_ec)
            pc_ee = pc_ee / 0.1

            cond_pc = np.stack(
                [uniform_sample_one_frame(pc_t, 2048, use_xyz_only=True) for pc_t in pc_ee],
                axis=0,
            ).astype(np.float32)

            # 注意： Dataset 训练时点云缩放了两次，这里保持一致；
            # 如果修正了 Dataset 的重复缩放，这里也要同步去掉下面这一行。
            cond_pc = cond_pc.astype(np.float32)

            # 6. 条件拼接
            cond = np.concatenate(
                [cond_x.reshape(1, -1), cond_fe.reshape(1, -1)], axis=-1
            )

            result = sample_velocity_trajectory(
                model=model,
                obs_encoder=obs_encoder,
                pose_model=pose_model,
                traj_len=traj_len,
                stats=stats,
                device=device,
                steps=steps,
                return_history=True,
                seed=seed,
                cfg=cfg,
                cond=cond,
                cond_pc_np=cond_pc,
                vel_key=vel_key
            )
            if separation_vp:
                # 7. pose_model 生成的是绝对期望位姿 [pd, Rd6d]，不是局部 delta pose
                desired_pose_pred_norm = result["desired_pose_pred_norm"]
                p_now_raw = demo["p"][i].astype(np.float32)
                R_now_raw = demo["R"][i].astype(np.float32)

                p_des_pred, R_des_pred = recover_desired_pose_from_abs(
                    desired_pose_pred_norm=desired_pose_pred_norm,
                    stats=stats,
                )

                pd_pred_list.append(p_des_pred.astype(np.float32))
                Rd_pred_list.append(R_des_pred.astype(np.float32))

                if demo["pd"] is not None and demo["Rd"] is not None:
                    # 你的 Dataset 当前 m=0，因此这里对齐 i
                    pd_gt_list.append(demo["pd"][i].astype(np.float32))
                    Rd_gt_list.append(demo["Rd"][i].astype(np.float32))

                # 8. FM 生成 Vd_body，并结合预测 pd/Rd 解析构造 Vd_star_pred
                Vd_body_pred = result["v_sample_final"]  # [H,6]，若 stats 有 vd_mean/vd_std，则为 Vd_body
                Vd_body_now = Vd_body_pred[0]

                pd = p_des_pred
                Rd = R_des_pred
                dpd, dRd = vd_body_to_dpd_dRd(Vd_body_now, Rd)

                g_now = np.eye(4, dtype=np.float32)
                g_now[:3, :3] = R_now_raw
                g_now[:3, 3] = p_now_raw

                Vd_star_pred = get_velocity_field(
                    g=g_now,
                    pd=pd,
                    Rd=Rd,
                    dpd=dpd,
                    dRd=dRd,
                )
                v_sample_final.append(Vd_star_pred)
            else:
                # Direct Vd_star mode: the sampled trajectory is [H, 6], but
                # rolling offline evaluation compares one current command per i.
                v_sample_final.append(result["v_sample_final"][0])
            v_sample_final_norm.append(result["v_sample_final_norm"][0])
            u_final_norm.append(result["u_final_norm"][0])
            v_sample_history.append(result["v_sample_history"][:, 0, :])
            v_sample_history_norm.append(result["v_sample_history_norm"][:, 0, :])
            u_history_norm.append(result["u_history_norm"][:, 0, :])
            step_t.append(result["step_t"])

        v_sample_pred = np.stack(v_sample_final, axis=0).astype(np.float32)
        v_sample_pred_norm = np.stack(v_sample_final_norm, axis=0).astype(np.float32)
        u_final_norm = np.stack(u_final_norm, axis=0).astype(np.float32)
        v_sample_history = np.stack(v_sample_history, axis=0).astype(np.float32)
        v_sample_history_norm = np.stack(v_sample_history_norm, axis=0).astype(np.float32)
        u_history_norm = np.stack(u_history_norm, axis=0).astype(np.float32)
        step_t = np.stack(step_t, axis=0).astype(np.float32)

        # 与当前 i 对齐，Vd_star_pred 是在每个 i 时刻用当前 g_now/pd/Rd 构造的
        v_gt = demo["v"][:len(v_sample_pred)]

        # 9. pd/Rd 预测对比可视化
        if len(pd_pred_list) > 0 and len(pd_gt_list) == len(pd_pred_list):
            pd_pred_arr = np.stack(pd_pred_list, axis=0).astype(np.float32)
            Rd_pred_arr = np.stack(Rd_pred_list, axis=0).astype(np.float32)
            pd_gt_arr = np.stack(pd_gt_list, axis=0).astype(np.float32)
            Rd_gt_arr = np.stack(Rd_gt_list, axis=0).astype(np.float32)

            pose_err = plot_desired_pose_comparison(
                pd_pred=pd_pred_arr,
                Rd_pred=Rd_pred_arr,
                pd_gt=pd_gt_arr,
                Rd_gt=Rd_gt_arr,
                save_dir=out_dir,
            )

            np.savez_compressed(
                os.path.join(out_dir, "desired_pose_prediction_comparison.npz"),
                pd_pred=pd_pred_arr,
                Rd_pred=Rd_pred_arr,
                pd_gt=pd_gt_arr,
                Rd_gt=Rd_gt_arr,
                pd_err=pose_err["pd_err"],
                rot_err_deg=pose_err["rot_err_deg"],
            )
        else:
            print("[Warning] No valid pd/Rd ground truth found for desired pose comparison.")

    else:
        raise ValueError(f"Unknown train_mode: {cfg.train_mode}")

    print("==== FM Velocity Generation ====")
    print(f"generated traj len : {len(v_sample_pred)}")

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

    # np.savez_compressed(
    #     os.path.join(out_dir, "generated_velocity_trajectory.npz"),
    #     v_pred=v_sample_pred,
    #     v_pred_norm=v_sample_pred_norm,
    #     v_gt=v_gt if v_gt is not None else np.zeros_like(v_sample_pred),
    #     step_t=step_t,
    #     v_mean=stats["v_mean"],
    #     v_std=stats["v_std"],
    #     pd_pred=pd_pred_arr if pd_pred_arr is not None else np.array([]),
    #     Rd_pred=Rd_pred_arr if Rd_pred_arr is not None else np.array([]),
    #     pd_gt=pd_gt_arr if pd_gt_arr is not None else np.array([]),
    #     Rd_gt=Rd_gt_arr if Rd_gt_arr is not None else np.array([]),
    # )

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

    # 可选：生成过程可视化
    # plot_generation_progress(
    #     step_t,
    #     v_sample_history,
    #     save_path=os.path.join(out_dir, "generation_progress.png"),
    # )

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
        "pd_pred": pd_pred_arr,
        "Rd_pred": Rd_pred_arr,
        "pd_gt": pd_gt_arr,
        "Rd_gt": Rd_gt_arr,
    }


if __name__ == "__main__":
    # type = "fixed_start"
    type = "random_start"
    robot_name = "indy7"
    robot_task = "bolt"

    if robot_task == "sphere":
        ckpt_path = f"/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_cfm_transformer_vis_pRFe_{type}/cfm_transformer_vis2pose_{type}_best19.pt"
        demo_path = "/home/zhou/autolab/GUFIC_mujoco-main/bolt_vis_demo/bolt_demo_0098.npz"
        out_dir = f"/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_cfm_transformer_vis_pRFe_{type}"

    elif robot_task == "insertion":
        ckpt_path = f"/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_cfm_transformer_insertion_vis_pRFe_{type}/cfm_transformer_{type}_best1.pt"
        demo_path = "/home/zhou/autolab/GUFIC_mujoco-main/insertion_vis_demo/bolt_demo_0171.npz"
        out_dir = f"/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_cfm_transformer_insertion_vis_pRFe_{type}"

    elif robot_task == "bolt":
        ckpt_path = f"/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_cfm_transformer_boltnut_vis_pRFe_{type}/cfm_transformer_{type}_best39.pt"
        demo_path = "/home/zhou/autolab/GUFIC_mujoco-main/boltnut3_vis_demo/bolt_demo_0000.npz"
        out_dir = f"/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_cfm_transformer_boltnut_vis_pRFe_{type}"

    else:
        raise ValueError(f"Unknown robot_task: {robot_task}")

    run_direct_field_inference(
        ckpt_path=ckpt_path,
        demo_path=demo_path,
        out_dir=out_dir,
        max_points=15000,
        steps=10,
        seed=42,
        robot_model=robot_name,
        robot_task=robot_task,
    )
