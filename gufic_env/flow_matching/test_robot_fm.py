import os
import glob
import math
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as RT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Utils
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wrap_angle_diff(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def unwrap_angle_series(arr_1d):
    arr = arr_1d.copy()
    for k in range(1, len(arr)):
        diff = arr[k] - arr[k - 1]
        if diff > np.pi:
            arr[k:] -= 2 * np.pi
        elif diff < -np.pi:
            arr[k:] += 2 * np.pi
    return arr


def interpolate_teacher_to_pred_time(teacher_t, teacher_x, pred_t):
    """
    将教师轨迹插值到预测轨迹时间轴上
    teacher_t: [T1]
    teacher_x: [T1, D]
    pred_t:    [T2]
    return:    [T2, D]
    """
    teacher_interp = np.zeros((len(pred_t), teacher_x.shape[1]), dtype=np.float32)
    for d in range(teacher_x.shape[1]):
        teacher_interp[:, d] = np.interp(pred_t, teacher_t, teacher_x[:, d])
    return teacher_interp


# ============================================================
# Part 1: Recorder
# ============================================================

class BoltTrajectoryRecorder:
    """
    在线记录一条拧螺栓演示轨迹：
      t, p, R, Vd_star, dVd_star

    保存后自动构造：
      euler, x=[p,euler], goal, total_time
    """
    def __init__(self, save_dir="./bolt_demos"):
        self.save_dir = save_dir
        ensure_dir(self.save_dir)
        self.reset()

    def reset(self):
        self.records = {
            "t": [],
            "p": [],
            "R": [],
            "euler": [],
            "Vd_star": [],
            "dVd_star": [],
        }

    def add(self, t, p, R, Vd_star, dVd_star):
        p = np.asarray(p).reshape(3)
        R = np.asarray(R).reshape(3, 3)
        Vd_star = np.asarray(Vd_star).reshape(6)
        dVd_star = np.asarray(dVd_star).reshape(6)

        euler = RT.from_matrix(R).as_euler("xyz", degrees=False)

        self.records["t"].append(float(t))
        self.records["p"].append(p.astype(np.float32))
        self.records["R"].append(R.astype(np.float32))
        self.records["euler"].append(euler.astype(np.float32))
        self.records["Vd_star"].append(Vd_star.astype(np.float32))
        self.records["dVd_star"].append(dVd_star.astype(np.float32))

    def __len__(self):
        return len(self.records["t"])

    def save(self, episode_name):
        if len(self) == 0:
            raise ValueError("Recorder is empty. Nothing to save.")

        save_path = os.path.join(self.save_dir, f"{episode_name}.npz")

        data = {}
        for k, v in self.records.items():
            data[k] = np.stack(v, axis=0).astype(np.float32)

        data["x"] = np.concatenate([data["p"], data["euler"]], axis=1).astype(np.float32)
        data["goal"] = np.concatenate([data["p"][-1], data["euler"][-1]], axis=0).astype(np.float32)
        data["total_time"] = np.array([data["t"][-1]], dtype=np.float32)

        np.savez_compressed(save_path, **data)
        print(f"[Recorder] saved demo -> {save_path}")
        return save_path

@torch.no_grad()
def integrate_recorded_velocity_full(demo):
    """
    直接对采集到的教师速度场 Vd_star 积分，
    看它是否能重建/跟踪采集轨迹 demo["x"]。

    返回:
      integ_x: [T, 6]
      t_arr:   [T]
    """
    t_arr = demo["t"]
    x = demo["x"][0].copy()
    traj = [x.copy()]

    for k in range(len(t_arr) - 1):
        dt = float(t_arr[k + 1] - t_arr[k])
        dx = demo["Vd_star"][k]   # [6]
        x = x + dx * dt
        traj.append(x.copy())

    integ_x = np.stack(traj, axis=0).astype(np.float32)
    return integ_x, t_arr.copy()


def evaluate_integrated_teacher_vs_demo(integ_x, teacher_x, goal):
    """
    对“教师速度场积分轨迹”与“采集轨迹”做误差评估
    """
    pos_err_curve = np.linalg.norm(integ_x[:, :3] - teacher_x[:, :3], axis=1)
    ori_err_curve = np.linalg.norm(
        wrap_angle_diff(integ_x[:, 3:6] - teacher_x[:, 3:6]), axis=1
    )

    terminal_pos_err_to_goal = np.linalg.norm(integ_x[-1, :3] - goal[:3])
    terminal_ori_err_to_goal = np.linalg.norm(
        wrap_angle_diff(integ_x[-1, 3:6] - goal[3:6])
    )

    terminal_pos_err_to_teacher = np.linalg.norm(integ_x[-1, :3] - teacher_x[-1, :3])
    terminal_ori_err_to_teacher = np.linalg.norm(
        wrap_angle_diff(integ_x[-1, 3:6] - teacher_x[-1, 3:6])
    )

    print("==== Integrate Recorded Teacher Velocity Field ====")
    print(f"mean position tracking error   : {pos_err_curve.mean():.6f} m")
    print(f"max  position tracking error   : {pos_err_curve.max():.6f} m")
    print(f"mean orientation tracking error: {ori_err_curve.mean():.6f} rad")
    print(f"max  orientation tracking error: {ori_err_curve.max():.6f} rad")
    print(f"terminal pos err to goal       : {terminal_pos_err_to_goal:.6f} m")
    print(f"terminal ori err to goal       : {terminal_ori_err_to_goal:.6f} rad")
    print(f"terminal pos err to teacher    : {terminal_pos_err_to_teacher:.6f} m")
    print(f"terminal ori err to teacher    : {terminal_ori_err_to_teacher:.6f} rad")

    return {
        "pos_err_curve": pos_err_curve,
        "ori_err_curve": ori_err_curve,
        "terminal_pos_err_to_goal": terminal_pos_err_to_goal,
        "terminal_ori_err_to_goal": terminal_ori_err_to_goal,
        "terminal_pos_err_to_teacher": terminal_pos_err_to_teacher,
        "terminal_ori_err_to_teacher": terminal_ori_err_to_teacher,
    }


def plot_integrated_teacher_3d(teacher_x, integ_x, goal, save_path):
    """
    采集轨迹 vs 对教师速度场积分得到的轨迹
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        teacher_x[:, 0], teacher_x[:, 1], teacher_x[:, 2],
        "--", linewidth=2.5, label="recorded trajectory"
    )
    ax.plot(
        integ_x[:, 0], integ_x[:, 1], integ_x[:, 2],
        linewidth=2.5, label="integrated teacher velocity"
    )

    ax.scatter(teacher_x[0, 0], teacher_x[0, 1], teacher_x[0, 2], marker="x", s=100, label="start")
    ax.scatter(goal[0], goal[1], goal[2], marker="o", s=80, label="goal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Recorded Trajectory vs Integrated Teacher Velocity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_integrated_teacher_position_error(t_arr, teacher_x, integ_x, save_path):
    pos_err = np.linalg.norm(integ_x[:, :3] - teacher_x[:, :3], axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(t_arr, pos_err, linewidth=2)
    plt.xlabel("time (s)")
    plt.ylabel("position error (m)")
    plt.title("Integrated Teacher Velocity vs Recorded Position Error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_integrated_teacher_orientation_error(t_arr, teacher_x, integ_x, save_path):
    ori_err = np.linalg.norm(
        wrap_angle_diff(integ_x[:, 3:6] - teacher_x[:, 3:6]),
        axis=1
    )

    plt.figure(figsize=(8, 4))
    plt.plot(t_arr, ori_err, linewidth=2)
    plt.xlabel("time (s)")
    plt.ylabel("orientation error (rad)")
    plt.title("Integrated Teacher Velocity vs Recorded Orientation Error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_integrated_teacher_component_curves(t_arr, teacher_x, integ_x, save_path):
    """
    位置和姿态分量逐维对比
    """
    labels = ["px", "py", "pz", "rx", "ry", "rz"]
    fig, axes = plt.subplots(6, 1, figsize=(10, 13), sharex=True)

    for i in range(6):
        if i < 3:
            teacher_curve = teacher_x[:, i]
            integ_curve = integ_x[:, i]
        else:
            teacher_curve = unwrap_angle_series(teacher_x[:, i])
            integ_curve = unwrap_angle_series(integ_x[:, i])

        axes[i].plot(t_arr, teacher_curve, "--", linewidth=2, label="recorded")
        axes[i].plot(t_arr, integ_curve, linewidth=2, label="integrated teacher velocity")
        axes[i].set_ylabel(labels[i])
        axes[i].grid(alpha=0.3)
        if i == 0:
            axes[i].legend()

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Recorded Trajectory vs Integrated Teacher Velocity Components")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

def controller_integration_example():
    example = r'''
# ===== 在你的控制器 __init__ 里加 =====
self.demo_recorder = BoltTrajectoryRecorder(save_dir="./bolt_demos_train")
self.episode_idx = 0

# ===== 在 geometric_unified_force_impedance_control() 里，在拿到 p,R,Vd_star,dVd_star 后加 =====
current_t = self.iter * self.dt

self.demo_recorder.add(
    t=current_t,
    p=p,
    R=R,
    Vd_star=np.asarray(Vd_star).reshape(6),
    dVd_star=np.asarray(dVd_star).reshape(6),
)

# ===== 一条拧螺栓 episode 成功结束后 =====
self.demo_recorder.save(f"bolt_demo_{self.episode_idx:04d}")
self.demo_recorder.reset()
self.episode_idx += 1
'''
    print(example)


# ============================================================
# Part 2: Dataset
# ============================================================

class BoltFlowMatchingFrameDataset(Dataset):
    def __init__(
        self,
        demo_dir="./bolt_demos_train",
        add_state_noise=True,
        pos_noise_std=0.001,
        ori_noise_std=0.01,
        use_dVd_star=False,
    ):
        self.samples = []
        self.add_state_noise = add_state_noise
        self.pos_noise_std = pos_noise_std
        self.ori_noise_std = ori_noise_std
        self.use_dVd_star = use_dVd_star

        demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.npz")))
        if len(demo_files) == 0:
            raise ValueError(f"No demo files found in {demo_dir}")

        for f in demo_files:
            data = np.load(f)

            x = data["x"].astype(np.float32)
            dx = data["Vd_star"].astype(np.float32)
            ddx = data["dVd_star"].astype(np.float32)
            t = data["t"].astype(np.float32)
            goal = data["goal"].astype(np.float32)
            total_time = float(data["total_time"][0])

            for i in range(len(t)):
                item = {
                    "x_t": x[i],
                    "dx_t": dx[i],
                    "goal": goal,
                    "t_norm": np.array([t[i] / max(total_time, 1e-8)], dtype=np.float32),
                }
                if self.use_dVd_star:
                    item["ddx_t"] = ddx[i]
                self.samples.append(item)

        print(f"[Dataset] loaded demos={len(demo_files)}, total_frames={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        x_t = s["x_t"].copy()
        dx_t = s["dx_t"].copy()
        goal = s["goal"].copy()
        t_norm = s["t_norm"].copy()

        if self.add_state_noise:
            x_t[:3] += np.random.normal(scale=self.pos_noise_std, size=(3,)).astype(np.float32)
            x_t[3:6] += np.random.normal(scale=self.ori_noise_std, size=(3,)).astype(np.float32)

        if self.use_dVd_star:
            ddx_t = s["ddx_t"].copy()
            return (
                torch.from_numpy(x_t),
                torch.from_numpy(dx_t),
                torch.from_numpy(goal),
                torch.from_numpy(t_norm),
                torch.from_numpy(ddx_t),
            )

        return (
            torch.from_numpy(x_t),
            torch.from_numpy(dx_t),
            torch.from_numpy(goal),
            torch.from_numpy(t_norm),
        )


# ============================================================
# Part 3: Model
# ============================================================

class TimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half,
                device=t.device,
                dtype=t.dtype,
            )
        )
        angles = t * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb


class BoltVelocityMLP(nn.Module):
    """
    学习连续速度场:
      dx/dt = v_theta(x, t, goal)
    """
    def __init__(self, x_dim=6, goal_dim=6, time_dim=64, hidden_dim=256, num_layers=4):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)

        in_dim = x_dim + goal_dim + time_dim
        layers = []

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [x_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x_t, t, goal):
        t_emb = self.time_emb(t)
        e_t = x_t - goal
        h = torch.cat([e_t, goal, t_emb], dim=-1)
        return self.net(h)


# ============================================================
# Part 4: Train
# ============================================================

@dataclass
class TrainConfig:
    train_demo_dir: str = "./bolt_demos_train"
    val_demo_dir: str = "./bolt_demos_val"
    save_dir: str = "./checkpoints_bolt_real"

    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100

    add_state_noise: bool = True
    pos_noise_std: float = 0.001
    ori_noise_std: float = 0.01

    hidden_dim: int = 256
    num_layers: int = 4
    time_dim: int = 64


def train_velocity_field(cfg: TrainConfig):
    set_seed(42)
    ensure_dir(cfg.save_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = BoltFlowMatchingFrameDataset(
        demo_dir=cfg.train_demo_dir,
        add_state_noise=cfg.add_state_noise,
        pos_noise_std=cfg.pos_noise_std,
        ori_noise_std=cfg.ori_noise_std,
        use_dVd_star=False,
    )

    val_dataset = BoltFlowMatchingFrameDataset(
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

    model = BoltVelocityMLP(
        x_dim=6,
        goal_dim=6,
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

        for x_t, dx_t, goal, t in train_loader:
            x_t = x_t.to(device).float()
            dx_t = dx_t.to(device).float()
            goal = goal.to(device).float()
            t = t.to(device).float()

            pred = model(x_t, t, goal)
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
            for x_t, dx_t, goal, t in val_loader:
                x_t = x_t.to(device).float()
                dx_t = dx_t.to(device).float()
                goal = goal.to(device).float()
                t = t.to(device).float()

                pred = model(x_t, t, goal)
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
                os.path.join(cfg.save_dir, "bolt_velocity_field_best.pt"),
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


# ============================================================
# Part 5: Inference / Rollout
# ============================================================

def load_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_cfg = ckpt["train_cfg"]

    model = BoltVelocityMLP(
        x_dim=6,
        goal_dim=6,
        time_dim=train_cfg["time_dim"],
        hidden_dim=train_cfg["hidden_dim"],
        num_layers=train_cfg["num_layers"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def load_one_demo(npz_path):
    data = np.load(npz_path)
    return {
        "t": data["t"].astype(np.float32),
        "x": data["x"].astype(np.float32),
        "Vd_star": data["Vd_star"].astype(np.float32),
        "dVd_star": data["dVd_star"].astype(np.float32),
        "goal": data["goal"].astype(np.float32),
        "total_time": float(data["total_time"][0]),
    }


@torch.no_grad()
def rollout_velocity_field(model, x0, goal, t_arr, device="cuda"):
    """
    直接使用 demo 原始时间轴 t_arr 做积分，
    保证 pred_x 与 teacher_x 长度一致
    """
    x = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)
    goal_t = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)

    total_time = float(t_arr[-1]) if len(t_arr) > 1 else 1.0

    traj = [x.squeeze(0).cpu().numpy().copy()]
    vel = []

    for k in range(len(t_arr) - 1):
        ti = float(t_arr[k])
        dt = float(t_arr[k + 1] - t_arr[k])

        t_norm = np.array([[ti / max(total_time, 1e-8)]], dtype=np.float32)
        t_tensor = torch.from_numpy(t_norm).to(device)

        dx = model(x, t_tensor, goal_t)
        dx_np = dx.squeeze(0).cpu().numpy().copy()
        vel.append(dx_np)

        x = x + dx * dt
        traj.append(x.squeeze(0).cpu().numpy().copy())

    if len(vel) > 0:
        vel.append(vel[-1].copy())
    else:
        vel.append(np.zeros((6,), dtype=np.float32))

    traj = np.stack(traj, axis=0)
    vel = np.stack(vel, axis=0)

    return traj, vel, t_arr.copy()


def integrate_recorded_velocity(demo):
    """
    sanity check:
    直接积分真实记录下来的 Vd_star，看它本身是否可重建教师轨迹
    """
    t_arr = demo["t"]
    x = demo["x"][0].copy()
    traj = [x.copy()]

    for k in range(len(t_arr) - 1):
        dt = float(t_arr[k + 1] - t_arr[k])
        dx = demo["Vd_star"][k]
        x = x + dx * dt
        traj.append(x.copy())

    traj = np.stack(traj, axis=0)

    pos_err = np.linalg.norm(traj[-1, :3] - demo["goal"][:3])
    ori_err = np.linalg.norm(wrap_angle_diff(traj[-1, 3:6] - demo["goal"][3:6]))

    print("==== Integrate Recorded Vd_star ====")
    print(f"terminal pos err: {pos_err:.6f} m")
    print(f"terminal ori err: {ori_err:.6f} rad")

    return traj


def evaluate_rollout_terminal_error(pred_x, teacher_x, goal):
    pred_final = pred_x[-1]
    teacher_final = teacher_x[-1]

    pos_err_to_goal = np.linalg.norm(pred_final[:3] - goal[:3])
    pos_err_to_teacher = np.linalg.norm(pred_final[:3] - teacher_final[:3])

    ori_err_to_goal = np.linalg.norm(wrap_angle_diff(pred_final[3:6] - goal[3:6]))
    ori_err_to_teacher = np.linalg.norm(wrap_angle_diff(pred_final[3:6] - teacher_final[3:6]))

    return {
        "pos_err_to_goal": pos_err_to_goal,
        "pos_err_to_teacher": pos_err_to_teacher,
        "ori_err_to_goal": ori_err_to_goal,
        "ori_err_to_teacher": ori_err_to_teacher,
    }


@torch.no_grad()
def check_pointwise_prediction(model, demo, device="cuda", num_points=20):
    idxs = np.linspace(0, len(demo["t"]) - 1, num_points).astype(int)

    x_t = torch.tensor(demo["x"][idxs], dtype=torch.float32, device=device)
    goal = torch.tensor(np.repeat(demo["goal"][None, :], len(idxs), axis=0), dtype=torch.float32, device=device)
    t_norm = torch.tensor((demo["t"][idxs] / demo["total_time"])[:, None], dtype=torch.float32, device=device)

    pred = model(x_t, t_norm, goal).cpu().numpy()
    gt = demo["Vd_star"][idxs]

    mse = np.mean((pred - gt) ** 2)
    print(f"[Pointwise check] Vd_star MSE = {mse:.6f}")


@torch.no_grad()
def batch_rollout_test(model, demo_dir, num_tests=20, pos_threshold=0.005, ori_threshold=0.08, device="cuda"):
    demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.npz")))
    if len(demo_files) == 0:
        raise ValueError(f"No demos in {demo_dir}")

    demo_files = demo_files[: min(num_tests, len(demo_files))]

    pos_errors = []
    ori_errors = []
    success_count = 0

    for f in demo_files:
        demo = load_one_demo(f)

        pred_x, pred_dx, pred_t = rollout_velocity_field(
            model=model,
            x0=demo["x"][0],
            goal=demo["goal"],
            t_arr=demo["t"],
            device=device,
        )

        metrics = evaluate_rollout_terminal_error(pred_x, demo["x"], demo["goal"])
        pos_err = metrics["pos_err_to_goal"]
        ori_err = metrics["ori_err_to_goal"]

        pos_errors.append(pos_err)
        ori_errors.append(ori_err)

        if pos_err < pos_threshold and ori_err < ori_threshold:
            success_count += 1

    pos_errors = np.array(pos_errors)
    ori_errors = np.array(ori_errors)

    print("==== Batch Rollout Test ====")
    print(f"num_tests      : {len(demo_files)}")
    print(f"success_rate   : {success_count / len(demo_files):.3f}")
    print(f"mean_pos_err   : {pos_errors.mean():.6f} m")
    print(f"mean_ori_err   : {ori_errors.mean():.6f} rad")

    return {
        "success_rate": success_count / len(demo_files),
        "pos_errors": pos_errors,
        "ori_errors": ori_errors,
    }


# ============================================================
# Part 6: Visualization
# ============================================================

def plot_3d_rollout(teacher_x, pred_x, goal, save_path):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(teacher_x[:, 0], teacher_x[:, 1], teacher_x[:, 2], "--", linewidth=2.5, label="teacher")
    ax.plot(pred_x[:, 0], pred_x[:, 1], pred_x[:, 2], linewidth=2.5, label="rollout")
    ax.scatter(teacher_x[0, 0], teacher_x[0, 1], teacher_x[0, 2], marker="x", s=100, label="start")
    ax.scatter(goal[0], goal[1], goal[2], marker="o", s=80, label="goal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Teacher vs Rollout Trajectory")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_position_error_curve(teacher_t, teacher_x, pred_t, pred_x, save_path):
    teacher_interp = interpolate_teacher_to_pred_time(teacher_t, teacher_x, pred_t)
    pos_err = np.linalg.norm(pred_x[:, :3] - teacher_interp[:, :3], axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(pred_t, pos_err, linewidth=2)
    plt.xlabel("time (s)")
    plt.ylabel("position error (m)")
    plt.title("Rollout Position Error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_velocity_component_curves(teacher_t, teacher_v, pred_t, pred_v, save_path):
    """
    可视化教师速度场 vs 推理速度场的 6 维分量
    """
    teacher_v_interp = interpolate_teacher_to_pred_time(teacher_t, teacher_v, pred_t)

    labels = ["vx", "vy", "vz", "wx", "wy", "wz"]
    fig, axes = plt.subplots(6, 1, figsize=(10, 13), sharex=True)

    for i in range(6):
        axes[i].plot(pred_t, teacher_v_interp[:, i], "--", linewidth=2, label="teacher Vd_star")
        axes[i].plot(pred_t, pred_v[:, i], linewidth=2, label="predicted field")
        axes[i].set_ylabel(labels[i])
        axes[i].grid(alpha=0.3)
        if i == 0:
            axes[i].legend()

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Teacher vs Predicted Velocity Components")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_velocity_magnitude_curves(teacher_t, teacher_v, pred_t, pred_v, save_path):
    """
    可视化线速度/角速度模长
    """
    teacher_v_interp = interpolate_teacher_to_pred_time(teacher_t, teacher_v, pred_t)

    teacher_lin = np.linalg.norm(teacher_v_interp[:, :3], axis=1)
    teacher_ang = np.linalg.norm(teacher_v_interp[:, 3:6], axis=1)
    pred_lin = np.linalg.norm(pred_v[:, :3], axis=1)
    pred_ang = np.linalg.norm(pred_v[:, 3:6], axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(pred_t, teacher_lin, "--", linewidth=2, label="teacher ||v||")
    axes[0].plot(pred_t, pred_lin, linewidth=2, label="predicted ||v||")
    axes[0].set_ylabel("linear speed")
    axes[0].set_title("Linear Speed Magnitude")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(pred_t, teacher_ang, "--", linewidth=2, label="teacher ||w||")
    axes[1].plot(pred_t, pred_ang, linewidth=2, label="predicted ||w||")
    axes[1].set_ylabel("angular speed")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("Angular Speed Magnitude")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_velocity_error_curve(teacher_t, teacher_v, pred_t, pred_v, save_path):
    """
    速度场误差范数曲线 ||v_pred - v_teacher||
    """
    teacher_v_interp = interpolate_teacher_to_pred_time(teacher_t, teacher_v, pred_t)
    vel_err = np.linalg.norm(pred_v - teacher_v_interp, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(pred_t, vel_err, linewidth=2)
    plt.xlabel("time (s)")
    plt.ylabel("velocity error norm")
    plt.title("Velocity Field Error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_velocity_quiver_3d(teacher_t, teacher_x, teacher_v, pred_t, pred_x, pred_v, save_path, stride=200):
    """
    在3D轨迹上画速度箭头
    虚线轨迹: teacher
    实线轨迹: rollout
    蓝色箭头: teacher velocity
    红色箭头: predicted velocity
    """
    teacher_x_interp = interpolate_teacher_to_pred_time(teacher_t, teacher_x, pred_t)
    teacher_v_interp = interpolate_teacher_to_pred_time(teacher_t, teacher_v, pred_t)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(teacher_x_interp[:, 0], teacher_x_interp[:, 1], teacher_x_interp[:, 2], "--", linewidth=2, label="teacher traj")
    ax.plot(pred_x[:, 0], pred_x[:, 1], pred_x[:, 2], linewidth=2, label="rollout traj")

    idx = np.arange(0, len(pred_t), stride)
    if len(idx) == 0:
        idx = np.array([0])

    # teacher arrows
    ax.quiver(
        teacher_x_interp[idx, 0], teacher_x_interp[idx, 1], teacher_x_interp[idx, 2],
        teacher_v_interp[idx, 0], teacher_v_interp[idx, 1], teacher_v_interp[idx, 2],
        length=0.02, normalize=True, alpha=0.8
    )

    # predicted arrows
    ax.quiver(
        pred_x[idx, 0], pred_x[idx, 1], pred_x[idx, 2],
        pred_v[idx, 0], pred_v[idx, 1], pred_v[idx, 2],
        length=0.02, normalize=True, alpha=0.8
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Teacher / Predicted Velocity Field on 3D Trajectory")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_terminal_hist(test_result, save_dir):
    plt.figure(figsize=(8, 4))
    plt.hist(test_result["pos_errors"], bins=20)
    plt.xlabel("terminal position error (m)")
    plt.ylabel("count")
    plt.title("Terminal Position Error Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "terminal_pos_error_hist.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(test_result["ori_errors"], bins=20)
    plt.xlabel("terminal orientation error (rad)")
    plt.ylabel("count")
    plt.title("Terminal Orientation Error Histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "terminal_ori_error_hist.png"), dpi=180)
    plt.close()


def run_inference_and_visualize(
    ckpt_path="./checkpoints_bolt_real/bolt_velocity_field_best.pt",
    demo_path=None,
    demo_dir="./bolt_demos_val",
    out_dir="./bolt_vfield_vis",
):
    ensure_dir(out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt = load_model(ckpt_path, device=device)

    if demo_path is None:
        demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.npz")))
        if len(demo_files) == 0:
            raise ValueError(f"No demos found in {demo_dir}")
        demo_path = demo_files[0]

    demo = load_one_demo(demo_path)

    # --------------------------------------------------------
    # 先检查：直接对采集到的教师速度场 Vd_star 积分，能否跟踪上采集轨迹
    # --------------------------------------------------------
    integ_teacher_x, integ_teacher_t = integrate_recorded_velocity_full(demo)

    teacher_integ_metrics = evaluate_integrated_teacher_vs_demo(
        integ_teacher_x,
        demo["x"],
        demo["goal"],
    )

    plot_integrated_teacher_3d(
        demo["x"],
        integ_teacher_x,
        demo["goal"],
        os.path.join(out_dir, "teacher_velocity_integrated_traj_3d.png"),
    )

    plot_integrated_teacher_position_error(
        integ_teacher_t,
        demo["x"],
        integ_teacher_x,
        os.path.join(out_dir, "teacher_velocity_integrated_position_error.png"),
    )

    plot_integrated_teacher_orientation_error(
        integ_teacher_t,
        demo["x"],
        integ_teacher_x,
        os.path.join(out_dir, "teacher_velocity_integrated_orientation_error.png"),
    )

    plot_integrated_teacher_component_curves(
        integ_teacher_t,
        demo["x"],
        integ_teacher_x,
        os.path.join(out_dir, "teacher_velocity_integrated_components.png"),
    )

    # sanity check 1: 模型在教师点上的拟合误差
    check_pointwise_prediction(model, demo, device=device)

    # sanity check 2: 真实记录的 Vd_star 本身能不能积分回目标
    integrate_recorded_velocity(demo)

    # 直接用 demo 原始时间轴 rollout
    pred_x, pred_dx, pred_t = rollout_velocity_field(
        model=model,
        x0=demo["x"][0],
        goal=demo["goal"],
        t_arr=demo["t"],
        device=device,
    )

    metrics = evaluate_rollout_terminal_error(pred_x, demo["x"], demo["goal"])
    print("==== Single Demo Rollout ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    plot_3d_rollout(demo["x"], pred_x, demo["goal"], os.path.join(out_dir, "traj_3d.png"))
    plot_position_error_curve(
        demo["t"],
        demo["x"],
        pred_t,
        pred_x,
        os.path.join(out_dir, "position_error_curve.png"),
    )

    # 新增：教师速度场 vs 推理速度场可视化
    plot_velocity_component_curves(
        demo["t"],
        demo["Vd_star"],
        pred_t,
        pred_dx,
        os.path.join(out_dir, "velocity_components.png"),
    )

    plot_velocity_magnitude_curves(
        demo["t"],
        demo["Vd_star"],
        pred_t,
        pred_dx,
        os.path.join(out_dir, "velocity_magnitudes.png"),
    )

    plot_velocity_error_curve(
        demo["t"],
        demo["Vd_star"],
        pred_t,
        pred_dx,
        os.path.join(out_dir, "velocity_error_curve.png"),
    )

    plot_velocity_quiver_3d(
        demo["t"],
        demo["x"],
        demo["Vd_star"],
        pred_t,
        pred_x,
        pred_dx,
        os.path.join(out_dir, "velocity_quiver_3d.png"),
        stride=max(1, len(pred_t) // 30),
    )

    test_result = batch_rollout_test(
        model=model,
        demo_dir=demo_dir,
        num_tests=20,
        device=device,
    )
    plot_terminal_hist(test_result, out_dir)

    print(f"Figures saved to: {out_dir}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    mode = "infer"   # record_template / train / infer

    if mode == "record_template":
        controller_integration_example()

    elif mode == "train":
        cfg = TrainConfig(
            train_demo_dir="/home/zhou/autolab/GUFIC_mujoco-main/bolt_demos",
            val_demo_dir="/home/zhou/autolab/GUFIC_mujoco-main/bolt_demos",
            save_dir="./checkpoints_bolt_real",
            batch_size=1024,
            lr=1e-3,
            weight_decay=1e-5,
            epochs=100,
            add_state_noise=True,
            pos_noise_std=0.001,
            ori_noise_std=0.01,
        )
        train_velocity_field(cfg)

    elif mode == "infer":
        run_inference_and_visualize(
            ckpt_path="/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_bolt_real/bolt_velocity_field_best.pt",
            demo_path=None,
            demo_dir="/home/zhou/autolab/GUFIC_mujoco-main/bolt_demos",
            out_dir="/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/bolt_vfield_vis",
        )