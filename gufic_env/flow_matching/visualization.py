import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from scipy.spatial.transform import Rotation as RT
# ============================================================
# Visualization
# ============================================================


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

@torch.no_grad()
def rollout_with_velocity(model, path_sampler, x0, x1, steps=100):
    """
    除了轨迹，还返回每一步的速度向量和速度模长。
    x0, x1: [B, 2]
    返回:
      traj: [steps+1, B, 2]
      vel:  [steps,   B, 2]
      speed:[steps,   B]
    """
    model.eval()
    x = x0.clone()
    traj = [x.clone()]
    vel_list = []
    speed_list = []
    dt = 1.0 / steps

    for i in range(steps):
        t_value = torch.full((x.shape[0], 1), i / steps, device=x.device, dtype=x.dtype)
        v = model(x, t_value, x1)
        vel_list.append(v.clone())
        speed_list.append(torch.norm(v, dim=1))
        x = x + dt * v
        traj.append(x.clone())

    traj = torch.stack(traj, dim=0)
    vel = torch.stack(vel_list, dim=0)
    speed = torch.stack(speed_list, dim=0)
    return traj, vel, speed

def pose_to_state_x(p, R):
    """
    把当前位姿重新转成模型输入状态 x=[p,euler]
    """
    euler = RT.from_matrix(R).as_euler("xyz", degrees=False).astype(np.float32)
    return np.concatenate([p.astype(np.float32), euler], axis=0).astype(np.float32)


def integrate_twist_step(p, R, twist, dt, linear_in_body=True):
    """
    方案A:
      p_{k+1} = p_k + R_k v_k dt      (如果 v 是 body-frame 线速度)
      R_{k+1} = R_k Exp(\hat{w} dt)

    Args:
      p: [3]
      R: [3,3]
      twist: [6] = [vx,vy,vz, wx,wy,wz]
      dt: scalar
      linear_in_body: True 表示线速度在 body frame；False 表示线速度已在 world frame

    Returns:
      p_next, R_next
    """
    p = np.asarray(p, dtype=np.float64).reshape(3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    twist = np.asarray(twist, dtype=np.float64).reshape(6)

    v = twist[:3]
    w = twist[3:]

    if linear_in_body:
        p_next = p + (R @ v) * dt
    else:
        p_next = p + v * dt

    dR = RT.from_rotvec(w * dt).as_matrix()
    R_next = R @ dR

    # 数值上重新正交化一下，避免累计漂移
    U, _, Vt = np.linalg.svd(R_next)
    R_next = U @ Vt
    if np.linalg.det(R_next) < 0:
        U[:, -1] *= -1
        R_next = U @ Vt

    return p_next.astype(np.float32), R_next.astype(np.float32)

@torch.no_grad()
def integrate_recorded_velocity_full(demo, linear_in_body=True):
    p = demo["x"][0, 0:3].copy()
    euler = demo["x"][0, 3:6].copy()
    R = RT.from_euler("xyz", euler, degrees=False).as_matrix().astype(np.float32)

    t_arr = demo["t"]

    traj_x = [pose_to_state_x(p, R)]
    traj_p = [p.copy()]
    traj_R = [R.copy()]

    for k in range(len(t_arr) - 1):
        dt = float(t_arr[k + 1] - t_arr[k])
        twist = demo["Vd_star"][k]

        p, R = integrate_twist_step(
            p, R, twist, dt,
            linear_in_body=linear_in_body,
        )

        traj_p.append(p.copy())
        traj_R.append(R.copy())
        traj_x.append(pose_to_state_x(p, R))

    result = {
        "x": np.stack(traj_x, axis=0).astype(np.float32),
        "p": np.stack(traj_p, axis=0).astype(np.float32),
        "R": np.stack(traj_R, axis=0).astype(np.float32),
        "t": t_arr.copy().astype(np.float32),
    }
    return result["x"], result["t"]

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


from scipy.spatial.transform import Rotation as RT
import numpy as np
import matplotlib.pyplot as plt

def plot_velocity_quiver_3d(
    teacher_t,
    teacher_x,
    teacher_v,
    pred_t,
    pred_x,
    pred_v,
    save_path,
    stride=200,
    linear_in_body=True,
):
    """
    在3D轨迹上画速度箭头

    虚线轨迹: teacher
    实线轨迹: rollout
    蓝色箭头: teacher linear velocity
    红色箭头: predicted linear velocity

    注意:
      - 这里只画线速度，不画角速度
      - 如果线速度在 body frame，需要先变换到 world frame:
            v_world = R @ v_body
    """
    teacher_x_interp = interpolate_teacher_to_pred_time(teacher_t, teacher_x, pred_t)
    teacher_v_interp = interpolate_teacher_to_pred_time(teacher_t, teacher_v, pred_t)

    teacher_pos = teacher_x_interp[:, 0:3]
    pred_pos = pred_x[:, 0:3]

    teacher_euler = teacher_x_interp[:, 3:6]
    pred_euler = pred_x[:, 3:6]

    teacher_R = RT.from_euler("xyz", teacher_euler, degrees=False).as_matrix()   # [T,3,3]
    pred_R = RT.from_euler("xyz", pred_euler, degrees=False).as_matrix()         # [T,3,3]

    teacher_lin = teacher_v_interp[:, 0:3].copy()
    pred_lin = pred_v[:, 0:3].copy()

    if linear_in_body:
        teacher_lin_world = np.einsum("nij,nj->ni", teacher_R, teacher_lin)
        pred_lin_world = np.einsum("nij,nj->ni", pred_R, pred_lin)
    else:
        teacher_lin_world = teacher_lin
        pred_lin_world = pred_lin

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        teacher_pos[:, 0], teacher_pos[:, 1], teacher_pos[:, 2],
        "--", linewidth=2, label="teacher traj"
    )
    ax.plot(
        pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2],
        linewidth=2, label="rollout traj"
    )

    idx = np.arange(0, len(pred_t), stride)
    if len(idx) == 0:
        idx = np.array([0])

    # teacher linear velocity arrows (world frame)
    ax.quiver(
        teacher_pos[idx, 0], teacher_pos[idx, 1], teacher_pos[idx, 2],
        teacher_lin_world[idx, 0], teacher_lin_world[idx, 1], teacher_lin_world[idx, 2],
        length=0.02, normalize=True, alpha=0.8
    )

    # predicted linear velocity arrows (world frame)
    ax.quiver(
        pred_pos[idx, 0], pred_pos[idx, 1], pred_pos[idx, 2],
        pred_lin_world[idx, 0], pred_lin_world[idx, 1], pred_lin_world[idx, 2],
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