import os
import random
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model import VelocityRegressiveMLP
from cfm import CurvedPathCFM
from visualization import integrate_recorded_velocity_full, evaluate_integrated_teacher_vs_demo, plot_integrated_teacher_3d, plot_integrated_teacher_position_error, plot_integrated_teacher_orientation_error
from visualization import plot_integrated_teacher_component_curves, check_pointwise_prediction, integrate_recorded_velocity, evaluate_rollout_terminal_error, plot_3d_rollout, plot_position_error_curve
from visualization import plot_velocity_component_curves, plot_velocity_error_curve, plot_velocity_quiver_3d, plot_terminal_hist, pose_to_state_x
import glob
from scipy.spatial.transform import Rotation as RT

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_one_demo(npz_path):
    data = np.load(npz_path)
    return {
        "t": data["t"].astype(np.float32),
        "x": data["x"].astype(np.float32),
        "Vd_star": data["Vd_star"].astype(np.float32),
        "dVd_star": data["dVd_star"].astype(np.float32),
        "fe": data["Fe"].astype(np.float32),
        "goal": data["goal"].astype(np.float32),
        "total_time": float(data["total_time"][0]),
    }

def load_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_cfg = ckpt["train_cfg"]

    model = VelocityRegressiveMLP(
        x_dim=6,
        cond_dim=6,
        time_dim=train_cfg["time_dim"],
        hidden_dim=train_cfg["hidden_dim"],
        num_layers=train_cfg["num_layers"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt

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
def rollout_velocity_field(
    model,
    p0,
    R0,
    cond,
    t_arr,
    device="cuda",
    linear_in_body=True,
):
    """
    方案A版 rollout:
      每步先把当前 (p,R) 转成 x=[p,euler] 喂给模型，
      模型输出 twist，
      然后在 p,R 上做物理一致的积分。

    Returns:
      result = {
        'x': [T,6],
        'p': [T,3],
        'R': [T,3,3],
        'V': [T,6],
        't': [T],
      }
    """
    p = np.asarray(p0, dtype=np.float32).reshape(3).copy()
    R0 = RT.from_euler("xyz", R0, degrees=False).as_matrix() 
    R = np.asarray(R0, dtype=np.float32).reshape(3, 3).copy()
    cond = np.asarray(cond, dtype=np.float32)

    total_time = float(t_arr[-1]) if len(t_arr) > 1 else 1.0

    traj_x = [pose_to_state_x(p, R)]
    traj_p = [p.copy()]
    traj_R = [R.copy()]
    traj_v = []

    for k in range(len(t_arr) - 1):
        ti = float(t_arr[k])
        dt = float(t_arr[k + 1] - t_arr[k])

        x_np = pose_to_state_x(p, R)
        x_t = torch.from_numpy(x_np[None, :]).to(device)

        t_norm = np.array([[ti / max(total_time, 1e-8)]], dtype=np.float32)
        t_tensor = torch.from_numpy(t_norm).to(device)
        cond_t = torch.from_numpy(cond[k, :]).to(device).unsqueeze(0)

        twist = model(x_t, t_tensor, cond_t).cpu().numpy().reshape(-1).astype(np.float32)
        traj_v.append(twist.copy())

        p, R = integrate_twist_step(
            p, R, twist, dt,
            linear_in_body=linear_in_body,
        )

        traj_p.append(p.copy())
        traj_R.append(R.copy())
        traj_x.append(pose_to_state_x(p, R))

    if len(traj_v) > 0:
        traj_v.append(traj_v[-1].copy())
    else:
        traj_v.append(np.zeros((6,), dtype=np.float32))

    result = {
        "x": np.stack(traj_x, axis=0).astype(np.float32),
        "p": np.stack(traj_p, axis=0).astype(np.float32),
        "R": np.stack(traj_R, axis=0).astype(np.float32),
        "V": np.stack(traj_v, axis=0).astype(np.float32),
        "t": t_arr.copy().astype(np.float32),
    }
    return result

@torch.no_grad()
def batch_rollout_test(
    model,
    demo_dir,
    num_tests=20,
    pos_threshold=0.005,
    ori_threshold=0.08,
    device="cuda",
    linear_in_body=True,
):
    demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.npz")))
    if len(demo_files) == 0:
        raise ValueError(f"No demos in {demo_dir}")

    demo_files = demo_files[: min(num_tests, len(demo_files))]

    pos_errors = []
    ori_errors = []
    success_count = 0

    for f in demo_files:
        demo = load_one_demo(f)
        cond = select_condition_from_demo(demo)

        pred = rollout_velocity_field(
            model=model,
            p0=demo["x"][:, 0:3][0],
            R0=demo["x"][:, 3:][0],
            cond=cond,
            t_arr=demo["t"],
            device=device,
            linear_in_body=linear_in_body,
        )

        metrics = evaluate_rollout_terminal_error(pred["x"], demo["x"], demo["goal"])
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

def select_condition_from_demo(demo):
    """
    自动选择模型条件:
    1) 如果 demo 有 fe，就优先用 fe
    2) 否则用 goal

    你当前代码里 rollout_velocity_field 已经改成 fe 条件了，
    但 batch_rollout_test 里还是按 goal 调，存在不一致。
    这里统一自动处理。
    """
    if "fe" in demo:
        fe = demo["fe"]
        if fe.ndim == 1:
            return fe.astype(np.float32)
        elif fe.ndim == 2:
            # 如果保存成逐时刻 fe，这里先取第一个时刻。
            # 若你想做时变条件，再单独扩展。
            return fe.astype(np.float32)

    return demo["goal"].astype(np.float32)

def run_inference_and_visualize(
    ckpt_path="./checkpoints_bolt_real/bolt_velocity_field_best.pt",
    demo_path=None,
    demo_dir="./bolt_demos_val",
    out_dir="./bolt_vfield_vis",
    linear_in_body=True,
):
    """
    linear_in_body=True:
      假设模型/教师速度场的前三维线速度是在 body frame 下。
      如果你确认它其实已经是 world frame，改成 False。
    """
    ensure_dir(out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt = load_model(ckpt_path, device=device)

    if demo_path is None:
        demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.npz")))
        if len(demo_files) == 0:
            raise ValueError(f"No demos found in {demo_dir}")
        demo_path = demo_files[0]

    demo = load_one_demo(demo_path)
    cond = select_condition_from_demo(demo)

    # --------------------------------------------------------
    # 先检查：直接对采集到的教师速度场 Vd_star 做 scheme A 积分
    # --------------------------------------------------------
    integ_teacher_x, integ_teacher_t = integrate_recorded_velocity_full(
        demo,
        linear_in_body=linear_in_body,
    )

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

    # rollout (方案A)
    pred = rollout_velocity_field(
        model=model,
        p0=demo["x"][:, 0:3][0],
        R0=demo["x"][:, 3:][0],
        cond=cond,
        t_arr=demo["t"],
        device=device,
        linear_in_body=linear_in_body,
    )

    pred_x = pred["x"]
    pred_dx = pred["V"]
    pred_t = pred["t"]

    metrics = evaluate_rollout_terminal_error(pred_x, demo["x"], demo["goal"])
    print("==== Single Demo Rollout ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    plot_3d_rollout(
        demo["x"],
        pred_x,
        demo["goal"],
        os.path.join(out_dir, "traj_3d.png"),
    )

    plot_position_error_curve(
        demo["t"],
        demo["x"],
        pred_t,
        pred_x,
        os.path.join(out_dir, "position_error_curve.png"),
    )

    # 教师速度场 vs 推理速度场
    plot_velocity_component_curves(
        demo["t"],
        demo["Vd_star"],
        pred_t,
        pred_dx,
        os.path.join(out_dir, "velocity_components.png"),
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
        linear_in_body=linear_in_body,
    )
    plot_terminal_hist(test_result, out_dir)

    print(f"Figures saved to: {out_dir}")


def main():
    run_inference_and_visualize(
        ckpt_path="/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_regressive/velocity_field_regressive_best.pt",
        demo_path=None,
        demo_dir="/home/zhou/autolab/GUFIC_mujoco-main/bolt_demos",
        out_dir="/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/infer_regressive",
    )

if __name__ == "__main__":
    main()
