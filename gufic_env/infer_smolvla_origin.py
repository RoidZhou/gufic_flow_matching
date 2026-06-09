import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as RT

try:
    import mujoco
    import mujoco.viewer
    from gufic_env.env_gufic_velocity_field_infer_smolvla import RobotEnv as SmolVLAGUFICEnv
    _MUJOCO_IMPORT_ERROR = None
except Exception as exc:
    mujoco = None
    SmolVLAGUFICEnv = object
    _MUJOCO_IMPORT_ERROR = exc


DEFAULT_POLICY_PATH = (
    "/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/"
    "checkpoints_smolvla_v2/020000/pretrained_model"
)
DEFAULT_VLM_MODEL_NAME = (
    "/home/zhou/.cache/huggingface/hub/"
    "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/"
    "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
)
DEFAULT_DATASET_ROOT = "/media/zhou/Elements SE/VLA/boltnut_pi0_lerobot_20HZ"
DEFAULT_DATASET_REPO_ID = "gufic_boltnut_smolvla"


def to_numpy(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def rot6d_to_rotmat_np(r6d):
    r6d = np.asarray(r6d, dtype=np.float32).reshape(6)
    a1 = r6d[:3]
    a2 = r6d[3:6]

    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2_orth = a2 - np.dot(b1, a2) * b1
    b2 = a2_orth / (np.linalg.norm(a2_orth) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float32)


def unpack_pose_action(sample):
    if "action.pd" in sample and "action.Rd6d" in sample:
        pd = to_numpy(sample["action.pd"]).astype(np.float32).reshape(3)
        Rd6d = to_numpy(sample["action.Rd6d"]).astype(np.float32).reshape(6)
        return pd, rot6d_to_rotmat_np(Rd6d), Rd6d

    if "action" not in sample:
        raise KeyError("Dataset sample has no action / action.pd, cannot compare pd/Rd.")

    action = to_numpy(sample["action"]).astype(np.float32).reshape(-1)
    if action.shape[0] < 9:
        raise ValueError(f"Action must contain at least [pd(3), Rd6d(6)], got {action.shape}")
    pd = action[:3]
    Rd6d = action[3:9]
    return pd, rot6d_to_rotmat_np(Rd6d), Rd6d


def rotation_geodesic_error_deg(R_pred, R_gt):
    R_pred = np.asarray(R_pred, dtype=np.float64).reshape(-1, 3, 3)
    R_gt = np.asarray(R_gt, dtype=np.float64).reshape(-1, 3, 3)
    R_err = np.einsum("nij,njk->nik", np.transpose(R_pred, (0, 2, 1)), R_gt)
    cos_theta = (np.trace(R_err, axis1=1, axis2=2) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_theta)).astype(np.float32)


def save_pose_prediction_plots(records, out_dir):
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = np.asarray(records["frame_indices"], dtype=np.int64)
    pd_pred = np.stack(records["pd_pred"], axis=0)
    pd_gt = np.stack(records["pd_gt"], axis=0)
    Rd_pred = np.stack(records["Rd_pred"], axis=0)
    Rd_gt = np.stack(records["Rd_gt"], axis=0)
    p_now = np.stack(records["p_now"], axis=0)

    pd_err = np.linalg.norm(pd_pred - pd_gt, axis=1)
    p_to_pd_gt_err = np.linalg.norm(p_now - pd_gt, axis=1)
    rot_err = rotation_geodesic_error_deg(Rd_pred, Rd_gt)

    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(frame_indices, pd_pred[:, i], label=f"pd_pred_{labels[i]}", linewidth=1.4)
        ax.plot(frame_indices, pd_gt[:, i], "--", label=f"pd_gt_{labels[i]}", linewidth=1.1)
        ax.set_ylabel("m")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("dataset frame")
    fig.suptitle("SmolVLA Origin Pose Prediction: pd")
    fig.tight_layout()
    fig.savefig(out_dir / "pd_pred_vs_gt.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))
    ax.plot(frame_indices, pd_err, label="||pd_pred - pd_gt||", linewidth=1.4)
    ax.plot(frame_indices, p_to_pd_gt_err, "--", label="||p_now - pd_gt||", linewidth=1.1)
    ax.set_xlabel("dataset frame")
    ax.set_ylabel("m")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "pd_error_norm.png", dpi=150)
    plt.close(fig)

    euler_pred = RT.from_matrix(Rd_pred).as_euler("xyz", degrees=True)
    euler_gt = RT.from_matrix(Rd_gt).as_euler("xyz", degrees=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(frame_indices, euler_pred[:, i], label=f"Rd_pred_euler_{labels[i]}", linewidth=1.4)
        ax.plot(frame_indices, euler_gt[:, i], "--", label=f"Rd_gt_euler_{labels[i]}", linewidth=1.1)
        ax.set_ylabel("deg")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("dataset frame")
    fig.suptitle("SmolVLA Origin Pose Prediction: Rd Euler")
    fig.tight_layout()
    fig.savefig(out_dir / "Rd_euler_pred_vs_gt.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))
    ax.plot(frame_indices, rot_err, linewidth=1.4)
    ax.set_xlabel("dataset frame")
    ax.set_ylabel("deg")
    ax.set_title("Rd Geodesic Error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "Rd_error_deg.png", dpi=150)
    plt.close(fig)

    np.savez_compressed(
        out_dir / "pose_prediction_records.npz",
        frame_indices=frame_indices,
        timestamp=np.asarray(records["timestamp"], dtype=np.float32),
        p_now=p_now,
        pd_pred=pd_pred,
        pd_gt=pd_gt,
        Rd_pred=Rd_pred,
        Rd_gt=Rd_gt,
        pd_err=pd_err,
        p_to_pd_gt_err=p_to_pd_gt_err,
        rot_err_deg=rot_err,
    )

    print(f"\n[SmolVLA-Origin] saved pose comparison to: {out_dir}")
    print(
        "[SmolVLA-Origin] pd_err(m): "
        f"mean={pd_err.mean():.6f}, median={np.median(pd_err):.6f}, "
        f"p95={np.percentile(pd_err, 95):.6f}, max={pd_err.max():.6f}"
    )
    print(
        "[SmolVLA-Origin] Rd_err(deg): "
        f"mean={rot_err.mean():.4f}, median={np.median(rot_err):.4f}, "
        f"p95={np.percentile(rot_err, 95):.4f}, max={rot_err.max():.4f}"
    )


def run_pose_prediction_comparison(
    policy_path=DEFAULT_POLICY_PATH,
    dataset_repo_id=DEFAULT_DATASET_REPO_ID,
    dataset_root=DEFAULT_DATASET_ROOT,
    out_dir="./infer_smolvla_origin_pose_compare",
    start_index=0,
    max_frames=None,
    stride=1,
    language="insert the bolt into the hole",
    device=None,
    action_mode="full",
    vlm_model_name=DEFAULT_VLM_MODEL_NAME,
    reset_each_frame=True,
):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from gufic_env.flow_matching.infer_smolvla import (
        load_smolvla_velocity_field_infer,
        sample_to_smolvla_inputs,
    )

    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)
    end_index = len(dataset) if max_frames is None else min(len(dataset), start_index + max_frames * stride)
    frame_indices = list(range(int(start_index), int(end_index), int(stride)))
    if len(frame_indices) == 0:
        raise ValueError(
            f"No frames selected: len(dataset)={len(dataset)}, start_index={start_index}, "
            f"max_frames={max_frames}, stride={stride}"
        )

    infer = load_smolvla_velocity_field_infer(
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        language=language,
        device=device,
        action_mode=action_mode,
        vlm_model_name=vlm_model_name,
    )

    records = {
        "frame_indices": [],
        "timestamp": [],
        "p_now": [],
        "pd_pred": [],
        "pd_gt": [],
        "Rd_pred": [],
        "Rd_gt": [],
    }

    print(f"[SmolVLA-Origin] dataset_root: {dataset_root}")
    print(f"[SmolVLA-Origin] dataset_repo_id: {dataset_repo_id}")
    print(f"[SmolVLA-Origin] len(dataset): {len(dataset)}")
    print(f"[SmolVLA-Origin] selected frames: {len(frame_indices)}")
    print(f"[SmolVLA-Origin] action_mode: {action_mode}, reset_each_frame: {reset_each_frame}")

    for n, frame_index in enumerate(frame_indices):
        sample = dataset[frame_index]
        wrist_image, external_image, p, R, Fe = sample_to_smolvla_inputs(sample)
        pd_gt, Rd_gt, _ = unpack_pose_action(sample)

        if reset_each_frame:
            infer.reset()

        pred = infer.predict_desired_motion(
            wrist_image=wrist_image,
            external_image=external_image,
            p=p,
            R=R,
            Fe=Fe,
        )

        records["frame_indices"].append(frame_index)
        records["timestamp"].append(float(to_numpy(sample.get("timestamp", 0.0))))
        records["p_now"].append(p.astype(np.float32).reshape(3))
        records["pd_pred"].append(np.asarray(pred["pd"], dtype=np.float32).reshape(3))
        records["pd_gt"].append(pd_gt.astype(np.float32).reshape(3))
        records["Rd_pred"].append(np.asarray(pred["Rd"], dtype=np.float32).reshape(3, 3))
        records["Rd_gt"].append(Rd_gt.astype(np.float32).reshape(3, 3))

        if n % 50 == 0:
            pd_err = np.linalg.norm(records["pd_pred"][-1] - records["pd_gt"][-1])
            rot_err = rotation_geodesic_error_deg(
                records["Rd_pred"][-1][None, ...],
                records["Rd_gt"][-1][None, ...],
            )[0]
            print(
                f"[{n + 1}/{len(frame_indices)}] frame={frame_index} "
                f"pd_err={pd_err:.6f}m Rd_err={rot_err:.4f}deg"
            )

    save_pose_prediction_plots(records, out_dir)
    return records


class SmolVLAOriginEnv(SmolVLAGUFICEnv):
    """
    SmolVLA original-style online inference.

    This version does not use GUFIC / geometric force-impedance control.
    SmolVLA predicts only the desired end-effector pose [pd, Rd6d]. The pose is
    converted to a joint target with IK, then sent directly to MuJoCo position
    actuators.
    """

    def __init__(
        self,
        *args,
        policy_hz=20,
        contact_policy_hz=50,
        contact_force_threshold=1.0,
        reset_each_update=True,
        debug_use_gt_pose=False,
        pose_lowpass_alpha=1.0,
        q_cmd_alpha=1.0,
        q_cmd_max_delta=0.12,
        q_cmd_startup_max_delta=0.04,
        q_cmd_startup_steps=200,
        ik_step_size=0.35,
        ik_damping=1e-3,
        ik_max_delta_q=0.08,
        ik_tol=2e-4,
        ik_max_cnt=500,
        **kwargs,
    ):
        if _MUJOCO_IMPORT_ERROR is not None:
            raise RuntimeError(
                "SmolVLAOriginEnv online mode requires mujoco and the GUFIC simulation imports. "
                "Activate the correct conda environment before using --mode online."
            ) from _MUJOCO_IMPORT_ERROR

        kwargs["model"] = "smolvla"
        kwargs["use_learned_velocity_field"] = True
        kwargs["test_offline_cond"] = False
        super().__init__(*args, **kwargs)

        self.policy_hz = float(policy_hz)
        self.policy_decimation = max(1, int(round(1.0 / (self.policy_hz * self.dt))))
        self.contact_policy_hz = float(contact_policy_hz)
        self.contact_policy_decimation = max(
            1,
            int(round(1.0 / (self.contact_policy_hz * self.dt))),
        )
        self.contact_force_threshold = float(contact_force_threshold)
        self.reset_each_update = bool(reset_each_update)
        self.debug_use_gt_pose = bool(debug_use_gt_pose)
        self.last_policy_contact = False
        self.pose_lowpass_alpha = float(pose_lowpass_alpha)
        self.q_cmd_alpha = float(q_cmd_alpha)
        self.q_cmd_max_delta = float(q_cmd_max_delta)
        self.q_cmd_startup_max_delta = float(q_cmd_startup_max_delta)
        self.q_cmd_startup_steps = int(q_cmd_startup_steps)
        self.ik_step_size = float(ik_step_size)
        self.ik_damping = float(ik_damping)
        self.ik_max_delta_q = float(ik_max_delta_q)
        self.ik_tol = float(ik_tol)
        self.ik_max_cnt = int(ik_max_cnt)

        self.pd_cmd = None
        self.Rd_cmd = None
        self.q_cmd = self.data.qpos.copy()[: self.robot_state.N].astype(np.float64)
        self.last_action = None
        self.last_ik_error_norm = None
        self.last_ik_iters = 0
        self.last_cmd_update_iter = -1

        print(
            "[SmolVLA-Origin] policy_hz:",
            self.policy_hz,
            "contact_policy_hz:",
            self.contact_policy_hz,
            "reset_each_update:",
            self.reset_each_update,
            "debug_use_gt_pose:",
            self.debug_use_gt_pose,
            "q_cmd_alpha:",
            self.q_cmd_alpha,
            "q_cmd_max_delta:",
            self.q_cmd_max_delta,
            "q_cmd_startup_max_delta:",
            self.q_cmd_startup_max_delta,
        )

    def load_xml(self):
        model_dir = os.path.join(os.getcwd(), "gufic_env", "mujoco_models")
        if self.robot_name == "indy7":
            if self.task == "bolt":
                model_path = os.path.join(model_dir, "Indy7_nutbolt_position.xml")
            elif self.task == "sphere":
                model_path = os.path.join(model_dir, "Indy7_wiping_sphere.xml")
            elif self.task == "insertion":
                model_path = os.path.join(model_dir, "Indy7_insertion.xml")
            else:
                model_path = os.path.join(model_dir, "Indy7_wiping.xml")
        else:
            raise NotImplementedError

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.init_camera_renderer()

        if self.show_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.fix_camera:
                self.viewer.cam.fixedcamid = 0
                self.viewer.cam.trackbodyid = -1
                self.viewer.cam.lookat = np.array([0.5, 0.0, 0.3])
                self.viewer.cam.distance = 1.5
                self.viewer.cam.azimuth = 180
                self.viewer.cam.elevation = -20
        else:
            self.viewer = None

    def _pose_error(self, p, R, pd, Rd):
        ep = (p - pd).reshape(3, 1)
        eR = -0.5 * (
            np.cross(R[:, 0], Rd[:, 0])
            + np.cross(R[:, 1], Rd[:, 1])
            + np.cross(R[:, 2], Rd[:, 2])
        ).reshape(3, 1)
        return np.vstack((ep, eR))

    def _solve_ik_qpos(self, pd, Rd):
        qpos_bak = self.data.qpos.copy()
        qvel_bak = self.data.qvel.copy()
        ctrl_bak = self.data.ctrl.copy()

        q_work = qpos_bak.copy()
        self.data.qpos[:] = q_work
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.robot_state.update()

        p, R = self.robot_state.get_pose()
        error = self._pose_error(p, R, pd, Rd)
        step_cnt = 0

        while np.linalg.norm(error) >= self.ik_tol and step_cnt < self.ik_max_cnt:
            jac = self.robot_state.get_jacobian()
            jj_t = jac @ jac.T
            damping = (self.ik_damping ** 2) * np.eye(jj_t.shape[0])
            delta_q = -jac.T @ np.linalg.solve(jj_t + damping, error)
            delta_q = self.ik_step_size * delta_q.reshape(-1)
            delta_norm = np.linalg.norm(delta_q)
            if delta_norm > self.ik_max_delta_q:
                delta_q *= self.ik_max_delta_q / (delta_norm + 1e-12)

            self.data.qpos[: self.robot_state.N] += delta_q
            self.robot_state.check_joint_limits(self.data.qpos[: self.robot_state.N])

            mujoco.mj_forward(self.model, self.data)
            self.robot_state.update()
            p, R = self.robot_state.get_pose()
            error = self._pose_error(p, R, pd, Rd)
            step_cnt += 1

        q_des = self.data.qpos.copy()[: self.robot_state.N]
        self.last_ik_error_norm = float(np.linalg.norm(error))
        self.last_ik_iters = step_cnt

        self.data.qpos[:] = qpos_bak
        self.data.qvel[:] = qvel_bak
        self.data.ctrl[:] = ctrl_bak
        mujoco.mj_forward(self.model, self.data)
        self.robot_state.update()
        return q_des

    def _smooth_pose(self, pd, Rd):
        if self.pd_cmd is None or self.Rd_cmd is None or self.pose_lowpass_alpha >= 1.0:
            return pd, Rd

        alpha = np.clip(self.pose_lowpass_alpha, 0.0, 1.0)
        pd_smooth = (1.0 - alpha) * self.pd_cmd + alpha * pd

        R_prev = RT.from_matrix(self.Rd_cmd)
        R_new = RT.from_matrix(Rd)
        rel = R_prev.inv() * R_new
        Rd_smooth = (R_prev * RT.from_rotvec(alpha * rel.as_rotvec())).as_matrix()
        return pd_smooth.astype(np.float32), Rd_smooth.astype(np.float32)

    def _smooth_q_command(self, q_des):
        q_des = np.asarray(q_des, dtype=np.float64).reshape(self.robot_state.N)
        q_prev = np.asarray(self.q_cmd, dtype=np.float64).reshape(self.robot_state.N)

        max_delta = self.q_cmd_startup_max_delta if self.iter < self.q_cmd_startup_steps else self.q_cmd_max_delta
        delta = q_des - q_prev
        delta_norm = np.linalg.norm(delta)
        if max_delta > 0.0 and delta_norm > max_delta:
            delta *= max_delta / (delta_norm + 1e-12)
            q_des = q_prev + delta

        alpha = np.clip(self.q_cmd_alpha, 0.0, 1.0)
        return ((1.0 - alpha) * q_prev + alpha * q_des).astype(np.float64)

    def infer_desired_pose(self):
        p, R = self.robot_state.get_pose()
        Fe = np.asarray(self.get_FT_value(), dtype=np.float32).reshape(6)
        t = self.iter * self.dt

        if self.debug_use_gt_pose:
            result = {"action": np.zeros(9, dtype=np.float32)}
            pd = self.pd_t(t).reshape(3).astype(np.float32)
            Rd = self.Rd_t(t).reshape(3, 3).astype(np.float32)
        else:
            wrist_image = self.resize_rgb(self.get_camera_rgb(self.cam_id))
            external_image = self.resize_rgb(self.get_external_rgb())
            if self.reset_each_update:
                self.smolvla_infer.reset()
            result = self.smolvla_infer.predict_desired_motion(
                wrist_image=wrist_image,
                external_image=external_image,
                p=p.astype(np.float32),
                R=R.astype(np.float32),
                Fe=Fe,
            )
            pd = np.asarray(result["pd"], dtype=np.float32).reshape(3)
            Rd = np.asarray(result["Rd"], dtype=np.float32).reshape(3, 3)

        pd, Rd = self._smooth_pose(pd, Rd)

        self.pd_cmd = pd.copy()
        self.Rd_cmd = Rd.copy()
        self.last_action = result["action"].copy()
        q_des = self._solve_ik_qpos(pd, Rd)
        self.q_cmd = self._smooth_q_command(q_des)

        if self.iter % max(1, self.policy_decimation * 5) == 0:
            pd_gt = self.pd_t(t).reshape(3)
            Rd_gt = self.Rd_t(t).reshape(3, 3)
            rd_err = rotation_geodesic_error_deg(Rd[None, ...], Rd_gt[None, ...])[0]
            print("[SmolVLA-Origin] p_now:", p.reshape(3))
            print("[SmolVLA-Origin] pd_cmd:", pd)
            print("[SmolVLA-Origin] pd_gt:", pd_gt)
            print("[SmolVLA-Origin] ||pd_cmd - pd_gt||:", np.linalg.norm(pd - pd_gt))
            print("[SmolVLA-Origin] Rd_err_to_gt_deg:", rd_err)
            print("[SmolVLA-Origin] p_err:", np.linalg.norm(p.reshape(3) - pd))
            print("[SmolVLA-Origin] ik_err:", self.last_ik_error_norm, "ik_iters:", self.last_ik_iters)
            print("[SmolVLA-Origin] q_des:", q_des)
            print("[SmolVLA-Origin] q_cmd:", self.q_cmd)

        return pd, Rd

    def set_control_position(self, q_cmd, gripper=0.03):
        q_cmd = np.asarray(q_cmd, dtype=np.float64).reshape(-1)
        if q_cmd.shape[0] != self.robot_state.N:
            raise ValueError(f"q_cmd dim must be {self.robot_state.N}, got {q_cmd.shape[0]}")

        ctrl_range = self.model.actuator_ctrlrange[: self.robot_state.N]
        q_cmd = np.clip(q_cmd, ctrl_range[:, 0], ctrl_range[:, 1])
        self.data.ctrl[: self.robot_state.N] = q_cmd

        if self.model.nu >= self.robot_state.N + 2:
            self.data.ctrl[self.robot_state.N] = -float(gripper)
            self.data.ctrl[self.robot_state.N + 1] = float(gripper)

    def _print_tracking_status(self, label, contact, decimation):
        q_now = self.data.qpos.copy()[: self.robot_state.N]
        steps_since_cmd = self.iter - self.last_cmd_update_iter
        print(f"[SmolVLA-Origin] {label} contact:", contact, "decimation:", decimation)
        print(f"[SmolVLA-Origin] {label} steps_since_cmd:", steps_since_cmd)
        print(f"[SmolVLA-Origin] {label} q_track_err:", np.linalg.norm(q_now - self.q_cmd))
        if self.pd_cmd is not None:
            p_now, R_now = self.robot_state.get_pose()
            R_track_err = rotation_geodesic_error_deg(
                R_now.reshape(1, 3, 3),
                self.Rd_cmd.reshape(1, 3, 3),
            )[0]
            print(
                f"[SmolVLA-Origin] {label} p_track_err:",
                np.linalg.norm(p_now.reshape(3) - self.pd_cmd),
            )
            print(f"[SmolVLA-Origin] {label} R_track_err_deg:", R_track_err)

    def step(self):
        self.robot_state.update()

        Fe_now = np.asarray(self.get_FT_value(), dtype=np.float32).reshape(-1)
        contact = abs(float(Fe_now[2])) > self.contact_force_threshold
        decimation = self.contact_policy_decimation if contact else self.policy_decimation
        contact_changed = contact != self.last_policy_contact
        update_policy = contact_changed or self.iter % decimation == 0
        print_period = max(1, self.policy_decimation * 5)

        if update_policy and self.pd_cmd is not None and self.iter % print_period == 0:
            self._print_tracking_status("pre_update", contact, decimation)

        if update_policy:
            if contact_changed and not self.debug_use_gt_pose:
                self.smolvla_infer.reset()
            self.infer_desired_pose()
            self.last_policy_contact = contact
            self.last_cmd_update_iter = self.iter

        self.set_control_position(self.q_cmd, gripper=0.03)
        self.robot_state.update_dynamic()

        mid_phase = max(1, decimation // 2)
        if self.iter % print_period == 0:
            self._print_tracking_status("post_step", contact, decimation)
        elif self.iter % decimation == mid_phase and self.iter % print_period < decimation:
            self._print_tracking_status("mid_cycle", contact, decimation)

        if self.show_viewer:
            self.viewer.sync()

        obs = {}
        p, R = self.robot_state.get_pose()
        Fe = self.get_FT_value()
        Fe_raw = self.get_FT_value_raw()
        t = self.iter * self.dt

        gd = np.eye(4)
        if self.Rd_cmd is not None:
            gd[:3, :3] = self.Rd_cmd
            gd[:3, 3] = self.pd_cmd.reshape(3)
        else:
            gd[:3, :3] = R
            gd[:3, 3] = p.reshape(3)

        for observable in self.observables:
            if observable == "p":
                obs[observable] = p.copy()
            elif observable == "pd":
                obs[observable] = gd[:3, 3].copy()
            elif observable == "R":
                obs[observable] = R.copy()
            elif observable == "Rd":
                obs[observable] = gd[:3, :3].copy()
            elif observable == "Fe":
                obs[observable] = Fe.copy()
            elif observable == "Fe_raw":
                obs[observable] = Fe_raw.copy()
            elif observable == "x_tf":
                obs[observable] = self.x_tf
            elif observable == "x_ti":
                obs[observable] = self.x_ti
            elif observable == "Fd":
                obs[observable] = np.zeros((6, 1))
            elif observable == "rho":
                obs[observable] = 0.0

        done = self.iter == self.max_iter - 1
        reward = 0
        info = {"t": t, "pd_cmd": gd[:3, 3].copy(), "Rd_cmd": gd[:3, :3].copy()}

        self.iter += 1
        return obs, reward, done, info

    def run(self):
        p_list, R_list, pd_list, Fe_list, Fe_raw_list = [], [], [], [], []

        for i in range(self.max_iter):
            self.golbal_steps = i
            obs, reward, done, info = self.step()

            p, R = self.robot_state.get_pose()
            Fe = self.get_FT_value()
            Fe_raw = self.get_FT_value_raw()

            p_list.append(p)
            R_list.append(R)
            pd_list.append(info["pd_cmd"])
            Fe_list.append(Fe)
            Fe_raw_list.append(Fe_raw)

            if i % 100 == 0:
                print(f"Time Step: {i}")

            if done:
                break

        return p_list, R_list, pd_list, Fe_list, Fe_raw_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="SmolVLA origin-style pose-only inference for pd/Rd."
    )
    parser.add_argument(
        "--mode",
        choices=["compare", "online"],
        default="online",
        help="compare: test pd/Rd on LeRobot dataset; online: send predicted pose to position-control robot.",
    )
    parser.add_argument("--policy_path", default=os.environ.get("SMOLVLA_POLICY_PATH", DEFAULT_POLICY_PATH))
    parser.add_argument("--vlm_model_name", default=os.environ.get("SMOLVLA_VLM_MODEL_NAME", DEFAULT_VLM_MODEL_NAME))
    parser.add_argument("--dataset_root", default=os.environ.get("SMOLVLA_DATASET_ROOT", DEFAULT_DATASET_ROOT))
    parser.add_argument("--dataset_repo_id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--language", default="insert the bolt into the hole")
    parser.add_argument("--device", default=None)
    parser.add_argument("--action_mode", choices=["pose", "full"], default="full")
    parser.add_argument("--out_dir", default="./infer_smolvla_origin_pose_compare")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=1000,
        help="Number of dataset frames to compare. Use -1 for all selected frames.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--keep_action_queue",
        action="store_true",
        default=False,
        help="Keep SmolVLA action queue across dataset frames. Default resets every frame for one-step pd/Rd evaluation.",
    )
    parser.add_argument("--show_viewer", action="store_true", default=True)
    parser.add_argument("--policy_hz", type=float, default=20.0)
    parser.add_argument("--contact_policy_hz", type=float, default=50.0)
    parser.add_argument("--contact_force_threshold", type=float, default=1.0)
    parser.add_argument("--q_cmd_alpha", type=float, default=1.0)
    parser.add_argument("--q_cmd_max_delta", type=float, default=0.12)
    parser.add_argument("--q_cmd_startup_max_delta", type=float, default=0.005)
    parser.add_argument("--q_cmd_startup_steps", type=int, default=200)
    parser.add_argument(
        "--reset_each_update",
        action="store_true",
        default=False,
        help="Reset SmolVLA action queue before every policy update. This is enabled by default.",
    )
    parser.add_argument(
        "--debug_use_gt_pose",
        action="store_true",
        default=True,
        help="Use analytic pd_t/Rd_t instead of SmolVLA prediction to test IK/position-control reachability.",
    )
    parser.add_argument("--max_time", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "compare":
        max_frames = None if args.max_frames is not None and args.max_frames < 0 else args.max_frames
        run_pose_prediction_comparison(
            policy_path=args.policy_path,
            dataset_repo_id=args.dataset_repo_id,
            dataset_root=args.dataset_root,
            out_dir=args.out_dir,
            start_index=args.start_index,
            max_frames=max_frames,
            stride=args.stride,
            language=args.language,
            device=args.device,
            action_mode=args.action_mode,
            vlm_model_name=args.vlm_model_name,
            reset_each_frame=not args.keep_action_queue,
        )
        return

    robot_name = "indy7"
    show_viewer = args.show_viewer
    randomized_start = True
    inertia_shaping = False
    task = "bolt"
    max_time = args.max_time
    fz = 2

    env = SmolVLAOriginEnv(
        robot_name=robot_name,
        show_viewer=show_viewer,
        max_time=max_time,
        fz=fz,
        fix_camera=True,
        task=task,
        randomized_start=randomized_start,
        inertia_shaping=inertia_shaping,
        record_demos=False,
        demo_path=None,
        seed=args.seed,
        save_tensorboard=False,
        visualize_delta_pose=False,
        smolvla_policy_path=args.policy_path,
        smolvla_dataset_repo_id=args.dataset_repo_id,
        smolvla_dataset_root=args.dataset_root,
        smolvla_vlm_model_name=args.vlm_model_name,
        smolvla_action_mode=args.action_mode,
        smolvla_language=args.language,
        policy_hz=args.policy_hz,
        contact_policy_hz=args.contact_policy_hz,
        contact_force_threshold=args.contact_force_threshold,
        reset_each_update=args.reset_each_update and not args.keep_action_queue,
        debug_use_gt_pose=args.debug_use_gt_pose,
        pose_lowpass_alpha=1.0,
        q_cmd_alpha=args.q_cmd_alpha,
        q_cmd_max_delta=args.q_cmd_max_delta,
        q_cmd_startup_max_delta=args.q_cmd_startup_max_delta,
        q_cmd_startup_steps=args.q_cmd_startup_steps,
        ik_damping=1e-3,
        ik_max_delta_q=0.08,
    )

    env.run()


if __name__ == "__main__":
    main()
