import argparse
import os
import shutil
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as RT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from gufic_env.env_gufic_velocity_field_infer_smolvla import RobotEnv


TASK_NAME = "insert the bolt into the hole"
DEFAULT_REPO_ID = "gufic_nutbolt_position_smolvla"
DEFAULT_ROOT = "/media/zhou/Elements SE/VLA/nutbolt_position_smolvla"


def resize_rgb(image, size=(256, 256)):
    image = np.asarray(image, dtype=np.uint8)
    if image.shape[0] == size[1] and image.shape[1] == size[0]:
        return image
    return np.asarray(Image.fromarray(image).resize(size), dtype=np.uint8)


def pose_to_xyzrpy(p, R):
    p = np.asarray(p, dtype=np.float32).reshape(3)
    euler = RT.from_matrix(np.asarray(R).reshape(3, 3)).as_euler("xyz", degrees=False)
    return np.concatenate([p, euler.astype(np.float32)], axis=0).astype(np.float32)


def smoothstep(t, duration):
    tau = np.clip(float(t) / max(float(duration), 1e-8), 0.0, 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def rotz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


class NutBoltPositionCollectEnv(RobotEnv):
    """
    Collect LeRobot demos in the same style as lerobot-mujoco-tutorial/1.collect_data.py.

    observation.image       : external camera image, 256x256x3
    observation.wrist_image : wrist camera image, 256x256x3
    observation.state       : end-effector pose [x, y, z, roll, pitch, yaw]
    action                  : joint position command [q1..q6, gripper]
    obj_init                : nut/hole pose [x, y, z, roll, pitch, yaw]
    """

    def __init__(
        self,
        *args,
        fps=20,
        gripper=0.03,
        q_cmd_alpha=1.0,
        q_cmd_max_delta=0.01,
        q_cmd_startup_max_delta=0.002,
        q_cmd_startup_steps=1000,
        ik_step_size=0.35,
        ik_damping=1e-3,
        ik_max_delta_q=0.08,
        ik_tol=2e-4,
        ik_max_cnt=1000,
        **kwargs,
    ):
        kwargs["model"] = kwargs.get("model", "mlp")
        kwargs["use_learned_velocity_field"] = False
        kwargs["test_offline_cond"] = False
        super().__init__(*args, **kwargs)

        self.collect_fps = float(fps)
        self.collect_decimation = max(1, int(round(1.0 / (self.collect_fps * self.dt))))
        self.collect_actual_fps = 1.0 / (self.collect_decimation * self.dt)
        self.gripper = float(gripper)

        self.q_cmd_alpha = float(q_cmd_alpha)
        self.q_cmd_max_delta = float(q_cmd_max_delta)
        self.q_cmd_startup_max_delta = float(q_cmd_startup_max_delta)
        self.q_cmd_startup_steps = int(q_cmd_startup_steps)

        self.ik_step_size = float(ik_step_size)
        self.ik_damping = float(ik_damping)
        self.ik_max_delta_q = float(ik_max_delta_q)
        self.ik_tol = float(ik_tol)
        self.ik_max_cnt = int(ik_max_cnt)

        self.q_cmd = self.data.qpos.copy()[: self.robot_state.N].astype(np.float64)
        self.last_ik_error_norm = None
        self.last_ik_iters = 0
        self.last_success_pos_err = None
        self.last_success_rot_err = None

        print(
            "[NutBolt-VLA] fps:",
            self.collect_fps,
            "dt:",
            self.dt,
            "decimation:",
            self.collect_decimation,
            "actual_fps:",
            self.collect_actual_fps,
        )

    def reset(self, angle_prefix=None):
        obs = super().reset(angle_prefix=angle_prefix)
        if self.task == "bolt":
            self._initialize_position_bolt_trajectory()
        return obs

    def load_xml(self):
        model_dir = Path(os.getcwd()) / "gufic_env" / "mujoco_models"
        if self.robot_name != "indy7":
            raise NotImplementedError(f"Unsupported robot_name: {self.robot_name}")

        if self.task == "bolt":
            model_path = model_dir / "Indy7_nutbolt_position.xml"
        elif self.task == "insertion":
            model_path = model_dir / "Indy7_insertion.xml"
        elif self.task == "sphere":
            model_path = model_dir / "Indy7_wiping_sphere.xml"
        else:
            model_path = model_dir / "Indy7_wiping.xml"

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
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

    def _initialize_position_bolt_trajectory(self):
        p0, _ = self.robot_state.get_pose()

        pd_mouth = np.array([self.p_plate[0], self.p_plate[1], 0.20], dtype=np.float64)
        Rd_base = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        u, _, vt = np.linalg.svd(Rd_base)
        Rd_base = u @ vt

        T1 = 4.0
        T2 = 2.0
        T3 = max(self.max_time - T1 - T2, 1e-8)
        h = -0.02
        z_axis_world = Rd_base @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        p_above = pd_mouth + h * z_axis_world

        total_radian = 4.0 * np.pi
        pitch = 0.0025
        s_per_rad = pitch / (2.0 * np.pi)

        def pd_t(t):
            t = float(t)
            if t <= T1:
                s = smoothstep(t, T1)
                return p0 + s * (p_above - p0)
            if t <= T1 + T2:
                s = smoothstep(t - T1, T2)
                return p_above + s * (pd_mouth - p_above)

            s = smoothstep(t - T1 - T2, T3)
            theta = total_radian * s
            p_loc = np.array([0.0, 0.0, s_per_rad * theta], dtype=np.float64)
            return pd_mouth + Rd_base @ p_loc

        def Rd_t(t):
            t = float(t)
            if t <= T1 + T2:
                return Rd_base
            s = smoothstep(t - T1 - T2, T3)
            theta = total_radian * s
            return Rd_base @ rotz(theta)

        def dpd_t(t):
            return np.zeros((3, 1), dtype=np.float64)

        def dRd_t(t):
            return np.zeros((3, 3), dtype=np.float64)

        self.pd_t = pd_t
        self.Rd_t = Rd_t
        self.dpd_t = dpd_t
        self.dRd_t = dRd_t
        self.ddpd_t = dpd_t
        self.ddRd_t = dRd_t

    def _solve_ik_qpos(self, pd, Rd):
        qpos_bak = self.data.qpos.copy()
        qvel_bak = self.data.qvel.copy()
        ctrl_bak = self.data.ctrl.copy()

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.robot_state.update()

        p, R = self.robot_state.get_pose()
        error = self._pose_error(p, R, pd, Rd)
        step_cnt = 0

        while np.linalg.norm(error) >= self.ik_tol and step_cnt < self.ik_max_cnt:
            jac = self.robot_state.get_jacobian()
            jj_t = jac @ jac.T
            damping = (self.ik_damping**2) * np.eye(jj_t.shape[0])
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

    def set_control_position(self, q_cmd):
        q_cmd = np.asarray(q_cmd, dtype=np.float64).reshape(self.robot_state.N)
        ctrl_range = self.model.actuator_ctrlrange[: self.robot_state.N]
        q_cmd = np.clip(q_cmd, ctrl_range[:, 0], ctrl_range[:, 1])
        self.data.ctrl[: self.robot_state.N] = q_cmd

        if self.model.nu >= self.robot_state.N + 2:
            self.data.ctrl[self.robot_state.N] = -self.gripper
            self.data.ctrl[self.robot_state.N + 1] = self.gripper

    def sync_position_command_to_current_q(self):
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.robot_state.update()
        self.q_cmd = self.data.qpos.copy()[: self.robot_state.N].astype(np.float64)
        self.set_control_position(self.q_cmd)

    def update_position_command_from_gt_pose(self):
        t = self.iter * self.dt
        pd = self.pd_t(t).reshape(3)
        Rd = self.Rd_t(t).reshape(3, 3)
        q_des = self._solve_ik_qpos(pd, Rd)
        self.q_cmd = self._smooth_q_command(q_des)
        return pd, Rd, q_des

    def step_position_control(self):
        self.robot_state.update()
        pd, Rd, q_des = self.update_position_command_from_gt_pose()
        self.set_control_position(self.q_cmd)
        self.robot_state.update_dynamic()

        if self.show_viewer and self.iter % 10 == 0:
            self.viewer.sync()

        self.iter += 1
        return pd, Rd, q_des

    def get_images(self):
        external_image = resize_rgb(self.get_external_rgb())
        wrist_image = resize_rgb(self.get_camera_rgb(self.cam_id))
        return external_image, wrist_image

    def get_observation_state(self):
        p, R = self.robot_state.get_pose()
        return pose_to_xyzrpy(p, R)

    def get_action(self):
        q_now = self.data.qpos.copy()[: self.robot_state.N].astype(np.float32)
        return np.concatenate(
            [
                q_now.reshape(self.robot_state.N),
                np.array([self.gripper], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

    def get_obj_init(self):
        p = np.asarray(self.p_plate, dtype=np.float32).reshape(3)
        euler = np.zeros(3, dtype=np.float32)
        return np.concatenate([p, euler], axis=0).astype(np.float32)

    def collect_episode(self, dataset, episode_idx):
        self.reset()
        self.sync_position_command_to_current_q()
        dataset.clear_episode_buffer()

        frame_count = 0
        done = False
        while self.iter < self.max_iter:
            pd, Rd, q_des = self.step_position_control()

            if self.iter % self.collect_decimation == 0:
                external_image, wrist_image = self.get_images()
                frame = {
                    "observation.image": external_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": self.get_observation_state(),
                    "action": self.get_action(),
                    "obj_init": self.get_obj_init(),
                }
                dataset.add_frame(frame, task=TASK_NAME)
                frame_count += 1

            if self.iter % 1000 == 0:
                p_now, _ = self.robot_state.get_pose()
                q_now = self.data.qpos.copy()[: self.robot_state.N]
                hole_mouth = np.array([self.p_plate[0], self.p_plate[1], 0.20], dtype=np.float64)
                print(
                    f"[NutBolt-VLA] episode={episode_idx} iter={self.iter} "
                    f"frames={frame_count} p_now={p_now} pd={pd} "
                    f"p_err={np.linalg.norm(p_now - pd):.6f} "
                    f"p_to_hole={np.linalg.norm(p_now - hole_mouth):.6f} "
                    f"pd_to_hole={np.linalg.norm(pd - hole_mouth):.6f} "
                    f"q_track_err={np.linalg.norm(q_now - self.q_cmd):.6f} "
                    f"ik_err={self.last_ik_error_norm:.6f} ik_iters={self.last_ik_iters}"
                )

            if self.check_task_success_quiet():
                done = True
                print(
                    f"[NutBolt-VLA] success at iter={self.iter}, frames={frame_count}, "
                    f"Final Pos Error={self.last_success_pos_err:.6f}, "
                    f"Final Rot Error={np.rad2deg(self.last_success_rot_err):.3f} deg"
                )
                break

        if frame_count == 0:
            raise RuntimeError("Episode produced zero frames; check fps/max_time settings.")

        dataset.save_episode()
        print(f"[NutBolt-VLA] saved episode {episode_idx}, frames={frame_count}, success={done}")
        return frame_count, done

    def check_task_success_quiet(self):
        p, R = self.robot_state.get_pose()
        t = self.max_time

        pd = self.pd_t(t).reshape(-1)
        Rd = self.Rd_t(t)

        pos_err = np.linalg.norm(p - pd)

        rot_err_mat = Rd.T @ R
        trace_val = np.clip(
            (np.trace(rot_err_mat) - 1) / 2,
            -1.0,
            1.0,
        )
        rot_err = np.arccos(trace_val)

        self.last_success_pos_err = float(pos_err)
        self.last_success_rot_err = float(rot_err)

        success = (pos_err < 0.002) and rot_err < np.deg2rad(10.0)
        return success


def create_or_load_dataset(repo_id, root, fps, overwrite=False, append=False):
    root = Path(root)
    create_new = True

    if root.exists():
        if overwrite:
            shutil.rmtree(root)
        elif append:
            create_new = False
        else:
            print(f"Directory {root} already exists.")
            ans = input("Do you want to delete it? (y/n) ")
            if ans.lower() == "y":
                shutil.rmtree(root)
            else:
                create_new = False

    if create_new:
        return LeRobotDataset.create(
            repo_id=repo_id,
            root=root,
            robot_type="indy7",
            fps=int(round(fps)),
            features={
                "observation.image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.wrist_image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action"],
                },
                "obj_init": {
                    "dtype": "float32",
                    "shape": (6,),
                    "names": ["obj_init"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

    print("Load from previous dataset")
    return LeRobotDataset(repo_id, root=root)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect pure position-control nut-bolt VLA demos.")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--num_demo", type=int, default=50)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--max_time", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomized_start", action="store_true", default=True)
    parser.add_argument("--fixed_start", action="store_true")
    parser.add_argument("--show_viewer", action="store_true", default=True)
    parser.add_argument("--no_viewer", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--keep_images", action="store_true")
    parser.add_argument("--q_cmd_alpha", type=float, default=1.0)
    parser.add_argument("--q_cmd_max_delta", type=float, default=0.001)
    parser.add_argument("--q_cmd_startup_max_delta", type=float, default=0.002)
    parser.add_argument("--q_cmd_startup_steps", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    randomized_start = args.randomized_start and not args.fixed_start
    show_viewer = args.show_viewer and not args.no_viewer

    dataset = create_or_load_dataset(
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        overwrite=args.overwrite,
        append=args.append,
    )

    env = NutBoltPositionCollectEnv(
        robot_name="indy7",
        model="mlp",
        show_viewer=show_viewer,
        max_time=args.max_time,
        fz=2,
        fix_camera=True,
        task="bolt",
        randomized_start=randomized_start,
        inertia_shaping=False,
        use_learned_velocity_field=False,
        record_demos=False,
        seed=args.seed,
        save_tensorboard=False,
        visualize_delta_pose=False,
        fps=args.fps,
        q_cmd_alpha=args.q_cmd_alpha,
        q_cmd_max_delta=args.q_cmd_max_delta,
        q_cmd_startup_max_delta=args.q_cmd_startup_max_delta,
        q_cmd_startup_steps=args.q_cmd_startup_steps,
    )

    try:
        total_frames = 0
        successes = 0
        for episode_idx in range(args.num_demo):
            if args.seed is not None:
                np.random.seed(args.seed + episode_idx)
            frames, success = env.collect_episode(dataset, episode_idx)
            total_frames += frames
            successes += int(success)

        print(
            f"[NutBolt-VLA] finished num_demo={args.num_demo}, "
            f"total_frames={total_frames}, successes={successes}"
        )
    finally:
        if env.viewer is not None:
            env.viewer.close()

        if not args.keep_images:
            images_dir = Path(dataset.root) / "images"
            if images_dir.exists():
                shutil.rmtree(images_dir)


if __name__ == "__main__":
    main()
