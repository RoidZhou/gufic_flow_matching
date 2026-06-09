import argparse
from pathlib import Path

import numpy as np

from gufic_env.env_gufic_velocity_field_infer_smolvla import RobotEnv
from gufic_env.flow_matching.infer_pi0 import (
    get_velocity_field,
    rot6d_to_rotmat_np,
    to_numpy,
    unpack_pi0_action,
)


def _as_int(x):
    if hasattr(x, "item"):
        return int(x.item())
    return int(x)


def get_episode_range(dataset, episode_index):
    episode_data_index = getattr(dataset, "episode_data_index", None)
    if episode_data_index is not None:
        for start_key, end_key in (("from", "to"), ("start", "end")):
            if start_key in episode_data_index and end_key in episode_data_index:
                return (
                    _as_int(episode_data_index[start_key][episode_index]),
                    _as_int(episode_data_index[end_key][episode_index]),
                )

    if episode_index != 0:
        raise ValueError(
            "Could not read dataset.episode_data_index, so only episode_index=0 "
            "can be replayed with the fallback path."
        )
    return 0, len(dataset)


def sample_pose_from_dataset_sample(sample):
    if "observation.state" in sample:
        state = to_numpy(sample["observation.state"]).astype(np.float32).reshape(-1)
    elif "observation.robot_state" in sample:
        state = to_numpy(sample["observation.robot_state"]).astype(np.float32).reshape(-1)
    else:
        raise KeyError("Dataset sample has no observation.state / observation.robot_state.")

    if state.shape[0] < 9:
        raise ValueError(f"Expected state=[p(3), R6d(6)], got shape {state.shape}.")

    p = state[:3]
    R = rot6d_to_rotmat_np(state[3:9])
    return p, R


def load_replay_actions(
    dataset_root,
    dataset_repo_id,
    episode_index,
    source_fps,
    start_offset=0,
    max_frames=None,
):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)
    ep_start, ep_end = get_episode_range(dataset, episode_index)

    start = ep_start + int(start_offset)
    if start >= ep_end:
        raise ValueError(
            f"start_offset={start_offset} is outside episode range [{ep_start}, {ep_end})."
        )

    indices = list(range(start, ep_end))
    if max_frames is not None:
        indices = indices[: int(max_frames)]
    if not indices:
        raise ValueError("No replay frames selected.")

    actions = []
    for idx in indices:
        sample = dataset[idx]
        if "action" not in sample:
            raise KeyError("Dataset sample has no action field.")
        actions.append(to_numpy(sample["action"]).astype(np.float32).reshape(-1))

    first_sample = dataset[indices[0]]
    p0, R0 = sample_pose_from_dataset_sample(first_sample)

    print(
        f"[ReplayDataset] episode={episode_index}, raw_range=[{ep_start}, {ep_end}), "
        f"selected_raw_frames={len(indices)}"
    )
    print(
        f"[ReplayDataset] source_fps={source_fps}, "
        f"source_duration ~= {len(indices) / float(source_fps):.3f}s"
    )
    print("[ReplayDataset] first_dataset_p:", p0)

    return np.stack(actions, axis=0), p0, R0, indices


class ReplayDatasetEnv(RobotEnv):
    def configure_dataset_replay(
        self,
        actions,
        source_fps=1000.0,
        default_fps=20.0,
        contact_fps=500.0,
        contact_force_threshold=1.0,
        phase_force_ref=8.0,
        phase_force_scale=8.0,
        min_phase_rate=0.02,
        replay_time_scale=2.0,
        init_from_dataset_pose=None,
        lowpass_alpha=None,
    ):
        self.replay_actions = np.asarray(actions, dtype=np.float32)
        if self.replay_actions.ndim != 2 or self.replay_actions.shape[1] < 9:
            raise ValueError(
                "actions must have shape [N, 9] or [N, 15], got "
                f"{self.replay_actions.shape}"
            )
        final_pd, final_Rd, _, _, _ = unpack_pi0_action(self.replay_actions[-1])
        self.replay_final_pd = final_pd.astype(np.float32)
        self.replay_final_Rd = final_Rd.astype(np.float32)
        self.replay_success_reported = False

        self.replay_source_fps = float(source_fps)
        self.replay_default_fps = float(default_fps)
        self.replay_contact_fps = float(contact_fps)
        self.replay_contact_force_threshold = float(contact_force_threshold)
        self.replay_phase_force_ref = float(phase_force_ref)
        self.replay_phase_force_scale = float(phase_force_scale)
        self.replay_min_phase_rate = float(min_phase_rate)
        self.replay_time_scale = float(replay_time_scale)

        self.replay_default_decimation = max(
            1,
            int(round(1.0 / (max(self.replay_default_fps, 1e-8) * self.dt))),
        )
        self.replay_contact_decimation = max(
            1,
            int(round(1.0 / (max(self.replay_contact_fps, 1e-8) * self.dt))),
        )
        self.replay_default_actual_fps = 1.0 / (self.replay_default_decimation * self.dt)
        self.replay_contact_actual_fps = 1.0 / (self.replay_contact_decimation * self.dt)
        self.replay_cursor = -1
        self.replay_phase_index = 0.0
        self.replay_phase_rate = 1.0
        self.replay_last_update_iter = -1
        self.replay_in_contact = False
        self.replay_prev_Vd_star = np.zeros(6, dtype=np.float32)
        self.replay_prev_dVd_star = np.zeros(6, dtype=np.float32)
        self.replay_lowpass_alpha = (
            self.fm_lowpass_alpha if lowpass_alpha is None else float(lowpass_alpha)
        )

        self.max_iter = int(
            np.ceil(
                len(self.replay_actions)
                / max(self.replay_source_fps, 1e-8)
                * max(self.replay_time_scale, 1.0)
                / self.dt
            )
        )
        self.max_time = self.max_iter * self.dt

        self._set_replay_action(0)
        self._install_replay_desired_functions()

        if init_from_dataset_pose is not None:
            p0, R0 = init_from_dataset_pose
            self.set_robot_to_pose(p0, R0)

        # Reuse the learned-field branch in the base controller, but do not
        # load any VLA/FLOW model. The override below supplies Vd_star directly.
        self.policy = "smolvla"
        self.use_learned_velocity_field = True

        print(
            f"[ReplayDataset] source_fps={self.replay_source_fps}, dt={self.dt}"
        )
        print(
            f"[ReplayDataset] default_fps={self.replay_default_fps}, "
            f"decimation={self.replay_default_decimation}, "
            f"actual_fps={self.replay_default_actual_fps:.3f}"
        )
        print(
            f"[ReplayDataset] contact_fps={self.replay_contact_fps}, "
            f"decimation={self.replay_contact_decimation}, "
            f"actual_fps={self.replay_contact_actual_fps:.3f}, "
            f"force_threshold={self.replay_contact_force_threshold}"
        )
        print(
            f"[ReplayDataset] phase_force_ref={self.replay_phase_force_ref}, "
            f"phase_force_scale={self.replay_phase_force_scale}, "
            f"min_phase_rate={self.replay_min_phase_rate}"
        )
        print(
            f"[ReplayDataset] raw_replay_frames={len(self.replay_actions)}, "
            f"max_iter={self.max_iter}, max_time={self.max_time:.3f}s, "
            f"time_scale={self.replay_time_scale}"
        )

    def _set_replay_action(self, action_index):
        action_index = int(np.clip(action_index, 0, len(self.replay_actions) - 1))
        self.replay_cursor = action_index
        action = self.replay_actions[action_index]
        pd, Rd, Vd_body, dpd, dRd = unpack_pi0_action(action)

        self.replay_pd = pd.astype(np.float32)
        self.replay_Rd = Rd.astype(np.float32)
        self.replay_Vd_body = Vd_body.astype(np.float32)
        self.replay_dpd = dpd.astype(np.float32)
        self.replay_dRd = dRd.astype(np.float32)

    def _install_replay_desired_functions(self):
        self.pd_t = lambda _t: self.replay_pd.reshape(3)
        self.Rd_t = lambda _t: self.replay_Rd.reshape(3, 3)
        self.dpd_t = lambda _t: self.replay_dpd.reshape(3)
        self.dRd_t = lambda _t: self.replay_dRd.reshape(3, 3)
        self.ddpd_t = lambda _t: np.zeros(3, dtype=np.float32)
        self.ddRd_t = lambda _t: np.zeros((3, 3), dtype=np.float32)

    def check_task_success(self, verbose=True):
        p, R = self.robot_state.get_pose()
        pd = self.replay_final_pd.reshape(3)
        Rd = self.replay_final_Rd.reshape(3, 3)

        pos_err = np.linalg.norm(p - pd)
        rot_err_mat = Rd.T @ R
        trace_val = np.clip((np.trace(rot_err_mat) - 1.0) / 2.0, -1.0, 1.0)
        rot_err = np.arccos(trace_val)

        success = (pos_err < 0.002) and rot_err < np.deg2rad(10.0)
        if verbose:
            print("================================")
            print(f"Final Pos Error : {pos_err:.6f}")
            print(f"Final Rot Error : {np.rad2deg(rot_err):.3f}")
            print(f"Task Success    : {success}")
            print("================================")
        return success

    def step(self):
        obs, reward, done, info = super().step()
        success = self.check_task_success(verbose=False)
        if success:
            done = True
            info["success"] = True
            if not self.replay_success_reported:
                print(f"[ReplayDataset] success reached at iter={self.iter}")
                self.check_task_success(verbose=True)
                self.replay_success_reported = True
        return obs, reward, done, info

    def _compute_phase_rate(self, Fe_now, contact):
        if not contact:
            return 1.0

        fz_abs = abs(float(Fe_now[2]))
        overload = max(0.0, fz_abs - self.replay_phase_force_ref)
        phase_rate = 1.0 / (1.0 + (overload / max(self.replay_phase_force_scale, 1e-8)) ** 2)
        return max(self.replay_min_phase_rate, phase_rate)

    def get_learned_velocity_field(self, p, R, t, Fe, point_cloud):
        Fe_now = np.asarray(Fe, dtype=np.float32).reshape(-1)
        contact = abs(float(Fe_now[2])) > self.replay_contact_force_threshold
        self.replay_phase_rate = self._compute_phase_rate(Fe_now, contact)
        if self.iter > 0:
            self.replay_phase_index += self.replay_source_fps * self.dt * self.replay_phase_rate
            self.replay_phase_index = min(
                self.replay_phase_index,
                float(len(self.replay_actions) - 1),
            )

        current_decimation = (
            self.replay_contact_decimation if contact else self.replay_default_decimation
        )
        source_index = int(round(self.replay_phase_index))
        contact_changed = contact != self.replay_in_contact
        should_update = (
            self.replay_last_update_iter < 0
            or contact_changed
            or (self.iter - self.replay_last_update_iter) >= current_decimation
        )
        if should_update:
            self._set_replay_action(source_index)
            self.replay_last_update_iter = self.iter
            self.replay_in_contact = contact

        p_now = np.asarray(p, dtype=np.float32).reshape(-1, 3)[-1]
        R_now = np.asarray(R, dtype=np.float32).reshape(-1, 3, 3)[-1]

        g_now = np.eye(4, dtype=np.float32)
        g_now[:3, :3] = R_now
        g_now[:3, 3] = p_now

        Vd_star = get_velocity_field(
            g=g_now,
            pd=self.replay_pd,
            Rd=self.replay_Rd,
            dpd=self.replay_dpd,
            dRd=self.replay_dRd,
            zeta_v=self.zeta_v,
            zeta_w=self.zeta_w,
        ).reshape(6)

        if self.iter == 0:
            dVd_star = np.zeros(6, dtype=np.float32)
        else:
            dVd_star = (Vd_star - self.replay_prev_Vd_star) / max(self.dt, 1e-8)

        alpha = self.replay_lowpass_alpha
        dVd_star = alpha * dVd_star + (1.0 - alpha) * self.replay_prev_dVd_star

        self.replay_prev_Vd_star = Vd_star.copy()
        self.replay_prev_dVd_star = dVd_star.copy()

        if contact_changed or self.iter % max(1, self.replay_default_decimation * 5) == 0:
            print(
                "[ReplayDataset] iter:",
                self.iter,
                "source_index:",
                self.replay_cursor,
                "phase:",
                f"{self.replay_phase_index:.2f}",
                "phase_rate:",
                f"{self.replay_phase_rate:.3f}",
                "contact:",
                contact,
                "update:",
                should_update,
                "decimation:",
                current_decimation,
            )
            print("[ReplayDataset] p_now:", p_now)
            print("[ReplayDataset] pd_cmd:", self.replay_pd)
            print("[ReplayDataset] ||p_now - pd_cmd||:", np.linalg.norm(p_now - self.replay_pd))
            print("[ReplayDataset] Vd_body_cmd:", self.replay_Vd_body)
            print("[ReplayDataset] Vd_star:", Vd_star)

        return Vd_star.astype(np.float32), dVd_star.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay a 1000Hz LeRobot bolt/nut dataset as 50Hz desired motion "
            "through the GUFIC force-impedance controller."
        )
    )
    parser.add_argument(
        "--dataset_root",
        default="/media/zhou/Elements SE/VLA/boltnut3_pi0_lerobot_random_start",
    )
    parser.add_argument("--dataset_repo_id", default="gufic_boltnut_pi0")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--source_fps", type=float, default=1000.0)
    parser.add_argument("--default_fps", type=float, default=20.0)
    parser.add_argument("--contact_fps", type=float, default=100.0)
    parser.add_argument("--contact_force_threshold", type=float, default=1.0)
    parser.add_argument(
        "--phase_force_ref",
        type=float,
        default=8.0,
        help="Reference |Fz| in N. Above this, replay phase slows down smoothly.",
    )
    parser.add_argument(
        "--phase_force_scale",
        type=float,
        default=8.0,
        help="Force scale in N for smooth phase slowdown.",
    )
    parser.add_argument(
        "--min_phase_rate",
        type=float,
        default=0.02,
        help="Lower bound for phase speed ratio to avoid completely freezing replay.",
    )
    parser.add_argument(
        "--replay_time_scale",
        type=float,
        default=2.0,
        help="Allow extra simulation time because force feedback can slow replay phase.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Deprecated; dynamic replay keeps the raw source sequence and ignores this option.",
    )
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--robot_name", default="indy7")
    parser.add_argument("--task", default="bolt")
    parser.add_argument("--fz", type=float, default=5.0)
    parser.add_argument("--show_viewer", default=True, action="store_true")
    parser.add_argument("--randomized_start", action="store_true")
    parser.add_argument("--inertia_shaping", action="store_true")
    parser.add_argument(
        "--no_init_from_dataset_pose",
        action="store_true",
        help="Do not set the robot to the first dataset observation pose before replay.",
    )
    parser.add_argument("--lowpass_alpha", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    actions, p0, R0, _indices = load_replay_actions(
        dataset_root=dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        episode_index=args.episode_index,
        source_fps=args.source_fps,
        start_offset=args.start_offset,
        max_frames=args.max_frames,
    )

    if args.stride is not None:
        print("[ReplayDataset] WARNING: --stride is ignored by dynamic phase replay.")

    max_time = len(actions) / float(args.source_fps) * max(args.replay_time_scale, 1.0)
    env = ReplayDatasetEnv(
        robot_name=args.robot_name,
        model="smolvla",
        max_time=max_time,
        show_viewer=args.show_viewer,
        fz=args.fz,
        fix_camera=True,
        task=args.task,
        randomized_start=args.randomized_start,
        inertia_shaping=args.inertia_shaping,
        use_learned_velocity_field=False,
        save_tensorboard=False,
        test_offline_cond=False,
        visualize_delta_pose=False,
    )

    init_pose = None if args.no_init_from_dataset_pose else (p0, R0)
    env.configure_dataset_replay(
        actions=actions,
        source_fps=args.source_fps,
        default_fps=args.default_fps,
        contact_fps=args.contact_fps,
        contact_force_threshold=args.contact_force_threshold,
        phase_force_ref=args.phase_force_ref,
        phase_force_scale=args.phase_force_scale,
        min_phase_rate=args.min_phase_rate,
        replay_time_scale=args.replay_time_scale,
        init_from_dataset_pose=init_pose,
        lowpass_alpha=args.lowpass_alpha,
    )

    env.run()
    if hasattr(env, "check_task_success"):
        success = env.check_task_success()
        print("[ReplayDataset] success:", success)
    else:
        print("[ReplayDataset] replay finished.")

    if args.show_viewer and env.viewer is not None:
        env.viewer.close()


if __name__ == "__main__":
    main()
