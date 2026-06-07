import argparse
import copy
import types
from pathlib import Path

import numpy as np
import torch


def hat_map(w):
    wx, wy, wz = np.asarray(w, dtype=np.float32).reshape(3)
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ],
        dtype=np.float32,
    )


def vee_map(R):
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    v3 = -R[0, 1]
    v1 = -R[1, 2]
    v2 = R[0, 2]
    return np.array([v1, v2, v3], dtype=np.float32).reshape(3)


def rotmat_to_rot6d_np(R):
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    return R[:, :2].T.reshape(6).astype(np.float32)


def rot6d_to_rotmat_np(r6d):
    r6d = np.asarray(r6d, dtype=np.float32).reshape(6)
    a1 = r6d[:3]
    a2 = r6d[3:6]

    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2_orth = a2 - np.dot(b1, a2) * b1
    b2 = a2_orth / (np.linalg.norm(a2_orth) + 1e-8)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=1).astype(np.float32)


def vd_body_to_dpd_dRd(Vd_body, Rd):
    """
    The dataset stores action.Vd_body = [Rd.T @ dpd, vee(Rd.T @ dRd)].
    This function restores world-frame dpd and matrix dRd.
    """
    Vd_body = np.asarray(Vd_body, dtype=np.float32).reshape(6)
    Rd = np.asarray(Rd, dtype=np.float32).reshape(3, 3)

    vd_body = Vd_body[:3]
    wd_body = Vd_body[3:]

    dpd = Rd @ vd_body
    dRd = Rd @ hat_map(wd_body)
    return dpd.astype(np.float32), dRd.astype(np.float32)


def get_velocity_field(g, pd, Rd, dpd, dRd, zeta_v=50.0, zeta_w=10.0):
    """
    Analytic GUFIC velocity field used after pi0 predicts desired pose and
    desired trajectory velocity.
    """
    g = np.asarray(g, dtype=np.float32).reshape(4, 4)
    pd = np.asarray(pd, dtype=np.float32).reshape(3)
    Rd = np.asarray(Rd, dtype=np.float32).reshape(3, 3)
    dpd = np.asarray(dpd, dtype=np.float32).reshape(3)
    dRd = np.asarray(dRd, dtype=np.float32).reshape(3, 3)

    p = g[:3, 3]
    R = g[:3, :3]

    Vd_star = np.zeros(6, dtype=np.float32)
    Vd_star[:3] = (
        R.T @ dRd @ Rd.T @ (p - pd)
        + R.T @ dpd
        - zeta_v * R.T @ (p - pd)
    )
    Vd_star[3:] = vee_map(
        R.T @ dRd @ Rd.T @ R
        - zeta_w * (Rd.T @ R - R.T @ Rd)
    )
    return Vd_star.astype(np.float32)


def resize_rgb_np(image, size=(256, 256)):
    from PIL import Image

    image = np.asarray(image, dtype=np.uint8)
    if image.shape[0] == size[1] and image.shape[1] == size[0]:
        return image
    return np.asarray(Image.fromarray(image).resize(size), dtype=np.uint8)


def rgb_to_tensor(image, device):
    """
    pi0 expects image tensors in [B, C, H, W] with float values in [0, 1].
    The policy itself will resize/pad to 224 and map to [-1, 1].
    """
    image = resize_rgb_np(image, size=(256, 256))
    image_t = torch.from_numpy(image).to(device=device, dtype=torch.float32) / 255.0
    image_t = image_t.permute(2, 0, 1).unsqueeze(0).contiguous()
    return image_t


def make_robot_state(p, R, Fe=None, expected_dim=None):
    p = np.asarray(p, dtype=np.float32).reshape(3)
    R6d = rotmat_to_rot6d_np(R)

    if expected_dim == 15:
        if Fe is None:
            Fe = np.zeros(6, dtype=np.float32)
        return np.concatenate([p, R6d, np.asarray(Fe, dtype=np.float32).reshape(6)], axis=0)

    if expected_dim == 6:
        if Fe is not None:
            return np.asarray(Fe, dtype=np.float32).reshape(6)
        return R6d

    return np.concatenate([p, R6d], axis=0).astype(np.float32)


def unpack_pi0_action(action):
    """
    action layout from env_gufic_velocity_field_collect_dataset_val.py:
        [pd(3), Rd6d(6), Vd_body(6)]
    If the policy was trained with PI0_ACTION_MODE=pose, the action layout is:
        [pd(3), Rd6d(6)]
    and Vd_body is set to zero.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] < 9:
        raise ValueError(f"pi0 action dim must be at least 9, got {action.shape[0]}")

    pd = action[:3].astype(np.float32)
    Rd6d = action[3:9].astype(np.float32)
    if action.shape[0] >= 15:
        Vd_body = action[9:15].astype(np.float32)
    else:
        Vd_body = np.zeros(6, dtype=np.float32)
    Rd = rot6d_to_rotmat_np(Rd6d)
    dpd, dRd = vd_body_to_dpd_dRd(Vd_body, Rd)
    return pd, Rd, Vd_body, dpd, dRd


class PI0VelocityFieldInfer:
    """
    Use a fine-tuned pi0 policy to predict:
        pd, Rd, Vd_body
    then restore:
        dpd, dRd
    and compute GUFIC Vd_star for the force-impedance controller.
    """

    def __init__(
        self,
        policy_path,
        dataset_repo_id,
        dataset_root,
        language="insert the bolt into the hole",
        device=None,
        zeta_v=50.0,
        zeta_w=10.0,
        action_mode="pose",
    ):
        self.policy_path = Path(policy_path)
        self.dataset_repo_id = dataset_repo_id
        self.dataset_root = Path(dataset_root)
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.zeta_v = float(zeta_v)
        self.zeta_w = float(zeta_w)
        self.action_mode = action_mode.lower()

        self.policy, self.state_key, self.force_key = self._load_policy()
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()

    def reset(self):
        self.policy.reset()

    def _load_policy(self):
        from lerobot.common.constants import OBS_STATE
        from lerobot.common.datasets.utils import dataset_to_policy_features
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy, pad_vector
        from lerobot.configs.types import FeatureType

        metadata = LeRobotDatasetMetadata(
            self.dataset_repo_id,
            root=self.dataset_root,
        )
        features = copy.deepcopy(metadata.features)
        stats = copy.deepcopy(metadata.stats)

        if self.action_mode in ("pose", "pose_only", "no_vd_body", "no_vd"):
            features["action"]["shape"] = (9,)
            features["action"]["names"] = ["pd_Rd6d"]
            if "action" in stats:
                for stat_name, value in list(stats["action"].items()):
                    if isinstance(value, torch.Tensor):
                        stats["action"][stat_name] = value[:9].clone()
                    else:
                        stats["action"][stat_name] = value[:9].copy()
        elif self.action_mode not in ("full", "pose_vd", "pose_vd_body", "with_vd_body"):
            raise ValueError(
                f"Unknown pi0 action mode {self.action_mode!r}. Use 'pose' or 'full'."
            )

        config = PreTrainedConfig.from_pretrained(str(self.policy_path))
        policy_features = dataset_to_policy_features(features)
        config.input_features = {
            key: ft for key, ft in policy_features.items() if ft.type is not FeatureType.ACTION
        }
        config.output_features = {
            key: ft for key, ft in policy_features.items() if ft.type is FeatureType.ACTION
        }
        config.device = self.device

        policy = PI0Policy.from_pretrained(
            str(self.policy_path),
            config=config,
            dataset_stats=stats,
        )

        state_keys = [
            key
            for key, ft in policy.config.input_features.items()
            if ft.type is FeatureType.STATE
        ]
        if not state_keys:
            raise ValueError("pi0 policy has no STATE input feature.")

        if OBS_STATE in state_keys:
            state_key = OBS_STATE
        elif "observation.robot_state" in state_keys:
            state_key = "observation.robot_state"
        else:
            state_key = state_keys[0]

        force_key = "observation.force" if "observation.force" in state_keys else None

        if state_key != OBS_STATE:
            def prepare_state(policy_self, batch):
                return pad_vector(batch[state_key], policy_self.config.max_state_dim)

            def prepare_language(policy_self, batch):
                device = batch[state_key].device
                tasks = batch["task"]
                tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]
                tokenized_prompt = policy_self.language_tokenizer.__call__(
                    tasks,
                    padding="max_length",
                    padding_side="right",
                    max_length=policy_self.config.tokenizer_max_length,
                    return_tensors="pt",
                )
                lang_tokens = tokenized_prompt["input_ids"].to(device=device)
                lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)
                return lang_tokens, lang_masks

            policy.prepare_state = types.MethodType(prepare_state, policy)
            policy.prepare_language = types.MethodType(prepare_language, policy)

        return policy, state_key, force_key

    def _select_image_for_key(self, key, wrist_image, external_image):
        if "wrist" in key:
            return wrist_image
        return external_image

    def build_batch(self, wrist_image, external_image, p, R, Fe):
        from lerobot.common.constants import OBS_STATE

        batch = {"task": [self.language]}

        for key in self.policy.config.image_features:
            image = self._select_image_for_key(key, wrist_image, external_image)
            batch[key] = rgb_to_tensor(image, self.device)

        expected_dim = int(self.policy.config.input_features[self.state_key].shape[0])
        state = make_robot_state(p, R, Fe=Fe, expected_dim=expected_dim)
        state_t = torch.from_numpy(state[None, :]).to(self.device, dtype=torch.float32)
        batch[self.state_key] = state_t

        # Some pi0 implementations still access the constant observation.state.
        # If state_key is different, prepare_state/prepare_language are patched,
        # but this alias keeps adapt_to_pi_aloha=False code paths harmless.
        if OBS_STATE not in batch:
            batch[OBS_STATE] = state_t

        if self.force_key is not None and self.force_key != self.state_key:
            force = np.asarray(Fe, dtype=np.float32).reshape(6)
            batch[self.force_key] = torch.from_numpy(force[None, :]).to(
                self.device,
                dtype=torch.float32,
            )

        return batch

    @torch.no_grad()
    def predict_action(self, wrist_image, external_image, p, R, Fe):
        batch = self.build_batch(
            wrist_image=wrist_image,
            external_image=external_image,
            p=p,
            R=R,
            Fe=Fe,
        )
        action = self.policy.select_action(batch)
        return action[0].detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict_desired_motion(self, wrist_image, external_image, p, R, Fe):
        action = self.predict_action(
            wrist_image=wrist_image,
            external_image=external_image,
            p=p,
            R=R,
            Fe=Fe,
        )
        pd, Rd, Vd_body, dpd, dRd = unpack_pi0_action(action)
        return {
            "action": action,
            "pd": pd,
            "Rd": Rd,
            "Vd_body": Vd_body,
            "dpd": dpd,
            "dRd": dRd,
        }

    @torch.no_grad()
    def predict_velocity_field(self, wrist_image, external_image, p, R, Fe):
        result = self.predict_desired_motion(
            wrist_image=wrist_image,
            external_image=external_image,
            p=p,
            R=R,
            Fe=Fe,
        )

        g = np.eye(4, dtype=np.float32)
        g[:3, :3] = np.asarray(R, dtype=np.float32).reshape(3, 3)
        g[:3, 3] = np.asarray(p, dtype=np.float32).reshape(3)

        Vd_star = get_velocity_field(
            g=g,
            pd=result["pd"],
            Rd=result["Rd"],
            dpd=result["dpd"],
            dRd=result["dRd"],
            zeta_v=self.zeta_v,
            zeta_w=self.zeta_w,
        )
        result["Vd_star"] = Vd_star
        return result


def load_pi0_velocity_field_infer(
    policy_path,
    dataset_repo_id,
    dataset_root,
    language="insert the bolt into the hole",
    device=None,
    zeta_v=50.0,
    zeta_w=10.0,
    action_mode="pose",
):
    return PI0VelocityFieldInfer(
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        language=language,
        device=device,
        zeta_v=zeta_v,
        zeta_w=zeta_w,
        action_mode=action_mode,
    )


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def image_to_uint8_hwc(image):
    image = to_numpy(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {image.shape}")

    # LeRobot samples are usually [C,H,W], while env renderers are [H,W,C].
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        if np.nanmax(image) <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def sample_to_pi0_inputs(sample):
    wrist_key = "observation.wrist_image"
    external_key = "observation.external_image"

    if wrist_key not in sample:
        raise KeyError(f"Dataset sample missing {wrist_key}")

    wrist_image = image_to_uint8_hwc(sample[wrist_key])
    if external_key in sample:
        external_image = image_to_uint8_hwc(sample[external_key])
    elif "observation.image" in sample:
        external_image = image_to_uint8_hwc(sample["observation.image"])
    else:
        external_image = wrist_image

    if "observation.robot_state" in sample:
        robot_state = to_numpy(sample["observation.robot_state"]).astype(np.float32).reshape(-1)
    elif "observation.state" in sample:
        robot_state = to_numpy(sample["observation.state"]).astype(np.float32).reshape(-1)
    else:
        raise KeyError("Dataset sample missing observation.robot_state / observation.state")

    if robot_state.shape[0] < 9:
        raise ValueError(
            "robot_state must contain [p(3), R6d(6)] for velocity-field inference, "
            f"got shape {robot_state.shape}"
        )

    p = robot_state[:3]
    R = rot6d_to_rotmat_np(robot_state[3:9])

    if "observation.force" in sample:
        Fe = to_numpy(sample["observation.force"]).astype(np.float32).reshape(6)
    elif robot_state.shape[0] >= 15:
        Fe = robot_state[9:15]
    else:
        Fe = np.zeros(6, dtype=np.float32)

    return wrist_image, external_image, p, R, Fe


def print_result(name, result):
    print(f"\n[{name}]")
    print("pd:", result["pd"])
    print("Rd:\n", result["Rd"])
    print("Vd_body:", result["Vd_body"])
    print("dpd:", result["dpd"])
    print("dRd:\n", result["dRd"])
    if "Vd_star" in result:
        print("Vd_star:", result["Vd_star"])


def rotation_geodesic_error_deg(R_pred, R_gt):
    R_pred = np.asarray(R_pred, dtype=np.float32).reshape(-1, 3, 3)
    R_gt = np.asarray(R_gt, dtype=np.float32).reshape(-1, 3, 3)
    R_err = np.einsum("nij,njk->nik", np.transpose(R_pred, (0, 2, 1)), R_gt)
    trace = np.trace(R_err, axis1=1, axis2=2)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_theta) * 180.0 / np.pi


def plot_vector_comparison(pred, gt, labels, title, ylabel, save_path):
    import matplotlib.pyplot as plt

    pred = np.asarray(pred)
    gt = np.asarray(gt)
    step_idx = np.arange(len(pred))

    fig, axes = plt.subplots(len(labels), 1, figsize=(12, 2.5 * len(labels)), sharex=True)
    if len(labels) == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(step_idx, pred[:, i], linewidth=1.4, label=f"pred_{labels[i]}")
        ax.plot(step_idx, gt[:, i], linewidth=1.1, linestyle="--", label=f"gt_{labels[i]}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[0].set_title(title)
    axes[-1].set_xlabel("sample index")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_scalar_error(values, title, ylabel, save_path):
    import matplotlib.pyplot as plt

    values = np.asarray(values)
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.5))
    ax.plot(np.arange(len(values)), values, linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("sample index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_rotation_matrix_comparison(R_pred, R_gt, save_path):
    import matplotlib.pyplot as plt

    R_pred = np.asarray(R_pred)
    R_gt = np.asarray(R_gt)
    step_idx = np.arange(len(R_pred))

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
    for r in range(3):
        for c in range(3):
            ax = axes[r, c]
            ax.plot(step_idx, R_pred[:, r, c], linewidth=1.2, label="pred")
            ax.plot(step_idx, R_gt[:, r, c], linewidth=1.0, linestyle="--", label="gt")
            ax.set_title(f"R[{r},{c}]")
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(loc="best")

    fig.suptitle("Rd Matrix Comparison")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def save_comparison_plots(records, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd_pred = np.stack(records["pd_pred"], axis=0)
    pd_gt = np.stack(records["pd_gt"], axis=0)
    Rd_pred = np.stack(records["Rd_pred"], axis=0)
    Rd_gt = np.stack(records["Rd_gt"], axis=0)
    Vd_body_pred = np.stack(records["Vd_body_pred"], axis=0)
    Vd_body_gt = np.stack(records["Vd_body_gt"], axis=0)
    Vd_star_pred = np.stack(records["Vd_star_pred"], axis=0)
    Vd_star_gt = np.stack(records["Vd_star_gt"], axis=0)

    plot_vector_comparison(
        pd_pred,
        pd_gt,
        labels=["x", "y", "z"],
        title="Desired Position pd",
        ylabel="m",
        save_path=out_dir / "pd_comparison.png",
    )
    plot_rotation_matrix_comparison(
        Rd_pred,
        Rd_gt,
        save_path=out_dir / "Rd_matrix_comparison.png",
    )
    plot_vector_comparison(
        Vd_body_pred,
        Vd_body_gt,
        labels=["vx", "vy", "vz", "wx", "wy", "wz"],
        title="Desired Body Velocity Vd_body",
        ylabel="m/s, rad/s",
        save_path=out_dir / "Vd_body_comparison.png",
    )
    plot_vector_comparison(
        Vd_star_pred,
        Vd_star_gt,
        labels=["vx", "vy", "vz", "wx", "wy", "wz"],
        title="Final GUFIC Velocity Field Vd_star",
        ylabel="m/s, rad/s",
        save_path=out_dir / "Vd_star_comparison.png",
    )

    pd_err = np.linalg.norm(pd_pred - pd_gt, axis=1)
    rot_err = rotation_geodesic_error_deg(Rd_pred, Rd_gt)
    Vd_body_err = np.linalg.norm(Vd_body_pred - Vd_body_gt, axis=1)
    Vd_star_err = np.linalg.norm(Vd_star_pred - Vd_star_gt, axis=1)

    plot_scalar_error(pd_err, "pd Error Norm", "m", out_dir / "pd_error_norm.png")
    plot_scalar_error(rot_err, "Rd Geodesic Error", "deg", out_dir / "Rd_error_deg.png")
    plot_scalar_error(Vd_body_err, "Vd_body Error Norm", "m/s, rad/s", out_dir / "Vd_body_error_norm.png")
    plot_scalar_error(Vd_star_err, "Vd_star Error Norm", "m/s, rad/s", out_dir / "Vd_star_error_norm.png")

    np.savez_compressed(
        out_dir / "pi0_comparison_arrays.npz",
        pd_pred=pd_pred,
        pd_gt=pd_gt,
        Rd_pred=Rd_pred,
        Rd_gt=Rd_gt,
        Vd_body_pred=Vd_body_pred,
        Vd_body_gt=Vd_body_gt,
        Vd_star_pred=Vd_star_pred,
        Vd_star_gt=Vd_star_gt,
        pd_err=pd_err,
        rot_err_deg=rot_err,
        Vd_body_err=Vd_body_err,
        Vd_star_err=Vd_star_err,
        frame_indices=np.asarray(records["frame_indices"], dtype=np.int64),
    )

    print(f"\nSaved comparison plots to: {out_dir}")
    print(f"pd_err mean/max: {pd_err.mean():.6f} / {pd_err.max():.6f} m")
    print(f"Rd_err mean/max: {rot_err.mean():.6f} / {rot_err.max():.6f} deg")
    print(f"Vd_body_err mean/max: {Vd_body_err.mean():.6f} / {Vd_body_err.max():.6f}")
    print(f"Vd_star_err mean/max: {Vd_star_err.mean():.6f} / {Vd_star_err.max():.6f}")


def run_dataset_comparison(
    policy_path,
    dataset_repo_id,
    dataset_root,
    out_dir,
    start_index=0,
    max_frames=1000,
    stride=1,
    language="insert the bolt into the hole",
    device=None,
    zeta_v=50.0,
    zeta_w=10.0,
    action_mode="pose",
    reset_each_frame=True,
):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)
    pi0_infer = load_pi0_velocity_field_infer(
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        language=language,
        device=device,
        zeta_v=zeta_v,
        zeta_w=zeta_w,
        action_mode=action_mode,
    )

    end_index = len(dataset) if max_frames is None else min(len(dataset), start_index + max_frames * stride)
    frame_indices = list(range(int(start_index), int(end_index), int(stride)))

    records = {
        "frame_indices": [],
        "pd_pred": [],
        "pd_gt": [],
        "Rd_pred": [],
        "Rd_gt": [],
        "Vd_body_pred": [],
        "Vd_body_gt": [],
        "Vd_star_pred": [],
        "Vd_star_gt": [],
    }

    for n, frame_index in enumerate(frame_indices):
        sample = dataset[frame_index]
        if "action" not in sample:
            raise KeyError("Dataset sample has no action, cannot draw pred/gt comparison.")

        wrist_image, external_image, p, R, Fe = sample_to_pi0_inputs(sample)

        if reset_each_frame:
            pi0_infer.reset()

        pred = pi0_infer.predict_velocity_field(
            wrist_image=wrist_image,
            external_image=external_image,
            p=p,
            R=R,
            Fe=Fe,
        )

        action_gt = to_numpy(sample["action"]).astype(np.float32).reshape(-1)
        pd_gt, Rd_gt, Vd_body_gt, dpd_gt, dRd_gt = unpack_pi0_action(action_gt)
        g = np.eye(4, dtype=np.float32)
        g[:3, :3] = R
        g[:3, 3] = p
        Vd_star_gt = get_velocity_field(
            g=g,
            pd=pd_gt,
            Rd=Rd_gt,
            dpd=dpd_gt,
            dRd=dRd_gt,
            zeta_v=zeta_v,
            zeta_w=zeta_w,
        )

        records["frame_indices"].append(frame_index)
        records["pd_pred"].append(pred["pd"])
        records["pd_gt"].append(pd_gt)
        records["Rd_pred"].append(pred["Rd"])
        records["Rd_gt"].append(Rd_gt)
        records["Vd_body_pred"].append(pred["Vd_body"])
        records["Vd_body_gt"].append(Vd_body_gt)
        records["Vd_star_pred"].append(pred["Vd_star"])
        records["Vd_star_gt"].append(Vd_star_gt)

        if n % 50 == 0:
            print(f"[{n + 1}/{len(frame_indices)}] frame={frame_index}")

    save_comparison_plots(records, out_dir)
    return records


def run_one_dataset_frame(
    policy_path,
    dataset_repo_id,
    dataset_root,
    frame_index=0,
    language="insert the bolt into the hole",
    device=None,
    zeta_v=50.0,
    zeta_w=10.0,
    save_npz=None,
    action_mode="pose",
):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)
    sample = dataset[int(frame_index)]
    wrist_image, external_image, p, R, Fe = sample_to_pi0_inputs(sample)

    pi0_infer = load_pi0_velocity_field_infer(
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        language=language,
        device=device,
        zeta_v=zeta_v,
        zeta_w=zeta_w,
        action_mode=action_mode,
    )
    result = pi0_infer.predict_velocity_field(
        wrist_image=wrist_image,
        external_image=external_image,
        p=p,
        R=R,
        Fe=Fe,
    )

    print(f"frame_index: {frame_index}")
    print("p_now:", p)
    print("Fe:", Fe)
    print_result("pi0_pred", result)

    gt_result = None
    if "action" in sample:
        action_gt = to_numpy(sample["action"]).astype(np.float32).reshape(-1)
        pd_gt, Rd_gt, Vd_body_gt, dpd_gt, dRd_gt = unpack_pi0_action(action_gt)
        g = np.eye(4, dtype=np.float32)
        g[:3, :3] = R
        g[:3, 3] = p
        gt_result = {
            "action": action_gt,
            "pd": pd_gt,
            "Rd": Rd_gt,
            "Vd_body": Vd_body_gt,
            "dpd": dpd_gt,
            "dRd": dRd_gt,
            "Vd_star": get_velocity_field(
                g=g,
                pd=pd_gt,
                Rd=Rd_gt,
                dpd=dpd_gt,
                dRd=dRd_gt,
                zeta_v=zeta_v,
                zeta_w=zeta_w,
            ),
        }
        print_result("dataset_gt", gt_result)
        print("\n[error]")
        print("||pd_pred - pd_gt||:", float(np.linalg.norm(result["pd"] - pd_gt)))
        print("||Vd_body_pred - Vd_body_gt||:", float(np.linalg.norm(result["Vd_body"] - Vd_body_gt)))
        print("||Vd_star_pred - Vd_star_gt||:", float(np.linalg.norm(result["Vd_star"] - gt_result["Vd_star"])))

    if save_npz is not None:
        save_path = Path(save_npz)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            p=p,
            R=R,
            Fe=Fe,
            pred_action=result["action"],
            pred_pd=result["pd"],
            pred_Rd=result["Rd"],
            pred_Vd_body=result["Vd_body"],
            pred_dpd=result["dpd"],
            pred_dRd=result["dRd"],
            pred_Vd_star=result["Vd_star"],
            gt_action=None if gt_result is None else gt_result["action"],
            gt_pd=None if gt_result is None else gt_result["pd"],
            gt_Rd=None if gt_result is None else gt_result["Rd"],
            gt_Vd_body=None if gt_result is None else gt_result["Vd_body"],
            gt_Vd_star=None if gt_result is None else gt_result["Vd_star"],
        )
        print(f"\nSaved: {save_path}")

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pi0 inference for pd, Rd, dpd, dRd and GUFIC velocity field."
    )
    parser.add_argument(
        "--policy_path",
        default="/media/zhou/Elements SE/VLA/checkpoints_pi0/checkpoints/001000/pretrained_model",
    )
    parser.add_argument(
        "--dataset_root",
        default="/media/zhou/Elements SE/VLA/boltnut3_pi0_lerobot_random_start",
        help="LeRobot dataset root used for metadata/stats and optional frame test.",
    )
    parser.add_argument("--dataset_repo_id", default="gufic_boltnut_pi0")
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--language", default="insert the bolt into the hole")
    parser.add_argument("--device", default=None)
    parser.add_argument("--zeta_v", type=float, default=50.0)
    parser.add_argument("--zeta_w", type=float, default=10.0)
    parser.add_argument(
        "--action_mode",
        default="pose",
        choices=["pose", "full"],
        help="Use 'pose' for [pd,Rd6d] checkpoints, or 'full' for [pd,Rd6d,Vd_body].",
    )
    parser.add_argument("--save_npz", default=None)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run multi-frame dataset comparison and save plots instead of only printing one frame.",
    )
    parser.add_argument(
        "--out_dir",
        default="./infer_pi0_compare",
        help="Output directory for --compare plots.",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--keep_action_queue",
        action="store_true",
        help="Keep pi0 action queue across frames. Default resets every frame for fair one-step pred/gt plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.compare:
        run_dataset_comparison(
            policy_path=args.policy_path,
            dataset_repo_id=args.dataset_repo_id,
            dataset_root=args.dataset_root,
            out_dir=args.out_dir,
            start_index=args.start_index,
            max_frames=args.max_frames,
            stride=args.stride,
            language=args.language,
            device=args.device,
            zeta_v=args.zeta_v,
            zeta_w=args.zeta_w,
            action_mode=args.action_mode,
            reset_each_frame=not args.keep_action_queue,
        )
    else:
        run_one_dataset_frame(
            policy_path=args.policy_path,
            dataset_repo_id=args.dataset_repo_id,
            dataset_root=args.dataset_root,
            frame_index=args.frame_index,
            language=args.language,
            device=args.device,
            zeta_v=args.zeta_v,
            zeta_w=args.zeta_w,
            save_npz=args.save_npz,
            action_mode=args.action_mode,
        )


if __name__ == "__main__":
    main()
