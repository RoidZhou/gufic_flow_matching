import argparse
import json
import os
import sys
import types
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as RT

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    try:
        from tensorboardX import SummaryWriter
    except Exception:
        SummaryWriter = None

try:
    import mujoco
    import mujoco.viewer
except Exception as exc:
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gufic_env.env_gufic_velocity_field_infer_smolvla import RobotEnv
from gufic_env.utils.misc_func import hat_map, vee_map


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


TASK_NAME = "insert the bolt into the hole"
DEFAULT_DATASET_REPO_ID = "gufic_nutbolt_position_smolvla"
DEFAULT_DATASET_ROOT = "/media/zhou/Elements SE/VLA/nutbolt_position_smolvla"
DEFAULT_POLICY_PATH = (
    "/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/"
    "checkpoints_smolvla_position/checkpoints/last/pretrained_model"
)
DEFAULT_VLM_MODEL_NAME = (
    "/home/zhou/.cache/huggingface/hub/"
    "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/"
    "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
)


def first_existing_path(paths):
    for path in paths:
        if path and Path(path).exists():
            return str(path)
    return None


def resolve_policy_path(policy_path=None):
    if policy_path:
        return str(policy_path)
    env_path = os.environ.get("SMOLVLA_POLICY_PATH")
    if env_path:
        return env_path
    path = first_existing_path([DEFAULT_POLICY_PATH])
    if path:
        return path
    raise FileNotFoundError(
        "No SmolVLA position policy checkpoint found. Pass --policy_path or set "
        "SMOLVLA_POLICY_PATH."
    )


def resolve_vlm_model_name(vlm_model_name=None):
    if vlm_model_name:
        if not Path(vlm_model_name).exists():
            raise FileNotFoundError(f"vlm_model_name path does not exist: {vlm_model_name}")
        return str(vlm_model_name)
    env_path = os.environ.get("SMOLVLA_VLM_MODEL_NAME")
    if env_path:
        if not Path(env_path).exists():
            raise FileNotFoundError(f"SMOLVLA_VLM_MODEL_NAME path does not exist: {env_path}")
        return env_path
    if Path(DEFAULT_VLM_MODEL_NAME).exists():
        return DEFAULT_VLM_MODEL_NAME
    raise FileNotFoundError(
        "SmolVLA needs a local SmolVLM snapshot. Pass --vlm_model_name or set "
        "SMOLVLA_VLM_MODEL_NAME."
    )


def load_smolvla_config(policy_path, device, vlm_model_name):
    from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.policies import PreTrainedConfig

    try:
        config = PreTrainedConfig.from_pretrained(str(policy_path))
    except Exception as exc:
        train_config_path = Path(policy_path) / "train_config.json"
        if not train_config_path.exists():
            raise RuntimeError(
                f"Failed to parse SmolVLA config from {policy_path}, and train_config.json was not found."
            ) from exc

        with open(train_config_path, "r", encoding="utf-8") as f:
            train_config = json.load(f)
        policy_config = train_config.get("policy", {})
        if policy_config.get("type") != "smolvla":
            raise RuntimeError(
                f"train_config.json does not contain a SmolVLA policy config: {train_config_path}"
            ) from exc

        valid_fields = {field.name for field in fields(SmolVLAConfig)}
        skip_fields = {"type", "input_features", "output_features", "normalization_mapping"}
        kwargs = {
            key: value
            for key, value in policy_config.items()
            if key in valid_fields and key not in skip_fields
        }
        config = SmolVLAConfig(**kwargs)

    config.vlm_model_name = vlm_model_name
    config.device = device
    return config


def resize_rgb(image, size=(256, 256)):
    image = np.asarray(image, dtype=np.uint8).copy()
    if image.shape[0] == size[1] and image.shape[1] == size[0]:
        return image
    return np.asarray(Image.fromarray(image).resize(size), dtype=np.uint8).copy()


def rgb_to_tensor(image, device):
    image = resize_rgb(image)
    image_t = torch.from_numpy(image).to(device=device, dtype=torch.float32) / 255.0
    return image_t.permute(2, 0, 1).unsqueeze(0).contiguous()


def pose_to_xyzrpy(p, R):
    p = np.asarray(p, dtype=np.float32).reshape(3)
    euler = RT.from_matrix(np.asarray(R).reshape(3, 3)).as_euler("xyz", degrees=False)
    return np.concatenate([p, euler.astype(np.float32)], axis=0).astype(np.float32)


class SmolVLAJointPositionInfer:
    """SmolVLA inference for datasets with action=[q1..q6, gripper]."""

    def __init__(
        self,
        policy_path=None,
        dataset_repo_id=DEFAULT_DATASET_REPO_ID,
        dataset_root=DEFAULT_DATASET_ROOT,
        language=TASK_NAME,
        device=None,
        vlm_model_name=None,
    ):
        self.policy_path = Path(resolve_policy_path(policy_path))
        self.dataset_repo_id = dataset_repo_id
        self.dataset_root = Path(dataset_root)
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vlm_model_name = resolve_vlm_model_name(vlm_model_name)

        self.policy, self.state_key = self._load_policy()
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()

    def reset(self):
        self.policy.reset()

    def _load_policy(self):
        from lerobot.common.constants import OBS_STATE
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.common.datasets.utils import dataset_to_policy_features
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
        from lerobot.configs.types import FeatureType

        metadata = LeRobotDatasetMetadata(self.dataset_repo_id, root=self.dataset_root)
        config = load_smolvla_config(self.policy_path, self.device, self.vlm_model_name)

        policy_features = dataset_to_policy_features(metadata.features)
        config.input_features = {
            key: ft for key, ft in policy_features.items() if ft.type is not FeatureType.ACTION
        }
        config.output_features = {
            key: ft for key, ft in policy_features.items() if ft.type is FeatureType.ACTION
        }

        policy = SmolVLAPolicy.from_pretrained(
            str(self.policy_path),
            config=config,
            dataset_stats=metadata.stats,
        )

        state_keys = [
            key
            for key, ft in policy.config.input_features.items()
            if ft.type is FeatureType.STATE
        ]
        if not state_keys:
            raise ValueError("SmolVLA policy has no STATE input feature.")
        state_key = OBS_STATE if OBS_STATE in state_keys else state_keys[0]

        if state_key != OBS_STATE:

            def prepare_state(policy_self, batch):
                state = batch[state_key][:, -1, :] if batch[state_key].ndim > 2 else batch[state_key]
                return pad_vector(state, policy_self.config.max_state_dim)

            def prepare_language(policy_self, batch):
                device = batch[state_key].device
                tasks = batch["task"]
                if len(tasks) == 1:
                    tasks = [tasks[0] for _ in range(batch[state_key].shape[0])]
                tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]
                tokenized_prompt = policy_self.language_tokenizer.__call__(
                    tasks,
                    padding=policy_self.config.pad_language_to,
                    padding_side="right",
                    max_length=policy_self.config.tokenizer_max_length,
                    return_tensors="pt",
                )
                lang_tokens = tokenized_prompt["input_ids"].to(device=device)
                lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)
                return lang_tokens, lang_masks

            policy.prepare_state = types.MethodType(prepare_state, policy)
            policy.prepare_language = types.MethodType(prepare_language, policy)

        return policy, state_key

    def _select_image_for_key(self, key, wrist_image, external_image):
        if "wrist" in key:
            return wrist_image
        return external_image

    def build_batch(self, wrist_image, external_image, p, R):
        from lerobot.common.constants import OBS_STATE

        batch = {"task": [self.language]}
        for key in self.policy.config.image_features:
            image = self._select_image_for_key(key, wrist_image, external_image)
            batch[key] = rgb_to_tensor(image, self.device)

        state = pose_to_xyzrpy(p, R)
        expected_dim = int(self.policy.config.input_features[self.state_key].shape[0])
        if expected_dim != state.shape[0]:
            raise ValueError(
                f"Policy state dim is {expected_dim}, but position dataset state is {state.shape[0]}."
            )
        state_t = torch.from_numpy(state[None, :]).to(self.device, dtype=torch.float32)
        batch[self.state_key] = state_t
        if OBS_STATE not in batch:
            batch[OBS_STATE] = state_t
        return batch

    @torch.no_grad()
    def predict_action(self, wrist_image, external_image, p, R):
        batch = self.build_batch(wrist_image=wrist_image, external_image=external_image, p=p, R=R)
        action = self.policy.select_action(batch)
        action = action[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
        if action.shape[0] < 7:
            raise ValueError(f"Expected action dim >= 7, got {action.shape[0]}")
        return action[:7]


class BoltNutPositionSmolVLAEnv(RobotEnv):
    """Online SmolVLA position-control deployment for action=[q1..q6, gripper]."""

    def __init__(
        self,
        *args,
        model="smolvla",
        policy_hz=20.0,
        contact_policy_hz=10.0,
        contact_force_threshold=1.0,
        reset_each_update=False,
        q_cmd_alpha=1.0,
        q_cmd_max_delta=0.01,
        q_cmd_startup_max_delta=0.002,
        q_cmd_startup_steps=1000,
        action_lowpass_alpha=1.0,
        joint_position_kp=(3000.0, 8000.0, 8000.0, 8000.0, 5000.0, 1000.0),
        joint_position_damping_scale=2.0,
        joint_position_tau_limit=8000.0,
        save_tensorboard=False,
        tensorboard_logdir="./gufic/tb_pose_smolvla",
        device=None,
        **kwargs,
    ):
        if _MUJOCO_IMPORT_ERROR is not None:
            raise RuntimeError("MuJoCo import failed; activate the correct environment.") from _MUJOCO_IMPORT_ERROR

        kwargs["model"] = model
        kwargs["use_learned_velocity_field"] = False
        kwargs["test_offline_cond"] = False
        kwargs["sim_mode"] = "smlovla_position_infer"
        kwargs["save_tensorboard"] = False
        super().__init__(*args, **kwargs)

        self.policy_hz = float(policy_hz)
        self.policy_decimation = max(1, int(round(1.0 / (self.policy_hz * self.dt))))
        self.contact_policy_hz = float(contact_policy_hz)
        self.contact_policy_decimation = max(1, int(round(1.0 / (self.contact_policy_hz * self.dt))))
        self.contact_force_threshold = float(contact_force_threshold)
        self.reset_each_update = bool(reset_each_update)
        self.q_cmd_alpha = float(q_cmd_alpha)
        self.q_cmd_max_delta = float(q_cmd_max_delta)
        self.q_cmd_startup_max_delta = float(q_cmd_startup_max_delta)
        self.q_cmd_startup_steps = int(q_cmd_startup_steps)
        self.action_lowpass_alpha = float(action_lowpass_alpha)
        self.joint_position_Kp = np.diag(np.asarray(joint_position_kp, dtype=np.float64).reshape(self.robot_state.N))
        joint_position_kd = np.sqrt(np.maximum(np.diag(self.joint_position_Kp), 0.0)) * float(joint_position_damping_scale)
        self.joint_position_Kd = np.diag(joint_position_kd)
        self.joint_position_tau_limit = float(joint_position_tau_limit)
        self.device = device
        self.contact = False
        self.save_pose_tensorboard = bool(save_tensorboard)
        self.tensorboard_logdir = str(tensorboard_logdir)
        self.tb_writer = None
        if self.save_pose_tensorboard:
            if SummaryWriter is None:
                raise ImportError("TensorBoard is not available. Install tensorboard or tensorboardX.")
            self.tb_writer = SummaryWriter(self.tensorboard_logdir)

        self.q_cmd = self.data.qpos.copy()[: self.robot_state.N].astype(np.float64)
        self.gripper_cmd = 0.03
        self.local_traj_t0 = 0.0
        self.local_traj_T = max(self.policy_decimation * self.dt, self.dt)
        self.local_p0 = None
        self.local_R0 = None
        self.local_p1 = None
        self.local_R1 = None
        self.local_rotvec = np.zeros(3, dtype=np.float64)
        self.latest_Fe = np.zeros(6, dtype=np.float64)
        self.gripper_cmd = 0.03
        self.last_action = None
        self.last_pred_pose_p = None
        self.last_pred_pose_R = None
        self.last_cmd_pose_p = None
        self.last_cmd_pose_R = None
        self.last_policy_contact = False
        self.last_cmd_update_iter = -1
        self.policy_segment_id = -1
        self.policy_segment_start_iter = 0
        self.next_segment_duration = self.policy_decimation * self.dt
        self._remove_parent_trajectory_callables()

        self.position_infer = SmolVLAJointPositionInfer(
            policy_path=self.smolvla_policy_path,
            dataset_repo_id=self.smolvla_dataset_repo_id,
            dataset_root=self.smolvla_dataset_root,
            language=self.smolvla_language,
            device=self.device,
            vlm_model_name=self.smolvla_vlm_model_name,
        )
        self.position_infer.reset()
        self.sync_position_command_to_current_q()

        print("[BoltNut-Position-SmVLA] policy_path:", self.position_infer.policy_path)
        print("[BoltNut-Position-SmVLA] dataset_root:", self.smolvla_dataset_root)
        print("[BoltNut-Position-SmVLA] dataset_repo_id:", self.smolvla_dataset_repo_id)
        print(
            "[BoltNut-Position-SmVLA] policy_hz:",
            self.policy_hz,
            "decimation:",
            self.policy_decimation,
            "interval_s:",
            self.policy_decimation * self.dt,
            "contact_policy_hz:",
            self.contact_policy_hz,
            "contact_decimation:",
            self.contact_policy_decimation,
            "contact_interval_s:",
            self.contact_policy_decimation * self.dt,
        )
        print("[BoltNut-Position-SmVLA] joint_position_Kp:", np.diag(self.joint_position_Kp))
        print("[BoltNut-Position-SmVLA] joint_position_Kd:", np.diag(self.joint_position_Kd))
        print("[BoltNut-Position-SmVLA] joint_position_tau_limit:", self.joint_position_tau_limit)
        if self.tb_writer is not None:
            print("[BoltNut-Position-SmVLA] tensorboard_logdir:", self.tensorboard_logdir)

    def load_xml(self):
        model_dir = Path(os.getcwd()) / "gufic_env" / "mujoco_models"
        if self.robot_name != "indy7":
            raise NotImplementedError(f"Unsupported robot_name: {self.robot_name}")
        if self.task == "bolt":
            model_path = model_dir / "Indy7_nutbolt_impedance.xml"
        else:
            model_path = model_dir / "Indy7_nutbolt_impedance.xml"

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

    def sync_position_command_to_current_q(self):
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.robot_state.update()
        self.q_cmd = self.data.qpos.copy()[: self.robot_state.N].astype(np.float64)
        p_now, R_now = self.robot_state.get_pose()
        self._set_local_pose_trajectory(
            p_now,
            R_now,
            p_now,
            R_now,
            duration=max(self.policy_decimation * self.dt, self.dt),
        )
        self.gd = np.eye(4)
        self.gd[:3, :3] = R_now
        self.gd[:3, 3] = p_now
        self.last_pred_pose_p = p_now.copy()
        self.last_pred_pose_R = R_now.copy()
        self.last_cmd_pose_p = p_now.copy()
        self.last_cmd_pose_R = R_now.copy()

    def _euler_deg(self, R):
        return RT.from_matrix(np.asarray(R, dtype=np.float64).reshape(3, 3)).as_euler("xyz", degrees=True)

    def _write_pose_tensorboard(self, p, R, Fe=None, control_mode="unknown"):
        if self.tb_writer is None:
            return

        step = int(self.iter)
        p = np.asarray(p, dtype=np.float64).reshape(3)
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        r = self._euler_deg(R)

        pred_p = p if self.last_pred_pose_p is None else np.asarray(self.last_pred_pose_p).reshape(3)
        pred_R = R if self.last_pred_pose_R is None else np.asarray(self.last_pred_pose_R).reshape(3, 3)
        pred_r = self._euler_deg(pred_R)

        cmd_p = p if self.last_cmd_pose_p is None else np.asarray(self.last_cmd_pose_p).reshape(3)
        cmd_R = R if self.last_cmd_pose_R is None else np.asarray(self.last_cmd_pose_R).reshape(3, 3)
        cmd_r = self._euler_deg(cmd_R)

        names = ("x", "y", "z")
        for i, name in enumerate(names):
            self.tb_writer.add_scalars(
                f"position/{name}",
                {
                    "actual": float(p[i]),
                    "pred_raw": float(pred_p[i]),
                    "cmd_smooth": float(cmd_p[i]),
                },
                step,
            )
            self.tb_writer.add_scalars(
                f"euler_deg/{name}",
                {
                    "actual": float(r[i]),
                    "pred_raw": float(pred_r[i]),
                    "cmd_smooth": float(cmd_r[i]),
                },
                step,
            )

        self.tb_writer.add_scalar("error/pred_position_norm", float(np.linalg.norm(p - pred_p)), step)
        self.tb_writer.add_scalar("error/cmd_position_norm", float(np.linalg.norm(p - cmd_p)), step)
        self.tb_writer.add_scalar(
            "error/pred_rotation_deg",
            float((RT.from_matrix(pred_R.T @ R).magnitude()) * 180.0 / np.pi),
            step,
        )
        self.tb_writer.add_scalar(
            "error/cmd_rotation_deg",
            float((RT.from_matrix(cmd_R.T @ R).magnitude()) * 180.0 / np.pi),
            step,
        )
        self.tb_writer.add_scalar("state/contact", float(self.contact), step)
        self.tb_writer.add_scalar("state/control_mode", 1.0 if control_mode == "gufic" else 0.0, step)
        self.tb_writer.add_scalar("state/last_cmd_update_iter", float(self.last_cmd_update_iter), step)

        if Fe is not None:
            Fe = np.asarray(Fe, dtype=np.float64).reshape(-1)
            force_names = ("fx", "fy", "fz", "tx", "ty", "tz")
            for i, name in enumerate(force_names[: min(len(Fe), len(force_names))]):
                self.tb_writer.add_scalar(f"force/{name}", float(Fe[i]), step)

        if step % 100 == 0:
            self.tb_writer.flush()

    def _write_segment_tracking_tensorboard(self, p, R, control_mode="unknown"):
        if self.tb_writer is None or not self.contact or self.local_p0 is None:
            return

        step = int(self.iter)
        segment_step = max(0, step - int(self.policy_segment_start_iter))
        t = step * self.dt
        pd, Rd, dpd, dRd, _, _ = self._local_traj_pose_derivatives(t)

        p = np.asarray(p, dtype=np.float64).reshape(3)
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        pd = np.asarray(pd, dtype=np.float64).reshape(3)
        Rd = np.asarray(Rd, dtype=np.float64).reshape(3, 3)
        dpd = np.asarray(dpd, dtype=np.float64).reshape(3)
        wd = vee_map(dRd @ Rd.T).reshape(3)

        actual_r = self._euler_deg(R)
        desired_r = self._euler_deg(Rd)
        position_error = p - pd
        rotation_error_deg = RT.from_matrix(Rd.T @ R).magnitude() * 180.0 / np.pi
        progress = np.clip((t - self.local_traj_t0) / self.local_traj_T, 0.0, 1.0)

        self.tb_writer.add_scalar("segment_tracking/meta/segment_id", float(self.policy_segment_id), step)
        self.tb_writer.add_scalar("segment_tracking/meta/segment_step", float(segment_step), step)
        self.tb_writer.add_scalar("segment_tracking/meta/progress", float(progress), step)
        self.tb_writer.add_scalar("segment_tracking/meta/control_mode", 1.0 if control_mode == "gufic" else 0.0, step)

        names = ("x", "y", "z")
        for i, name in enumerate(names):
            self.tb_writer.add_scalars(
                f"segment_tracking/position/{name}",
                {"actual": float(p[i]), "desired": float(pd[i]), "error": float(position_error[i])},
                step,
            )
            self.tb_writer.add_scalars(
                f"segment_tracking_by_step/position/{name}",
                {"actual": float(p[i]), "desired": float(pd[i]), "error": float(position_error[i])},
                segment_step,
            )
            self.tb_writer.add_scalars(
                f"segment_tracking/euler_deg/{name}",
                {"actual": float(actual_r[i]), "desired": float(desired_r[i])},
                step,
            )
            self.tb_writer.add_scalars(
                f"segment_tracking_by_step/euler_deg/{name}",
                {"actual": float(actual_r[i]), "desired": float(desired_r[i])},
                segment_step,
            )
            self.tb_writer.add_scalar(f"segment_tracking/velocity/{name}", float(dpd[i]), step)
            self.tb_writer.add_scalar(f"segment_tracking/angular_velocity/{name}", float(wd[i]), step)

        self.tb_writer.add_scalar("segment_tracking/error/position_norm", float(np.linalg.norm(position_error)), step)
        self.tb_writer.add_scalar("segment_tracking/error/rotation_deg", float(rotation_error_deg), step)
        self.tb_writer.add_scalar(
            "segment_tracking_by_step/error/position_norm",
            float(np.linalg.norm(position_error)),
            segment_step,
        )
        self.tb_writer.add_scalar(
            "segment_tracking_by_step/error/rotation_deg",
            float(rotation_error_deg),
            segment_step,
        )

    def resize_rgb(self, image):
        return resize_rgb(image)

    def _remove_parent_trajectory_callables(self):
        for name in ("pd_t", "Rd_t", "dpd_t", "dRd_t", "ddpd_t", "ddRd_t"):
            if name in self.__dict__:
                delattr(self, name)

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

    def qpos_to_pose(self, q_arm):
        q_arm = np.asarray(q_arm, dtype=np.float64).reshape(self.robot_state.N)
        qpos_bak = self.data.qpos.copy()
        qvel_bak = self.data.qvel.copy()
        ctrl_bak = self.data.ctrl.copy()

        self.data.qpos[: self.robot_state.N] = q_arm
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.robot_state.update()
        p_des, R_des = self.robot_state.get_pose()
        p_des = p_des.copy().reshape(3)
        R_des = R_des.copy().reshape(3, 3)

        self.data.qpos[:] = qpos_bak
        self.data.qvel[:] = qvel_bak
        self.data.ctrl[:] = ctrl_bak
        mujoco.mj_forward(self.model, self.data)
        self.robot_state.update()
        return p_des, R_des

    def _set_local_pose_trajectory(self, p0, R0, p1, R1, duration):
        self.local_traj_t0 = self.iter * self.dt
        self.local_traj_T = max(float(duration), self.dt)
        self.local_p0 = np.asarray(p0, dtype=np.float64).reshape(3)
        self.local_R0 = np.asarray(R0, dtype=np.float64).reshape(3, 3)
        self.local_p1 = np.asarray(p1, dtype=np.float64).reshape(3)
        self.local_R1 = np.asarray(R1, dtype=np.float64).reshape(3, 3)
        R_rel = self.local_R0.T @ self.local_R1
        self.local_rotvec = RT.from_matrix(R_rel).as_rotvec().astype(np.float64)

    def start_q_cmd_trajectory(self, q_old, q_new, duration=None):
        p0, R0 = self.qpos_to_pose(q_old)
        p1, R1 = self.qpos_to_pose(q_new)
        duration = self.policy_decimation * self.dt if duration is None else duration
        self._set_local_pose_trajectory(p0, R0, p1, R1, duration=duration)
        self.gd = np.eye(4)
        self.gd[:3, :3] = R0
        self.gd[:3, 3] = p0

    def _local_traj_scalars(self, t):
        if self.local_p0 is None:
            p, R = self.robot_state.get_pose()
            self._set_local_pose_trajectory(p, R, p, R, duration=max(self.policy_decimation * self.dt, self.dt))
        T = self.local_traj_T
        u = np.clip((float(t) - self.local_traj_t0) / T, 0.0, 1.0)
        u_dot = 1.0 / T

        s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        ds_du = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
        dds_du2 = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
        ds = ds_du * u_dot
        dds = dds_du2 * u_dot * u_dot
        return s, ds, dds

    def _local_traj_pose_derivatives(self, t):
        s, ds, dds = self._local_traj_scalars(t)
        dp = self.local_p1 - self.local_p0
        pd = self.local_p0 + s * dp
        dpd = ds * dp
        ddpd = dds * dp

        A = hat_map(self.local_rotvec)
        Rd = self.local_R0 @ expm(A * s)
        dRd = Rd @ (A * ds)
        ddRd = Rd @ (A @ A * ds * ds + A * dds)
        return pd, Rd, dpd, dRd, ddpd, ddRd

    def pd_t(self, t):
        return self._local_traj_pose_derivatives(t)[0].reshape(3, 1)

    def Rd_t(self, t):
        return self._local_traj_pose_derivatives(t)[1]

    def dpd_t(self, t):
        return self._local_traj_pose_derivatives(t)[2].reshape(3, 1)

    def dRd_t(self, t):
        return self._local_traj_pose_derivatives(t)[3]

    def ddpd_t(self, t):
        return self._local_traj_pose_derivatives(t)[4].reshape(3, 1)

    def ddRd_t(self, t):
        return self._local_traj_pose_derivatives(t)[5]

    def infer_joint_action(self, build_gufic_trajectory=False):
        p, R = self.robot_state.get_pose()
        wrist_image = self.resize_rgb(self.get_camera_rgb(self.cam_id))
        external_image = self.resize_rgb(self.get_external_rgb())
        if self.reset_each_update:
            self.position_infer.reset()
        action = self.position_infer.predict_action(
            wrist_image=wrist_image,
            external_image=external_image,
            p=p.astype(np.float32),
            R=R.astype(np.float32),
        )

        q_des = action[: self.robot_state.N].astype(np.float64)
        gripper = float(action[-1])
        if self.last_action is not None and self.action_lowpass_alpha < 1.0:
            alpha = np.clip(self.action_lowpass_alpha, 0.0, 1.0)
            q_des = (1.0 - alpha) * self.last_action[: self.robot_state.N] + alpha * q_des
            gripper = float((1.0 - alpha) * self.last_action[-1] + alpha * gripper)

        self.last_action = np.concatenate([q_des, np.array([gripper])]).astype(np.float64)
        q_old = self.q_cmd.copy()
        q_new = self._smooth_q_command(q_des)
        self.last_pred_pose_p, self.last_pred_pose_R = self.qpos_to_pose(q_des)
        self.last_cmd_pose_p, self.last_cmd_pose_R = self.qpos_to_pose(q_new)
        self.q_cmd = q_new
        if build_gufic_trajectory:
            self.start_q_cmd_trajectory(q_old, q_new, duration=self.next_segment_duration)
        self.gripper_cmd = gripper

        print_period = max(1, self.policy_decimation * 5)
        if self.iter % print_period == 0:
            q_now = self.data.qpos.copy()[: self.robot_state.N]
            print("[BoltNut-Position-SmVLA] p_now:", p.reshape(3))
            print("[BoltNut-Position-SmVLA] q_pred:", q_des)
            print("[BoltNut-Position-SmVLA] q_cmd:", self.q_cmd)
            print("[BoltNut-Position-SmVLA] gripper:", self.gripper_cmd)
            print("[BoltNut-Position-SmVLA] q_track_err:", np.linalg.norm(q_now - self.q_cmd))

    def joint_position_torque_control(self):
        q = self.data.qpos.copy()[: self.robot_state.N].reshape(-1)
        dq = self.data.qvel.copy()[: self.robot_state.N].reshape(-1)
        q_des = np.asarray(self.q_cmd, dtype=np.float64).reshape(self.robot_state.N)
        G = self.robot_state.get_bias_torque().reshape(-1)

        tau_cmd = self.joint_position_Kp @ (q_des - q) - self.joint_position_Kd @ dq + G
        if self.joint_position_tau_limit > 0.0:
            tau_cmd = np.clip(tau_cmd, -self.joint_position_tau_limit, self.joint_position_tau_limit)
        return tau_cmd.reshape(-1)

    def get_velocity_field(self, g, V, t):
        zeta_v = self.zeta_v
        zeta_w = self.zeta_w
        pd = self.pd_t(t).reshape((-1,))
        Rd = self.Rd_t(t)
        dpd = self.dpd_t(t).reshape((-1,))
        dRd = self.dRd_t(t)
        ddpd = self.ddpd_t(t).reshape((-1,))
        ddRd = self.ddRd_t(t)

        p = g[:3, 3]
        R = g[:3, :3]
        v = V[:3]
        w = V[3:]

        Vd_star = np.zeros(6,)
        vd_star = R.T @ dRd @ Rd.T @ (p - pd) + R.T @ dpd - zeta_v * R.T @ (p - pd)
        wd_star = vee_map(R.T @ dRd @ Rd.T @ R - zeta_w * (Rd.T @ R - R.T @ Rd)).reshape((-1,))
        Vd_star[:3] = vd_star
        Vd_star[3:] = wd_star

        term1 = -hat_map(w) @ R.T @ dRd @ Rd.T @ R + R.T @ ddRd @ Rd.T @ R + R.T @ dRd @ dRd.T @ R + R.T @ dRd @ Rd.T @ R @ hat_map(w)
        term2 = -hat_map(w) @ R.T @ dRd @ Rd.T @ (p - pd) + R.T @ ddRd @ Rd.T @ (p - pd) + R.T @ dRd @ dRd.T @ (p - pd) \
                + R.T @ dRd @ Rd.T @ (R.T @ v - pd) - hat_map(w) @ R.T @ dpd + R.T @ ddpd
        term3 = dRd.T @ R + Rd.T @ R @ hat_map(w) + hat_map(w) @ R.T @ Rd - R.T @ dRd
        term4 = -hat_map(w) @ R.T @ (p - pd) + v - R.T @ dpd
        dVd_star = np.zeros(6,)
        dVd_star[:3] = term2 - zeta_v * term4
        dVd_star[3:] = vee_map(term1 - zeta_w * term3).reshape((-1,))
        return Vd_star, dVd_star

    def step(self):
        self.robot_state.update()
        Fe_now = np.asarray(self.get_FT_value(), dtype=np.float32).reshape(-1)
        self.latest_Fe = Fe_now.astype(np.float64, copy=True)
        print("[BoltNut-Position-SmVLA] Fe:", Fe_now)
        if not self.contact:
            self.contact = abs(float(Fe_now[2])) > self.contact_force_threshold
        decimation = self.contact_policy_decimation if self.contact else self.policy_decimation
        contact_changed = self.contact != self.last_policy_contact
        steps_since_update = (
            decimation if self.last_cmd_update_iter < 0 else int(self.iter) - int(self.last_cmd_update_iter)
        )
        update_policy = contact_changed or self.last_cmd_update_iter < 0 or steps_since_update >= decimation

        if update_policy:
            if contact_changed:
                self.position_infer.reset()
            self.next_segment_duration = decimation * self.dt
            self.policy_segment_id += 1
            self.policy_segment_start_iter = int(self.iter)
            self.infer_joint_action(build_gufic_trajectory=self.contact)
            self.last_policy_contact = self.contact
            self.last_cmd_update_iter = self.iter

        if self.contact:
            control_mode = "gufic"
            tau_cmd = self.geometric_unified_force_impedance_control()
        else:
            control_mode = "joint_position"
            tau_cmd = self.joint_position_torque_control()
        self.robot_state.set_control_torque(tau_cmd, self.gripper_cmd)
        self.robot_state.update_dynamic()

        if self.iter % max(1, self.policy_decimation * 5) == 0:
            q_now = self.data.qpos.copy()[: self.robot_state.N]
            print("[BoltNut-Position-SmVLA] control_mode:", control_mode)
            print("[BoltNut-Position-SmVLA] q_track_err_after:", np.linalg.norm(q_now - self.q_cmd))
            print("[BoltNut-Position-SmVLA] tau_norm:", np.linalg.norm(tau_cmd))
            print("[BoltNut-Position-SmVLA] tau_max_abs:", np.max(np.abs(tau_cmd)))

        if self.show_viewer and self.iter % 10 == 0:
            self.viewer.sync()

        obs = {}
        p, R = self.robot_state.get_pose()
        Fe = self.get_FT_value()
        Fe_raw = self.get_FT_value_raw()
        self._write_pose_tensorboard(p, R, Fe=Fe, control_mode=control_mode)
        self._write_segment_tracking_tensorboard(p, R, control_mode=control_mode)
        for observable in self.observables:
            if observable == "p":
                obs[observable] = p.copy()
            elif observable == "pd":
                obs[observable] = self.pd_t(self.iter * self.dt).reshape((-1,)).copy() if self.contact else p.copy()
            elif observable == "R":
                obs[observable] = R.copy()
            elif observable == "Rd":
                obs[observable] = self.Rd_t(self.iter * self.dt).copy() if self.contact else R.copy()
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
        reward = 0.0
        info = {
            "contact": self.contact,
            "q_cmd": self.q_cmd.copy(),
            "gripper": self.gripper_cmd,
            "tau_cmd": tau_cmd.copy(),
            "control_mode": control_mode,
        }
        self.iter += 1
        return obs, reward, done, info

    def run(self):
        p_list, R_list, Fe_list, Fe_raw_list = [], [], [], []
        for i in range(self.max_iter):
            self.golbal_steps = i
            obs, reward, done, info = self.step()
            p, R = self.robot_state.get_pose()
            p_list.append(p)
            R_list.append(R)
            Fe_list.append(self.get_FT_value())
            Fe_raw_list.append(self.get_FT_value_raw())

            if i % 100 == 0:
                print(f"Time Step: {i}")
            if done:
                break
        return p_list, R_list, Fe_list, Fe_raw_list

DEFAULT_SMOLVLA_POLICY_PATHS = (
    "/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/"
    "checkpoints_smolvla_v_position/015000/pretrained_model"
)
def parse_args():
    parser = argparse.ArgumentParser(description="Online SmolVLA joint-position inference for nut-bolt.")
    parser.add_argument("--policy_path", default=DEFAULT_SMOLVLA_POLICY_PATHS)
    parser.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset_repo_id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--vlm_model_name", default=None)
    parser.add_argument("--language", default=TASK_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_time", type=float, default=52.0)
    parser.add_argument("--policy_hz", type=float, default=50.0)
    parser.add_argument("--contact_policy_hz", type=float, default=10.0)
    parser.add_argument("--contact_force_threshold", type=float, default=1.0)
    parser.add_argument("--reset_each_update", action="store_true")
    parser.add_argument("--action_lowpass_alpha", type=float, default=1.0)
    parser.add_argument("--q_cmd_alpha", type=float, default=1.0)
    parser.add_argument("--q_cmd_max_delta", type=float, default=0.01)
    parser.add_argument("--q_cmd_startup_max_delta", type=float, default=0.002)
    parser.add_argument("--q_cmd_startup_steps", type=int, default=1000)
    parser.add_argument(
        "--joint_position_kp",
        type=float,
        nargs=6,
        default=[3000.0, 8000.0, 8000.0, 8000.0, 5000.0, 1000.0],
    )
    parser.add_argument("--joint_position_damping_scale", type=float, default=2.0)
    parser.add_argument("--joint_position_tau_limit", type=float, default=8000.0)
    parser.add_argument("--save_tensorboard", default=True, action="store_true")
    parser.add_argument("--tensorboard_logdir", default="./gufic/tb_pose_smolvla")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed_start", action="store_true")
    parser.add_argument("--show_viewer", action="store_true", default=True)
    parser.add_argument("--no_viewer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    env = BoltNutPositionSmolVLAEnv(
        robot_name="indy7",
        model="smolvla",
        show_viewer=args.show_viewer and not args.no_viewer,
        max_time=args.max_time,
        fz=20,
        fix_camera=True,
        task="bolt",
        randomized_start=not args.fixed_start,
        inertia_shaping=False,
        use_learned_velocity_field=False,
        record_demos=False,
        seed=args.seed,
        save_tensorboard=args.save_tensorboard,
        tensorboard_logdir=args.tensorboard_logdir,
        visualize_delta_pose=False,
        smolvla_policy_path=args.policy_path,
        smolvla_dataset_repo_id=args.dataset_repo_id,
        smolvla_dataset_root=args.dataset_root,
        smolvla_vlm_model_name=args.vlm_model_name,
        smolvla_language=args.language,
        policy_hz=args.policy_hz,
        contact_policy_hz=args.contact_policy_hz,
        contact_force_threshold=args.contact_force_threshold,
        reset_each_update=args.reset_each_update,
        action_lowpass_alpha=args.action_lowpass_alpha,
        device=args.device,
        q_cmd_alpha=args.q_cmd_alpha,
        q_cmd_max_delta=args.q_cmd_max_delta,
        q_cmd_startup_max_delta=args.q_cmd_startup_max_delta,
        q_cmd_startup_steps=args.q_cmd_startup_steps,
        joint_position_kp=args.joint_position_kp,
        joint_position_damping_scale=args.joint_position_damping_scale,
        joint_position_tau_limit=args.joint_position_tau_limit,
    )

    try:
        env.run()
    finally:
        if env.viewer is not None:
            env.viewer.close()
        if getattr(env, "tb_writer", None) is not None:
            env.tb_writer.flush()
            env.tb_writer.close()


if __name__ == "__main__":
    main()
