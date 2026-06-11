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
from scipy.spatial.transform import Rotation as RT

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


def vee_map(R):
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    v3 = -R[0, 1]
    v1 = -R[1, 2]
    v2 = R[0, 2]
    return np.array([v1, v2, v3], dtype=np.float64).reshape(3, 1)


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
    """Online SmolVLA joint-position policy with task-space impedance execution."""

    def __init__(
        self,
        *args,
        policy_hz=20.0,
        contact_policy_hz=50.0,
        contact_force_threshold=1.0,
        reset_each_update=False,
        q_cmd_alpha=1.0,
        q_cmd_max_delta=0.01,
        q_cmd_startup_max_delta=0.002,
        q_cmd_startup_steps=1000,
        action_lowpass_alpha=1.0,
        free_impedance_kp=(2500.0, 2500.0, 1200.0),
        free_impedance_kr=(1000.0, 1000.0, 1000.0),
        free_impedance_damping_scale=8.0,
        free_tau_limit=1000.0,
        contact_impedance_kp=(500.0, 500.0, 80.0),
        contact_impedance_kr=(250.0, 250.0, 250.0),
        contact_impedance_damping_scale=6.0,
        contact_tau_limit=300.0,
        free_control_mode="joint_pd",
        free_joint_kp=(3000.0, 8000.0, 8000.0, 8000.0, 5000.0, 1000.0),
        free_joint_damping_scale=2.0,
        free_joint_tau_limit=8000.0,
        use_contact_force_tracking=True,
        contact_force_target=(0.0, 0.0, -10.0, 0.0, 0.0, 0.0),
        contact_force_kp=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        contact_force_ki=(0.0, 0.0, 4.0, 0.0, 0.0, 0.0),
        contact_force_kd=(0.0, 0.0, 0.02, 0.0, 0.0, 0.0),
        contact_force_int_limit=20.0,
        contact_force_output_limit=80.0,
        contact_force_sign=1.0,
        device=None,
        **kwargs,
    ):
        if _MUJOCO_IMPORT_ERROR is not None:
            raise RuntimeError("MuJoCo import failed; activate the correct environment.") from _MUJOCO_IMPORT_ERROR

        kwargs["model"] = "mlp"
        kwargs["use_learned_velocity_field"] = False
        kwargs["test_offline_cond"] = False
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
        self.free_impedance_Kp = np.diag(np.asarray(free_impedance_kp, dtype=np.float64).reshape(3))
        self.free_impedance_KR = np.diag(np.asarray(free_impedance_kr, dtype=np.float64).reshape(3))
        self.free_impedance_damping_scale = float(free_impedance_damping_scale)
        self.free_tau_limit = float(free_tau_limit)
        self.contact_impedance_Kp = np.diag(np.asarray(contact_impedance_kp, dtype=np.float64).reshape(3))
        self.contact_impedance_KR = np.diag(np.asarray(contact_impedance_kr, dtype=np.float64).reshape(3))
        self.contact_impedance_damping_scale = float(contact_impedance_damping_scale)
        self.contact_tau_limit = float(contact_tau_limit)
        if free_control_mode not in ("joint_pd", "task_impedance"):
            raise ValueError(f"Unsupported free_control_mode: {free_control_mode}")
        self.free_control_mode = free_control_mode
        self.free_joint_Kp = np.diag(np.asarray(free_joint_kp, dtype=np.float64).reshape(self.robot_state.N))
        free_joint_kd = np.sqrt(np.maximum(np.diag(self.free_joint_Kp), 0.0)) * float(free_joint_damping_scale)
        self.free_joint_Kd = np.diag(free_joint_kd)
        self.free_joint_tau_limit = float(free_joint_tau_limit)
        self.use_contact_force_tracking = bool(use_contact_force_tracking)
        self.contact_force_target = np.asarray(contact_force_target, dtype=np.float64).reshape(6, 1)
        self.contact_force_Kp = np.diag(np.asarray(contact_force_kp, dtype=np.float64).reshape(6))
        self.contact_force_Ki = np.diag(np.asarray(contact_force_ki, dtype=np.float64).reshape(6))
        self.contact_force_Kd = np.diag(np.asarray(contact_force_kd, dtype=np.float64).reshape(6))
        self.contact_force_int_limit = float(contact_force_int_limit)
        self.contact_force_output_limit = float(contact_force_output_limit)
        self.contact_force_sign = float(contact_force_sign)
        self.contact_force_int = np.zeros((6, 1), dtype=np.float64)
        self.last_contact_force_error = np.zeros((6, 1), dtype=np.float64)
        self.last_contact_force_tracking = np.zeros((6, 1), dtype=np.float64)
        self.prev_contact = False
        self.device = device

        self.q_cmd = self.data.qpos.copy()[: self.robot_state.N].astype(np.float64)
        self.q_target = self.q_cmd.copy()
        self.gripper_cmd = 0.03
        self.last_action = None
        self.last_policy_contact = False
        self.last_cmd_update_iter = -1
        self.pd_cmd = None
        self.Rd_cmd = None

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
            "contact_policy_hz:",
            self.contact_policy_hz,
            "contact_decimation:",
            self.contact_policy_decimation,
        )
        print("[BoltNut-Position-SmVLA] free_impedance_Kp:", np.diag(self.free_impedance_Kp))
        print("[BoltNut-Position-SmVLA] free_impedance_KR:", np.diag(self.free_impedance_KR))
        print("[BoltNut-Position-SmVLA] free_damping_scale:", self.free_impedance_damping_scale)
        print("[BoltNut-Position-SmVLA] free_tau_limit:", self.free_tau_limit)
        print("[BoltNut-Position-SmVLA] contact_impedance_Kp:", np.diag(self.contact_impedance_Kp))
        print("[BoltNut-Position-SmVLA] contact_impedance_KR:", np.diag(self.contact_impedance_KR))
        print("[BoltNut-Position-SmVLA] contact_damping_scale:", self.contact_impedance_damping_scale)
        print("[BoltNut-Position-SmVLA] contact_tau_limit:", self.contact_tau_limit)
        print("[BoltNut-Position-SmVLA] free_control_mode:", self.free_control_mode)
        print("[BoltNut-Position-SmVLA] free_joint_Kp:", np.diag(self.free_joint_Kp))
        print("[BoltNut-Position-SmVLA] free_joint_Kd:", np.diag(self.free_joint_Kd))
        print("[BoltNut-Position-SmVLA] free_joint_tau_limit:", self.free_joint_tau_limit)
        print("[BoltNut-Position-SmVLA] use_contact_force_tracking:", self.use_contact_force_tracking)
        print("[BoltNut-Position-SmVLA] contact_force_target:", self.contact_force_target.reshape(-1))
        print("[BoltNut-Position-SmVLA] contact_force_Kp:", np.diag(self.contact_force_Kp))
        print("[BoltNut-Position-SmVLA] contact_force_Ki:", np.diag(self.contact_force_Ki))
        print("[BoltNut-Position-SmVLA] contact_force_Kd:", np.diag(self.contact_force_Kd))
        print("[BoltNut-Position-SmVLA] contact_force_output_limit:", self.contact_force_output_limit)
        print("[BoltNut-Position-SmVLA] contact_force_sign:", self.contact_force_sign)

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
        self.q_target = self.q_cmd.copy()
        self.pd_cmd, self.Rd_cmd = self.qpos_to_pose(self.q_cmd)
        self.robot_state.set_control_torque(np.zeros(self.robot_state.N), self.gripper_cmd)

    def resize_rgb(self, image):
        return resize_rgb(image)

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

    def infer_joint_action(self):
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
        self.q_target = q_des.copy()
        self.gripper_cmd = gripper

        print_period = max(1, self.policy_decimation * 5)
        if self.iter % print_period == 0:
            q_now = self.data.qpos.copy()[: self.robot_state.N]
            p_now, R_now = self.robot_state.get_pose()
            print("[BoltNut-Position-SmVLA] p_now:", p.reshape(3))
            print("[BoltNut-Position-SmVLA] q_pred:", q_des)
            print("[BoltNut-Position-SmVLA] q_target:", self.q_target)
            print("[BoltNut-Position-SmVLA] q_cmd:", self.q_cmd)
            print("[BoltNut-Position-SmVLA] q_target_cmd_gap:", np.linalg.norm(self.q_target - self.q_cmd))
            print("[BoltNut-Position-SmVLA] pd_cmd:", self.pd_cmd)
            print("[BoltNut-Position-SmVLA] p_err_to_pd_cmd:", np.linalg.norm(p_now - self.pd_cmd))
            print("[BoltNut-Position-SmVLA] gripper:", self.gripper_cmd)
            print("[BoltNut-Position-SmVLA] q_track_err:", np.linalg.norm(q_now - self.q_cmd))

    def _select_impedance_gains(self, contact):
        if contact:
            return (
                self.contact_impedance_Kp,
                self.contact_impedance_KR,
                self.contact_impedance_damping_scale,
                self.contact_tau_limit,
            )
        return (
            self.free_impedance_Kp,
            self.free_impedance_KR,
            self.free_impedance_damping_scale,
            self.free_tau_limit,
        )

    def _contact_force_tracking_wrench(self, Fe, dFe=None):
        Fe = np.asarray(Fe, dtype=np.float64).reshape(6, 1)
        dFe = np.zeros((6, 1), dtype=np.float64) if dFe is None else np.asarray(dFe, dtype=np.float64).reshape(6, 1)

        Fd_star = self.contact_force_target
        force_error = -Fe - Fd_star
        de_force = -dFe
        self.last_contact_force_error = force_error.copy()
        self.contact_force_int = self.contact_force_int + force_error * self.dt
        if self.contact_force_int_limit > 0.0:
            self.contact_force_int = np.clip(
                self.contact_force_int,
                -self.contact_force_int_limit,
                self.contact_force_int_limit,
            )

        F_force = (
            -self.contact_force_Kp @ (-Fe)
            - self.contact_force_Ki @ self.contact_force_int
            - self.contact_force_Kd @ de_force
            + Fd_star
        )
        F_force = self.contact_force_sign * F_force
        if self.contact_force_output_limit > 0.0:
            F_force = np.clip(
                F_force,
                -self.contact_force_output_limit,
                self.contact_force_output_limit,
            )
        self.last_contact_force_tracking = F_force.copy()
        return F_force

    def impedance_control(self, contact=False, Fe=None, dFe=None):
        if self.pd_cmd is None or self.Rd_cmd is None:
            self.pd_cmd, self.Rd_cmd = self.qpos_to_pose(self.q_cmd)

        Kp, KR, damping_scale, tau_limit = self._select_impedance_gains(contact)
        Jb = self.robot_state.get_body_jacobian()
        G = self.robot_state.get_bias_torque().reshape(-1, 1)
        p, R = self.robot_state.get_pose()
        pd = np.asarray(self.pd_cmd, dtype=np.float64).reshape(3)
        Rd = np.asarray(self.Rd_cmd, dtype=np.float64).reshape(3, 3)

        fp = R.T @ Rd @ Kp @ Rd.T @ (p - pd).reshape(3, 1)
        fR = vee_map(KR @ Rd.T @ R - R.T @ Rd @ KR)
        fg = np.vstack((fp, fR))

        eV = self.robot_state.get_body_ee_velocity()
        gains = np.concatenate(
            [np.diag(Kp), np.diag(KR)],
            axis=0,
        )
        Kd = np.diag(np.sqrt(np.maximum(gains, 0.0))) * damping_scale

        tau_tilde = -fg - Kd @ eV

        F_force = np.zeros((6, 1), dtype=np.float64)
        if contact and self.use_contact_force_tracking and Fe is not None:
            F_force = self._contact_force_tracking_wrench(Fe, dFe=dFe)
            tau_tilde = tau_tilde + F_force
        tau_cmd = Jb.T @ tau_tilde + G
        tau_cmd = tau_cmd.reshape(-1)
        if tau_limit > 0.0:
            tau_cmd = np.clip(tau_cmd, -tau_limit, tau_limit)
        return tau_cmd

    def joint_position_torque_control(self):
        q = self.data.qpos.copy()[: self.robot_state.N].reshape(-1)
        dq = self.data.qvel.copy()[: self.robot_state.N].reshape(-1)
        q_des = np.asarray(self.q_cmd, dtype=np.float64).reshape(self.robot_state.N)
        G = self.robot_state.get_bias_torque().reshape(-1)

        tau_cmd = self.free_joint_Kp @ (q_des - q) - self.free_joint_Kd @ dq + G
        if self.free_joint_tau_limit > 0.0:
            tau_cmd = np.clip(tau_cmd, -self.free_joint_tau_limit, self.free_joint_tau_limit)
        return tau_cmd.reshape(-1)

    def step(self):
        self.robot_state.update()

        Fe_now, dFe_now = self.get_FT_value(return_derivative=True)
        Fe_now = np.asarray(Fe_now, dtype=np.float32).reshape(-1)
        dFe_now = np.asarray(dFe_now, dtype=np.float32).reshape(-1)
        print("[BoltNut-Position-SmVLA] Fe:", Fe_now)
        contact = abs(float(Fe_now[2])) > self.contact_force_threshold
        decimation = self.contact_policy_decimation if contact else self.policy_decimation
        contact_changed = contact != self.last_policy_contact
        update_policy = contact_changed or self.iter % decimation == 0
        if contact_changed and contact:
            self.contact_force_int[:] = 0.0
        elif not contact:
            self.contact_force_int[:] = 0.0

        if update_policy:
            if contact_changed:
                self.position_infer.reset()
            self.infer_joint_action()
            self.last_policy_contact = contact
            self.last_cmd_update_iter = self.iter

        self.q_cmd = self._smooth_q_command(self.q_target)
        self.pd_cmd, self.Rd_cmd = self.qpos_to_pose(self.q_cmd)

        # control_mode = "task_impedance" if contact or self.free_control_mode == "task_impedance" else "joint_pd"
        control_mode = "joint_pd"
        if control_mode == "task_impedance":
            tau_cmd = self.impedance_control(contact=contact, Fe=Fe_now, dFe=dFe_now)
        else:
            tau_cmd = self.joint_position_torque_control()
        self.robot_state.set_control_torque(tau_cmd, self.gripper_cmd)
        self.robot_state.update_dynamic()

        if self.iter % max(1, self.policy_decimation * 5) == 0:
            q_now = self.data.qpos.copy()[: self.robot_state.N]
            print("[BoltNut-Position-SmVLA] control_mode:", control_mode)
            print("[BoltNut-Position-SmVLA] q_target_cmd_gap:", np.linalg.norm(self.q_target - self.q_cmd))
            print("[BoltNut-Position-SmVLA] q_track_err_after:", np.linalg.norm(q_now - self.q_cmd))
            print("[BoltNut-Position-SmVLA] tau_norm:", np.linalg.norm(tau_cmd))
            print("[BoltNut-Position-SmVLA] tau_max_abs:", np.max(np.abs(tau_cmd)))
            if contact and self.use_contact_force_tracking:
                print("[BoltNut-Position-SmVLA] force_target:", self.contact_force_target.reshape(6))
                print("[BoltNut-Position-SmVLA] force_error:", self.last_contact_force_error.reshape(6))
                print("[BoltNut-Position-SmVLA] F_force_tracking:", self.last_contact_force_tracking.reshape(6))

        if self.show_viewer and self.iter % 10 == 0:
            self.viewer.sync()

        obs = {}
        p, R = self.robot_state.get_pose()
        Fe = self.get_FT_value()
        Fe_raw = self.get_FT_value_raw()
        for observable in self.observables:
            if observable == "p":
                obs[observable] = p.copy()
            elif observable == "pd":
                obs[observable] = self.pd_cmd.copy() if self.pd_cmd is not None else p.copy()
            elif observable == "R":
                obs[observable] = R.copy()
            elif observable == "Rd":
                obs[observable] = self.Rd_cmd.copy() if self.Rd_cmd is not None else R.copy()
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
            "contact": contact,
            "q_cmd": self.q_cmd.copy(),
            "pd_cmd": None if self.pd_cmd is None else self.pd_cmd.copy(),
            "Rd_cmd": None if self.Rd_cmd is None else self.Rd_cmd.copy(),
            "tau_cmd": tau_cmd.copy(),
            "control_mode": control_mode,
            "gripper": self.gripper_cmd,
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
    "checkpoints_smolvla_v_position/020000/pretrained_model"
)
def parse_args():
    parser = argparse.ArgumentParser(description="Online SmolVLA joint-position inference for nut-bolt.")
    parser.add_argument("--policy_path", default=DEFAULT_SMOLVLA_POLICY_PATHS)
    parser.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset_repo_id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--vlm_model_name", default=None)
    parser.add_argument("--language", default=TASK_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_time", type=float, default=12.0)
    parser.add_argument("--policy_hz", type=float, default=50.0)
    parser.add_argument("--contact_policy_hz", type=float, default=50.0)
    parser.add_argument("--contact_force_threshold", type=float, default=0.5)
    parser.add_argument("--reset_each_update", action="store_true")
    parser.add_argument("--action_lowpass_alpha", type=float, default=1.0)
    parser.add_argument("--q_cmd_alpha", type=float, default=1.0)
    parser.add_argument("--q_cmd_max_delta", type=float, default=0.001)
    parser.add_argument("--q_cmd_startup_max_delta", type=float, default=0.002)
    parser.add_argument("--q_cmd_startup_steps", type=int, default=1000)
    parser.add_argument("--free_impedance_kp", type=float, nargs=3, default=[5500.0, 5500.0, 5000.0])
    parser.add_argument("--free_impedance_kr", type=float, nargs=3, default=[1000.0, 1000.0, 1000.0])
    parser.add_argument("--free_impedance_damping_scale", type=float, default=8.0)
    parser.add_argument("--free_tau_limit", type=float, default=1000.0)
    parser.add_argument("--contact_impedance_kp", type=float, nargs=3, default=[500.0, 500.0, 80.0])
    parser.add_argument("--contact_impedance_kr", type=float, nargs=3, default=[250.0, 250.0, 250.0])
    parser.add_argument("--contact_impedance_damping_scale", type=float, default=6.0)
    parser.add_argument("--contact_tau_limit", type=float, default=300.0)
    parser.add_argument("--free_control_mode", choices=["joint_pd", "task_impedance"], default="joint_pd")
    parser.add_argument(
        "--free_joint_kp",
        type=float,
        nargs=6,
        default=[3000.0, 8000.0, 8000.0, 8000.0, 5000.0, 1000.0],
    )
    parser.add_argument("--free_joint_damping_scale", type=float, default=2.0)
    parser.add_argument("--free_joint_tau_limit", type=float, default=8000.0)
    parser.add_argument("--enable_contact_external_force", action="store_true")
    parser.add_argument("--disable_contact_force_tracking", action="store_true")
    parser.add_argument(
        "--contact_force_target",
        type=float,
        nargs=6,
        default=[0.0, 0.0, -10.0, 0.0, 0.0, 0.0],
    )
    parser.add_argument(
        "--contact_force_kp",
        type=float,
        nargs=6,
        default=[0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
    )
    parser.add_argument(
        "--contact_force_ki",
        type=float,
        nargs=6,
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    parser.add_argument(
        "--contact_force_kd",
        type=float,
        nargs=6,
        default=[0.0, 0.0, 0.02, 0.0, 0.0, 0.0],
    )
    parser.add_argument("--contact_force_int_limit", type=float, default=20.0)
    parser.add_argument("--contact_force_output_limit", type=float, default=80.0)
    parser.add_argument("--contact_force_sign", type=float, default=1.0)
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
        model="mlp",
        show_viewer=args.show_viewer and not args.no_viewer,
        max_time=args.max_time,
        fz=2,
        fix_camera=True,
        task="bolt",
        randomized_start=not args.fixed_start,
        inertia_shaping=False,
        use_learned_velocity_field=False,
        record_demos=False,
        seed=args.seed,
        save_tensorboard=False,
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
        free_impedance_kp=args.free_impedance_kp,
        free_impedance_kr=args.free_impedance_kr,
        free_impedance_damping_scale=args.free_impedance_damping_scale,
        free_tau_limit=args.free_tau_limit,
        contact_impedance_kp=args.contact_impedance_kp,
        contact_impedance_kr=args.contact_impedance_kr,
        contact_impedance_damping_scale=args.contact_impedance_damping_scale,
        contact_tau_limit=args.contact_tau_limit,
        free_control_mode=args.free_control_mode,
        free_joint_kp=args.free_joint_kp,
        free_joint_damping_scale=args.free_joint_damping_scale,
        free_joint_tau_limit=args.free_joint_tau_limit,
        use_contact_force_tracking=not args.disable_contact_force_tracking,
        contact_force_target=args.contact_force_target,
        contact_force_kp=args.contact_force_kp,
        contact_force_ki=args.contact_force_ki,
        contact_force_kd=args.contact_force_kd,
        contact_force_int_limit=args.contact_force_int_limit,
        contact_force_output_limit=args.contact_force_output_limit,
        contact_force_sign=args.contact_force_sign,
    )

    try:
        env.run()
    finally:
        if env.viewer is not None:
            env.viewer.close()


if __name__ == "__main__":
    main()
