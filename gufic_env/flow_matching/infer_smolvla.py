import argparse
import copy
import os
import types
from pathlib import Path

import numpy as np
import torch

try:
    from .infer_pi0 import (
        get_velocity_field,
        image_to_uint8_hwc,
        make_robot_state,
        print_result,
        rgb_to_tensor,
        rot6d_to_rotmat_np,
        save_comparison_plots,
        to_numpy,
        unpack_pi0_action,
    )
except ImportError:
    from infer_pi0 import (
        get_velocity_field,
        image_to_uint8_hwc,
        make_robot_state,
        print_result,
        rgb_to_tensor,
        rot6d_to_rotmat_np,
        save_comparison_plots,
        to_numpy,
        unpack_pi0_action,
    )


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


DEFAULT_SMOLVLA_POLICY_PATHS = (
    "/root/vla/gufic_flow_matching/gufic_env/flow_matching/"
    "checkpoints_smolvla_boltnut/checkpoints/last/pretrained_model",
)
DEFAULT_SMOLVLM_PATHS = (
    "/root/autodl-tmp/hub/"
    "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467",
)


def first_existing_path(paths):
    for path in paths:
        if path and Path(path).exists():
            return str(path)
    return None


def resolve_policy_path(policy_path):
    if policy_path is not None:
        return str(policy_path)
    env_path = os.environ.get("SMOLVLA_POLICY_PATH")
    if env_path:
        return env_path
    path = first_existing_path(DEFAULT_SMOLVLA_POLICY_PATHS)
    if path:
        return path
    raise FileNotFoundError(
        "No SmolVLA policy checkpoint found. Pass --policy_path or set SMOLVLA_POLICY_PATH.\n"
        "Expected one of:\n"
        + "\n".join(f"  - {p}" for p in DEFAULT_SMOLVLA_POLICY_PATHS)
    )


def resolve_vlm_model_name(vlm_model_name=None):
    if vlm_model_name and Path(str(vlm_model_name)).exists():
        return str(vlm_model_name)
    env_path = os.environ.get("SMOLVLA_VLM_MODEL_NAME")
    if env_path:
        if not Path(env_path).exists():
            raise FileNotFoundError(
                "SMOLVLA_VLM_MODEL_NAME points to a missing path:\n"
                f"{env_path}\n"
                "It must be a local SmolVLM snapshot containing processor_config.json."
            )
        return env_path
    path = first_existing_path(DEFAULT_SMOLVLM_PATHS)
    if path:
        return path
    raise FileNotFoundError(
        "SmolVLA inference needs SmolVLM locally, but no local snapshot was found.\n"
        "Pass --vlm_model_name or set SMOLVLA_VLM_MODEL_NAME.\n"
        "Expected one of:\n"
        + "\n".join(f"  - {p}" for p in DEFAULT_SMOLVLM_PATHS)
    )


def patch_action_features_for_mode(features, stats, action_mode):
    action_mode = action_mode.lower()
    if action_mode in ("full", "pose_vd", "pose_vd_body", "with_vd_body"):
        return features, stats
    if action_mode not in ("pose", "pose_only", "no_vd_body", "no_vd"):
        raise ValueError(f"Unknown SmolVLA action mode {action_mode!r}. Use 'pose' or 'full'.")

    features = copy.deepcopy(features)
    stats = copy.deepcopy(stats)
    features["action"]["shape"] = (9,)
    features["action"]["names"] = ["pd_Rd6d"]

    if "action" in stats:
        for stat_name, value in list(stats["action"].items()):
            if isinstance(value, torch.Tensor):
                stats["action"][stat_name] = value[:9].clone()
            else:
                stats["action"][stat_name] = value[:9].copy()
    return features, stats


class SmolVLAVelocityFieldInfer:
    """
    Use a fine-tuned SmolVLA policy to predict:
        pose mode: [pd, Rd6d]
        full mode: [pd, Rd6d, Vd_body]
    then compute the GUFIC velocity field Vd_star.
    """

    def __init__(
        self,
        policy_path=None,
        dataset_repo_id="gufic_boltnut_pi0",
        dataset_root="/media/zhou/Elements SE/VLA/boltnut3_pi0_lerobot_random_start",
        language="insert the bolt into the hole",
        device=None,
        zeta_v=50.0,
        zeta_w=10.0,
        action_mode="pose",
        vlm_model_name=None,
    ):
        self.policy_path = Path(resolve_policy_path(policy_path))
        self.dataset_repo_id = dataset_repo_id
        self.dataset_root = Path(dataset_root)
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.zeta_v = float(zeta_v)
        self.zeta_w = float(zeta_w)
        self.action_mode = action_mode.lower()
        self.vlm_model_name = resolve_vlm_model_name(vlm_model_name)

        self.policy, self.state_key, self.force_key = self._load_policy()
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
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.configs.types import FeatureType

        metadata = LeRobotDatasetMetadata(self.dataset_repo_id, root=self.dataset_root)
        features, stats = patch_action_features_for_mode(
            metadata.features,
            metadata.stats,
            self.action_mode,
        )

        config = PreTrainedConfig.from_pretrained(str(self.policy_path))
        config.vlm_model_name = self.vlm_model_name
        config.device = self.device

        policy_features = dataset_to_policy_features(features)
        config.input_features = {
            key: ft for key, ft in policy_features.items() if ft.type is not FeatureType.ACTION
        }
        config.output_features = {
            key: ft for key, ft in policy_features.items() if ft.type is FeatureType.ACTION
        }

        policy = SmolVLAPolicy.from_pretrained(
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
            raise ValueError("SmolVLA policy has no STATE input feature.")

        if OBS_STATE in state_keys:
            state_key = OBS_STATE
        elif "observation.robot_state" in state_keys:
            state_key = "observation.robot_state"
        else:
            state_key = state_keys[0]

        force_key = "observation.force" if "observation.force" in state_keys else None

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

        result["Vd_star"] = get_velocity_field(
            g=g,
            pd=result["pd"],
            Rd=result["Rd"],
            dpd=result["dpd"],
            dRd=result["dRd"],
            zeta_v=self.zeta_v,
            zeta_w=self.zeta_w,
        )
        return result


def load_smolvla_velocity_field_infer(
    policy_path=None,
    dataset_repo_id="gufic_boltnut_pi0",
    dataset_root="/media/zhou/Elements SE/VLA/boltnut3_pi0_lerobot_random_start",
    language="insert the bolt into the hole",
    device=None,
    zeta_v=50.0,
    zeta_w=10.0,
    action_mode="pose",
    vlm_model_name=None,
):
    return SmolVLAVelocityFieldInfer(
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        language=language,
        device=device,
        zeta_v=zeta_v,
        zeta_w=zeta_w,
        action_mode=action_mode,
        vlm_model_name=vlm_model_name,
    )


def sample_to_smolvla_inputs(sample):
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


def run_one_dataset_frame(
    policy_path=None,
    dataset_repo_id="gufic_boltnut_pi0",
    dataset_root="/media/zhou/Elements SE/VLA/boltnut3_pi0_lerobot_random_start",
    frame_index=0,
    language="insert the bolt into the hole",
    device=None,
    zeta_v=50.0,
    zeta_w=10.0,
    save_npz=None,
    action_mode="pose",
    vlm_model_name=None,
):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)
    sample = dataset[int(frame_index)]
    wrist_image, external_image, p, R, Fe = sample_to_smolvla_inputs(sample)

    infer = load_smolvla_velocity_field_infer(
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        language=language,
        device=device,
        zeta_v=zeta_v,
        zeta_w=zeta_w,
        action_mode=action_mode,
        vlm_model_name=vlm_model_name,
    )
    result = infer.predict_velocity_field(
        wrist_image=wrist_image,
        external_image=external_image,
        p=p,
        R=R,
        Fe=Fe,
    )

    print(f"frame_index: {frame_index}")
    print("policy_path:", infer.policy_path)
    print("vlm_model_name:", infer.vlm_model_name)
    print("p_now:", p)
    print("Fe:", Fe)
    print_result("smolvla_pred", result)

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


def run_dataset_comparison(
    policy_path=None,
    dataset_repo_id="gufic_boltnut_pi0",
    dataset_root="/media/zhou/Elements SE/VLA/boltnut3_pi0_lerobot_random_start",
    out_dir="./infer_smolvla_compare",
    start_index=0,
    max_frames=1000,
    stride=1,
    language="insert the bolt into the hole",
    device=None,
    zeta_v=50.0,
    zeta_w=10.0,
    action_mode="pose",
    vlm_model_name=None,
    reset_each_frame=True,
):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)
    infer = load_smolvla_velocity_field_infer(
        policy_path=policy_path,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        language=language,
        device=device,
        zeta_v=zeta_v,
        zeta_w=zeta_w,
        action_mode=action_mode,
        vlm_model_name=vlm_model_name,
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

        wrist_image, external_image, p, R, Fe = sample_to_smolvla_inputs(sample)
        if reset_each_frame:
            infer.reset()

        pred = infer.predict_velocity_field(
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SmolVLA inference for pd, Rd, dpd, dRd and GUFIC velocity field."
    )
    parser.add_argument(
        "--policy_path",
        default="/root/autodl-tmp/checkpoints_smolvla_v2/checkpoints/005000/pretrained_model",
        help="Path to SmolVLA pretrained_model directory. Defaults to SMOLVLA_POLICY_PATH or local checkpoints.",
    )
    parser.add_argument(
        "--vlm_model_name",
        default=None,
        help="Local SmolVLM snapshot path. Defaults to SMOLVLA_VLM_MODEL_NAME or known local cache paths.",
    )
    parser.add_argument(
        "--dataset_root",
        default="/root/autodl-tmp/boltnut_pi0_lerobot_20HZ",
        help="LeRobot dataset root used for metadata/stats and optional frame test.",
    )
    parser.add_argument("--dataset_repo_id", default="gufic_boltnut_smolvla")
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--language", default="insert the bolt into the hole")
    parser.add_argument("--device", default=None)
    parser.add_argument("--zeta_v", type=float, default=50.0)
    parser.add_argument("--zeta_w", type=float, default=10.0)
    parser.add_argument(
        "--action_mode",
        default="full",
        choices=["pose", "full"],
        help="Use 'pose' for [pd,Rd6d] checkpoints, or 'full' for [pd,Rd6d,Vd_body].",
    )
    parser.add_argument("--save_npz", default=None)
    parser.add_argument(
        "--compare",
        default=True,
    )
    parser.add_argument("--out_dir", default="./infer_smolvla_compare")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=360)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--keep_action_queue",
        action="store_true",
        help="Keep SmolVLA action queue across frames. Default resets every frame for fair one-step pred/gt plots.",
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
            vlm_model_name=args.vlm_model_name,
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
            vlm_model_name=args.vlm_model_name,
        )


if __name__ == "__main__":
    main()
