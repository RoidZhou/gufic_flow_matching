#!/usr/bin/env python
import logging
import copy
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy

try:
    from lerobot.common.optim.factory import make_optimizer_and_scheduler as lerobot_make_optimizer_and_scheduler
except ModuleNotFoundError:
    lerobot_make_optimizer_and_scheduler = None


DEFAULT_CONFIG_PATH = Path(__file__).with_name("pi0_boltnut.yaml")
DEFAULT_SMOLVLA_CONFIG_PATH = Path(__file__).with_name("smolvla_boltnut.yaml")
DEFAULT_PI0_PATH = (
    "/home/zhou/.cache/huggingface/hub/"
    "models--lerobot--pi0/snapshots/e4ed526af508e58f6008b29e9e48f1098278fdb5"
)
DEFAULT_VLA_ACTION_MODE = "pose"  # "pose": [pd, Rd6d], "full": [pd, Rd6d, Vd_body]
DEFAULT_VLA_FRAME_STRIDE = 4
# Backward-compatible env names used by the earlier pi0-only training path.
DEFAULT_PI0_ACTION_MODE = DEFAULT_VLA_ACTION_MODE
DEFAULT_PI0_FRAME_STRIDE = DEFAULT_VLA_FRAME_STRIDE


def maybe_set_pretrained_path(cfg: TrainPipelineConfig) -> None:
    if cfg.policy.type == "pi0" and not getattr(cfg.policy, "pretrained_path", None):
        cfg.policy.pretrained_path = DEFAULT_PI0_PATH
    elif cfg.policy.type == "smolvla" and not getattr(cfg.policy, "pretrained_path", None):
        cfg.policy.pretrained_path = "lerobot/smolvla_base"


def get_default_config_path() -> Path:
    """Select a default config when --config_path is omitted.

    Examples:
        VLA_CONFIG=pi0      -> pi0_boltnut.yaml
        VLA_CONFIG=smolvla  -> smolvla_boltnut.yaml
        VLA_CONFIG=/path/to/custom.yaml -> that yaml
    """
    config_name = (
        os.environ.get("VLA_CONFIG")
        or os.environ.get("VLA_POLICY_TYPE")
        or os.environ.get("VLA_POLICY")
        or "pi0"
    )
    normalized = config_name.lower().replace("-", "_")
    if normalized in ("pi0", "pi_0"):
        return DEFAULT_CONFIG_PATH
    if normalized in ("smolvla", "smol_vla", "smooth_vla", "smoth_vla"):
        return DEFAULT_SMOLVLA_CONFIG_PATH

    config_path = Path(config_name)
    if config_path.suffix in (".yaml", ".yml"):
        return config_path
    raise ValueError(
        "Unknown VLA_CONFIG "
        f"{config_name!r}. Use 'pi0', 'smolvla', or a yaml path."
    )


def log_runtime_paths() -> None:
    try:
        import lerobot

        lerobot_path = getattr(lerobot, "__file__", None)
    except Exception as exc:
        lerobot_path = f"<failed to import lerobot: {exc}>"

    logging.info(f"Python executable: {sys.executable}")
    logging.info(f"lerobot path: {lerobot_path}")
    logging.info(
        "optimizer factory: "
        + ("lerobot.common.optim.factory" if lerobot_make_optimizer_and_scheduler is not None else "torch.optim.AdamW fallback")
    )


def get_cfg_attr(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def validate_dataset_root(cfg: TrainPipelineConfig) -> None:
    dataset_root = get_cfg_attr(getattr(cfg, "dataset", None), "root", None)
    if dataset_root is None:
        return

    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        return

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(
            "Incomplete LeRobot dataset root: "
            f"{dataset_root}\n"
            "Missing meta/episodes.jsonl. This usually means add_frame() wrote images, "
            "but save_episode() was not executed successfully. Re-collect into a new "
            "dataset directory, or remove/rename this partial directory before collecting."
        )


class ActionModeDataset(torch.utils.data.Dataset):
    """Wrap a LeRobotDataset and optionally reduce batch['action'] dimensions."""

    def __init__(self, dataset, action_mode="full"):
        self.dataset = dataset
        self.action_mode = action_mode
        self.meta = dataset.meta
        self.episode_data_index = dataset.episode_data_index
        self.episodes = getattr(dataset, "episodes", None)
        self._patch_meta_for_action_mode()

    def _action_slice(self):
        if self.action_mode in ("full", "pose_vd", "pose_vd_body", "with_vd_body"):
            return slice(0, 15)
        if self.action_mode in ("pose", "pose_only", "no_vd_body", "no_vd"):
            return slice(0, 9)
        raise ValueError(
            "Unknown VLA action mode "
            f"{self.action_mode!r}. Use 'full' or 'pose'."
        )

    def _patch_stat(self, stats, key, action_slice):
        if key not in stats:
            return
        for stat_name, value in list(stats[key].items()):
            if isinstance(value, torch.Tensor):
                stats[key][stat_name] = value[action_slice].clone()
            else:
                stats[key][stat_name] = value[action_slice].copy()

    def _patch_meta_for_action_mode(self):
        action_slice = self._action_slice()
        action_dim = action_slice.stop - action_slice.start
        if action_dim == 15:
            return

        # Avoid mutating nested dicts shared with the original dataset object.
        self.meta.info["features"] = copy.deepcopy(self.meta.features)
        self.meta.stats = copy.deepcopy(self.meta.stats)
        self.meta.episodes_stats = copy.deepcopy(self.meta.episodes_stats)

        self.meta.features["action"]["shape"] = (action_dim,)
        self.meta.features["action"]["names"] = ["pd_Rd6d"] if action_dim == 9 else ["action"]

        self._patch_stat(self.meta.stats, "action", action_slice)
        for ep_stats in self.meta.episodes_stats.values():
            self._patch_stat(ep_stats, "action", action_slice)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        action_slice = self._action_slice()
        if "action" in item:
            item["action"] = item["action"][..., action_slice]
        return item

    @property
    def num_frames(self):
        return self.dataset.num_frames

    @property
    def num_episodes(self):
        return self.dataset.num_episodes

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class StridedDataset(torch.utils.data.Dataset):
    """Use one training sample every `stride` frames, preserving episode boundaries."""

    def __init__(self, dataset, stride=1):
        self.dataset = dataset
        self.stride = int(stride)
        if self.stride < 1:
            raise ValueError(f"frame stride must be >= 1, got {self.stride}")

        self.meta = dataset.meta
        self.episodes = getattr(dataset, "episodes", None)
        self.index_map, self.episode_data_index = self._build_index_map()

    def _build_index_map(self):
        old_index = self.dataset.episode_data_index
        mapped_indices = []
        new_from = []
        new_to = []

        cursor = 0
        for start, stop in zip(old_index["from"].tolist(), old_index["to"].tolist()):
            ep_indices = list(range(start, stop, self.stride))
            new_from.append(cursor)
            cursor += len(ep_indices)
            new_to.append(cursor)
            mapped_indices.extend(ep_indices)

        return (
            torch.as_tensor(mapped_indices, dtype=torch.long),
            {
                "from": torch.as_tensor(new_from, dtype=torch.long),
                "to": torch.as_tensor(new_to, dtype=torch.long),
            },
        )

    def __len__(self):
        return int(self.index_map.numel())

    def __getitem__(self, index):
        source_index = int(self.index_map[int(index)].item())
        return self.dataset[source_index]

    @property
    def num_frames(self):
        return len(self)

    @property
    def num_episodes(self):
        return self.dataset.num_episodes

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def get_vla_action_mode(cfg: TrainPipelineConfig) -> str:
    return (
        get_cfg_attr(cfg, "vla_action_mode", None)
        or os.environ.get("VLA_ACTION_MODE")
        or get_cfg_attr(cfg, "pi0_action_mode", None)
        or os.environ.get("PI0_ACTION_MODE", DEFAULT_PI0_ACTION_MODE)
    ).lower()


def get_vla_frame_stride(cfg: TrainPipelineConfig) -> int:
    return int(
        get_cfg_attr(cfg, "vla_frame_stride", None)
        or os.environ.get("VLA_FRAME_STRIDE")
        or get_cfg_attr(cfg, "pi0_frame_stride", None)
        or os.environ.get("PI0_FRAME_STRIDE", DEFAULT_PI0_FRAME_STRIDE)
    )


def maybe_wrap_action_mode_dataset(dataset, cfg: TrainPipelineConfig):
    policy_type = getattr(cfg.policy, "type", "vla")
    if policy_type not in ("pi0", "smolvla"):
        return dataset

    action_mode = get_vla_action_mode(cfg)
    if action_mode in ("full", "pose_vd", "pose_vd_body", "with_vd_body"):
        logging.info(f"{policy_type} action mode: full [pd, Rd6d, Vd_body] (15 dims)")
        return dataset

    wrapped = ActionModeDataset(dataset, action_mode=action_mode)
    logging.info(f"{policy_type} action mode: pose only [pd, Rd6d] (9 dims)")
    return wrapped


def maybe_wrap_frame_stride_dataset(dataset, cfg: TrainPipelineConfig):
    frame_stride = get_vla_frame_stride(cfg)
    if frame_stride <= 1:
        logging.info("VLA frame stride: 1 (use every frame)")
        return dataset

    wrapped = StridedDataset(dataset, stride=frame_stride)
    logging.info(
        "VLA frame stride: "
        f"{frame_stride} ({dataset.num_frames} -> {wrapped.num_frames} frames)"
    )
    return wrapped


def make_optimizer_and_scheduler_compat(cfg: TrainPipelineConfig, policy: PreTrainedPolicy):
    if lerobot_make_optimizer_and_scheduler is not None:
        return lerobot_make_optimizer_and_scheduler(cfg, policy)

    optimizer_cfg = getattr(cfg, "optimizer", None)
    lr = get_cfg_attr(optimizer_cfg, "lr", 1e-4)
    weight_decay = get_cfg_attr(optimizer_cfg, "weight_decay", 1e-5)
    betas = get_cfg_attr(optimizer_cfg, "betas", (0.9, 0.95))
    eps = get_cfg_attr(optimizer_cfg, "eps", 1e-8)

    logging.warning(
        "lerobot.common.optim.factory is not available; falling back to "
        f"torch.optim.AdamW(lr={lr}, weight_decay={weight_decay})."
    )
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    return optimizer, None


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()

    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)

    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    log_runtime_paths()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    validate_dataset_root(cfg)
    dataset = make_dataset(cfg)
    dataset = maybe_wrap_action_mode_dataset(dataset, cfg)
    dataset = maybe_wrap_frame_stride_dataset(dataset, cfg)

    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    maybe_set_pretrained_path(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler_compat(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
    )

    logging.info("Start offline VLA training")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        step += 1
        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if eval_env is not None and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("sum_rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size,
                dataset.num_frames,
                dataset.num_episodes,
                eval_metrics,
                initial_step=step,
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env is not None:
        eval_env.close()
    logging.info("End of VLA training")


if __name__ == "__main__":
    init_logging()
    if "--config_path" not in sys.argv:
        sys.argv.extend(["--config_path", str(get_default_config_path())])
    train()
