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
import copy
import os
import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
import argparse
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
import argparse

try:
    from lerobot.common.optim.factory import make_optimizer_and_scheduler as lerobot_make_optimizer_and_scheduler
except ModuleNotFoundError:
    lerobot_make_optimizer_and_scheduler = None

DEFAULT_PI0_PATH = (
    "/root/autodl-tmp/hub/"
    "models--lerobot--pi0/snapshots/e4ed526af508e58f6008b29e9e48f1098278fdb5"
)

DEFAULT_PI0_PATHS = (
    "/root/autodl-tmp/hub/"
    "paligemma-3b-pt-224"
)
DEFAULT_SMOLVLM_PATHS = (
    "/root/autodl-tmp/hub/"
    "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
)
DEFAULT_SMOLVLA_PATH = (
    "/root/autodl-tmp/hub/"
    "models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
)

def maybe_set_pretrained_path(cfg: TrainPipelineConfig) -> None:
    if cfg.policy.type == "pi0" and not getattr(cfg.policy, "pretrained_path", None):
        cfg.policy.pretrained_path = DEFAULT_PI0_PATH
        cfg.policy.vlm_model_name = DEFAULT_PI0_PATHS
    elif cfg.policy.type == "smolvla" and not getattr(cfg.policy, "pretrained_path", None):
        cfg.policy.pretrained_path = DEFAULT_SMOLVLA_PATH
        cfg.policy.vlm_model_name = DEFAULT_SMOLVLM_PATHS

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
    validate_parquet_files(dataset_root)


def validate_parquet_files(dataset_root: Path) -> None:
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(
            "Incomplete LeRobot dataset root: "
            f"{dataset_root}\n"
            "Missing data/ directory."
        )

    parquet_paths = sorted(data_dir.rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            "No parquet episode files found under "
            f"{data_dir}. Check dataset.root / dataset.repo_id."
        )

    bad_files = []
    for parquet_path in parquet_paths:
        try:
            size = parquet_path.stat().st_size
            with parquet_path.open("rb") as f:
                head = f.read(4)
                if size >= 4:
                    f.seek(-4, os.SEEK_END)
                    tail = f.read(4)
                else:
                    tail = b""
        except OSError as exc:
            bad_files.append((parquet_path, f"read failed: {exc}"))
            continue

        if size < 8:
            bad_files.append((parquet_path, f"too small ({size} bytes)"))
        elif head != b"PAR1" or tail != b"PAR1":
            bad_files.append((parquet_path, "missing parquet PAR1 magic bytes"))

    if bad_files:
        details = "\n".join(f"  - {path}: {reason}" for path, reason in bad_files[:20])
        if len(bad_files) > 20:
            details += f"\n  ... and {len(bad_files) - 20} more"
        raise RuntimeError(
            "Corrupted or non-parquet episode files were found before training:\n"
            f"{details}\n"
            "Fix by deleting/re-collecting these episodes or restoring them from a "
            "complete copy. The original pyarrow error is usually: "
            "'Parquet magic bytes not found in footer'."
        )


def log_smolvla_force_config(cfg: TrainPipelineConfig, dataset) -> None:
    policy_cfg = getattr(cfg, "policy", None)
    if policy_cfg is None or getattr(policy_cfg, "type", None) != "smolvla":
        return

    effort_type = getattr(policy_cfg, "effort_type", "none")
    effort_tokenizer = getattr(policy_cfg, "effort_tokenizer", "raw")
    effort_key = getattr(policy_cfg, "effort_key", None)
    if effort_type in {"none", "no"}:
        return

    logging.info(
        "SmolVLA force config: "
        f"effort_key={effort_key}, effort_type={effort_type}, "
        f"effort_tokenizer={effort_tokenizer}"
    )
    if effort_key not in dataset.meta.features:
        raise KeyError(
            f"Configured effort_key {effort_key!r} is not in dataset features. "
            f"Available features: {list(dataset.meta.features)}"
        )

    if effort_tokenizer == "force_vqvae":
        ckpt = Path(getattr(policy_cfg, "force_vqvae_ckpt", "")).expanduser()
        if not ckpt.is_file():
            raise FileNotFoundError(f"policy.force_vqvae_ckpt does not exist: {ckpt}")
        logging.info(
            "SmolVLA force VQ-VAE enabled: "
            f"ckpt={ckpt}, window={getattr(policy_cfg, 'force_vqvae_window', None)}, "
            f"force_refine_enabled={getattr(policy_cfg, 'force_refine_enabled', False)}"
        )

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
    log_smolvla_force_config(cfg, dataset)
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pi0 inference for pd, Rd, dpd, dRd and GUFIC velocity field."
    )
    parser.add_argument(
        "--config_path",
        default="/root/vla/gufic_flow_matching/gufic_env/flow_matching/smolvla_boltnut.yaml",
        help="pi0_boltnut.yaml or smolvla_boltnut.yaml.",
    )
    parser.add_argument(
        "--resume",
        default=False,
        help="pi0_boltnut.yaml or smolvla_boltnut.yaml.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    init_logging()
    args = parse_args()
    if "--config_path" not in sys.argv:
        sys.argv.extend(["--config_path", args.config_path])
    train()
