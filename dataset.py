import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
import glob
import os   

def rotmat_batch_to_rot6d(R: np.ndarray) -> np.ndarray:
    """
    R: [N, 3, 3]
    return: [N, 6]
    """
    R = np.asarray(R, dtype=np.float32)
    assert R.ndim == 3 and R.shape[-2:] == (3, 3)
    rot6d = R[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    return rot6d.astype(np.float32)

class FlowMatchingDataset(Dataset):
    """
    相邻点对数据集
    每个样本对应 (k, k+1)

    返回:
      x0, x1:   [6], [6]
      v0, v1:   [6], [6]      # 采集到的 Vd_star
      fe0, fe1: [C], [C]
      t0, t1:   [1], [1]
      dt:       [1]
    """
    def __init__(self, demo_dir, cond_key="fe", normalize = True, eps=1e-6):
        self.samples = []
        self.cond_key = cond_key
        raw_v_list = []

        demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.npz")))
        if len(demo_files) == 0:
            raise ValueError(f"No demo files found in {demo_dir}")

        for f in demo_files:
            data = np.load(f)

            x = data["x"].astype(np.float32)               # [T,6]
            v = data["Vd_star"].astype(np.float32)         # [T,6]
            t = data["t"].astype(np.float32)               # [T]
            total_time = float(data["total_time"][0])

            if cond_key in data:
                cond = data[cond_key].astype(np.float32)
            else:
                # 如果没有 fe，就退化成用 goal 作为条件
                goal = data["goal"].astype(np.float32)
                cond = np.repeat(goal[None, :], len(t), axis=0).astype(np.float32)

            raw_v_list.append(v)
            self.samples.append({
                "x": x,
                "v_raw": v,
                "fe": cond,
                "t": np.array([t / max(total_time, 1e-8)], dtype=np.float32),
                })

        all_v = np.concatenate(raw_v_list, axis=0)      # [sum(T), 6]
        v_mean = all_v.mean(axis=0, keepdims=True).astype(np.float32)
        v_std = all_v.std(axis=0, keepdims=True).astype(np.float32)
        v_std = np.clip(v_std, eps, None)
        self.stats = {"v_mean": v_mean, "v_std": v_std}

        for s in self.samples:
            if normalize:
                s["v"] = ((s["v_raw"] - self.stats["v_mean"]) / self.stats["v_std"]).astype(np.float32)
            else:
                s["v"] = s["v_raw"]

        print(f"[PairDataset] loaded {len(demo_files)} demos, total pairs = {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["x"]),
            torch.from_numpy(s["v"]),
            torch.from_numpy(s["fe"]),
            torch.from_numpy(s["t"]),
        )
    def get_stats(self):
        return self.stats

class RollingForceHistoryFMDataset(Dataset):
    """
    每个样本:
      fe_hist:  [K, 6]
      v_future: [H, 6]

    训练目标:
      用最近 K 步力序列作为条件，生成未来 H 步速度场
    """
    def __init__(
        self,
        demo_dir,
        force_hist_len=16,
        pred_horizon=100,
        stride=5,
        normalize_v=True,
        v_stats=None,
        eps=1e-6,
    ):
        self.samples = []
        self.force_hist_len = force_hist_len
        self.pred_horizon = pred_horizon
        self.normalize_v = normalize_v
        self.eps = eps

        demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.npz")))
        if len(demo_files) == 0:
            raise ValueError(f"No demo files found in {demo_dir}")

        all_v = []

        for f in demo_files:
            data = np.load(f)
            if "p" not in data:
                raise ValueError(f"p not found in {f}")
            if "R" not in data:
                raise ValueError(f"R not found in {f}")
            if "Vd_star" not in data:
                raise ValueError(f"Vd_star not found in {f}")
            if "Fe" not in data:
                raise ValueError(f"Fe not found in {f}")
            p = data["p"].astype(np.float32)       # [T,3]
            R = data["R"].astype(np.float32)       # [T,3]
            R6d = rotmat_batch_to_rot6d(R)                        # [T,6]
            x = np.concatenate([p, R6d], axis=-1)        # [T,6]
            v = data["Vd_star"].astype(np.float32)   # [T,6]
            fe = data["Fe"].astype(np.float32)       # [T,6]

            T = len(v)
            all_v.append(v)

            # 需要至少有 K 步历史 + H 步未来
            # start_k = force_hist_len - 1
            end_k = T - pred_horizon - 1

            for k in range(0, end_k + 1, stride):
                left = max(0, k - force_hist_len + 1)
                fe_hist = fe[left : k + 1]      # [K,6]
                x_hist = x[left : k + 1]      # [K,6]
                if fe_hist.shape[0] < force_hist_len:
                    # 如果不足 K 步历史，就在前面补零
                    pad_len = force_hist_len - fe_hist.shape[0]
                    fe_hist = np.pad(fe_hist, ((pad_len, 0), (0, 0)), mode="constant")
                if x_hist.shape[0] < force_hist_len:
                    # 如果整个序列都不足 K 步，就在前面补零
                    pad_len = force_hist_len - x_hist.shape[0]
                    pad = np.repeat(x_hist[0:1], pad_len, axis=0)
                    x_hist = np.concatenate([pad, x_hist], axis=0)
                v_future = v[k + 1 : k + 1 + pred_horizon]        # [H,6]

                self.samples.append({
                    "x_hist":x_hist,
                    "fe_hist": fe_hist,
                    "v_future_raw": v_future,
                })

        if len(self.samples) == 0:
            raise ValueError("No valid rolling-horizon samples found.")

        # 只对 v 做标准化
        if v_stats is None:
            all_v_cat = np.concatenate(all_v, axis=0)  # [sum(T), 6]
            v_mean = all_v_cat.mean(axis=0, keepdims=True).astype(np.float32)
            v_std = all_v_cat.std(axis=0, keepdims=True).astype(np.float32)
            v_std = np.clip(v_std, eps, None)
            self.v_stats = {
                "v_mean": v_mean,
                "v_std": v_std,
            }
        else:
            self.v_stats = {
                "v_mean": v_stats["v_mean"].astype(np.float32),
                "v_std": np.clip(v_stats["v_std"].astype(np.float32), eps, None),
            }

        for s in self.samples:
            if normalize_v:
                s["v_future"] = (
                    (s["v_future_raw"] - self.v_stats["v_mean"]) / self.v_stats["v_std"]
                ).astype(np.float32)
            else:
                s["v_future"] = s["v_future_raw"].astype(np.float32)

        print(f"[RollingForceHistoryFMDataset] samples = {len(self.samples)}")
        print(f"[RollingForceHistoryFMDataset] K = {force_hist_len}, H = {pred_horizon}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x_hist = s["x_hist"]
        x_hist_flat = x_hist.reshape(-1)        # [6K]
        fe_hist = s["fe_hist"]                    # [K,6]
        fe_hist_flat = fe_hist.reshape(-1)        # [6K]
        cond_hist = np.concatenate([x_hist_flat, fe_hist_flat], axis=-1)   # [12K]

        return (
            torch.from_numpy(cond_hist.astype(np.float32)),   # [6K]
            torch.from_numpy(s["v_future"].astype(np.float32)),  # [H,6]
        )

    def get_v_stats(self):
        return self.v_stats

class FlowMatchingHybridDataset(Dataset):
    """
    相邻点对数据集
    每个样本对应 (k, k+1)

    返回:
      x0, x1:   [6], [6]
      v0, v1:   [6], [6]      # 采集到的 Vd_star
      fe0, fe1: [C], [C]
      t0, t1:   [1], [1]
      dt:       [1]
    """
    def __init__(self, demo_dir, cond_key="fe"):
        self.samples = []
        self.cond_key = cond_key

        demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.npz")))
        if len(demo_files) == 0:
            raise ValueError(f"No demo files found in {demo_dir}")

        for f in demo_files:
            data = np.load(f)

            x = data["x"].astype(np.float32)               # [T,6]
            v = data["Vd_star"].astype(np.float32)         # [T,6]
            t = data["t"].astype(np.float32)               # [T]
            total_time = float(data["total_time"][0])

            if cond_key in data:
                cond = data[cond_key].astype(np.float32)
            else:
                # 如果没有 fe，就退化成用 goal 作为条件
                goal = data["goal"].astype(np.float32)
                cond = np.repeat(goal[None, :], len(t), axis=0).astype(np.float32)

            for k in range(len(t) - 1):
                dt = float(t[k + 1] - t[k])
                if dt <= 1e-8:
                    continue

                self.samples.append({
                    "x0": x[k],
                    "x1": x[k + 1],
                    "v0": v[k],
                    "v1": v[k + 1],
                    "fe0": cond[k],
                    "fe1": cond[k + 1],
                    "t0": np.array([t[k] / max(total_time, 1e-8)], dtype=np.float32),
                    "t1": np.array([t[k + 1] / max(total_time, 1e-8)], dtype=np.float32),
                    "dt": np.array([dt], dtype=np.float32),
                })

        print(f"[PairDataset] loaded {len(demo_files)} demos, total pairs = {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["x0"]),
            torch.from_numpy(s["x1"]),
            torch.from_numpy(s["v0"]),
            torch.from_numpy(s["v1"]),
            torch.from_numpy(s["fe0"]),
            torch.from_numpy(s["fe1"]),
            torch.from_numpy(s["t0"]),
            torch.from_numpy(s["t1"]),
            torch.from_numpy(s["dt"]),
        )
    
class VelocityFieldRegressiveDataset(Dataset):
    def __init__(
        self,
        demo_dir="./bolt_demos_train",
        add_state_noise=True,
        pos_noise_std=0.001,
        ori_noise_std=0.01,
        use_dVd_star=False,
    ):
        self.samples = []
        self.add_state_noise = add_state_noise
        self.pos_noise_std = pos_noise_std
        self.ori_noise_std = ori_noise_std
        self.use_dVd_star = use_dVd_star

        demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.npz")))
        if len(demo_files) == 0:
            raise ValueError(f"No demo files found in {demo_dir}")

        for f in demo_files:
            data = np.load(f)

            x = data["x"].astype(np.float32)
            dx = data["Vd_star"].astype(np.float32)
            ddx = data["dVd_star"].astype(np.float32)
            Fe = data["Fe"].astype(np.float32)
            t = data["t"].astype(np.float32)
            goal = data["goal"].astype(np.float32)
            total_time = float(data["total_time"][0])

            for i in range(len(t)):
                item = {
                    "x_t": x[i],
                    "dx_t": dx[i],
                    "fe_t": Fe[i],
                    "goal": goal,
                    "t_norm": np.array([t[i] / max(total_time, 1e-8)], dtype=np.float32),
                }
                if self.use_dVd_star:
                    item["ddx_t"] = ddx[i]
                self.samples.append(item)

        print(f"[Dataset] loaded demos={len(demo_files)}, total_frames={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        x_t = s["x_t"].copy()
        dx_t = s["dx_t"].copy()
        fe_t = s["fe_t"].copy()
        goal = s["goal"].copy()
        t_norm = s["t_norm"].copy()

        if self.add_state_noise:
            x_t[:3] += np.random.normal(scale=self.pos_noise_std, size=(3,)).astype(np.float32)
            x_t[3:6] += np.random.normal(scale=self.ori_noise_std, size=(3,)).astype(np.float32)

        if self.use_dVd_star:
            ddx_t = s["ddx_t"].copy()
            return (
                torch.from_numpy(x_t),
                torch.from_numpy(dx_t),
                torch.from_numpy(fe_t),
                torch.from_numpy(goal),
                torch.from_numpy(t_norm),
                torch.from_numpy(ddx_t),
            )

        return (
            torch.from_numpy(x_t),
            torch.from_numpy(dx_t),
            torch.from_numpy(fe_t),
            torch.from_numpy(goal),
            torch.from_numpy(t_norm),
        )
