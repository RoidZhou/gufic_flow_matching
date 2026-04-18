import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
import glob
import os   

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
