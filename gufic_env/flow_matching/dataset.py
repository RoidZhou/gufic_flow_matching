import mujoco
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import glob
import os
from tqdm.auto import tqdm  

def load_xml(robot_name, task):
    dir = os.getcwd() + '/'
    if robot_name == 'ur5e':
        raise NotImplementedError
    elif robot_name == 'indy7':
        if task == "sphere":
            model_path = dir + "gufic_env/mujoco_models/Indy7_wiping_sphere.xml"
        else:
            model_path = dir + "gufic_env/mujoco_models/Indy7_wiping.xml"
    elif robot_name == 'panda':
        raise NotImplementedError
    else:
        raise NotImplementedError

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    MuJoCo quaternion format: [w, x, y, z]
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R


def get_hand_eye_from_xml(
    robot_name: str,
    task: str,
    camera_name: str = "eye_in_hand",
    ee_site_name: str = "end_effector",
):
    """
    读取 XML 后，自动计算固定手眼外参：
        x_e = R_ec @ x_c + t_ec

    返回:
        R_ec: [3,3]  camera -> end_effector 的旋转
        t_ec: [3]    camera 原点在 end_effector 坐标系下的位置
        extra: dict  调试信息
    """
    model, data = load_xml(robot_name, task)

    # 让 data.xpos/xmat/site_xpos/site_xmat 有效
    mujoco.mj_forward(model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)

    if cam_id < 0:
        raise ValueError(f"Camera '{camera_name}' not found.")
    if site_id < 0:
        raise ValueError(f"Site '{ee_site_name}' not found.")

    # 相机的父 body
    cam_body_id = int(model.cam_bodyid[cam_id])

    # 父 body 在世界系下位姿
    p_wb = np.array(data.xpos[cam_body_id], dtype=np.float32).copy()
    R_wb = np.array(data.xmat[cam_body_id], dtype=np.float32).reshape(3, 3).copy()

    # 相机相对父 body 的局部外参
    p_bc = np.array(model.cam_pos[cam_id], dtype=np.float32).copy()
    q_bc = np.array(model.cam_quat[cam_id], dtype=np.float32).copy()
    R_bc = quat_wxyz_to_rotmat(q_bc)

    # 相机在世界系下位姿
    p_wc = p_wb + R_wb @ p_bc
    R_wc = R_wb @ R_bc

    # 末端 site 在世界系下位姿
    p_we = np.array(data.site_xpos[site_id], dtype=np.float32).copy()
    R_we = np.array(data.site_xmat[site_id], dtype=np.float32).reshape(3, 3).copy()

    # camera -> end_effector
    # x_e = R_ec @ x_c + t_ec
    R_ec = R_we.T @ R_wc
    t_ec = R_we.T @ (p_wc - p_we)

    extra = {
        "p_wc": p_wc,
        "R_wc": R_wc,
        "p_we": p_we,
        "R_we": R_we,
        "cam_body_id": cam_body_id,
        "p_bc": p_bc,
        "R_bc": R_bc,
    }
    return R_ec.astype(np.float32), t_ec.astype(np.float32), extra

def pointcloud_cam_to_world_batch(
    pc: np.ndarray,   # [T, P, C]
    p: np.ndarray,    # [T, 3]      end-effector world position
    R: np.ndarray,    # [T, 3, 3]   end-effector world rotation
    R_ec: np.ndarray, # [3, 3]      camera -> ee
    t_ec: np.ndarray, # [3]
) -> np.ndarray:
    """
    利用固定手眼外参，把相机系点云批量变到世界系。

    x_e = R_ec @ x_c + t_ec
    x_w = R_we @ x_e + p_we
    """
    pc = np.asarray(pc, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    R_ec = np.asarray(R_ec, dtype=np.float32).reshape(3, 3)
    t_ec = np.asarray(t_ec, dtype=np.float32).reshape(3)

    xyz_cam = pc[..., :3]                                      # [T,P,3]
    xyz_ee = np.einsum("ij,tpj->tpi", R_ec, xyz_cam) + t_ec    # [T,P,3]
    xyz_w = np.einsum("tij,tpj->tpi", R, xyz_ee) + p[:, None]  # [T,P,3]

    if pc.shape[-1] > 3:
        return np.concatenate([xyz_w, pc[..., 3:]], axis=-1).astype(np.float32)
    return xyz_w.astype(np.float32)

def rotmat_batch_to_rot6d(R: np.ndarray) -> np.ndarray:
    """
    R: [N, 3, 3]
    return: [N, 6]
    """
    R = np.asarray(R, dtype=np.float32)
    assert R.ndim == 3 and R.shape[-2:] == (3, 3)
    rot6d = R[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    return rot6d.astype(np.float32)

def uniform_sample_one_frame(point_cloud, num_points, use_xyz_only=True):
    """
    对单帧点云采样。
    输入:
        point_cloud: [N, C]
    输出:
        sampled_points: [num_points, C]
    """
    point_cloud = np.asarray(point_cloud, dtype=np.float32)

    if use_xyz_only:
        point_cloud = point_cloud[:, :3]

    if point_cloud.shape[0] == 0:
        C = 3 if use_xyz_only else point_cloud.shape[1]
        return np.zeros((num_points, C), dtype=np.float32)

    replace = point_cloud.shape[0] < num_points

    idx = np.random.choice(
        point_cloud.shape[0],
        size=num_points,
        replace=replace
    )

    return point_cloud[idx].astype(np.float32)

def rotmat_to_rot6d_one(R: np.ndarray) -> np.ndarray:
    """
    R: [3,3]
    return: [6]
    """
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    return R[:, :2].T.reshape(-1).astype(np.float32)

def build_delta_pose_target(
    p_now: np.ndarray,
    R_now: np.ndarray,
    p_next: np.ndarray,
    R_next: np.ndarray,
) -> np.ndarray:
    """
    return: [9] = [delta_p(3), delta_R6d(6)]
    """
    delta_p = (p_next - p_now).astype(np.float32)     # [3]
    R_rel = R_now.T @ R_next                          # [3,3]
    delta_r6d = rotmat_to_rot6d_one(R_rel)           # [6]
    return np.concatenate([delta_p, delta_r6d], axis=0).astype(np.float32)

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
    def __init__(
        self,
        demo_dir,
        x_hist_len=1,
        pc_hist_len=1,
        force_hist_len=16,
        pred_horizon=100,
        stride=5,
        normalize_v=True,
        cond_stats=None,
        use_pc_color=False,
        eps=1e-6,
        robot_model = None,
        robot_task = None
    ):
        self.samples = []
        self.force_hist_len = force_hist_len
        self.x_hist_len = x_hist_len
        self.pred_horizon = pred_horizon
        self.normalize_v = normalize_v
        self.eps = eps
        self.use_pc_color = use_pc_color
        self.pc_scale = 0.1
        demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.npz")))
        if len(demo_files) == 0:
            raise ValueError(f"No demo files found in {demo_dir}")

        all_p = []
        all_R = []
        all_Fe = []
        all_v = []
        all_pc = []
        all_delta_p = []
        all_delta_R = []
        for f in tqdm(demo_files, desc="Loading demo files", ncols=100):
            try:
                data = np.load(f)
            except Exception as e:
                print(f"[Dataset] Skip bad file: {f}, error: {e}")
                continue

            if "p" not in data:
                raise ValueError(f"p not found in {f}")
            if "R" not in data:
                raise ValueError(f"R not found in {f}")
            if "Vd_star" not in data:
                raise ValueError(f"Vd_star not found in {f}")
            if "Fe" not in data:
                raise ValueError(f"Fe not found in {f}")
            if "point_cloud" not in data:
                raise ValueError(f"point_cloud not found in {f}")

            p = data["p"].astype(np.float32)
            R = data["R"].astype(np.float32)
            R6d = rotmat_batch_to_rot6d(R)
            fe = data["Fe"].astype(np.float32)
            v = data["Vd_star"].astype(np.float32)
            pc = data["point_cloud"].astype(np.float32)
            R_ec, t_ec, _ = get_hand_eye_from_xml(robot_model, robot_task)
            pc_world = pointcloud_cam_to_world_batch(pc, p, R, R_ec, t_ec)

            pc_ee = np.einsum("tji,tpj->tpi", R, pc_world[..., :3] - p[:, None, :])  # R^T (x_w - p)
            pc_ee = pc_ee / self.pc_scale

            if self.use_pc_color:
                pc = pc_ee
            else:
                pc = pc_ee[:,:,:3]
            # 上采样
            pc = np.stack(
                [uniform_sample_one_frame(pc_t, 2048, use_xyz_only=True) for pc_t in pc],
                axis=0
            ).astype(np.float32)  # [T, 2048, 3]
            # We shuffle the points, i.e. shuffle pcd along dim=2 (T, P, 3)
            idx = torch.randperm(pc.shape[1])
            pc = pc[:, idx, :]

            T = len(v)
            all_p.append(p)
            all_R.append(R6d)
            all_Fe.append(fe)
            all_v.append(v)
            all_pc.append(pc)

            end_k = T - pred_horizon - 1

            for k in tqdm(
                range(0, end_k + 1, stride),
                desc=f"Building samples: {os.path.basename(f)}",
                leave=False,
                ncols=100,
            ):
                fe_left = max(0, k - force_hist_len + 1)
                p_left = max(0, k - x_hist_len + 1)
                R_left = max(0, k - x_hist_len + 1)
                pc_left = max(0, k - pc_hist_len + 1)

                # 历史状态
                fe_hist = fe[fe_left: k + 1]
                p_hist = p[p_left: k + 1]
                R_hist = R6d[R_left: k + 1]
                pc_hist = pc[pc_left: k + 1]

                if fe_hist.shape[0] < force_hist_len:
                    pad_len = force_hist_len - fe_hist.shape[0]
                    fe_hist = np.pad(fe_hist, ((pad_len, 0), (0, 0)), mode="constant")

                if p_hist.shape[0] < x_hist_len:
                    pad_len = x_hist_len - p_hist.shape[0]
                    pad_p = np.repeat(p_hist[0:1], pad_len, axis=0)
                    pad_R = np.repeat(R_hist[0:1], pad_len, axis=0)
                    p_hist = np.concatenate([pad_p, p_hist], axis=0)
                    R_hist = np.concatenate([pad_R, R_hist], axis=0)

                if pc_hist.shape[0] < pc_hist_len:
                    pad_len = pc_hist_len - pc_hist.shape[0]
                    pad_pc = np.repeat(pc_hist[0:1], pad_len, axis=0)
                    pc_hist = np.concatenate([pad_pc, pc_hist], axis=0)

                # 当前状态
                p_now = p[k].astype(np.float32)          # [3]
                R_now = R[k].astype(np.float32)          # [3,3]
                R6d_now = R6d[k].astype(np.float32)      # [6]

                # 下一步状态，给视觉分支监督
                # p_next = p[k + 1].astype(np.float32)
                # R_next = R[k + 1].astype(np.float32)
                # 未来第10步状态，给视觉分支监督
                m = 10
                p_next = p[min(k + m, T - 1)].astype(np.float32)
                R_next = R[min(k + m, T - 1)].astype(np.float32)
                delta_pose_target = build_delta_pose_target(p_now, R_now, p_next, R_next)  # [9]
                all_delta_p.append(delta_pose_target[:3][None, :])   # [1,3]
                all_delta_R.append(delta_pose_target[3:][None, :])   # [1,6]

                v_future = v[k + 1: k + 1 + pred_horizon]

                self.samples.append({
                    # "p_now_raw": p_now,
                    # "R6d_now_raw": R6d_now,
                    "delta_pose_target": delta_pose_target,
                    "p_hist_raw": p_hist,
                    "R_hist_raw": R_hist,
                    "pc_hist_raw": pc_hist,
                    "fe_hist_raw": fe_hist,
                    "v_future_raw": v_future,
                })

        if len(self.samples) == 0:
            raise ValueError("No valid rolling-horizon samples found.")

        # 只对 v 做标准化
        if cond_stats is None:
            all_p_cat = np.concatenate(all_p, axis=0)  # [sum(T), 6]
            p_mean = all_p_cat.mean(axis=0, keepdims=True).astype(np.float32)
            p_std = all_p_cat.std(axis=0, keepdims=True).astype(np.float32)
            p_std = np.clip(p_std, eps, None)

            all_R_cat = np.concatenate(all_R, axis=0)  # [sum(T), 6]
            R_mean = all_R_cat.mean(axis=0, keepdims=True).astype(np.float32)
            R_std = all_R_cat.std(axis=0, keepdims=True).astype(np.float32)
            R_std = np.clip(R_std, eps, None)

            all_fe_cat = np.concatenate(all_Fe, axis=0)  # [sum(T), 6]
            fe_mean = all_fe_cat.mean(axis=0, keepdims=True).astype(np.float32)
            fe_std = all_fe_cat.std(axis=0, keepdims=True).astype(np.float32)
            fe_std = np.clip(fe_std, eps, None)
            
            all_v_cat = np.concatenate(all_v, axis=0)  # [sum(T), 6]
            v_mean = all_v_cat.mean(axis=0, keepdims=True).astype(np.float32)
            v_std = all_v_cat.std(axis=0, keepdims=True).astype(np.float32)
            v_std = np.clip(v_std, eps, None)

            all_delta_p_cat = np.concatenate(all_delta_p, axis=0)   # [num_samples, 3]
            delta_p_mean = all_delta_p_cat.mean(axis=0, keepdims=True).astype(np.float32)
            delta_p_std = all_delta_p_cat.std(axis=0, keepdims=True).astype(np.float32)
            print("delta_p_std =", delta_p_std)
            delta_p_std = np.clip(delta_p_std,  5e-4, None)

            all_delta_R_cat = np.concatenate(all_delta_R, axis=0)   # [num_samples, 6]
            delta_R_mean = all_delta_R_cat.mean(axis=0, keepdims=True).astype(np.float32)
            delta_R_std = all_delta_R_cat.std(axis=0, keepdims=True).astype(np.float32)
            print("delta_R_std =", delta_R_std)
            delta_R_std = np.clip(delta_R_std, 1e-3, None)

            self.cond_stats = {
                "p_mean": p_mean,
                "p_std": p_std,
                "R_mean": R_mean,
                "R_std": R_std,
                "fe_mean": fe_mean,
                "fe_std": fe_std,
                "v_mean": v_mean,
                "v_std": v_std,
                "delta_p_mean": delta_p_mean,
                "delta_p_std": delta_p_std,
                "delta_R_mean": delta_R_mean,
                "delta_R_std": delta_R_std,
            }
        else:
            self.cond_stats = {
                "p_mean": cond_stats["p_mean"].astype(np.float32),
                "p_std": np.clip(cond_stats["p_std"].astype(np.float32), eps, None),
                "R_mean": cond_stats["R_mean"].astype(np.float32),
                "R_std": np.clip(cond_stats["R_std"].astype(np.float32), eps, None),
                "fe_mean": cond_stats["fe_mean"].astype(np.float32),
                "fe_std": np.clip(cond_stats["fe_std"].astype(np.float32), eps, None),
                "v_mean": cond_stats["v_mean"].astype(np.float32),
                "v_std": np.clip(cond_stats["v_std"].astype(np.float32), eps, None),
                "delta_p_mean": cond_stats["delta_p_mean"].astype(np.float32),
                "delta_p_std": np.clip(cond_stats["delta_p_std"].astype(np.float32), eps, None),
                "delta_R_mean": cond_stats["delta_R_mean"].astype(np.float32),
                "delta_R_std": np.clip(cond_stats["delta_R_std"].astype(np.float32), eps, None),
            }
        for s in self.samples:
            if normalize_v:
                s["p_hist"] = (
                    (s["p_hist_raw"] - self.cond_stats["p_mean"]) / self.cond_stats["p_std"]
                ).astype(np.float32)
                s["R_hist"] = (
                    (s["R_hist_raw"] - self.cond_stats["R_mean"]) / self.cond_stats["R_std"]
                ).astype(np.float32)
                s["fe_hist"] = (
                    (s["fe_hist_raw"] - self.cond_stats["fe_mean"]) / self.cond_stats["fe_std"]
                ).astype(np.float32)
                s["v_future"] = (
                    (s["v_future_raw"] - self.cond_stats["v_mean"]) / self.cond_stats["v_std"]
                ).astype(np.float32)

                pc_raw = s["pc_hist_raw"].astype(np.float32)
                s["pc_hist"] = (pc_raw / self.pc_scale).astype(np.float32)

                delta_p_raw = s["delta_pose_target"][:3].astype(np.float32)[None, :]
                delta_R_raw = s["delta_pose_target"][3:].astype(np.float32)[None, :]

                delta_p = (
                    (delta_p_raw - self.cond_stats["delta_p_mean"]) / self.cond_stats["delta_p_std"]
                ).astype(np.float32)
                delta_R = (
                    (delta_R_raw - self.cond_stats["delta_R_mean"]) / self.cond_stats["delta_R_std"]
                ).astype(np.float32)

                s["delta_pose_target_norm"] = np.concatenate(
                    [delta_p.reshape(-1), delta_R.reshape(-1)],
                    axis=0
                ).astype(np.float32)

            else:
                s["p_hist"] = s["p_hist_raw"].astype(np.float32)
                s["R_hist"] = s["R_hist_raw"].astype(np.float32)
                s["fe_hist"] = s["fe_hist_raw"].astype(np.float32)
                s["v_future"] = s["v_future_raw"].astype(np.float32)
                s["pc_hist"] = s["pc_hist_raw"].astype(np.float32)
                s["delta_pose_target_norm"] = s["delta_pose_target"].astype(np.float32)

        print(f"[RollingForceHistoryFMDataset] samples = {len(self.samples)}")
        print(f"[RollingForceHistoryFMDataset] K = {force_hist_len}, H = {pred_horizon}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        p_hist = s["p_hist"]
        R_hist = s["R_hist"]
        fe_hist = s["fe_hist"]                    # [K,6]
        fe_hist_flat = fe_hist.reshape(-1)        # [6K]
        x_hist = np.concatenate([p_hist, R_hist], axis=-1)   # [K, 9]
        x_hist_flat = x_hist.reshape(-1)                     # [9K]

        p_now = s["p_hist"][-1]      # [3]
        R_now = s["R_hist"][-1]      # [6]
        x_now = np.concatenate([p_now, R_now], axis=-1).astype(np.float32)
        cond_hist = np.concatenate([x_hist_flat, fe_hist_flat], axis=-1)   # [12K]
        return (
            torch.from_numpy(cond_hist.astype(np.float32)),   # [6K]
            torch.from_numpy(x_now.astype(np.float32)),   # [6K]
            torch.from_numpy(s["pc_hist"].astype(np.float32)),  # [H,6]
            torch.from_numpy(s["delta_pose_target_norm"].astype(np.float32)),
            torch.from_numpy(s["v_future"].astype(np.float32)),  # [H,6]
        )

    def get_cond_stats(self):
        return self.cond_stats

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
