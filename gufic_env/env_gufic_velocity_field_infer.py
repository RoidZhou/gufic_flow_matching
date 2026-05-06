import os
import sys
import time
import csv
import copy
import math
import random
import pickle
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.linalg import expm
from scipy.spatial.transform import Rotation as RT
from collections import deque
import torch
import torch.nn as nn
# sys.path.append(r"/home/zhou/autolab/GUFIC_mujoco-main")
import open3d as o3d
from gufic_env.flow_matching.model import VelocityRegressiveMLP, VelocityFMTransformer, VisionDeltaPoseNet, VelocityFMTransformer, VelocityFMCondUnet1D
from gufic_env.flow_matching.diffusion_model.vision.pointnet import PointNetBackbone
from gufic_env.flow_matching.dataset import rotmat_batch_to_rot6d, pointcloud_cam_to_world_batch, get_hand_eye_from_xml, uniform_sample_one_frame
from tensorboardX import SummaryWriter

from gufic_env.utils.robot_state import RobotState
from gufic_env.utils.mujoco import set_state, set_body_pose_rotm
from gufic_env.utils.misc_func import *
from gufic_env.flow_matching.infer_fm import load_one_demo 
# ============================================================
# Optional recorder for collecting demos
# ============================================================

class BoltTrajectoryRecorder:
    """
    记录一条 episode 的:
      t, p, R, Vd_star, dVd_star
    保存后自动补充:
      euler, x=[p,euler], goal, total_time
    """
    def __init__(self, save_dir="./bolt_demos"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.reset()

    def reset(self):
        self.records = {
            "t": [],
            "p": [],
            "R": [],
            "euler": [],
            "Vd_star": [],
            "dVd_star": [],
            "Fe": [],
        }

    def add(self, t, p, R, Vd_star, dVd_star):
        p = np.asarray(p).reshape(3)
        R = np.asarray(R).reshape(3, 3)
        Vd_star = np.asarray(Vd_star).reshape(6)
        dVd_star = np.asarray(dVd_star).reshape(6)
        euler = RT.from_matrix(R).as_euler("xyz", degrees=False)

        self.records["t"].append(float(t))
        self.records["p"].append(p.astype(np.float32))
        self.records["R"].append(R.astype(np.float32))
        self.records["euler"].append(euler.astype(np.float32))
        self.records["Vd_star"].append(Vd_star.astype(np.float32))
        self.records["dVd_star"].append(dVd_star.astype(np.float32))

    def __len__(self):
        return len(self.records["t"])

    def save(self, episode_name):
        if len(self) == 0:
            raise ValueError("Recorder is empty. Nothing to save.")

        save_path = os.path.join(self.save_dir, f"{episode_name}.npz")
        data = {}
        for k, v in self.records.items():
            data[k] = np.stack(v, axis=0).astype(np.float32)

        data["x"] = np.concatenate([data["p"], data["euler"]], axis=1).astype(np.float32)
        data["goal"] = np.concatenate([data["p"][-1], data["euler"][-1]], axis=0).astype(np.float32)
        data["total_time"] = np.array([data["t"][-1]], dtype=np.float32)

        np.savez_compressed(save_path, **data)
        print(f"[Recorder] saved demo -> {save_path}")
        return save_path

def normalize_data(data, stats, key="v"):
    return (data - stats[f"{key}_mean"]) / stats[f"{key}_std"]


def denormalize_data(data, stats, key="v"):
    return data * stats[f"{key}_std"] + stats[f"{key}_mean"]

# ============================================================
# Learned velocity field model
# ============================================================

class RobotEnv:
    def __init__(
        self,
        robot_name='indy7',
        model='mlp',
        max_time=20,
        show_viewer=False,
        fz=5,
        observables=None,
        fix_camera=False,
        task='regulation',
        randomized_start=False,
        inertia_shaping=False,
        # ===== FM infer related =====
        use_learned_velocity_field=False,
        fm_ckpt_path="./checkpoints_bolt_real/bolt_velocity_field_best.pt",
        fm_goal=None,
        fm_time_horizon=None,
        fm_lowpass_alpha=0.2,
        # ===== optional demo recording =====
        record_demos=False,
        demo_save_dir="./bolt_demos",
        seed=None,
        save_tensorboard=False,
        test_offline_cond=False,
    ):
        self.robot_name = robot_name
        self.task = task
        self.randomized_start = randomized_start
        self.inertia_shaping = inertia_shaping
        self.policy = model
        self.save_tensorboard = save_tensorboard
        self.test_offline_cond = test_offline_cond

        if observables is not None:
            self.observables = observables
        else:
            self.observables = ['p', 'pd', 'R', 'Rd', 'x_tf', 'x_ti', 'Fe', 'Fe_raw', 'Fd', 'rho']

        demo_path = "/home/zhou/autolab/GUFIC_mujoco-main/bolt_vis_demo/bolt_demo_0000.npz"
        self.demo = load_one_demo(demo_path)
        self.fz = fz
        self.fix_camera = fix_camera
        self.fz_mode = "other"
        self.golbal_steps = 0
        self.writer = SummaryWriter('./gufic/logs')
        self.writer1 = SummaryWriter('./gufic/logs1')
        self.writer2 = SummaryWriter('./gufic/logs2')
        # ==============================
        # Point cloud camera settings
        # ==============================
        self.vis_point = False
        self.camera_name = "eye_in_hand"   # 你 XML 里末端相机的名字
        self.cam_id = -1

        self.rgb_renderer = None
        self.depth_renderer = None

        self.cam_height = 128
        self.cam_width = 128
        self.num_points = 2048

        self.camera_matrix = np.eye(3, dtype=np.float32)
        self.camera_matrix_inv = np.eye(3, dtype=np.float32)
        self.pc_scale = 0.1
        # 每几步采集一次；1 表示每一帧都采集
        self.pointcloud_capture_every = 1
        self.use_pc_color = False
        print('==============================================')
        print('USING GEOMETRIC UNIFED FORCE IMPEDANCE CONTROL')
        print('==============================================')

        self.p_plate = np.array([0.50, 0.00, 0.11])
        self.R_plate = np.array([[0, 1, 0],
                                 [1, 0, 0],
                                 [0, 0, -1]])

        if self.task == 'sphere':
            self.p_plate = np.array([0.40, 0.00, 0.0])

        self.z_init_offset = -0.1

        self.pd_t, self.Rd_t, self.dpd_t, self.dRd_t, self.ddpd_t, self.ddRd_t = initialize_trajectory(task=self.task)

        self.show_viewer = show_viewer
        self.load_xml()

        self.robot_state = RobotState(self.model, self.data, "end_effector", self.robot_name)

        self.dt = self.model.opt.timestep
        self.max_iter = int(max_time / self.dt)
        self.iter = 0
        self.velocity_model = None
        self.prev_Vd_star_fm = np.zeros((6,), dtype=np.float32)
        self.prev_dVd_star_fm = np.zeros((6,), dtype=np.float32)
        self.record_demos = record_demos
        self.R_ec, self.t_ec, _ = get_hand_eye_from_xml(self.robot_name, self.task)

        self.Fe = np.zeros((6, 1))
        self.reset()

        self.Kp, self.KR, self.Kd, self.kp_force, self.kd_force, self.ki_force, self.zeta = set_gains(
            controller='GUFIC', task=self.task
        )

        self.int_sat = 50
        self.e_force_prev = np.zeros((6, 1))
        self.int_force_prev = np.zeros((6, 1))

        self.T_f_low = 0.5
        self.T_f_high = 20
        self.delta_f = 1

        self.T_i_low = 0.5
        self.T_i_high = 20
        self.delta_i = 1

        T_i_init = 10
        T_f_init = 10

        if self.task == 'sphere':
            T_i_init = 90
            self.T_i_high = 100

        self.x_tf = np.sqrt(2 * T_f_init)
        self.x_ti = np.sqrt(2 * T_i_init)

        self.T_f = 0.5 * self.x_tf**2
        self.T_i = 0.5 * self.x_ti**2

        self.d_max = 0.03
        self.eR_norm_max = 0.05

        self.Ff_list = []
        self.Vb_list = []
        self.Ff_activation = []
        self.rho_list = []
        self.Fd_star_list = []
        self.Fi_activation = []

        # ======================================================
        # FM infer related
        # ======================================================
        self.use_learned_velocity_field = use_learned_velocity_field
        self.fm_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fm_ckpt_path = fm_ckpt_path
        self.fm_time_horizon = fm_time_horizon if fm_time_horizon is not None else max_time
        self.fm_lowpass_alpha = fm_lowpass_alpha
        self.pred_vt = None
        self.seed = seed
        # 默认 goal：沿用任务起始时刻参考位姿。
        # 也可以在外部直接传入 fm_goal=[p_goal(3), euler_goal(3)]
        if fm_goal is None:
            pd0 = self.pd_t(0).reshape(-1)
            Rd0 = self.Rd_t(0)
            euler_goal = RT.from_matrix(Rd0).as_euler("xyz", degrees=False).astype(np.float32)
            self.fm_goal = np.concatenate([pd0.astype(np.float32), euler_goal], axis=0)
        else:
            self.fm_goal = np.asarray(fm_goal, dtype=np.float32).reshape(6)

        if self.use_learned_velocity_field:
            self._load_fm_model()

        # ======================================================
        # optional demo recorder
        # ======================================================
        self.episode_idx = 0
        self.demo_recorder = BoltTrajectoryRecorder(save_dir=demo_save_dir) if record_demos else None

    def _load_fm_model(self):
        if not os.path.exists(self.fm_ckpt_path):
            raise FileNotFoundError(f"FM checkpoint not found: {self.fm_ckpt_path}")

        ckpt = torch.load(self.fm_ckpt_path, map_location=self.fm_device, weights_only=False)
        if self.policy == 'mlp':
            train_cfg = ckpt["train_cfg"]
            self.velocity_model = VelocityRegressiveMLP(
                x_dim=6,
                cond_dim=6,
                time_dim=train_cfg["time_dim"],
                hidden_dim=train_cfg["hidden_dim"],
                num_layers=train_cfg["num_layers"],
            ).to(self.fm_device)
        elif self.policy == 'transformer':
            train_cfg = ckpt["train_cfg"]
            self.use_cond = train_cfg["use_cond"]
            self.force_hist_len = train_cfg["force_hist_len"]
            self.x_hist_len = train_cfg["x_hist_len"]
            self.pc_hist_len = train_cfg["pc_hist_len"]
            self.pred_horizon = train_cfg["pred_horizon"]
            # self.steps = train_cfg.steps
            self.steps =10
            self.fe_queue = deque(maxlen=self.force_hist_len)

            self.velocity_model = VelocityFMTransformer(
                x_dim=6,
                cond_dim=train_cfg["cond_dim"],
                time_dim=train_cfg["time_dim"],
                hidden_dim=train_cfg["hidden_dim"],
                num_layers=train_cfg["num_layers"],
                use_cond=self.use_cond,
            ).to(self.fm_device)
        elif self.policy == 'unet':
            train_cfg = ckpt["train_cfg"]
            self.velocity_model = VelocityFMCondUnet1D(
                x_dim=6,
                cond_dim=train_cfg["cond_dim"],
                time_dim=train_cfg["time_dim"],
                kernel_size=5,
                use_cond=False,
            ).to(self.fm_device)
            
        self.velocity_model.load_state_dict(ckpt["model"])
        self.velocity_model.eval()

        self.obs_encoder = VisionDeltaPoseNet(
            state_dim=train_cfg["state_dim"],
            guide_dim=train_cfg["guide_dim"],
            embed_dim=train_cfg["embed_dim"],
            input_channels=train_cfg["input_channels"],
            input_transform=train_cfg["input_transform"],
        ).to(self.fm_device)
        self.obs_encoder.load_state_dict(ckpt["obs_encoder"])
        self.obs_encoder.eval()

        print(f"[FM] loaded checkpoint from {self.fm_ckpt_path}")
        print(f"[FM] goal = {self.fm_goal}")

        if self.policy == 'transformer' or self.policy == 'unet':
            if self.use_cond:
                self.stats = ckpt.get("cond_stats", None)
            else:
                steps = 100
                dt = 1.0 / steps
                traj_len = self.max_iter
                stats = ckpt.get("v_stats", None)
                # 初始噪声：速度轨迹样本（normalized space）
                v_t = torch.randn(1, traj_len, 6, device=self.fm_device)
                with torch.no_grad():
                    for i in range(steps):
                        # flow time，对整条轨迹共用一个标量
                        t_value = torch.full(
                            (1, 1, 1),
                            i / steps,
                            device=v_t.device,
                            dtype=v_t.dtype
                        )

                        # 模型输出的是 normalized space 里的 flow velocity
                        u_pred = self.velocity_model(x_t=v_t, t=t_value)   # [1, T, 6]

                        # Euler 更新
                        v_t = v_t + u_pred * dt
                
                    v_sample = v_t.squeeze(0).detach().cpu().numpy().astype(np.float32)   # [T,6]
                    self.Vd_star = denormalize_data(v_sample, stats, "v").astype(np.float32)


    def _build_fm_state(self, p, R):
        euler = RT.from_matrix(R).as_euler("xyz", degrees=False).astype(np.float32)
        x_t = np.concatenate([np.asarray(p, dtype=np.float32).reshape(3), euler], axis=0)
        return x_t

    @torch.no_grad()
    def get_learned_velocity_field(self, p, R, t, Fe, point_cloud):
        if self.policy == 'mlp':
            """
            用训练好的速度场模型 infer:
            Vd_star = model(x_t, t, goal)
            再用差分得到 dVd_star。
            """
            x_t_np = self._build_fm_state(p, R)
            goal_np = self.fm_goal

            t_norm = np.array([[t / max(self.fm_time_horizon, 1e-8)]], dtype=np.float32)

            x_t = torch.from_numpy(x_t_np[None, :]).to(self.fm_device)
            goal = torch.from_numpy(goal_np[None, :]).to(self.fm_device)
            Fe = torch.from_numpy(Fe[None, :]).to(self.fm_device)
            Fe_t = Fe.squeeze(-1).to(torch.float32)
            t_tensor = torch.from_numpy(t_norm).to(self.fm_device)

            Vd_star = self.velocity_model(x_t, t_tensor, Fe_t).cpu().numpy().reshape(6).astype(np.float32)

            if self.iter == 0:
                dVd_star = np.zeros((6,), dtype=np.float32)
            else:
                dVd_star = (Vd_star - self.prev_Vd_star_fm) / max(self.dt, 1e-8)

            # 轻微低通，避免差分加速度过抖
            alpha = self.fm_lowpass_alpha
            dVd_star = alpha * dVd_star + (1.0 - alpha) * self.prev_dVd_star_fm

            self.prev_Vd_star_fm = Vd_star.copy()
            self.prev_dVd_star_fm = dVd_star.copy()
        elif self.policy == 'transformer' or self.policy == 'unet':
            if self.use_cond:
                if self.seed is not None:
                    torch.manual_seed(self.seed)
                    torch.cuda.manual_seed_all(self.seed)

                dt = 1.0 / self.steps
                # 初始噪声：速度轨迹样本（normalized space）
                traj_len = self.pred_horizon
                if self.pred_vt is None:
                    v_t = torch.randn(1, traj_len, 6, device=self.fm_device)
                else:
                    pred = self.pred_vt.copy()
                    shift = np.concatenate([pred[1:], pred[-1:]], axis=0)
                    v_t = torch.from_numpy(shift[None, :]).to(self.fm_device).float()
                            
                R6d = rotmat_batch_to_rot6d(R)                        # [T,3,3]

                p_raw = p
                R_raw = R
                # 对 p 做归一化
                p = normalize_data(p, self.stats, "p").astype(np.float32)
                # 对 R 做归一化
                R6d = normalize_data(R6d, self.stats, "R").astype(np.float32)
                cond_x = np.concatenate([p, R6d], axis=-1)        # [T,9]

                # 对 fe 做归一化
                Fe = normalize_data(Fe.reshape(-1, 6), self.stats, "fe").astype(np.float32)
                self.fe_queue.append(Fe.squeeze(0))
                fe_cond = np.stack(list(self.fe_queue), axis=0)   # [n,6]

                if self.use_pc_color:
                    cond_pc = point_cloud.astype(np.float32)
                else:
                    cond_pc = point_cloud[:, :3].astype(np.float32)
                cond_pc = cond_pc[None, :, :]   # [1, P, C]

                if fe_cond.shape[0] < self.force_hist_len:
                    # 如果不足 K 步历史，就在前面补零
                    pad_len = self.force_hist_len - fe_cond.shape[0]
                    fe_cond = np.pad(fe_cond, ((pad_len, 0), (0, 0)), mode="constant")
                if cond_x.shape[0] < self.x_hist_len:
                    # 如果整个序列都不足 K 步，就在前面补零
                    pad_len = self.x_hist_len - cond_x.shape[0]
                    pad = np.repeat(cond_x[0:1], pad_len, axis=0)
                    cond_x = np.concatenate([pad, cond_x], axis=0)
                if cond_pc.shape[0] < self.pc_hist_len:
                    # 如果整个序列都不足 K 步，就在前面补pc
                    pad_len = self.pc_hist_len - cond_pc.shape[0]
                    pad_pc = np.repeat(cond_pc[0:1], pad_len, axis=0)
                    cond_pc = np.concatenate([pad_pc, cond_pc], axis=0)

                pc_world = pointcloud_cam_to_world_batch(cond_pc, p_raw, R_raw, self.R_ec, self.t_ec)
                pc_ee = np.einsum("tji,tpj->tpi", R_raw, pc_world[..., :3] - p_raw[:, None, :])  # R^T (x_w - p)
                cond_pc = pc_ee / 0.1
                cond_pc = (cond_pc / 0.1).astype(np.float32)

                cond = np.concatenate([cond_x.reshape(1, -1), fe_cond.reshape(1, -1)], axis=-1)  # [1, cond_dim]
                # cond: [K,6] -> [6K]
                if cond.ndim == 2:
                    cond = cond.reshape(-1)

                # cond: [6K] -> [1,6K]
                if cond.ndim == 1:
                    cond = cond[None, :]

                cond = torch.from_numpy(cond).to(self.fm_device).float()
                x_now = cond[:, :9]
                cond_pc = torch.from_numpy(cond_pc).to(self.fm_device).float()
                cond_pc = cond_pc.unsqueeze(0)   # [B, P, C]
                # pc_feat = self.obs_encoder(cond_pc)   # [1, embed_dim]
                # cond = torch.cat([cond, pc_feat], dim=-1)  # [1, cond_dim]
                guide_feat, delta_pose_pred = self.obs_encoder(cond_pc, x_now)   # [B,guide_dim], [B,9]
                # cond = torch.cat([cond, guide_feat], dim=-1)         # [B, cond_dim]

                for i in range(self.steps):
                    # flow time，对整条轨迹共用一个标量
                    t_value = torch.full(
                        (1, 1, 1),
                        i / self.steps,
                        device=v_t.device,
                        dtype=v_t.dtype
                    )

                    # 内部会自动扩成 [B, T, cond_dim]
                    # u_pred = self.velocity_model(x_t=v_t, t=t_value, fe=cond)   # [1, T, 6]
                    u_pred = self.velocity_model(
                        x_t=v_t,
                        t=t_value,
                        cond_main=cond,
                        guide=guide_feat,
                    )
                    # Euler 更新
                    v_t = v_t + u_pred * dt
            
                v_sample = v_t.squeeze(0).detach().cpu().numpy().astype(np.float32)   # [T,6]
                Vd_star_horizon = denormalize_data(v_sample, self.stats, "v").astype(np.float32)
                Vd_star = Vd_star_horizon[0]  # 取第一步的速度作为当前时刻的输出
                # self.pred_vt = v_sample.copy()

                if self.iter == 0:
                    dVd_star = np.zeros((6,), dtype=np.float32)
                else:
                    dVd_star = (Vd_star - self.prev_Vd_star_fm) / max(self.dt, 1e-8)

                # 轻微低通，避免差分加速度过抖
                alpha = self.fm_lowpass_alpha
                dVd_star = alpha * dVd_star + (1.0 - alpha) * self.prev_dVd_star_fm
                self.prev_Vd_star_fm = Vd_star.copy()
                self.prev_dVd_star_fm = dVd_star.copy() 
            else:
                """
                用训练好的速度场模型 infer:
                Vd_star = model(x_t, t)
                再用差分得到 dVd_star。
                """
                Vd_star = self.Vd_star[self.iter]
                if self.iter == 0:
                    dVd_star = np.zeros((6,), dtype=np.float32)
                else:
                    dVd_star = (Vd_star - self.prev_Vd_star_fm) / max(self.dt, 1e-8)
                
                # 轻微低通，避免差分加速度过抖
                alpha = self.fm_lowpass_alpha
                dVd_star = alpha * dVd_star + (1.0 - alpha) * self.prev_dVd_star_fm

                self.prev_Vd_star_fm = Vd_star.copy()
                self.prev_dVd_star_fm = dVd_star.copy()

        return Vd_star, dVd_star

    def load_xml(self):
        dir = os.getcwd() + '/'
        if self.robot_name == 'ur5e':
            raise NotImplementedError

        elif self.robot_name == 'indy7':
            if self.task == "sphere":
                model_path = dir + "gufic_env/mujoco_models/Indy7_wiping_sphere.xml"
            else:
                model_path = dir + "gufic_env/mujoco_models/Indy7_wiping.xml"

        elif self.robot_name == 'panda':
            raise NotImplementedError

        else:
            raise NotImplementedError

        self.model = mujoco.MjModel.from_xml_path(model_path)
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

    def init_camera_renderer(self):
        """
        初始化 RGB 和 Depth renderer，并计算相机内参。
        """
        self.cam_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.camera_name
        )

        if self.cam_id < 0:
            raise ValueError(f"Camera '{self.camera_name}' not found in XML.")

        self.rgb_renderer = mujoco.Renderer(
            self.model,
            height=self.cam_height,
            width=self.cam_width
        )

        self.depth_renderer = mujoco.Renderer(
            self.model,
            height=self.cam_height,
            width=self.cam_width
        )
        self.depth_renderer.enable_depth_rendering()

        # MuJoCo cam_fovy 单位是 degree
        fovy = np.deg2rad(self.model.cam_fovy[self.cam_id])

        fy = self.cam_height / (2.0 * np.tan(fovy / 2.0))
        fx = fy
        cx = self.cam_width / 2.0
        cy = self.cam_height / 2.0

        self.camera_matrix = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix).astype(np.float32)

        print(f"[Camera] name={self.camera_name}, id={self.cam_id}, fovy={np.rad2deg(fovy):.2f}")
        print(f"[Camera] K=\n{self.camera_matrix}")

    def get_camera_rgbd(self):
        """
        从指定相机采集 RGB 和深度图。
        """
        self.rgb_renderer.update_scene(self.data, camera=self.cam_id)
        rgb = self.rgb_renderer.render()

        self.depth_renderer.update_scene(self.data, camera=self.cam_id)
        depth = self.depth_renderer.render()

        return rgb, depth

    def rgbd_to_point_cloud(self, rgb, depth):
        """
        将 RGB-D 图像反投影为点云。
        输出 shape: [N, 6]，每个点为 [x, y, z, r, g, b]
        注意：这里的 xyz 是相机坐标系下的点云。
        """
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        ones = np.ones_like(u, dtype=np.float32)
        pixels = np.stack([u, v, ones], axis=-1).reshape(-1, 3).astype(np.float32)

        z = depth.reshape(-1, 1).astype(np.float32)

        xyz = (pixels @ self.camera_matrix_inv.T) * z
        rgb_flat = rgb.reshape(-1, 3).astype(np.float32)

        point_cloud = np.concatenate([xyz, rgb_flat], axis=1)

        # 过滤无效点和过远点
        valid = np.isfinite(point_cloud).all(axis=1)
        valid &= point_cloud[:, 2] > 0.0
        valid &= point_cloud[:, 2] < 0.3

        point_cloud = point_cloud[valid]

        return self.uniform_sample_point_cloud(point_cloud)

    def uniform_sample_point_cloud(self, point_cloud):
        """
        随机采样固定数量点。
        如果有效点不足，则允许重复采样。
        """
        if point_cloud.shape[0] == 0:
            return np.zeros((self.num_points, 6), dtype=np.float32)

        replace = point_cloud.shape[0] < self.num_points
        idx = np.random.choice(
            point_cloud.shape[0],
            size=self.num_points,
            replace=replace
        )

        return point_cloud[idx].astype(np.float32)

    def capture_point_cloud(self):
        """
        采集当前帧点云。
        """
        rgb, depth = self.get_camera_rgbd()
        point_cloud = self.rgbd_to_point_cloud(rgb, depth)
        return point_cloud

    def visualize_point_cloud_once(self):
        """
        采集当前相机的一帧点云并用 Open3D 可视化。
        点云格式：[x, y, z, r, g, b]
        """
        point_cloud = self.capture_point_cloud()

        if point_cloud is None or point_cloud.shape[0] == 0:
            print("[PointCloud] Empty point cloud.")
            return

        xyz = point_cloud[:, :3]
        rgb = point_cloud[:, 3:] / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        print("[PointCloud] shape:", point_cloud.shape)
        print("[PointCloud] xyz min:", xyz.min(axis=0))
        print("[PointCloud] xyz max:", xyz.max(axis=0))

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1,
            origin=[0, 0, 0]
        )

        o3d.visualization.draw_geometries(
            [pcd, frame],
            window_name="Eye-in-hand Point Cloud",
            width=960,
            height=720
        )
    
    def reset(self, angle_prefix=None):
        self.iter = 0
        self.prev_Vd_star_fm[:] = 0.0
        self.prev_dVd_star_fm[:] = 0.0

        pd = self.pd_t(0)
        Rd = self.Rd_t(0)

        if self.randomized_start:
            rand_xy = 2 * (np.random.rand(2,) - 0.5) * 0.05
            rand_rpy = 2 * (np.random.rand(3,) - 0.5) * 15 / 180 * np.pi
        else:
            rand_xy = np.array([0.05, -0.05])
            rand_rpy = np.array([15, -15, 15]) * np.pi / 180

        Rx = np.array([[1, 0, 0], [0, np.cos(rand_rpy[0]), -np.sin(rand_rpy[0])], [0, np.sin(rand_rpy[0]), np.cos(rand_rpy[0])]])
        Ry = np.array([[np.cos(rand_rpy[1]), 0, np.sin(rand_rpy[1])], [0, 1, 0], [-np.sin(rand_rpy[1]), 0, np.cos(rand_rpy[1])]])
        Rz = np.array([[np.cos(rand_rpy[2]), -np.sin(rand_rpy[2]), 0], [np.sin(rand_rpy[2]), np.cos(rand_rpy[2]), 0], [0, 0, 1]])

        p_init = pd.reshape((-1, 1)) + Rd @ np.array([rand_xy[0], rand_xy[1], self.z_init_offset]).reshape(-1, 1)
        R_init = Rd @ Rz @ Ry @ Rx

        p_init = p_init.reshape((-1,))

        if self.model.nv == 8:
            q0 = np.array([0, 0, -np.pi / 2, 0, -np.pi / 2, np.pi / 2, 0, 0])
        elif self.model.nv == 6:
            q0 = np.array([0, 0, -np.pi / 2, 0, -np.pi / 2, np.pi / 2])

        self.robot_state.gauss_newton_IK(p_init, R_init, q0)

        self.Fe = np.zeros((6, 1))
        obs = np.zeros((6, 1))

        Rt = np.eye(3)
        self.set_hole_pose(self.p_plate, Rt)

        self.robot_state.update()

        p, R = self.robot_state.get_pose()

        self.gd = np.eye(4)
        self.gd[:3, 3] = p
        self.gd[:3, :3] = R

        if self.show_viewer:
            self.viewer.sync()

        if self.record_demos and self.demo_recorder is not None:
            self.demo_recorder.reset()

        print('Initialization Complete')
        time.sleep(2)

        return obs

    def run(self):
        p_list = []
        R_list = []
        x_tf_list = []
        x_ti_list = []
        Fe_list = []
        Fd_list = []
        Fe_raw_list = []
        pd_list = []

        for i in range(self.max_iter):
            self.golbal_steps = i

            pd, Rd, vd, wd, dvd, dwd = self.update_desired_trajectory()
            obs, reward, done, info = self.step()

            p, R = self.robot_state.get_pose()
            Fe = self.get_FT_value()
            Fe_raw = self.get_FT_value_raw()

            p_list.append(p)
            R_list.append(R)
            x_tf_list.append(self.x_tf)
            x_ti_list.append(self.x_ti)
            Fe_list.append(Fe)
            Fe_raw_list.append(Fe_raw)
            Fd_list.append(0)
            pd_list.append(pd)

            if self.show_viewer:
                if i % 10 == 0:
                    self.viewer.sync()

            if i % 1000 == 0:
                print(f"Time Step: {i}")

            if done:
                break

        # optional save demo after one run
        if self.record_demos and self.demo_recorder is not None and len(self.demo_recorder) > 0:
            self.demo_recorder.save(f"bolt_demo_{self.episode_idx:04d}")
            self.episode_idx += 1

        return p_list, R_list, x_tf_list, x_ti_list, Fe_list, Fd_list, pd_list, Fe_raw_list

    def update_desired_trajectory(self):
        t = self.iter * self.dt
        pd = self.pd_t(t)
        Rd = self.Rd_t(t)

        dpd = self.dpd_t(t)
        dRd = self.dRd_t(t)

        ddpd = self.ddpd_t(t)
        ddRd = self.ddRd_t(t)

        vd = Rd.T @ dpd.reshape((-1, 1))
        wd = vee_map(Rd.T @ dRd)

        dvd = Rd.T @ ddpd.reshape((-1, 1)) - hat_map(wd) @ Rd.T @ dpd.reshape((-1, 1))
        dwd = vee_map(Rd.T @ ddRd - hat_map(wd) @ Rd.T @ dRd)

        return pd.reshape((-1,)), Rd, vd.reshape((-1,)), wd.reshape((-1,)), dvd.reshape((-1,)), dwd.reshape((-1,))

    def get_velocity_field(self, g, V, t):
        zeta = self.zeta
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
        vd_star = R.T @ dRd @ Rd.T @ (p - pd) + R.T @ dpd - zeta * R.T @ (p - pd)
        wd_star = vee_map(R.T @ dRd @ Rd.T @ R - zeta * (Rd.T @ R - R.T @ Rd)).reshape((-1,))

        Vd_star[:3] = vd_star
        Vd_star[3:] = wd_star

        dVd_star = np.zeros(6,)
        term1 = -hat_map(w) @ R.T @ dRd @ Rd.T @ R + R.T @ ddRd @ Rd.T @ R + R.T @ dRd @ dRd.T @ R + R.T @ dRd @ Rd.T @ R @ hat_map(w)
        term2 = -hat_map(w) @ R.T @ dRd @ Rd.T @ (p - pd) + R.T @ ddRd @ Rd.T @ (p - pd) + R.T @ dRd @ dRd.T @ (p - pd) \
                + R.T @ dRd @ Rd.T @ (R.T @ v - pd) - hat_map(w) @ R.T @ dpd + R.T @ ddpd
        term3 = dRd.T @ R + Rd.T @ R @ hat_map(w) + hat_map(w) @ R.T @ Rd - R.T @ dRd
        term4 = -hat_map(w) @ R.T @ (p - pd) + v - R.T @ dpd
        dvd_star = term2 - zeta * term4
        dwd_star = vee_map(term1 - zeta * term3).reshape((-1,))

        dVd_star[:3] = dvd_star
        dVd_star[3:] = dwd_star

        return Vd_star, dVd_star

    def torque_to_velocity(self, tau, dt):
        q = self.data.qpos.copy()[:self.model.nv]
        dq = self.data.qvel.copy()[:self.model.nv]

        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)

        qfrc_bias = self.data.qfrc_bias.copy()[:self.model.nv]
        tau_array = np.array(tau).reshape(-1,)
        ddq = np.linalg.pinv(M) @ (tau_array - qfrc_bias)
        dq_new = dq + ddq * dt
        return dq_new

    def step(self):
        self.robot_state.update()

        tau_cmd = self.geometric_unified_force_impedance_control()
        V = self.torque_to_velocity(tau_cmd, self.dt)
        gripper = 0.03

        self.robot_state.set_control_torque(tau_cmd, gripper)
        self.robot_state.update_dynamic()

        if self.show_viewer:
            self.viewer.sync()

        obs = {}
        p, R = self.robot_state.get_pose()

        pd = self.pd_t(self.iter * self.dt).reshape((-1,))
        Rd = self.Rd_t(self.iter * self.dt)

        for obs_name in self.observables:
            if obs_name == 'p':
                obs['p'] = p
            elif obs_name == 'pd':
                obs['pd'] = pd
            elif obs_name == 'R':
                obs['R'] = R
            elif obs_name == 'Rd':
                obs['Rd'] = Rd
            elif obs_name == 'x_tf':
                obs['x_tf'] = self.x_tf
            elif obs_name == 'x_ti':
                obs['x_ti'] = self.x_ti
            elif obs_name == 'Fe':
                obs['Fe'] = self.get_FT_value()
            elif obs_name == 'Fe_raw':
                obs['Fe_raw'] = self.get_FT_value_raw()
            elif obs_name == 'Fd':
                obs['Fd'] = self.get_force_field(self.gd, self.gd)
            elif obs_name == 'rho':
                obs['rho'] = self.rho
            elif obs_name == 'Psi':
                obs['Psi'] = 0.5 * np.linalg.norm(p - pd)**2 + np.trace(np.eye(3) - Rd.T @ R)
            else:
                raise ValueError('Invalid observable name')

        done = self.iter == self.max_iter - 1
        reward = 0
        info = dict()

        self.iter += 1
        return obs, reward, done, info

    def get_FT_value(self, return_derivative=False):
        Fe, dFe = self.robot_state.get_ee_force()
        if return_derivative:
            return -Fe, -dFe
        else:
            return -Fe

    def get_FT_value_raw(self):
        Fe, dFe = self.robot_state.get_ee_force_raw()
        return -Fe

    def get_eg(self, g, gd):
        p = g[:3, 3]
        R = g[:3, :3]

        pd = gd[:3, 3]
        Rd = gd[:3, :3]

        ep = R.T @ (p - pd)
        eR = vee_map(Rd.T @ R - R.T @ Rd).reshape((-1,))

        return np.hstack((ep, eR)).reshape((-1, 1))

    def get_force_field(self, g, gd):
        fz = self.fz
        Fd = np.array([0, 0, fz, 0, 0, 0])
        return Fd

    def geometric_unified_force_impedance_control(self):
        time_start = time.time()
        Jb = self.robot_state.get_body_jacobian()
        qfrc_bias = self.robot_state.get_bias_torque()
        M = self.robot_state.get_full_inertia()

        Kp = self.Kp
        KR = self.KR

        p, R = self.robot_state.get_pose()

        g = np.eye(4)
        g[:3, :3] = R
        g[:3, 3] = p

        Vb = self.robot_state.get_body_ee_velocity()

        Fe, d_Fe = self.get_FT_value(return_derivative=True)
        print("Fe : ", Fe)
        # print("d_Fe : ", d_Fe)
        Fe = Fe.reshape((-1, 1))
        d_Fe = d_Fe.reshape((-1, 1))
        if self.save_tensorboard:
            force_x = Fe[0]
            force_y = Fe[1]
            force_z = Fe[2]
            torque_x = Fe[3]
            torque_y = Fe[4]
            torque_z = Fe[5]
            self.writer.add_scalars("force_x",
                                    {"force_x": force_x}, self.golbal_steps)
            self.writer.add_scalars("force_y",
                                    {"force_y": force_y}, self.golbal_steps)
            self.writer.add_scalars("force_z",
                                    {"force_z": force_z}, self.golbal_steps)
            self.writer.add_scalars("torque_x",
                                    {"torque_x": torque_x}, self.golbal_steps)
            self.writer.add_scalars("torque_y",
                                    {"torque_y": torque_y}, self.golbal_steps)
            self.writer.add_scalars("torque_z",
                                    {"torque_z": torque_z}, self.golbal_steps)  
        if self.test_offline_cond:
            cond_fe = self.demo["fe"][self.golbal_steps].reshape(1, -1)      # [K,6]，滚动取最近 K 步 (Fe + x) 作为条件
            cond_R = np.expand_dims(self.demo["R"][self.golbal_steps], axis=0)  # [K,3,3] -> [K,6]
            cond_p = self.demo["p"][self.golbal_steps].reshape(1, -1)
        else:
            cond_fe = Fe
            cond_p = p.reshape(1, -1)
            cond_R = R.reshape(1, 3, 3)
        # pointcloud
        if self.iter % self.pointcloud_capture_every == 0:
            point_cloud = self.capture_point_cloud()
            if self.use_pc_color:
                point_cloud = point_cloud.astype(np.float32)
            else:
                point_cloud = point_cloud[:, :3].astype(np.float32)
            
        else:
            point_cloud = None
        # ======================================================
        # Source of desired velocity field
        # ======================================================
        current_t = self.iter * self.dt
        if self.use_learned_velocity_field:
            Vd_star, dVd_star = self.get_learned_velocity_field(cond_p, cond_R, current_t, cond_fe, point_cloud)
        else:
            Vd_star, dVd_star = self.get_velocity_field(g, Vb.reshape((-1,)), t=current_t)

        # optional recording of actual desired field used by controller
        if self.record_demos and self.demo_recorder is not None:
            self.demo_recorder.add(
                t=current_t,
                p=p,
                R=R,
                Vd_star=Vd_star,
                dVd_star=dVd_star,
            )
        if self.save_tensorboard:
            self.writer.add_scalars("Vd_star_x", {"Vd_star_x": Vd_star[0]}, self.golbal_steps)
            self.writer.add_scalars("Vd_star_y", {"Vd_star_y": Vd_star[1]}, self.golbal_steps)
            self.writer.add_scalars("Vd_star_z", {"Vd_star_z": Vd_star[2]}, self.golbal_steps)
            self.writer.add_scalars("Vd_star_rx", {"Vd_star_rx": Vd_star[3]}, self.golbal_steps)
            self.writer.add_scalars("Vd_star_ry", {"Vd_star_ry": Vd_star[4]}, self.golbal_steps)
            self.writer.add_scalars("Vd_star_rz", {"Vd_star_rz": Vd_star[5]}, self.golbal_steps)

        Vd_star = Vd_star.reshape((-1, 1))
        dVd_star = dVd_star.reshape((-1, 1))

        gd = self.gd
        Rd = gd[:3, :3]
        pd = gd[:3, 3]

        g_ed = np.linalg.inv(g) @ gd

        fp = R.T @ Rd @ Kp @ Rd.T @ (p - pd).reshape((-1, 1))
        fR = vee_map(KR @ Rd.T @ R - R.T @ Rd @ KR)
        fg = np.vstack((fp, fR))

        gd_bar = np.eye(4)
        t = self.iter * self.dt
        gd_bar[:3, :3] = self.Rd_t(t)
        gd_bar[:3, 3] = self.pd_t(t).reshape((-1,))
        Fd_star = self.get_force_field(g, gd_bar).reshape((-1, 1))

        e_force = -Fe - Fd_star
        de_force = -d_Fe
        int_force = self.int_force_prev + e_force * self.dt
        int_force = np.clip(int_force, -self.int_sat, self.int_sat)

        if self.fz_mode == "time-varying":
            F_f = - self.kp_force * e_force - self.kd_force * de_force - self.ki_force * int_force + Fd_star
        else:
            F_f = - self.kp_force * (-Fe) - self.ki_force * int_force - self.kd_force * de_force + Fd_star

        f_d = Fd_star[:3].reshape((-1,))
        m_d = Fd_star[3:].reshape((-1,))

        t = self.iter * self.dt
        gd_t = np.eye(4)
        gd_t[:3, :3] = self.Rd_t(t)
        gd_t[:3, 3] = self.pd_t(t).reshape((-1,))
        eg = self.get_eg(g, gd_t)
        if self.save_tensorboard:
            self.writer1.add_scalars("p_x", {"p_x": p[0]}, self.golbal_steps)
            self.writer1.add_scalars("p_y", {"p_y": p[1]}, self.golbal_steps)
            self.writer1.add_scalars("p_z", {"p_z": p[2]}, self.golbal_steps)

            self.writer2.add_scalars("p_x", {"pd_x": self.pd_t(t)[0]}, self.golbal_steps)
            self.writer2.add_scalars("p_y", {"pd_y": self.pd_t(t)[1]}, self.golbal_steps)
            self.writer2.add_scalars("p_z", {"pd_z": self.pd_t(t)[2]}, self.golbal_steps)

        ep = eg[:3, 0]
        eR = eg[3:, 0]

        rho_p = np.zeros((3,))
        rho_R = np.zeros((3,))

        if ep @ f_d <= 0:
            rho_p[:3] = 1
        elif ep @ f_d > 0:
            for i in range(3):
                if np.abs(ep[i]) <= self.d_max:
                    rho_p[i] = 0.5 * (1 + np.cos(np.pi * ep[i] / self.d_max))
                elif np.abs(f_d[i]) <= 0.05:
                    rho_p[i] = 0
        else:
            rho_p[:3] = 0

        eR_norm = np.linalg.norm(eR)
        if eR @ m_d <= 0:
            rho_R[:3] = 1
        elif eR @ m_d > 0:
            if eR_norm >= self.eR_norm_max:
                rho_R[:3] = 0.5 * (1 + np.cos(np.pi * eR_norm / self.eR_norm_max))
        else:
            rho_R[:3] = 0

        # self.writer.add_scalars("rho_Rx", {"rho_Rx": rho_R[0]}, self.golbal_steps)
        # self.writer.add_scalars("rho_Ry", {"rho_Ry": rho_R[1]}, self.golbal_steps)
        # self.writer.add_scalars("rho_Rz", {"rho_Rz": rho_R[2]}, self.golbal_steps)

        rho = np.block([rho_p, rho_R]).reshape((-1, 1))
        self.rho = rho
        F_f = F_f * rho

        self.e_force_prev = e_force
        self.int_force_prev = int_force

        inner_product_f = (Vb.T @ F_f).reshape((-1,))[0]

        self.T_f = 0.5 * self.x_tf**2
        gamma_f = 1 if inner_product_f < 0 else 0
        beta_f = 1 if self.T_f <= self.T_f_high else 0

        if self.T_f >= self.T_f_low + self.delta_f:
            alpha_f = 1
        elif self.T_f_low <= self.T_f <= self.T_f_low + self.delta_f:
            alpha_f = 0.5 * (1 - np.cos(np.pi * (self.T_f - self.T_f_low) / self.delta_f))
        else:
            alpha_f = 0

        dx_tf = - (beta_f / self.x_tf) * gamma_f * inner_product_f + (alpha_f / self.x_tf) * (gamma_f - 1) * inner_product_f
        self.x_tf = self.x_tf + dx_tf * self.dt
        self.T_f = 0.5 * self.x_tf**2

        activation_force = gamma_f + alpha_f * (1 - gamma_f)
        F_f_mod = activation_force * F_f

        inner_product_i = (Vd_star.T @ (F_f_mod + Fe)).reshape((-1,))[0]
        self.T_i = 0.5 * self.x_ti**2

        gamma_i = 1 if inner_product_i > 0 else 0
        beta_i = 1 if self.T_i <= self.T_i_high else 0

        if self.T_i >= self.T_i_low + self.delta_i:
            alpha_i = 1
        elif self.T_i_low <= self.T_i <= self.T_i_low + self.delta_i:
            alpha_i = 0.5 * (1 - np.cos(np.pi * (self.T_i - self.T_i_low) / self.delta_i))
        else:
            alpha_i = 0

        activation_impedance = gamma_i + alpha_i * (1 - gamma_i)
        Vd_star_mod = activation_impedance * Vd_star
        dVd_star_mod = activation_impedance * dVd_star
        ev_mod = Vb - Vd_star_mod
        if self.save_tensorboard:
            self.writer.add_scalars("activation_impedance", {"activation_impedance": activation_impedance}, self.golbal_steps)

        Vd_mod = adjoint_g_ed(np.linalg.inv(g_ed)) @ Vd_star_mod
        Vd_mod_hat = np.zeros((4, 4))
        Vd_mod_hat[:3, :3] = hat_map(Vd_mod[3:, 0])
        Vd_mod_hat[:3, 3] = Vd_mod[:3, 0]
        self.gd = gd @ expm(Vd_mod_hat * self.dt)

        Kd = self.Kd
        energy_dissipation = (ev_mod.T @ Kd @ ev_mod)[0, 0]
        if energy_dissipation > 10:
            energy_dissipation = 0.1

        dx_ti = (beta_i / self.x_ti) * (gamma_i * inner_product_i + energy_dissipation) \
                + (alpha_i / self.x_ti) * (1 - gamma_i) * inner_product_i
        self.x_ti = self.x_ti + dx_ti * self.dt

        M_tilde_inv = Jb @ np.linalg.pinv(M) @ Jb.T
        M_tilde = np.linalg.pinv(M_tilde_inv)
        M_d = np.eye(6) * 10

        Fe_raw = self.get_FT_value_raw().reshape((-1, 1))
        if self.inertia_shaping:
            tau_tilde = M_tilde @ (dVd_star_mod + np.linalg.inv(M_d) @ (- Kd @ ev_mod - fg + F_f_mod + Fe_raw)) - Fe_raw
        else:
            tau_tilde = M_tilde @ dVd_star_mod - Kd @ ev_mod - fg + F_f_mod

        tau_cmd = Jb.T @ tau_tilde + qfrc_bias.reshape((-1, 1))
        # print("qfrc_bias : ", qfrc_bias)

        self.Fd_star_list.append(Fd_star)
        self.Ff_list.append(F_f)
        self.Vb_list.append(Vb)
        self.Ff_activation.append(activation_force)
        self.Fi_activation.append(activation_impedance)
        self.rho_list.append(rho)
        time_end = time.time()
        print(f"time : {time_end-time_start:.4f} seconds")

        return tau_cmd.reshape((-1,))

    def set_hole_pose(self, pos, R):
        set_body_pose_rotm(self.model, 'hole', pos, R)


if __name__ == "__main__":
    robot_name = 'indy7'
    show_viewer = True
    randomized_start = True
    inertia_shaping = False
    task = 'sphere'
    if randomized_start:
        type = "random_start"
    else:
        type = "fixed_start"
        
    # ============================================================
    # Flow Matching version or Regressive version
    # FM Model :
    #   mlp: 
    #   transformer:
    # Regressive:
    #   mlp:
    # ============================================================
    model = 'transformer' # unet, transformer, mlp
    # ckpt_path="/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_regressive/velocity_field_regressive_best.pt"
    # ckpt_path="/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_fm/fm_best.pt"
    # ckpt_path="/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_fm_transformer_fixed_start/fm_best_0.025.pt"
    # ckpt_path="/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_fm_transformer_fixed_start/fm_best_4.21.pt"
    ckpt_path=f"/home/zhou/autolab/GUFIC_mujoco-main/gufic_env/flow_matching/checkpoints_cfm_transformer_vis_pRFe_{type}/cfm_transformer_vis2pose_{type}_best19.pt"

    assert task in ['regulation', 'circle', 'line', 'sphere']

    if task == 'regulation':
        max_time = 6
    elif task == 'line':
        max_time = 8
    elif task == 'circle':
        max_time = 10
    elif task == 'sphere':
        max_time = 10
    save_tensorboard = True

    RE = RobotEnv(
        robot_name,
        model,
        show_viewer=show_viewer,
        max_time=max_time,
        fz=10,
        fix_camera=True,
        task=task,
        randomized_start=randomized_start,
        inertia_shaping=inertia_shaping,
        # ===== 切换这里 =====
        use_learned_velocity_field=True,
        fm_ckpt_path=ckpt_path,
        fm_goal=None,                 # None 表示默认用任务初始参考位姿
        fm_time_horizon=max_time,
        fm_lowpass_alpha=0.2,
        # ===== 是否顺便记录当前实际使用的 Vd_star =====
        record_demos=False,
        demo_save_dir="./bolt_demos_fm_runtime",
        seed=42,
        save_tensorboard=True,
        test_offline_cond=False
    )

    RE.run()

    if show_viewer:
        RE.viewer.close()
