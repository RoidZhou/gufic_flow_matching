import os
import numpy as np
from scipy.spatial.transform import Rotation as RT


class BoltTrajectoryRecorder:
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

    def add(self, t, p, R, Vd_star, dVd_star, Fe):
        """
        t: scalar
        p: (3,)
        R: (3,3)
        Vd_star: (6,) or (6,1)
        dVd_star: (6,) or (6,1)
        Fe: (6,), (6,1)
        """
        p = np.asarray(p).reshape(3)
        R = np.asarray(R).reshape(3, 3)
        Vd_star = np.asarray(Vd_star).reshape(6)
        dVd_star = np.asarray(dVd_star).reshape(6)
        Fe = np.asarray(Fe).reshape(6)
        euler = RT.from_matrix(R).as_euler("xyz", degrees=False)

        self.records["t"].append(float(t))
        self.records["p"].append(p.astype(np.float32))
        self.records["R"].append(R.astype(np.float32))
        self.records["euler"].append(euler.astype(np.float32))
        self.records["Vd_star"].append(Vd_star.astype(np.float32))
        self.records["dVd_star"].append(dVd_star.astype(np.float32))
        self.records["Fe"].append(Fe.astype(np.float32))

    def save(self, episode_name):
        save_path = os.path.join(self.save_dir, f"{episode_name}.npz")

        data = {}
        for k, v in self.records.items():
            data[k] = np.stack(v, axis=0).astype(np.float32)

        # x = [p, euler]
        data["x"] = np.concatenate([data["p"], data["euler"]], axis=1).astype(np.float32)

        # goal 取最后一个时刻的位姿
        goal = np.concatenate([data["p"][-1], data["euler"][-1]], axis=0).astype(np.float32)
        data["goal"] = goal

        # total_time
        data["total_time"] = np.array([data["t"][-1]], dtype=np.float32)

        np.savez_compressed(save_path, **data)
        print(f"[Recorder] saved demo to {save_path}")
        return save_path