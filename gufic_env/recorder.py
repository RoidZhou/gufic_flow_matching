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
            "point_cloud": [],
        }

    def add(self, t, p, R, Vd_star, dVd_star, Fe=None, point_cloud=None):
        from scipy.spatial.transform import Rotation as RT
        import numpy as np

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

        if Fe is None:
            Fe = np.zeros(6, dtype=np.float32)
        self.records["Fe"].append(np.asarray(Fe).reshape(6).astype(np.float32))

        if point_cloud is not None:
            self.records["point_cloud"].append(
                np.asarray(point_cloud, dtype=np.float32)
            )

    def save(self, episode_name):
        save_path = os.path.join(self.save_dir, f"{episode_name}.npz")

        data = {}
        for k, v in self.records.items():
            if len(v) > 0:
                data[k] = np.stack(v, axis=0).astype(np.float32)

        data["x"] = np.concatenate([data["p"], data["euler"]], axis=1).astype(np.float32)
        data["goal"] = np.concatenate([data["p"][-1], data["euler"][-1]], axis=0).astype(np.float32)
        data["total_time"] = np.array([data["t"][-1]], dtype=np.float32)

        np.savez_compressed(save_path, **data)
        print(f"[Recorder] saved demo -> {save_path}")
        return save_path