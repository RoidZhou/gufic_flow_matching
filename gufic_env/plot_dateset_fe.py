import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_force_from_npz(npz_path):
    data = np.load(npz_path)

    if "Fe" in data:
        force = data["Fe"].astype(np.float32)
        key = "Fe"
    elif "fe" in data:
        force = data["fe"].astype(np.float32)
        key = "fe"
    elif "Fe_raw" in data:
        force = data["Fe_raw"].astype(np.float32)
        key = "Fe_raw"
    else:
        raise KeyError(f"{npz_path} 里没有找到 Fe / fe / Fe_raw")

    if "t" in data:
        t = data["t"].astype(np.float32).reshape(-1)
    else:
        t = np.arange(len(force), dtype=np.float32)

    return t, force, key


def plot_one_demo_force(npz_path, save_dir=None, show=True):
    t, Fe, key = load_force_from_npz(npz_path)

    labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for i in range(6):
        axes[i].plot(t, Fe[:, i], linewidth=1.5)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(alpha=0.3)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"{os.path.basename(npz_path)} - {key}")
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            os.path.splitext(os.path.basename(npz_path))[0] + f"_{key}.png"
        )
        plt.savefig(save_path, dpi=180)
        print(f"saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_all_demos_force(demo_dir, save_dir=None, show=False, max_files=None):
    demo_files = sorted(glob.glob(os.path.join(demo_dir, "*.npz")))
    if len(demo_files) == 0:
        raise ValueError(f"{demo_dir} 下面没有 .npz 文件")

    if max_files is not None:
        demo_files = demo_files[:max_files]

    print(f"found {len(demo_files)} demo files")
    for f in demo_files:
        try:
            plot_one_demo_force(f, save_dir=save_dir, show=show)
        except Exception as e:
            print(f"skip {f}: {e}")


if __name__ == "__main__":
    demo_dir = "./bolt_demos"   # 改成你的目录
    save_dir = "./force_plots"  # 图片保存目录

    # 画单个 demo
    # plot_one_demo_force("./bolt_demos/bolt_demo_0000.npz", save_dir=save_dir, show=True)

    # 批量画所有 demo
    plot_all_demos_force(demo_dir, save_dir=save_dir, show=False, max_files=None)