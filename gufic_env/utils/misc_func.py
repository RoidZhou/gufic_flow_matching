import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation as RT

def vee_map(R):
    v3 = -R[0,1]
    v1 = -R[1,2]
    v2 = R[0,2]
    return np.array([v1,v2,v3]).reshape((-1,1))

def hat_map(w):
    w = w.reshape((-1,))
    w_hat = np.array([[0, -w[2], w[1]],
                        [w[2], 0, -w[0]],
                        [-w[1], w[0], 0]])
    return w_hat

def rotmat_x(th):
    R = np.array([[1,0,0],
                    [0,np.cos(th),-np.sin(th)],
                    [0,np.sin(th), np.cos(th)]])

    return R

def adjoint_g_ed(g_ed):
    p = g_ed[:3,3]
    R = g_ed[:3,:3]

    p_hat = hat_map(p)
    # translation part first adjoint map
    adj = np.zeros((6,6))
    adj[:3,:3] = R
    adj[3:,3:] = R
    adj[:3,3:] = p_hat @ R

    return adj

def adjoint_g_ed_dual(g_ed):
    mat = adjoint_g_ed(np.linalg.inv(g_ed))

    return mat.T

def adjoint_g_ed_deriv(g, gd, v, w, vd, wd):
    v = v.reshape((-1,1))
    w = w.reshape((-1,1))
    vd = vd.reshape((-1,1))
    wd = wd.reshape((-1,1))

    g_ed = np.linalg.inv(g) @ gd
    p_ed = g_ed[:3,3]
    R_ed = g_ed[:3,:3]

    mat = np.zeros((6,6))

    dR_ed = hat_map(w) @ R_ed - R_ed @ hat_map(wd)
    dp_ed = -v - hat_map(w) @ p_ed + R_ed @ vd

    mat[:3, :3] = dR_ed
    mat[:3, 3:] = hat_map(p_ed)@ dR_ed + hat_map(dp_ed) @ R_ed
    mat[3:, 3:] = dR_ed

    return mat

def initialize_trajectory(task, max_time = 10, robot_state = None):
    t = sp.symbols('t')

    if task == "regulation":
        pd_default = np.array([0.50, 0.0, 0.125])
        Rd_default = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, -1]])
        
    elif task == "circle":
        pd_default = np.array([0.50, 0.0, 0.125])
        Rd_default = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, -1]])
        
    elif task == "line":
        pd_default = np.array([0.50, 0.0, 0.125])
        Rd_default = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, -1]])
        
    elif task == "sphere":
        pd_default = np.array([0.40, 0.0, 0.0]) #center of the sphere
        Rd_default = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, -1]])
    elif task == "insertion":
        pd_default = np.array([0.50, 0.0, 0.17])
        Rd_default = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, -1]])
    elif task == "bolt":
        pd_default = np.array([0.50, 0.0, 0.20]) # 螺栓孔口
        # pd_default = np.array([0.50, 0.0, 0.22])
        Rd_default = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, -1]])
        U, _, Vt = np.linalg.svd(Rd_default)
        Rd_default = U @ Vt
    else:
        raise ValueError("Invalid task")

    pd_default_sym = sp.Matrix(pd_default)
    Rd_default_sym = sp.Matrix(Rd_default)

    if task == 'regulation':
        pd_t_sim = pd_default_sym
        Rd_t_sim = Rd_default_sym
        # 绕z轴旋转任意角度phi的姿态矩阵 
        # R_z_phi = sp.Matrix([[sp.cos(1*t), -sp.sin(1*t), 0],
        #                      [sp.sin(1*t), sp.cos(1*t),  0],
        #                      [0,          0,            1]])
        # Rd_t_sim = R_z_phi * Rd_default_sym

    elif task == 'circle':
        r = 0.1
        pd_t_sim = pd_default_sym + sp.Matrix([r * sp.cos(t), r * sp.sin(t), 0])
        Rd_t_sim = Rd_default_sym
        
    elif task == 'line':
        pd_t_sim = pd_default_sym + sp.Matrix([0, 0.05 * t, 0])
        Rd_t_sim = Rd_default_sym

    elif task == 'sphere':

        total_radian = 1/2*np.pi
        omega_value = total_radian / max_time

        theta_y = omega_value * t - total_radian * 0.5


        r_sphere = 0.304
        pd_t_sim = pd_default_sym + sp.Matrix([0, r_sphere * sp.sin(theta_y), -0.10 + r_sphere * sp.cos(theta_y)])
        rotmat_y = sp.Matrix([[sp.cos(-theta_y), 0, sp.sin(-theta_y)], [0, 1, 0], [-sp.sin(-theta_y), 0, sp.cos(-theta_y)]])
        Rd_t_sim = Rd_default_sym @ rotmat_y

    elif task == 'insertion':
        # --- 1) 基准位姿（轴系） ---
        T1 = 2.0          # 从当前位置到孔口上方
        T2 = 2.0          # 从孔口上方下落到孔口
        T3 = max_time - T1 - T2   # 沿孔轴插入
        h = 0.06         # 孔口上方 4 cm

        # 符号变量
        t = sp.symbols('t', real=True, nonnegative=True)

        pd_sym = sp.Matrix(pd_default)
        Rd_sym = sp.Matrix(Rd_default)

        # 世界系下孔轴方向（局部 +Z）
        z_axis_world = Rd_sym * sp.Matrix([0, 0, 1])

        # 孔口上方预对准位置
        # p_above = pd_sym + h * z_axis_world
        # 先验证轨迹逻辑是否正确
        p_above = pd_sym + sp.Matrix(np.array([0.0, 0.0, h]))
        def s_curve(tt, T):
            tau = tt / T
            return 10*tau**3 - 15*tau**4 + 6*tau**5   # C2 连续

        # ===== 第1段：当前位置 -> 孔口上方 =====
        s1 = s_curve(t, T1)
        p0_np, R0_np = robot_state.get_pose()
        # p0_np, R0_np = pd_default, Rd_default
        pd_current_sym = sp.Matrix(p0_np)
        p_seg1 = pd_current_sym + s1 * (p_above - pd_current_sym)

        # ===== 第2段：孔口上方 -> 孔口 =====
        t2 = t - T1
        s2 = s_curve(t2, T2)
        p_seg2 = p_above + s2 * (pd_sym - p_above)

        # ===== 第3段：沿孔轴纯插入 =====
        t3 = t - (T1 + T2)
        s3 = s_curve(t3, T3)

        # 插入深度（根据你的孔深和装配需求设置）
        insert_depth = 0.03   # 例如 3 cm

        # 局部系纯 z 方向推进
        p_insert_loc = sp.Matrix([0, 0, insert_depth * s3])
        p_insert = pd_sym + Rd_sym * p_insert_loc

        # 姿态保持不变
        R_insert = Rd_sym

        # ===== 拼接三段 =====
        pd_default_sym = sp.Piecewise(
            (p_seg1, t <= T1),
            (p_seg2, (t > T1) & (t <= T1 + T2)),
            (p_insert, t > T1 + T2)
        )

        Rd_default_sym = sp.Piecewise(
            (Rd_sym, t <= T1 + T2),
            (R_insert, t > T1 + T2)
        )

        pd_t_sim = pd_default_sym
        Rd_t_sim = Rd_default_sym

    elif task == 'bolt':
        # --- 1) 基准位姿（轴系） ---
        T1 = 4.0          # 从当前位置到孔口上方
        T2 = 2.0          # 从孔口上方下落到孔口
        T3 = max_time - T1 - T2   # 沿孔轴插入
        h = -0.02         # 孔口上方 4 cm

        # 符号变量
        t = sp.symbols('t', real=True, nonnegative=True)

        pd_sym = sp.Matrix(pd_default)
        Rd_sym = sp.Matrix(Rd_default)

        # 世界系下的螺栓轴方向（local +Z）
        z_axis_world = Rd_sym * sp.Matrix([0, 0, 1])

        p_above = pd_sym + h * z_axis_world   # 螺栓口正上方 5 cm

        def s_curve(tt, T):
            tau = tt/T
            return 10*tau**3 - 15*tau**4 + 6*tau**5   # C2 连续

        # —— 段 1: pd_default -> p_above （任意空间直线）——
        s1 = s_curve(t, T1)              # 0->1
        # 从 pd_default 出发
        # p_seg1 = pd_sym + s1 * (p_above - pd_sym) 
        # 从 当前位置 出发
        p0_np, R0_np = robot_state.get_pose()
        pd_current_sym = sp.Matrix(p0_np)
        p_seg1 = pd_current_sym + s1 * (p_above - pd_current_sym) # 从 pd_default 出发

        # —— 段 2: p_above -> pd_default （沿螺栓轴垂直下落）——
        t2 = t - T1
        s2 = s_curve(t2, T2)                 # 0->1
        p_seg2 = p_above + s2 * (pd_sym - p_above)   # 最后回到 pd_sym（螺栓口）

        # —— 段 3:沿原来的螺旋/旋转轨迹（时间从 T1 开始重新计）——
        # 旋转总角度与时间
        total_radian = 4*np.pi          # 例如旋转180°
        theta0 = 0                        # 以中心对齐（起点为 0，终点 +Δθ/2）

        # 螺旋参数（半径r与螺距pitch
        r = 0.0000                        # 真实拧紧时通常为0；找牙可设0.3~0.5mm
        pitch = 0.0025                    # M20常见螺距(米/圈)=2.5mm/turn
        s_per_rad = pitch / (2*np.pi)     # 每弧度的轴向位移

        t3 = t - (T1 + T2)
        s3 = s_curve(t3, T3)
        theta_t3 = theta0 + total_radian * s3

        x_loc = r*sp.cos(theta_t3)
        y_loc = r*sp.sin(theta_t3)
        z_loc = s_per_rad * (theta_t3 - theta0)

        p_loc = sp.Matrix([x_loc, y_loc, z_loc])
        p_spiral = sp.Matrix(pd_default) + sp.Matrix(Rd_default) * p_loc

        ct2, st2 = sp.cos(theta_t3-theta0), sp.sin(theta_t3-theta0)
        Rz2 = sp.Matrix([[ct2, -st2, 0],
                        [st2,  ct2, 0],
                        [  0,    0, 1]])
        # R_spiral = Rz2.T * sp.Matrix(Rd_default)
        R_spiral = sp.Matrix(Rd_default) * Rz2
        # R_spiral = sp.Matrix(Rd_default)

        # --- 用 Piecewise 把两段拼在一起 ---
        pd_default_sym = sp.Piecewise(
            (p_seg1, t <= T1),
            (p_seg2, (t > T1) & (t <= T1 + T2)),
            (p_spiral, t > T1 + T2)   # 之后保持在螺栓口不动（你可以后面再接第二段轨迹）
        )

        # 姿态第一段先不变，始终保持为 Rd_default
        Rd_default_sym = sp.Piecewise(
            (Rd_sym, t <= T1 + T2),
            (R_spiral,  t > T1 + T2)   # 或者以后在这里接你原来的螺旋旋转 Rz 段
        )
        pd_t_sim = pd_default_sym
        Rd_t_sim = Rd_default_sym

    # Differentiate with symbolic expressions
    dpd_t_sim = sp.diff(pd_t_sim, t)
    dRd_t_sim = sp.diff(Rd_t_sim, t)
    ddpd_t_sim = sp.diff(dpd_t_sim, t)
    ddRd_t_sim = sp.diff(dRd_t_sim, t)

    # Convert symbolic to numpy expressions
    pd_t = sp.lambdify(t, pd_t_sim, "numpy")
    Rd_t = sp.lambdify(t, Rd_t_sim, "numpy")
    dpd_t = sp.lambdify(t, dpd_t_sim, "numpy")
    dRd_t = sp.lambdify(t, dRd_t_sim, "numpy")
    ddpd_t = sp.lambdify(t, ddpd_t_sim, "numpy")
    ddRd_t = sp.lambdify(t, ddRd_t_sim, "numpy")

    return pd_t, Rd_t, dpd_t, dRd_t, ddpd_t, ddRd_t

def set_gains(controller = "GUFIC", task = "regulation"):
    assert controller in ["GUFIC", "GIC"], "Invalid controller"
    assert task in ["regulation", "circle", "line", "sphere", "insertion", "bolt"], "Invalid task"
    
    if controller == "GIC":
        if task == "regulation":
            Kp = np.eye(3) * np.array([1500, 1500, 1500])
            KR = np.eye(3) * np.array([1500, 1500, 1500])
            Kd = np.eye(6) * np.array([500, 500, 500, 500, 500, 500])

        elif task in ["circle", "line", "sphere"]:
            Kp = np.eye(3) * np.array([2500, 2500, 1500])
            KR = np.eye(3) * np.array([2000, 2000, 2000])
            Kd = np.eye(6) * np.array([500, 500, 500, 500, 500, 500])

        return Kp, KR, Kd

    elif controller == "GUFIC":
        if task in ["regulation", "insertion"]:
            Kp = np.eye(3) * np.array([1500, 1500, 1500])
            KR = np.eye(3) * np.array([1500, 1500, 1500])
            Kd = np.eye(6) * np.array([500, 500, 500, 500, 500, 500])

            kp_force = 1.0
            kd_force = 0.5
            ki_force = 4.0

            zeta_v = 10
            zeta_w = 10
        elif task in ["bolt"]:
            Kp = np.eye(3) * np.array([2000, 2000, 2000])
            KR = np.eye(3) * np.array([2000, 2000, 2000])
            Kd = np.eye(6) * np.array([500, 500, 500, 500, 500, 500])

            kp_force = 1.0
            kd_force = 0.0
            ki_force = 4.0

            zeta_v = 50
            zeta_w = 10

        elif task in ["circle", "line", "sphere"]:
            Kp = np.eye(3) * np.array([2000, 2000, 10])
            KR = np.eye(3) * np.array([2000, 2000, 2000])
            Kd = np.eye(6) * np.array([500, 500, 500, 500, 500, 500])

            if task == "sphere":
                kp_force = 1.5
                kd_force = 0.75
                ki_force = 6.0

            else:
                kp_force = 1.0
                kd_force = 0.5
                ki_force = 4.0

            zeta_v = 5
            zeta_w = 5

        return Kp, KR, Kd, kp_force, kd_force, ki_force, zeta_v, zeta_w






