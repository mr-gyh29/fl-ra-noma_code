import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

############################################################
# 公共工具函数：角度 ↔ 指向向量，几何量与信道参数计算
############################################################

def angles_to_pointing(phi_e, phi_a):
    """
    根据论文中的定义：
    l(φ_e, φ_a) = [sin φ_e sin φ_a,  sin φ_e cos φ_a,  cos φ_e]^T
    """
    l_x = np.sin(phi_e) * np.sin(phi_a)
    l_y = np.sin(phi_e) * np.cos(phi_a)
    l_z = np.cos(phi_e)
    l = np.array([l_x, l_y, l_z], dtype=float)
    return l / np.linalg.norm(l)


def pointing_to_angles(l):
    """
    反映射：
      φ_e = arccos(l^T e3)
      φ_a = atan2(l^T e1, l^T e2)
    其中 e1=[1,0,0]^T, e2=[0,1,0]^T, e3=[0,0,1]^T
    """
    l = l / np.linalg.norm(l)
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])

    phi_e = np.arccos(np.clip(l @ e3, -1.0, 1.0))
    phi_a = np.arctan2(l @ e1, l @ e2)
    return phi_e, phi_a


class RAChannelParams:
    """
    存放与 RA 信道相关的所有几何与物理参数，并预计算 b1_k, b2_km 等。

    参数说明：
      lambda_: 波长 lambda_
      kappa:   kappa = 2(2β+1)
      beta:    天线 directivity factor β（directional gain 指数）
      q_B:     BS 位置, shape (3,)
      q_k:     K 个设备位置, shape (K,3)
      q_m:     M 个散射体位置, shape (M,3)
      sigma_m: 散射体 RCS, shape (M,)
      tau_m:   散射体相位 τ_m, shape (M,), ~ Uniform[-π, π)
    """
    def __init__(self, lambda_, kappa, beta,
                 q_B, q_k, q_m, sigma_m, tau_m):
        self.lambda_ = float(lambda_)
        self.kappa = float(kappa)
        self.beta = float(beta)

        self.q_B = np.asarray(q_B, dtype=float).reshape(3)
        self.q_k = np.asarray(q_k, dtype=float)   # (K,3)
        self.q_m = np.asarray(q_m, dtype=float)   # (M,3)

        self.sigma_m = np.asarray(sigma_m, dtype=float)  # (M,)
        self.tau_m = np.asarray(tau_m, dtype=float)      # (M,)

        self.K = self.q_k.shape[0]
        self.M = self.q_m.shape[0]

        self._precompute_geo_and_channel()

    def _precompute_geo_and_channel(self):
        """预计算 q_hat、距离、b1_k、b2_km 等常数。"""
        K, M = self.K, self.M
        lambda_ = self.lambda_
        kappa = self.kappa

        # 单位方向向量和距离
        self.qhat_kB = np.zeros((K, 3))
        self.d_kB = np.zeros(K)
        for k in range(K):
            diff = self.q_B - self.q_k[k]
            dkB = np.linalg.norm(diff)
            self.d_kB[k] = dkB
            self.qhat_kB[k] = diff / dkB

        self.qhat_km = np.zeros((K, M, 3))
        self.d_km = np.zeros((K, M))
        self.d_mB = np.zeros(M)
        for m in range(M):
            diff_MB = self.q_m[m] - self.q_B
            self.d_mB[m] = np.linalg.norm(diff_MB)
            for k in range(self.K):
                diff_km = self.q_m[m] - self.q_k[k]
                dkm = np.linalg.norm(diff_km)
                self.d_km[k, m] = dkm
                self.qhat_km[k, m] = diff_km / dkm

        # b1_k, b2_km (对应你论文里 b_1 和 b_{2,m})
        self.b1 = np.zeros(self.K, dtype=complex)
        self.b2 = np.zeros((self.K, self.M), dtype=complex)

        for k in range(self.K):
            d_kB = self.d_kB[k]
            phase_kB = -1j * 2.0 * np.pi / lambda_ * d_kB
            self.b1[k] = (lambda_ * np.sqrt(kappa)) / (4.0 * np.pi * d_kB) * np.exp(phase_kB)

            for m in range(self.M):
                d_km = self.d_km[k, m]
                d_mB = self.d_mB[m]
                phase_km = -1j * 2.0 * np.pi / lambda_ * (d_km + d_mB) + 1j * self.tau_m[m]
                # print("k is", k)
                # print("m is", m)
                # print("phase_km is", phase_km)
                self.b2[k, m] = (lambda_ * np.sqrt(kappa * self.sigma_m[m])) / (4.0 * np.pi * d_km * d_mB) * np.exp(phase_km)

    def h_kB(self, l_k, k):
        """
        h_{k,B}(l_k) = b1_k (l_k^T q̂_{k,B})^β + Σ_m b2_{k,m} (l_k^T q̂_{k,m})^β
        """
        l_k = np.asarray(l_k, dtype=float).reshape(3)
        beta = self.beta

        proj_B = max(l_k @ self.qhat_kB[k], 0.0)
        term_LoS = self.b1[k] * (proj_B ** beta)
        # print("k is", k)
        # print("self.b1[k]", self.b1[k])

        proj_m = self.qhat_km[k] @ l_k          # shape (M,)
        proj_m = np.maximum(proj_m, 0.0)        # 对应 φ>π/2 时 Ge=0
        term_NLoS = np.sum(self.b2[k] * (proj_m ** beta))
        # print("self.b2[k]", self.b2[k])

        return term_LoS + term_NLoS

    def tilde_h_k(self, l_k, k):
        """tilde{h}_k(l_k) = |h_{k,B}(l_k)|^2"""
        h = self.h_kB(l_k, k)
        return np.abs(h) ** 2

    def grad_tilde_h_k(self, l_k, k):
        l_k = np.asarray(l_k, dtype=float).reshape(3)
        beta = self.beta

        # 当前真实信道（已包含裁剪）
        h = self.h_kB(l_k, k)

        # 梯度中的复向量部分 v(l)
        v = np.zeros(3, dtype=complex)

        # LoS 部分
        proj_B = l_k @ self.qhat_kB[k]
        if proj_B > 0:  # 只有在 cos φ > 0 的那一半空间才有导数
            v += beta * self.b1[k] * (proj_B ** (beta - 1)) * self.qhat_kB[k]

        # NLoS 部分
        proj_m = self.qhat_km[k] @ l_k  # (M,)
        for m in range(self.M):
            if proj_m[m] > 0:
                v += beta * self.b2[k, m] * (proj_m[m] ** (beta - 1)) * self.qhat_km[k, m]

        grad = 2.0 * np.real(h * np.conjugate(v))
        return grad



############################################################
# Part 1: RA Orientation Optimization (RO)
############################################################

def plot_3d_positions(q_B, q_k, q_m, show=True):
    """
    绘制三维散点图：基站 q_B（红星）、设备 q_k（蓝点）、散射体 q_m（绿三角）。
    若 matplotlib 不可用则打印提示并返回 None。
    返回 (fig, ax) 或 None。
    """
    # try:
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    # except Exception:
    #     print("matplotlib not available, skipping 3D plot")
    #     return None

#     K = 10
#     M = 5
#     D_k = 2000 + 500 * np.random.randint(0, 5, size=K) 

#     # print("T_cp", args.CPUcycles_per_sample * D_k / args.client_computation_capacity)
# # 1) 设备位置 q_k 按 t_k = [x, 20, 0], x ~ U[0, 80]
#     # 固定随机种子以保证 x_tx 和 x_sc 可复现
#     rng = np.random.default_rng(42)

#     x_tx = rng.uniform(0.0, 80.0, size=K)
#     q_k = np.zeros((K, 3))
#     q_k[:, 0] = x_tx          # x
#     q_k[:, 1] = 2.0           # y 固定为 2
#     q_k[:, 2] = 0.0           # z 固定为 0

#     # 2) 散射体位置 q_m 按 s_q = [x, 6, z], x ~ U[0,100], z ~ U[20,40]
#     x_sc = rng.uniform(0.0, 100.0, size=M)
#     z_sc = rng.uniform(20.0, 40.0, size=M)
#     q_m = np.zeros((M, 3))
#     q_m[:, 0] = x_sc          # x
#     q_m[:, 1] = 6.0           # y 固定为 6
#     q_m[:, 2] = z_sc          # z in [20, 40]

#     # 3) BS 位置 q_B 按 u_k = [x, 2, z], x ~ U[0,100], z ~ U[80,100]
#     x_B = np.random.uniform(0.0, 100.0)
#     z_B = np.random.uniform(80.0, 100.0)
#     q_B = np.array([50, 20.0, 50])   # shape (3,)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(q_k[:, 0], q_k[:, 1], q_k[:, 2], marker='^', s=60, label='Devices q_k')
    ax.scatter(q_m[:, 0], q_m[:, 1], q_m[:, 2], marker='o', s=80, label='Scatterers q_m')
    ax.scatter(q_B[0], q_B[1], q_B[2], marker='*', s=150, c='red', label='BS q_B')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 110)
    ax.view_init(elev=25, azim=45)
    ax.legend()

    plt.title('3D Positions of q_B, q_m, q_k')
    plt.tight_layout()
    fig.savefig('2q_positions.png', dpi=200)
    plt.show()

def init_l0_phi_max_towards_BS(params, k, phi_max):
    """
    用 φ_e = φ_max，且方位角朝向 BS 的方式初始化 l0：
      l0 = [sin φ_max sin φ_B^a, sin φ_max cos φ_B^a, cos φ_max]
    其中 φ_B^a 由 qhat_kB[k] 的投影方向决定。
    """

    v = params.qhat_kB[k]          # shape (3,), 单位向量
    vx, vy, vz = v

    # BS 方向在 xy 平面上的方位角（注意顺序为 (x,y)）
    phi_a_B = np.arctan2(vx, vy)

    phi_e_0 = phi_max
    phi_a_0 = phi_a_B

    l0 = np.array([
        np.sin(phi_e_0) * np.sin(phi_a_0),
        np.sin(phi_e_0) * np.cos(phi_a_0),
        np.cos(phi_e_0)
    ], dtype=float)

    # 理论上已经是单位向量，数值上可以再归一化一次
    l0 = l0 / np.linalg.norm(l0)

    return l0

#。原始版本
def optimize_single_RA_orientation(
    params: RAChannelParams,
    k: int,
    phi_max: float,
    max_iter: int = 50,
    tol: float = 1e-4,
    rho: float = 1e-10
):
    """
    解决第 k 个设备的 RO 子问题：
        max_{l_k} |h_{k,B}(l_k)|^2
        s.t. ||l_k|| <= 1,  (l_k)_3 >= cos(phi_max)

    使用 SCA: 在 l_k^{(i)} 处做一阶泰勒展开，得到线性目标：
        tilde{h}_k(l) ≈ tilde{h}_k(l^{(i)}) + ∇^T (l - l^{(i)})

    => 等价于在同样约束下
        max g^T l - 0.5 * rho * ||l - l^{(i)}||^2

    返回：
        l_star       : 优化后的单位指向向量
        phi_e_star   : 对应的偏心角（度）
        phi_a_star   : 对应的方位角（弧度）
        h_curr_val   : 最终原始目标值 |h|^2
    """
    cos_phi_max = np.cos(phi_max)

    # ===== 初始点：取朝向 BS 的方向，再修正到满足偏心角约束 =====
    # l0 = params.qhat_kB[k].copy()
    # l0 = np.array([1.0, 0.0, 0.0]) 

    # #用一个较差但可行的初始方向，以测试 SCA 的提升效果
    qhat = params.qhat_kB[k].copy()
    qxy = qhat[:2]
    xy_norm = np.linalg.norm(qxy)

    if xy_norm > 1e-12:
        l0 = np.array([-qxy[0] / xy_norm, -qxy[1] / xy_norm, 0.0])
    else:
        l0 = np.array([1.0, 0.0, 0.0])  # 退化情形备用

    # 确保满足 l_z >= cos(phi_max)
    if l0[2] < cos_phi_max:
        # 将 z 分量抬到 cos_phi_max，然后重新归一化
        l0[2] = cos_phi_max

        # 保持 x,y 方向比例不变
        xy_norm = np.linalg.norm(l0[:2])
        if xy_norm > 1e-12:
            scale = np.sqrt(1.0 - cos_phi_max**2) / xy_norm
            l0[0] *= scale
            l0[1] *= scale
        else:
            # 如果 x,y 近似为 0，就直接设为竖直
            l0[0] = 0.0
            l0[1] = 0.0

        l0 = l0 / np.linalg.norm(l0)

    l_curr = l0
    h_curr_val = params.tilde_h_k(l_curr, k)
    obj_history = [h_curr_val]

    for i in range(max_iter):
        # 计算当前梯度
        g = params.grad_tilde_h_k(l_curr, k)

        # SCA 子问题：max g^T l - 0.5 * rho * ||l - l_curr||^2
        l = cp.Variable(3)
        constraints = [
            cp.norm(l) <= 1.0,
            l[2] >= np.cos(phi_max),
            l[2] <= 1.0
        ]

        objective = cp.Maximize(
            g @ l - 0.5 * rho * cp.sum_squares(l - l_curr)
        )
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)

        if l.value is None:
            rho *= 10
            # 若求解失败，退出
            break

        l_next = np.array(l.value).reshape(3)

        # 计算新的目标值
        h_next_val = params.tilde_h_k(l_next, k)

        if h_next_val >= h_curr_val:
            # 真 |h|^2 没有变差，就接受
            rho = rho
            obj_history.append(h_next_val)
        else:
            rho *= 10
        l_curr = l_next
        h_curr_val = h_next_val
        # if h_next_val >= h_curr_val:
        #     # 真 |h|^2 没有变差，就接受
        #     l_curr = l_next
        #     h_curr_val = h_next_val
        #     obj_history.append(h_next_val)
        #     # if abs(obj_history[-1] - obj_history[-2]) <= tol:
        #     #     break
        # else:
        #     rho *= 10
        #     continue



    # obj_history_db = 10.0 * np.log10(np.maximum(np.array(obj_history, dtype=float), 1e-15))
    obj_history_db = 10.0 * np.log10(np.array(obj_history, dtype=float))
    # ===== 归一化得到最终指向向量 =====
    l_star = l_curr / np.linalg.norm(l_curr)

    # ===== 反映射到 (φ_e, φ_a) =====
    phi_e_star, phi_a_star = pointing_to_angles(l_star)
    phi_e_star = np.degrees(phi_e_star)

    print("优化后的角度:", phi_e_star)

    return l_star, phi_e_star, phi_a_star, h_curr_val, obj_history_db


# def optimize_single_RA_orientation(
#     params: RAChannelParams,
#     k: int,
#     phi_max: float,
#     max_iter: int = 200,
#     tol: float = 1e-4,
#     rho: float = 1e-10,
#     rho_max: float = 1e6,
#     max_backtrack: int = 20,
# ):
#     """
#     解决第 k 个设备的 RO 子问题：
#         max_{l_k} |h_{k,B}(l_k)|^2
#         s.t. ||l_k|| <= 1,  l_{k,3} >= cos(phi_max)

#     返回：
#         l_star       : 优化后的单位指向向量
#         phi_e_star   : 偏心角（度）
#         phi_a_star   : 方位角（弧度）
#         h_curr_val   : 最终原始目标值 |h|^2
#         obj_history_db : 外层 SCA 被接受迭代的原始目标值历史（dB）
#     """
#     cos_phi_max = np.cos(phi_max)

#     l0 = params.qhat_kB[k].copy()
#     # ===== 初始化 =====
#     # 用一个较差但可行的初始方向，以测试 SCA 的提升效果
#     # qhat = params.qhat_kB[k].copy()
#     # qxy = qhat[:2]
#     # xy_norm = np.linalg.norm(qxy)

#     # if xy_norm > 1e-12:
#     #     l0 = np.array([-qxy[0] / xy_norm, -qxy[1] / xy_norm, 0.0])
#     # else:
#     #     l0 = np.array([1.0, 0.0, 0.0])  # 退化情形备用

#     # 若初始方向不满足偏心角约束，则投到可行边界附近
#     if l0[2] < cos_phi_max:
#         l0[2] = cos_phi_max
#         xy_norm = np.linalg.norm(l0[:2])
#         if xy_norm > 1e-12:
#             scale = np.sqrt(1.0 - cos_phi_max**2) / xy_norm
#             l0[0] *= scale
#             l0[1] *= scale
#         else:
#             l0[0] = 0.0
#             l0[1] = 0.0
#         l0 = l0 / np.linalg.norm(l0)

#     l_curr = l0
#     h_curr_val = float(np.real(params.tilde_h_k(l_curr, k)))

#     # 记录外层 SCA 被接受迭代的原始目标值
#     obj_history = [h_curr_val]
#     rho_history = [rho]

#     # ===== 外层 SCA 迭代 =====
#     for i in range(max_iter):
#         g = params.grad_tilde_h_k(l_curr, k)

#         accepted = False
#         backtrack_count = 0

#         # ===== 内层 backtracking：直到原始目标不下降 =====
#         while (not accepted) and (backtrack_count < max_backtrack) and (rho <= rho_max):
#             l = cp.Variable(3)

#             constraints = [
#                 cp.norm(l, 2) <= 1.0,
#                 l[2] >= cos_phi_max,
#             ]

#             objective = cp.Maximize(
#                 g @ l - 0.5 * rho * cp.sum_squares(l - l_curr)
#             )

#             prob = cp.Problem(objective, constraints)
#             prob.solve(solver=cp.ECOS, verbose=False)

#             if l.value is None:
#                 rho *= 10
#                 backtrack_count += 1
#                 continue

#             l_next = np.array(l.value).reshape(3)

#             # 为了和最终输出一致，这里归一化后再评价原始目标
#             l_next = l_next / np.linalg.norm(l_next)
#             h_next_val = float(np.real(params.tilde_h_k(l_next, k)))

#             if h_next_val >= h_curr_val:
#                 accepted = True
#             else:
#                 rho *= 10
#                 backtrack_count += 1

#         # 如果这一轮没有找到可接受点，则停止
#         if not accepted:
#             print(f"device {k}: SCA stopped at iter {i+1} (no accepted update).")
#             break

#         # 接受新点
#         l_curr = l_next
#         h_curr_val = h_next_val
#         obj_history.append(h_curr_val)
#         rho_history.append(rho)

#         # # 收敛判据：只对 accepted iterates 判断
#         # if len(obj_history) >= 2 and abs(obj_history[-1] - obj_history[-2]) <= tol:
#         #     break

#     # ===== 输出最终结果 =====
#     l_star = l_curr / np.linalg.norm(l_curr)
#     phi_e_star, phi_a_star = pointing_to_angles(l_star)
#     phi_e_star = np.degrees(phi_e_star)

#     obj_history_db = 10.0 * np.log10(
#         np.maximum(np.array(obj_history, dtype=float), 1e-15)
#     )

#     return l_star, phi_e_star, phi_a_star, h_curr_val, obj_history_db

# def optimize_single_RA_orientation(params: RAChannelParams,
#                                    k: int,
#                                    phi_max: float,
#                                    max_iter: int = 200,
#                                    tol: float = 1e-4,
#                                    rho: float = 1e-10,
#                                    rho_max: float = 1e6,
#                                    max_backtrack: int = 20):
#     cos_phi_max = np.cos(phi_max)
#     # l0 = params.qhat_kB[k].copy()

#     qhat = params.qhat_kB[k].copy()
#     qxy = qhat[:2]
#     xy_norm = np.linalg.norm(qxy)

#     if xy_norm > 1e-12:
#         l0 = np.array([-qxy[0] / xy_norm, -qxy[1] / xy_norm, 0.0])
#     else:
#         l0 = np.array([1.0, 0.0, 0.0])

#     if l0[2] < cos_phi_max:
#         l0[2] = cos_phi_max
#         xy_norm = np.linalg.norm(l0[:2])
#         if xy_norm > 1e-12:
#             scale = np.sqrt(1.0 - cos_phi_max**2) / xy_norm
#             l0[0] *= scale
#             l0[1] *= scale
#         else:
#             l0[0] = 0.0
#             l0[1] = 0.0
#         l0 = l0 / np.linalg.norm(l0)

#     l_curr = l0
#     h_curr_val = float(np.real(params.tilde_h_k(l_curr, k)))

#     obj_history = [h_curr_val]
#     rho_history = [rho]

#     for i in range(max_iter):
#         g = params.grad_tilde_h_k(l_curr, k)

#         accepted = False
#         backtrack_count = 0

#         while (not accepted) and (backtrack_count < max_backtrack) and (rho <= rho_max):
#             l = cp.Variable(3)

#             constraints = [
#                 cp.norm(l, 2) <= 1.0,
#                 l[2] >= cos_phi_max
#             ]

#             objective = cp.Maximize(
#                 g @ l - 0.5 * rho * cp.sum_squares(l - l_curr)
#             )
#             prob = cp.Problem(objective, constraints)
#             prob.solve(solver=cp.ECOS, verbose=False)

#             if l.value is None:
#                 rho *= 10
#                 backtrack_count += 1
#                 continue

#             l_next = np.array(l.value).reshape(3)
#             l_next = l_next / np.linalg.norm(l_next)
#             h_next_val = float(np.real(params.tilde_h_k(l_next, k)))

#             if h_next_val >= h_curr_val:
#                 accepted = True
#             else:
#                 rho *= 10
#                 backtrack_count += 1

#         if not accepted:
#             print(f"device {k}: SCA stopped at iter {i+1} (no accepted update).")
#             break

#         l_curr = l_next
#         h_curr_val = h_next_val
#         obj_history.append(h_curr_val)
#         rho_history.append(rho)

#         if len(obj_history) >= 2 and abs(obj_history[-1] - obj_history[-2]) <= tol:
#             break

#     l_star = l_curr / np.linalg.norm(l_curr)
#     phi_e_star, phi_a_star = pointing_to_angles(l_star)
#     phi_e_star = np.degrees(phi_e_star)

#     obj_history_db = 10.0 * np.log10(np.maximum(np.array(obj_history, dtype=float), 1e-15))

#     return l_star, phi_e_star, phi_a_star, h_curr_val, obj_history_db

def optimize_without_scatterer(params: RAChannelParams,
                               k: int,
                               phi_max: float):
    """
    将l_star的方向固定为朝向BS的方向，优化偏心角φ_e。如果φ_e超过φ_max，则取φ_e=φ_max。
    解决第 k 个设备的 RO 子问题：
      max_{l_k} |h_{k,B}(l_k)|^2
      s.t. ||l_k|| <= 1,  (l_k)_3 >= cos(phi_max)
    使用直接计算的方式。
    返回：
      l_k_star: 优化后的单位指向向量
      phi_e_star, phi_a_star: 对应的偏心角和方位角
    """
    # 初始点：朝向 BS 的方向
    l0 = params.qhat_kB[k].copy()
    phi_e_0, phi_a_0 = pointing_to_angles(l0)
    if phi_e_0 < phi_max:
        # 若初始偏心角已经小于 phi_max，则直接返回
        l_star = l0 / np.linalg.norm(l0)
        h_val = params.tilde_h_k(l_star, k)
        print("优化后的角度", np.degrees(phi_e_0))
        return l_star, np.degrees(phi_e_0), phi_a_0, h_val
    else:
        print("k is", k)
        print(f"初始偏心角 φ_e_0 = {np.degrees(phi_e_0):.4f}° 超过最大值 {np.degrees(phi_max):.4f}°，进行优化")
        # 优化偏心角 φ_e
        phi_e_star = min(phi_e_0, phi_max)
        # 重新计算指向向量 l_star
        l_star = np.array([
            np.sin(phi_e_star) * np.sin(phi_a_0),
            np.sin(phi_e_star) * np.cos(phi_a_0),
            np.cos(phi_e_star)
        ], dtype=float)
        l_star = l_star / np.linalg.norm(l_star)
        h_val = params.tilde_h_k(l_star, k)
        print("优化后的角度", np.degrees(phi_e_star))
        return l_star, np.degrees(phi_e_star), phi_a_0, h_val
    

def optimize_all_RA_orientations(args, params: RAChannelParams,
                                 phi_max: float,
                                 max_iter: int = 50,
                                 tol: float = 1e-4):
    """
    对所有 K 个设备分别求解 RO 子问题，得到：
      l_star[k], phi_e_star[k], phi_a_star[k], |h_k|^2
    """
    K = params.K
    l_stars = np.zeros((K, 3))
    phi_e_stars = np.zeros(K)
    phi_a_stars = np.zeros(K)
    h_gain_stars = np.zeros(K)

    devices_obj_history = []

    for k in range(K):
        if args.LoS_method == 'closed':
            # print(f"--- 优化第 {k} 个设备的 RA 方向（无散射体）---")
            l_star, phi_e_star, phi_a_star, h_val = optimize_without_scatterer(
                params, k, phi_max)
        else:
            l_star, phi_e_star, phi_a_star, h_val, obj_history = optimize_single_RA_orientation(
                params, k, phi_max, max_iter, tol
            )
            devices_obj_history.append(obj_history)
        l_stars[k] = l_star
        phi_e_stars[k] = phi_e_star
        phi_a_stars[k] = phi_a_star
        h_gain_stars[k] = h_val
    # 绘制所有设备的SCA收敛曲线
    for k in range(K):
        # 保存所有设备数据到txt文件
        np.savetxt(f'device_{k}_obj_history.txt', devices_obj_history[k])
        plt.figure()
        plt.plot(range(len(devices_obj_history[k])), devices_obj_history[k], marker='o', label=f'Device {k}')
        plt.xlabel('SCA iteration')
        plt.ylabel(r'Original objective $|h_{k,B}(\mathbf{l}_k^{(i)})|^2$ (dB)')
        plt.title('SCA Convergence for Each Device')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(f'device_{k}_sca_convergence.png', dpi=200)
        
    # 对devices_obj_history进行平均
    # devices_obj_history_mean = np.mean(devices_obj_history, axis=0)
    # print("devices_obj_history_mean:", devices_obj_history_mean)
    # np.savetxt('devices_obj_history_mean.txt', devices_obj_history_mean)
    # plt.figure()
    # # plt.plot(range(len(devices_obj_history_mean)), devices_obj_history_mean, marker='o')
    # plt.xlabel('SCA iteration')
    # plt.ylabel(r'Original objective $|h_{k,B}(\mathbf{l}_k^{(i)})|^2$')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    # 保存devices_obj_history_mean到txt文件

    return l_stars, phi_e_stars, phi_a_stars, h_gain_stars


############################################################
# Part 2: Power Allocation and Device Selection Optimization (PS)
############################################################

def power_allocation_and_device_selection(ps_method, h_gains,
                                          T_cp,
                                          noise_var,
                                          S,
                                          B,
                                          T_th,
                                          p_max):
    """
    对应你论文的 PS 子问题：

    已知：
      - 每个设备的 |h_{k,B}(φ_k^*)|^2 = h_gains[k]
      - 本地训练时延 T_k^{cp} = T_cp[k]
      - 噪声方差 σ_k^2 = noise_var[k]
      - 模型大小 S，比特；带宽 B
      - 最大允许时延 T_th
      - 最大发射功率 p_max

    利用式 (xx)：
      T_k^{cp} + S / [ B log2(1 + SINR_k) ] ≤ T_th
      SINR_k = p_k |h_k|^2 / ( Σ_{j=k+1} p_j |h_j|^2 + σ_k^2 )

    得到：
      p_k ≥ p_k^{min} = ((2^{S/[B(T_th - T_k^{cp})]} - 1) *
                        ( Σ_{j=k+1} p_j |h_j|^2 + σ_k^2 )) / |h_k|^2

    采用逆序 SIC（从最差信道开始），迭代计算 p_k^{min} 并与 p_max 比较：
      - 若 p_k^{min} ≤ p_max，则设置 p_k^* = p_k^{min}, a_k = 1
      - 否则 p_k^* = 0, a_k = 0

    注意：我们在排序后的索引空间里做计算，再映射回原始设备顺序。
    """

    h_gains = np.asarray(h_gains, dtype=float).reshape(-1)
    T_cp = np.asarray(T_cp, dtype=float).reshape(-1)
    noise_var = np.asarray(noise_var, dtype=float).reshape(-1)

    K = h_gains.shape[0]

    # 按信道增益从大到小排序（确定 SIC 顺序）
    order = np.argsort(h_gains)[::-1]   # order[0] 是最强设备
    h_sorted = h_gains[order]
    T_cp_sorted = T_cp[order]
    noise_sorted = noise_var[order]

    p_star_sorted = np.zeros(K)
    a_sorted = np.zeros(K, dtype=int)
    # 对排序后的h进行归一化
    # h_guiyi = h_sorted / np.linalg.norm(h_sorted)
    h_guiyi = h_sorted
    # print("归一化后的信道增益:", h_guiyi)
    # print(" ")
    # print("原始信道增益:", h_gains)
    # print("排序后的信道增益:", h_sorted)
    # print("排序后的用户索引顺序:", order)
    # print("输出反序的用户索引顺序:", order[::-1])

    for idx in range(K - 1, -1, -1):  # 逆序：最后解码的用户先算
        # 若本地训练时延已超过门限，则无法满足时延约束
        denom_time = T_th - T_cp_sorted[idx]
        if denom_time <= 0:
            p_star_sorted[idx] = 0.0
            a_sorted[idx] = 0
            continue

        # 需要的速率（bit/s/Hz）
        R_req = S / (B * denom_time)
        # print("用户", order[idx], "需要的最小速率 R_req =", R_req)
        gamma_req = 2.0 ** R_req - 1.0  # 所需 SINR

        # 后面（干扰）用户的实际发射功率（已经决定 p_star_sorted[j]）
        if idx < K - 1:
            interference = np.sum(
                p_star_sorted[idx + 1:] * h_sorted[idx + 1:]
            )
        else:
            interference = 0.0

        # 理论最小需要的功率
        p_req =  gamma_req * (interference + 1e-9) / h_sorted[idx]
        print("用户", order[idx], "需要的最小功率 p_req =", p_req)

        if ps_method == 'proposed_PS':
            if (p_req <= p_max) and (p_req > 0):
                # print("分配功率给用户", order[idx], "为", p_req)
                p_star_sorted[idx] = p_req
                # 用p_req计算传输时延，验证是否满足时延约束
                # print("干扰为", interference)
                T_tx = S / (B * np.log2(1 + p_req * h_sorted[idx] / (interference + noise_sorted[idx])))
                print("!!!!!!!!!", np.log2(1 + p_req * h_sorted[idx] / (interference + 1e-9)))
                total_time = T_cp_sorted[idx] + T_tx
                # print("总时延为", total_time)
                # print(" ")
                a_sorted[idx] = 1
            else:
                p_star_sorted[idx] = 0.0
                a_sorted[idx] = 0
        elif ps_method == 'maximum_PS':
            if (p_req <= p_max):
                p_star_sorted[idx] = p_max
                a_sorted[idx] = 1
            else:
                p_star_sorted[idx] = 0.0
                a_sorted[idx] = 0
        elif ps_method == 'random_PS':
            p_random = np.random.uniform(0, p_max)
            if (p_req <= p_random):
                p_star_sorted[idx] = p_random
                a_sorted[idx] = 1
            else:
                p_star_sorted[idx] = 0.0
                a_sorted[idx] = 0
        elif ps_method == 'channel_inversion_PS':
            p_inv = h_guiyi[K-1-idx] * p_max
            if (p_req <= p_inv) and (p_inv <= p_max):
                p_star_sorted[idx] = p_inv
                a_sorted[idx] = 1
            else:
                p_star_sorted[idx] = 0.0
                a_sorted[idx] = 0
        else:
            raise ValueError("Unknown PS method:", ps_method)


    # 映射回原始用户索引
    p_star = np.zeros(K)
    a = np.zeros(K, dtype=int)
    p_star[order] = p_star_sorted
    a[order] = a_sorted

    return p_star, a, order

# def Get_h_based_ro_method(params, ro_method):
#     K = params.K
#     h_gain_stars = np.zeros(K)
#     for k in range(K):
#         if ro_method == 'random_RO':
#             phi_e = np.radians(np.random.uniform(0, args.phi_max))
#             phi_a = np.radians(np.random.uniform(-180, 180))
#         elif ro_method == 'fixed_RO':
#             phi_e = np.radians(0.0)
#             phi_a = np.radians(0.0)  # 固定方位角为 0
#         l = angles_to_pointing(phi_e, phi_a)
#         h_gain_stars[k] = params.tilde_h_k(l, k)
#     return h_gain_stars

def overall_optimization_RA_and_PS(args, model_size=7e5):
    """
    整体优化流程示例：先 RO 再 PS
    """

    K = args.num_clients
    M = args.num_scatterers
    D_k = 2000 + 500 * np.random.randint(0, 5, size=K) 

    # print("T_cp", args.CPUcycles_per_sample * D_k / args.client_computation_capacity)
# 1) 设备位置 q_k 按 t_k = [x, 20, 0], x ~ U[0, 80]
    # 固定随机种子以保证 x_tx 和 x_sc 可复现 42
    if args.random_not == 'random':
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(1)

    x_tx = rng.uniform(0.0, 80.0, size=K)
    q_k = np.zeros((K, 3))
    q_k[:, 0] = x_tx          # x
    q_k[:, 1] = 2.0           # y 固定为 2
    q_k[:, 2] = 0.0           # z 固定为 0
    if K > 5:
        q_k[5, 0] = 0.88
        q_k[5, 1] = 2.0
        q_k[5, 2] = 49.0
    # q_k[4, 0] = 42.0
    # q_k[4, 1] = 2.0
    # q_k[4, 2] = 0.0
    # q_k[9, 0] = 43.0
    # q_k[9, 1] = 2.0
    # q_k[9, 2] = 0.0
    # q_k[0, 0] = 41.0
    # q_k[0, 1] = 2.0
    # q_k[0, 2] = 0.0
    # q_k[3, 0] = -100.0
    # q_k[3, 1] = -40.0
    # q_k[3, 2] = -50.0
    # q_k[2, 0] = -100.0
    # q_k[2, 1] = -30.0
    # q_k[2, 2] = -50.0
    # q_k[1, 0] = -100.0
    # q_k[1, 1] = -20.0
    # q_k[1, 2] = -50.0
    # q_k[0, 0] = -100.0
    # q_k[0, 1] = -10.0
    # q_k[0, 2] = -50.0

    rng = np.random.default_rng(1)
    # 2) 散射体位置 q_m 按 s_q = [x, 6, z], x ~ U[0,100], z ~ U[20,40]
    x_sc = rng.uniform(0.0, 100.0, size=M)
    z_sc = rng.uniform(20.0, 40.0, size=M)
    q_m = np.zeros((M, 3))
    q_m[:, 0] = x_sc          # x
    q_m[:, 1] = 6.0           # y 固定为 6
    q_m[:, 2] = z_sc          # z in [20, 40]

    # 3) BS 位置 q_B 按 u_k = [x, 2, z], x ~ U[0,100], z ~ U[80,100]
    x_B = np.random.uniform(0.0, 100.0)
    z_B = np.random.uniform(80.0, 100.0)
    q_B = np.array([50.0, 50.0, 50.0])   # shape (3,)

    # 调用绘图函数（你可以根据需要将 show=False，用于在脚本中延后显示或保存）
    plot_3d_positions(q_B, q_k, q_m, show=True)
    print("3D positions plotted.")

    # 准备 RA 信道参数
    params = RAChannelParams(
        lambda_=args.wavelength,
        kappa=2 * (2 * args.mi + 1),
        beta=args.mi,
        q_B=q_B,
        q_k=q_k,
        q_m=q_m,
        sigma_m=np.ones(M),
        tau_m=np.full(M, np.pi / 2),
    )
    #np.random.uniform(-np.pi, np.pi, size=M)

    # Part 1: RA Orientation Optimization (RO)
    if args.RO_method == 'proposed_RO':
        l_stars, phi_e_stars, phi_a_stars, h_gain_stars = optimize_all_RA_orientations(
            args, params, args.phi_max, max_iter=50, tol=1e-17
        )
    elif args.RO_method == 'random_RO':
        K = params.K
        h_gain_stars = np.zeros(K)
        for k in range(K):
            phi_e = np.radians(np.random.uniform(0, args.phi_max))
            phi_a = np.radians(np.random.uniform(-180, 180))
            l = angles_to_pointing(phi_e, phi_a)
            h_gain_stars[k] = params.tilde_h_k(l, k)

    elif args.RO_method == 'fixed_RO':
        K = params.K
        h_gain_stars = np.zeros(K)
        for k in range(K):
            phi_e = np.radians(0.0)
            phi_a = np.radians(0.0)  # 固定方位角为 0
            l = angles_to_pointing(phi_e, phi_a)
            # 2) BS-用户k 的单位方向向量 qhat_kB[k]
            qhat_kB_k = params.qhat_kB[k]  # shape (3,)

            # 3) 计算夹角 φ_{k,B}
            cos_phi_kB = np.clip(np.dot(l, qhat_kB_k), -1.0, 1.0)  # 数值上做个 clip 比较安全
            phi_kB = np.arccos(cos_phi_kB)       # 单位：弧度
            phi_kB_deg = np.degrees(phi_kB)
            print(f"用户 {k} 固定方向与 BS 夹角 φ_kB = {phi_kB_deg:.2f} 度")
            h_gain_stars[k] = params.tilde_h_k(l, k)
    else:
        raise ValueError("Unknown RA orientation method:", args.ro_method)

    print("h参数", h_gain_stars)
    print("RA orientations optimized.")

    
    # print("Optimized RA orientations (phi_e):", phi_e_stars)

    # Part 2: Power Allocation and Device Selection Optimization (PS)
    p_star, a, sic_order = power_allocation_and_device_selection(
        h_gains=h_gain_stars,
        T_cp=args.CPUcycles_per_sample * D_k / args.client_computation_capacity,
        noise_var=1e-9 * np.ones(K),
        S=model_size,
        B=args.bandwidth,
        T_th=args.latency_threshold,
        p_max=args.transmit_power_max,
        ps_method=args.PS_method
    )


    return a, h_gain_stars, D_k, p_star, sic_order

############################################################
# 示例：如何串联 RO 和 PS（仅示意）
############################################################

# if __name__ == "__main__":
#     # 以下只是示例参数，你可以用自己论文仿真中的配置替换


#     K = 10
#     cell_radius = 100.0  # m
#     r_min = 10.0         # m

#     r = r_min + (cell_radius - r_min) * np.random.rand(K)
#     theta = 2 * np.pi * np.random.rand(K)

#     q_k = np.zeros((K, 3))
#     q_k[:, 0] = r * np.cos(theta)  # x
#     q_k[:, 1] = r * np.sin(theta)  # y
#     q_k[:, 2] = 0.0                # z


#     # 几何与信道参数
#     K = 4       # 设备数
#     M = 3       # 散射体数
#     lambda_ = 0.125
#     beta = 4.0
#     kappa = 2 * (2 * beta + 1)  # 与论文一致

#     q_B = np.array([0.0, 0.0, 10.0])
#     q_k = np.random.randn(K, 3) * 10.0  # 随便放，实际中请换成你的布局
#     q_m = np.random.randn(M, 3) * 20.0
#     sigma_m = np.ones(M)
#     tau_m = np.random.uniform(-np.pi, np.pi, size=M)

#     params = RAChannelParams(
#         lambda_=lambda_,
#         kappa=kappa,
#         beta=beta,
#         q_B=q_B,
#         q_k=q_k,
#         q_m=q_m,
#         sigma_m=sigma_m,
#         tau_m=tau_m,
#     )

#     # Part 1: RA Orientation Optimization (RO)
#     phi_max = np.pi / 3  # 例如
#     l_stars, phi_e_stars, phi_a_stars, h_gain_stars = optimize_all_RA_orientations(
#         params, phi_max, max_iter=30, tol=1e-4
#     )

#     # Part 2: Power Allocation and Device Selection Optimization (PS)
#     # 计算本地训练时延（这里随便设一个例子，实际按 T_k^{cp} = ε C_k D_k / f_k）
#     T_cp = np.full(K, 0.02)  # 每个用户本地训练 20ms（举例）

#     # 噪声功率（举例）
#     noise_var = np.full(K, 1e-9)

#     # 模型大小、带宽、时延门限和最大功率
#     S = 1e5        # 比特
#     B = 1e6        # Hz
#     T_th = 0.1     # s
#     p_max = 1.0    # W

#     p_star, a, sic_order = power_allocation_and_device_selection(
#         h_gains=h_gain_stars,
#         T_cp=T_cp,
#         noise_var=noise_var,
#         S=S,
#         B=B,
#         T_th=T_th,
#         p_max=p_max,
#     )

#     print("Optimized RA orientations (phi_e, phi_a):")
#     for k in range(K):
#         print(f"User {k}: phi_e={phi_e_stars[k]:.3f}, phi_a={phi_a_stars[k]:.3f}")

#     print("\nOptimized powers p_k*: ", p_star)
#     print("Device selection a_k:   ", a)
#     print("SIC decoding order (from strongest to weakest):", sic_order)
