# # import torch
import numpy as np

# # print(torch.__version__)
# # print(torch.cuda.is_available())

# # proportions = np.random.dirichlet(np.repeat(0.5, 10))
# # print(proportions)
# # idx_batch = [[] for _ in range(10)]
# # # for p, idx_j in zip(proportions, idx_batch):
# # #     print(p, idx_j)
# # #     print('')

# # a = np.array([p * (len(idx_j) < 200 / 10)for p, idx_j in zip(proportions, idx_batch)])
# # a = a / a.sum()
# # print(a)



# import torch
# import numpy as np
# import os,sys,os.path
# from tensorboardX import SummaryWriter
# import pickle
# from torch import nn
# import hashlib
# import argparse


# from models import CNNFemnist,ResNet18,ShuffLeNet
# from sampling import LocalDataset, LocalDataloaders, partition_data
# from option import args_parser

# from Server.ServerFedAvg import ServerFedAvg
# from Server.ServerFedProx import ServerFedProx
# from Server.ServerFedMD import ServerFedMD 
# from Server.ServerFedProto import ServerFedProto
# from Server.ServerFedHKD import ServerFedHKD



# print(torch.__version__)
# torch.cuda.is_available()
# np.set_printoptions(threshold=np.inf)
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print(device.type)

# args = args_parser()
# print(args)
# args_hash = ''
# for k,v in vars(args).items():
#     if k == 'eval_only':
#         continue
#     args_hash += str(k)+str(v)
    
# args_hash = hashlib.sha256(args_hash.encode()).hexdigest()





# train_dataset, testset, dict_users, dict_users_test = partition_data(n_users = args.num_clients, alpha=args.beta,rand_seed = args.seed, dataset=str(args.dataset))




# Loaders_train = LocalDataloaders(train_dataset,dict_users,args.batch_size,ShuffleorNot = True,frac=args.part)
# Loaders_test = LocalDataloaders(testset,dict_users_test,args.batch_size,ShuffleorNot = True,frac=2*args.part)
# global_loader_test = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,shuffle=True, num_workers=2)


# # for idx, (X,Y) in enumerate(Loaders_train[0]):
# #     print(idx)

# n_users = 5
# N=20
# idx_k = np.where(np.array([1,1,1,1,1,1,1,1,2,2,2,2,2,23,3,3,3,3,3,3]) == 1)[0]
# a = [2, 2, 3, 1]
# print(np.split(idx_k, a))
# print(idx_k)
# idx_batch = [[] for _ in range(n_users)]
# proportions = np.random.dirichlet(np.repeat(0.5, n_users)) 
# proportions_train = np.array([p * (len(idx_j) < N / n_users) for p, idx_j in zip(proportions, idx_batch)])
# proportions_train = (np.cumsum(proportions_train) * len(idx_k)).astype(int)[:-1]
# print(proportions_train)


# idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, a))] 
# print(idx_batch)


# Y = [1, 2]
# X = [3, 4]
# before = id(Y)
# Y= X+Y
# print(before==id(Y))
# print(Y)

# J = 1
# K = 4
# N = 2
# V2 = np.random.randn(J, K, N)
# S1 = np.random.randint(0, 2, (J, K))
# S2 = np.random.randint(0, 2, (J, K, N))
# print(S2)
# for j in range(J):
#     prob = 1/(1+np.exp(-V2[j]))
#     S2[j][:] = 0
#     print(S2)
#     S2[j][np.arange(K), np.argmax(prob, axis=1)] = 1
# print(S2)


# print("------------------")
# a = np.array([[1, 2], [1, 1]])
# b = np.array([[2],[3]])
# print(a.shape, b.shape)
# print(a * b)
# print(b * a)

# a = np.array([[2, 3], [4, 7], [2,3]])
# b = np.array([[3, 5], [4, 7], [2, 3]])
# # result = (a[:, None] * b)
# print(a.shape, b.shape)
# print(a / b)
# # print(1+a)


# def save_records(self, list):
#     filename = f"{list.__name__ if hasattr(list, '__name__') else 'record'}.txt"
# names = [1, 2]
# filename = f"{names.__name__ if hasattr(names, '__name__') else 'record'}.txt"
# print(filename)

# S2 = np.random.randint(0, 2, (K, N))
# print("S2:", S2)
# active_uav = np.where(S2.sum(axis=0) > 0)[0]
# print("active_uav:", active_uav)


# a = [1, 2, 3, 4]
# b = [1,0,1,0]
# c= [a, b]
# print(np.mean(np.array(c), axis=0).tolist())

# np.random.seed(0)
# a = np.random.uniform(100, 500, (10, 3))

# b = np.random.uniform(500, 2000, 3)
# print(a)
# print(b)

# a1 = [112, 12, 15, 98, 199, 4, 0, 5, 11, 56]

# a2 = [51, 7, 8, 29, 8, 107, 57, 85, 38, 58]

# a3 = [17, 96, 60, 120, 22, 45, 61, 15, 1, 11]

# a4 = [19, 28, 123, 35, 27, 26, 21, 153, 7, 137]

# a5 = [10, 11, 145, 11, 18, 79, 24, 40, 213, 25]

# a6 = [80, 114, 22, 92, 29, 41, 0, 198, 0, 0]

# a7 = [28, 84, 39, 24, 16, 108, 39, 28, 86, 60]

# a8 = [17, 37, 5, 31, 17, 45, 146, 8, 243, 219]

# a9 = [151, 101, 55, 24, 115, 82, 176, 0, 0, 0]

# a10 = [84, 70, 43, 139, 98, 27, 71, 44, 0, 0]
# print(np.sum(a1))
# print(np.sum(a2))
# print(np.sum(a3))
# print(np.sum(a4))
# print(np.sum(a5))
# print(np.sum(a6))
# print(np.sum(a7))
# print(np.sum(a8))
# print(np.sum(a9))
# print(np.sum(a10))


# # Fig. 3 reproduction: Non-IID modeling (Dirichlet for class heterogeneity + Zipf for data imbalance)
# # Author: (you)
# # Requirements: numpy, matplotlib
# import numpy as np
# import matplotlib.pyplot as plt

# def draw_fig3(
#     U=20,            # number of users
#     N=10,            # number of classes (q̄_n = 1/N)
#     D_per_user=500,  # samples per user for the left column (class-heterogeneity)
#     D_total=600,     # total samples across users for the right column (amount-imbalance)
#     thetas=(0.01, 10.0, np.inf),   # Dirichlet concentration settings
#     etas=(0.0, 0.3, 1.0),          # Zipf skew settings
#     seed=0
# ):
#     """
#     Left column (heterogeneous data classes):
#       q ~ Dirichlet(θ * q̄), 其中 q̄ = (1/N,...,1/N)
#       实现方式：v_n ~ Gamma(θ q̄_n, 1), q_n = v_n / Σ v_n
#       对每个用户按 q 分配 D_per_user 个样本到 N 个类别，画为离散色块热图。

#     Right column (unbalanced data amounts):
#       D_u ∝ u^{-η}, u = 1..U
#       令 p_u = u^{-η} / Σ_u u^{-η}，四舍五入/剩余分配得到整数样本量，折线+圆点显示。
#     """
#     rng = np.random.default_rng(seed)
#     qbar = np.ones(N) / N

#     def sample_dirichlet_row(theta):
#         """为一个用户在 N 个类别上生成 q，并据此得到一行 D_per_user 的类别索引"""
#         if np.isinf(theta):
#             q = qbar.copy()  # θ -> ∞ 时，各类均衡
#         else:
#             # v_n ~ Gamma(θ q̄_n, 1), q_n = v_n / sum(v_n)
#             v = rng.gamma(shape=np.maximum(theta * qbar, 1e-12), scale=1.0)
#             q = v / (v.sum() + 1e-12)

#         # 将 q 转成整数计数并精确凑到 D_per_user
#         raw = q * D_per_user
#         counts = np.floor(raw).astype(int)
#         rem = D_per_user - counts.sum()
#         if rem > 0:
#             # 把余数加到小数部分最大的若干类
#             idx = np.argsort(raw - np.floor(raw))[-rem:]
#             counts[idx] += 1
#         elif rem < 0:
#             # 极端情况下（浮点误差），从最大计数里扣
#             idx = np.argsort(counts)[-(-rem):]
#             counts[idx] -= 1

#         # 构造一行类别索引（连续块显示更直观）
#         row = np.empty(D_per_user, dtype=int)
#         start = 0
#         for cls, cnt in enumerate(counts):
#             if cnt > 0:
#                 row[start:start+cnt] = cls
#                 start += cnt
#         if start < D_per_user:
#             row[start:] = 0
#         return row

#     # ===== 左列：三种 θ =====
#     left_mats = []
#     for theta in thetas:
#         mat = np.zeros((U, D_per_user), dtype=int)
#         for u in range(U):
#             mat[u] = sample_dirichlet_row(theta)
#         left_mats.append(mat)

#     # ===== 右列：三种 η =====
#     def zipf_counts(eta):
#         weights = 1.0 / (np.arange(1, U + 1) ** eta)  # u^{-η}
#         p = weights / weights.sum()
#         raw = D_total * p
#         counts = np.floor(raw).astype(int)
#         rem = D_total - counts.sum()
#         if rem > 0:
#             idx = np.argsort(raw - np.floor(raw))[-rem:]
#             counts[idx] += 1
#         return counts

#     right_counts = [zipf_counts(eta) for eta in etas]

#     # ===== 画图 =====
#     fig, axes = plt.subplots(3, 2, figsize=(10, 8), constrained_layout=True)

#     # 离散色图（更像分类色带）
#     cmap = plt.get_cmap('tab20', N)  # 离散 N 色

#     # 左列
#     for i, (theta, mat) in enumerate(zip(thetas, left_mats)):
#         ax = axes[i, 0]
#         im = ax.imshow(mat, aspect='auto', interpolation='nearest', cmap=cmap, vmin=0, vmax=N-1)
#         ax.set_ylabel('User Index')
#         ax.set_xlabel('Data Samples')
#         if np.isinf(theta):
#             title = r'Data Samples, when $\theta \to \infty$'
#         elif theta < 0.1:
#             title = r'Data Samples, when $\theta \approx 0$'
#         else:
#             title = rf'Data Samples, when $\theta={theta}$'
#         ax.set_title(title)

#     # 右列
#     for i, (eta, counts) in enumerate(zip(etas, right_counts)):
#         ax = axes[i, 1]
#         x = np.arange(1, U + 1)
#         ax.plot(x, counts, marker='o')
#         ax.set_xlabel('User Index')
#         ax.set_ylabel('Data Samples')
#         ax.set_title(rf'User Index, when $\eta={eta}$')
#         ax.grid(True, linestyle='--', alpha=0.4)

#     # 可选：显示类别色条（对应左列 N 类）
#     cbar = fig.colorbar(
#         plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=N-1)),
#         ax=[axes[0,0], axes[1,0], axes[2,0]],
#         orientation='vertical', fraction=0.015, pad=0.02
#     )
#     cbar.set_label('Class ID')
#     cbar.set_ticks(np.arange(N) + 0.5)
#     cbar.set_ticklabels([str(i) for i in range(N)])

#     # 保存 & 展示
#     fig.suptitle('Illustration of non-IID modeling (Dirichlet vs Zipf)', fontsize=12)
#     fig.savefig('fig3_demo.png', dpi=200)
#     plt.show()
#     print('Saved to fig3_demo.png')

# if __name__ == '__main__':
#     # draw_fig3()
#     zipf_probs = np.random.zipf(a=1.3, size=3).astype(float)
#     zipf_probs = zipf_probs / zipf_probs.sum()
#     probs = zipf_probs
#     idx_batch = [[], [], []]
#     print(probs)

#     for i, a, b in zip(range(3), probs, idx_batch):
#         print(i, a, b)

    
#     a = np.array([[1, 2, 3]])
#     b = np.array([[1,2,3],[1,1,1]])
#     # a = np.array(a).reshape(2, 3)
#     # a = np.tile(a, (1,2))
#     print(a[:,np.newaxis])
#     print(a[:,np.newaxis] * b)

#     c = np.random.uniform(0,1,2)
#     c = np.tile(c, (3,1))
#     print(c.T)

#     a = [1.010416666666666546e-01, 1.010416666666666546e-01, 1.010416666666666546e-01, 1.248697916666666463e-01, 1.169270833333333204e-01, 1.169270833333333204e-01, 1.010416666666666546e-01, 1.169270833333333204e-01, 1.089843749999999944e-01, 8.515625000000000278e-02, 1.328124999999999722e-01]
#     filename = "equal/10_20250926-161833_global_test_accuracy_equal_0.1_0_zipf_1.0th_6000000.0_10K_2U.txt"
#     # 读取filename的数据
#     a = np.loadtxt(filename)
#     print(np.mean(a))
#     # 平均前十个最大的值
#     a = np.sort(a)[-10:]
#     print(np.mean(a))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 只为触发 3D 支持

Nt, Nq, Nu = 30, 50, 30

# t_k = [x, 20, 0], x ~ U[0,80]
x_t = 80 * np.random.rand(Nt)
t_k = np.column_stack((x_t, 20*np.ones(Nt), np.zeros(Nt)))

# s_q = [x, 6, z], x ~ U[0,100], z ~ U[20,40]
x_s = 100 * np.random.rand(Nq)
z_s = 20 + 20*np.random.rand(Nq)
s_q = np.column_stack((x_s, 6*np.ones(Nq), z_s))

# u_k = [x, 2, z], x ~ U[0,100], z ~ U[80,100]
x_u = 100 * np.random.rand(Nu)
z_u = 80 + 20*np.random.rand(Nu)
u_k = np.column_stack((x_u, 2*np.ones(Nu), z_u))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(t_k[:,0], t_k[:,1], t_k[:,2], marker='^', s=60, label='Transmitter t_k')
ax.scatter(s_q[:,0], s_q[:,1], s_q[:,2], marker='o', s=40, label='Scatterer clusters s_q')
ax.scatter(u_k[:,0], u_k[:,1], u_k[:,2], marker='s', s=60, label='Receiver u_k')

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_xlim(0, 100)
ax.set_ylim(0, 25)
ax.set_zlim(0, 110)
ax.view_init(elev=25, azim=45)
ax.legend()
plt.title('3D Positions of t_k, s_q, u_k')
plt.show()


K = 10   # 设备数，可以按你的仿真要求改
M = 5   # 散射体数

# # 1) 设备位置 q_k 按 t_k = [x, 20, 0], x ~ U[0, 80]
# x_tx = np.random.uniform(0.0, 80.0, size=K)
# q_k = np.zeros((K, 3))
# q_k[:, 0] = x_tx          # x
# q_k[:, 1] = 2.0          # y 固定为 20
# q_k[:, 2] = 0.0           # z 固定为 0

# # 2) 散射体位置 q_m 按 s_q = [x, 6, z], x ~ U[0,100], z ~ U[20,40]
# x_sc = np.random.uniform(0.0, 100.0, size=M)
# z_sc = np.random.uniform(20.0, 40.0, size=M)
# q_m = np.zeros((M, 3))
# q_m[:, 0] = x_sc          # x
# q_m[:, 1] = 6.0           # y 固定为 6
# q_m[:, 2] = z_sc          # z in [20, 40]

rng = np.random.default_rng(42)

x_tx = rng.uniform(0.0, 80.0, size=K)
q_k = np.zeros((K, 3))
q_k[:, 0] = x_tx          # x
q_k[:, 1] = 2.0           # y 固定为 2
q_k[:, 2] = 0.0           # z 固定为 0

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
q_B = np.array([50, 20.0, 50])   # shape (3,)

# 3) BS 位置 q_B 按 u_k = [x, 2, z], x ~ U[0,100], z ~ U[80,100]
x_B = np.random.uniform(0.0, 100.0)
z_B = np.random.uniform(80.0, 100.0)
q_B = np.array([x_B, 20.0, z_B])   # shape (3,)

rng = np.random.default_rng(42)

x_tx = rng.uniform(0.0, 80.0, size=K)
q_k = np.zeros((K, 3))
q_k[:, 0] = x_tx          # x
q_k[:, 1] = 2.0           # y 固定为 2
q_k[:, 2] = 0.0           # z 固定为 0

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
q_B = np.array([50.0, 100.0, 50.0])   # shape (3,)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
print(q_B[0])
print(q_B[1])
print(q_B[2])

ax.scatter(q_k[:, 0], q_k[:, 1], q_k[:, 2], marker='^', s=60, label='Devices q_k')
ax.scatter(q_m[:, 0], q_m[:, 1], q_m[:, 2], marker='o', s=40, label='Scatterers q_m')
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
fig.savefig('q_positions.png', dpi=200)
plt.show()

a = [[1,2],[3,4]]
print(np.mean(np.array(a), axis=0).tolist())
b = np.mean(np.array(a), axis=0).tolist()
print(b)
print(np.min(b).tolist()) 


filename = "data.txt"  # 请根据需要改为你的txt文件路径
try:
    data = np.loadtxt(filename)
except Exception as e:
    print(f"无法读取文件 {filename}: {e}")
else:
    arr = np.asarray(data).ravel()
    if arr.size == 0:
        print("文件中没有数据。")
    else:
        overall_mean = arr.mean()
        k = min(10, arr.size)
        topk_mean = np.sort(arr)[-k:].mean()
        print(f"全部数据的平均值: {overall_mean}")
        print(f"前{k}个最大数据值的平均值: {topk_mean}")

# pro_mi
a1 = 7.055206739368405e-08
a3 = 1.6527650676528152e-07
a5 = 2.579776259549082e-07
a7 = 3.512555668781452e-07
a9 = 4.461224869411574e-07
print(10*np.log10(a1))
print(10*np.log10(a3))
print(10*np.log10(a5))
print(10*np.log10(a7))
print(10*np.log10(a9))
print("-----")

#ran_mi
q1 = 2.6892862495469827e-11
q3 = 4.614452258146064e-17
q5 = 1.0704263591253309e-22
q7 = 2.7435119008633143e-28
q9 = 7.395952137882013e-34
print(10*np.log10(q1))
print(10*np.log10(q3))
print(10*np.log10(q5))
print(10*np.log10(q7))
print(10*np.log10(q9))
print("-----")

#fix_mi
b1 = 1.7070930076058992e-11
b3 = 1.7896150766854614e-18
b5 = 1.2635136442562187e-25
b7 = 7.74112661820376e-33
b9 = 4.405469248434784e-40
print(10*np.log10(b1))
print(10*np.log10(b3))
print(10*np.log10(b5))
print(10*np.log10(b7))
print(10*np.log10(b9))
print("-----")

#non
non = 2.3550912601851262e-08
print(10*np.log10(non))
print("-----")

#pro_angle
ag1 = 1.5344876906022405e-16
ag2 = 4.007536080072812e-09
ag3 = 4.792661548938059e-08
ag4 = 1.6660478438619171e-07
ag5 = 2.120758700696183e-07
print(10*np.log10(ag1))
print(10*np.log10(ag2))
print(10*np.log10(ag3))
print(10*np.log10(ag4))
print(10*np.log10(ag5))
print("-----")

#fix_angle
agfix1 = 4.877159174936739e-22
print(10*np.log10(agfix1))
print("-----")

#ran_angle
agran1 = 8.653673712043273e-22
agran2 = 2.7066298561219547e-21
agran3 = 8.792586131856487e-21
agran4 = 2.5986507310679037e-20
agran5 = 7.077441319809767e-20
print(10*np.log10(agran1))
print(10*np.log10(agran2))
print(10*np.log10(agran3))
print(10*np.log10(agran4))
print(10*np.log10(agran5))


print(10*np.log10(1.62228845e-07))

