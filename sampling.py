import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
from torchvision import datasets, transforms


class LocalDataset(Dataset):
    """
    because torch.dataloader need override __getitem__() to iterate by index
    this class is map the index to local dataloader into the whole dataloader
    """
    def __init__(self, dataset, Dict):
        self.dataset = dataset
        self.idxs = [int(i) for i in Dict]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        X, y = self.dataset[self.idxs[item]]
        return X, y
# 载入数据集


def LocalDataloaders(dataset, dict_users, batch_size, ShuffleorNot = True, BatchorNot = True, frac = 1):
    """
    dataset: the same dataset object
    dict_users: dictionary of index of each local model
    batch_size: batch size for each dataloader
    ShuffleorNot: Shuffle or Not
    BatchorNot: if False, the dataloader will give the full length of data instead of a batch, for testing
    """
    num_users = len(dict_users)
    loaders = []
    for i in range(num_users):
        num_data = len(dict_users[i])
        frac_num_data = int(frac*num_data)
        if frac_num_data < batch_size:
            frac_num_data = num_data
        whole_range = range(num_data)
        frac_range = np.random.choice(whole_range, frac_num_data) # choice data index for users
        frac_dict_users = [dict_users[i][j] for j in frac_range] # choice data for users
        if BatchorNot== True:
            loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,frac_dict_users),
                        batch_size=batch_size,
                        shuffle = ShuffleorNot,
                        num_workers=0,
                        drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,frac_dict_users),
                        batch_size=len(LocalDataset(dataset,dict_users[i])),
                        shuffle = ShuffleorNot,
                        num_workers=0,
                        drop_last=True)
        loaders.append(loader)

    return loaders


def partition_data_fixed_m_noniid(
    n_users: int,
    m: int,
    alpha: float = 0.5,
    rand_seed: int = 0,
    dataset: str = 'FMNIST',
):
    """
    Fixed-m per client + controllable non-IID via Dirichlet over classes.
    Train indices are disjoint across clients.

    Returns:
      train_dataset, test_dataset, net_dataidx_map, net_dataidx_map_test
    """
    dataset_upper = dataset.upper()

    # -----------------------------
    # 1) Load dataset
    # -----------------------------
    if dataset_upper == 'FMNIST':
        K = 10
        data_dir = '../data/FMNIST/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test  = np.array(test_dataset.targets)

    elif dataset_upper == 'CIFAR10':
        K = 10
        data_dir = '../data/cifar10/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset  = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test  = np.array(test_dataset.targets)

    elif dataset_upper == 'CIFAR100':
        K = 100
        data_dir = '../data/cifar100/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset  = datasets.CIFAR100(data_dir, train=False, download=True, transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test  = np.array(test_dataset.targets)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    N = len(train_dataset)
    N_test = len(test_dataset)

    if n_users * m > N:
        raise ValueError(f"Require n_users*m <= N. Got {n_users}*{m}={n_users*m} > {N}.")

    rng = np.random.default_rng(rand_seed)

    # -----------------------------
    # 2) Build per-class index pools (train)
    # -----------------------------
    class_pools = []
    for k in range(K):
        idx_k = np.where(y_train == k)[0]
        rng.shuffle(idx_k)
        class_pools.append(idx_k.tolist())  # list for pop()

    # -----------------------------
    # 3) Decide each client's per-class demand via Dirichlet + Multinomial
    #    demand[i, k] are integers, sum_k demand[i,k] == m
    # -----------------------------
    demand = np.zeros((n_users, K), dtype=int)
    for i in range(n_users):
        q = rng.dirichlet(alpha * np.ones(K))
        demand[i, :] = rng.multinomial(m, q)

    # -----------------------------
    # 4) Allocate disjoint train indices according to demand (best-effort)
    #    If a class pool is insufficient, we allocate what we can and leave remainder to be filled later.
    # -----------------------------
    net_dataidx_map = {i: [] for i in range(n_users)}
    remaining_need = np.full(n_users, m, dtype=int)

    # First pass: allocate by class
    for k in range(K):
        # total requested from class k
        req_k = demand[:, k].copy()
        total_req = int(req_k.sum())
        avail = len(class_pools[k])
        if avail == 0 or total_req == 0:
            continue

        # If requests exceed availability, scale down proportionally (deterministic tie-break by fractional part)
        if total_req > avail:
            scale = avail / total_req
            scaled = req_k * scale
            base = np.floor(scaled).astype(int)
            remainder = avail - int(base.sum())
            frac = scaled - base

            # distribute leftover 1-by-1 to largest fractional parts
            order = np.argsort(-frac)
            for t in range(remainder):
                base[order[t % n_users]] += 1
            req_k = base

        # Allocate req_k[i] samples of class k to each client i
        for i in range(n_users):
            take = min(req_k[i], remaining_need[i], len(class_pools[k]))
            if take > 0:
                # pop from class pool
                picked = [class_pools[k].pop() for _ in range(take)]
                net_dataidx_map[i].extend(picked)
                remaining_need[i] -= take

    # Second pass: fill any remaining_need from the leftover global pool (all classes)
    # Build leftover pool
    leftover = []
    for k in range(K):
        leftover.extend(class_pools[k])
    rng.shuffle(leftover)

    ptr = 0
    for i in range(n_users):
        need = remaining_need[i]
        if need > 0:
            if ptr + need > len(leftover):
                raise RuntimeError("Not enough leftover samples to fill fixed m. This should not happen when n_users*m <= N.")
            net_dataidx_map[i].extend(leftover[ptr:ptr+need])
            ptr += need
            remaining_need[i] = 0

    # (Optional) shuffle each client's indices
    for i in range(n_users):
        rng.shuffle(net_dataidx_map[i])

    # -----------------------------
    # 5) Test set partition: approx IID & balanced (your方案B)
    # -----------------------------
    net_dataidx_map_test = {i: [] for i in range(n_users)}
    for k in range(K):
        idx_k_test = np.where(y_test == k)[0]
        rng.shuffle(idx_k_test)
        splits = np.array_split(idx_k_test, n_users)
        for i in range(n_users):
            net_dataidx_map_test[i].extend(splits[i].tolist())
    for i in range(n_users):
        rng.shuffle(net_dataidx_map_test[i])

    # -----------------------------
    # 6) Sanity print (optional)
    # -----------------------------
    sizes = [len(net_dataidx_map[i]) for i in range(n_users)]
    print("per-client train sizes:", sizes[:min(n_users,10)], "...")
    print("min/mean/max:", min(sizes), sum(sizes)/len(sizes), max(sizes))
    print("total train used:", sum(sizes), f"(should be {n_users*m})")

    return train_dataset, test_dataset, net_dataidx_map, net_dataidx_map_test


def partition_data_fixed_m_noniid_onenine(
    n_users: int,
    m: int,
    alpha: float = 0.5,
    rand_seed: int = 0,
    dataset: str = 'FMNIST',
):
    """
    Fixed-m per client + non-IID via Dirichlet allocation for other categories, with main class fixed.
    Each user has a dominant class, and other categories' samples are allocated with non-IID distribution.

    Returns:
      train_dataset, test_dataset, net_dataidx_map, net_dataidx_map_test
    """
    dataset_upper = dataset.upper()
    n_users = 10
    # -----------------------------
    # 1) Load dataset
    # -----------------------------
    if dataset_upper == 'FMNIST':
        K = 10  # 10 classes for FMNIST
        data_dir = '../data/FMNIST/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test  = np.array(test_dataset.targets)
    
    N = len(train_dataset)  # Total number of training samples
    N_test = len(test_dataset)  # Total number of test samples

    if n_users * m > N:
        raise ValueError(f"Require n_users * m <= N. Got {n_users} * {m} = {n_users * m} > {N}.")

    # rand_seed = np.random.seed()
    rng = np.random.default_rng(rand_seed)

    # -----------------------------
    # 2) Build per-class index pools (train)
    # -----------------------------
    class_pools = []
    for k in range(K):
        idx_k = np.where(y_train == k)[0]
        rng.shuffle(idx_k)
        class_pools.append(idx_k.tolist())  # list for pop()

    # -----------------------------
    # 3) Decide each user's main class and other samples
    # -----------------------------
    net_dataidx_map = {i: [] for i in range(n_users)}
    remaining_need = np.full(n_users, m, dtype=int)

    # Step 1: Assign each user a dominant class and some other class samples
    all_classes = list(range(K))
    main_classes = rng.choice(all_classes, size=n_users, replace=False)  # Assign a unique main class for each user
    
    for i in range(n_users):
        main_class = main_classes[i]
        net_dataidx_map[i].extend(class_pools[main_class][:m * 2 // 3])  # Assign main class samples (two-thirds of m)

        # Step 2: Distribute remaining samples among other classes using Dirichlet distribution
        remaining_samples = m - len(net_dataidx_map[i])  # Remaining number of samples for the user
        remaining_classes = [k for k in range(K) if k != main_class]  # Other classes except the user's main class
        
        # Use Dirichlet distribution to assign proportions to the remaining classes
        proportions = rng.dirichlet(np.repeat(alpha, len(remaining_classes)))
        # Scale the proportions to fit the remaining samples
        proportions = proportions * remaining_samples
        proportions = np.floor(proportions).astype(int)  # Allocate the floor of the proportions
        remainder = remaining_samples - np.sum(proportions)  # Calculate remaining samples after floor allocation
        
        # Randomly distribute the remainder
        for j in range(remainder):
            proportions[j % len(proportions)] += 1
        
        # Assign samples from remaining classes
        for idx, class_id in enumerate(remaining_classes):
            num_samples = proportions[idx]
            if num_samples > 0:
                net_dataidx_map[i].extend(class_pools[class_id][:num_samples])
                class_pools[class_id] = class_pools[class_id][num_samples:]

    # -----------------------------
    # 4) Test set partition: each class 200 samples (2000 samples total)
    # -----------------------------
    net_dataidx_map_test = {i: [] for i in range(n_users)}
    test_samples_per_class = 200  # Each class will have 200 samples in the test set
    
    for k in range(K):
        idx_k_test = np.where(y_test == k)[0]
        rng.shuffle(idx_k_test)
        net_dataidx_map_test[k] = idx_k_test[:test_samples_per_class]  # Select 200 samples per class
    
    # Shuffle test data
    for k in range(K):
        rng.shuffle(net_dataidx_map_test[k])

    # -----------------------------
    # 5) Sanity print (optional)
    # -----------------------------
    sizes = [len(net_dataidx_map[i]) for i in range(n_users)]
    print("per-client train sizes:", sizes[:min(n_users, 10)], "...")
    print("min/mean/max:", min(sizes), sum(sizes) / len(sizes), max(sizes))
    print("total train used:", sum(sizes), f"(should be {n_users * m})")

    return train_dataset, test_dataset, net_dataidx_map, net_dataidx_map_test

def partition_data(args, n_users, n_clients, zipf=1.3, alpha=0.5, rand_seed=0, dataset='cifar10'):
    """
    Partitions the dataset into non-IID training subsets for federated learning,
    and approximately IID & balanced test subsets (方案 B).

    Parameters:
    ----------
    n_users : int
        Number of users/clients to partition the data among.
    n_clients : int
        Number of clients for Zipf distribution (should typically equal n_users).
    zipf : float, optional
        If 0, use Dirichlet-only partition. If >0, use Zipf-based heterogeneous sizes
        for training data. Default is 1.3.
    alpha : float, optional
        Dirichlet concentration parameter controlling non-IID degree for training data.
        Smaller alpha -> more non-IID. Default is 0.5.
    rand_seed : int, optional
        Random seed for reproducibility. Default is 0.
    dataset : str, optional
        Dataset name: 'CIFAR10', 'CIFAR100', 'EMNIST', 'SVHN', 'FMNIST' (case-insensitive).
        Default is 'cifar10'.

    Returns:
    -------
    train_dataset : torch.utils.data.Dataset
        The training dataset.
    test_dataset : torch.utils.data.Dataset
        The testing dataset.
    net_dataidx_map : dict
        Mapping: user_id -> list of training indices.
    net_dataidx_map_test : dict
        Mapping: user_id -> list of test indices (approx. IID & balanced across users).
    """

    # -----------------------------
    # 1) 加载数据集 & 标签
    # -----------------------------
    dataset_upper = dataset.upper()

    if dataset_upper == 'CIFAR10':
        K = 10
        data_dir = '../data/cifar10/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)

    elif dataset_upper == 'CIFAR100':
        K = 100
        data_dir = '../data/cifar100/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                          transform=apply_transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                         transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)

    elif dataset_upper == 'EMNIST':
        K = 62
        data_dir = '../data/EMNIST/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )
        train_dataset = datasets.EMNIST(data_dir, train=True, split='byclass', download=True,
                                        transform=apply_transform)
        test_dataset = datasets.EMNIST(data_dir, train=False, split='byclass', download=True,
                                       transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)

    elif dataset_upper == 'SVHN':
        K = 10
        data_dir = '../data/SVHN/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        train_dataset = datasets.SVHN(data_dir, split='train', download=True,
                                      transform=apply_transform)
        test_dataset = datasets.SVHN(data_dir, split='test', download=True,
                                     transform=apply_transform)
        y_train = np.array(train_dataset.labels)
        y_test = np.array(test_dataset.labels)

    elif dataset_upper == 'FMNIST':
        K = 10
        data_dir = '../data/FMNIST/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                              transform=apply_transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                             transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if args.dataspilt == 'original':
        # -----------------------------
        # 2) 初始化
        # -----------------------------
        N = len(train_dataset)
        N_test = len(test_dataset)

        net_dataidx_map = {}

        np.random.seed(rand_seed)

        # -----------------------------
        # 3) 训练集划分（非 IID）
        # -----------------------------
        min_size = 0  # 每个 client 至少 10 个训练样本

        # zipf == 0: Dirichlet 非 IID，容量约束为 N / n_users
        if zipf == 0:
            while min_size < 10:
                idx_batch = [[] for _ in range(n_users)]
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)

                    # Dirichlet 分布生成类别 k 在各用户间的比例
                    proportions = np.random.dirichlet(np.repeat(alpha, n_users))

                    # 按当前已分配数量进行一个“软”容量约束
                    proportions = np.array([
                        p * (len(idx_j) < N / n_users)
                        for p, idx_j in zip(proportions, idx_batch)
                    ])

                    # 防止全为 0 的极端情况
                    if proportions.sum() == 0:
                        proportions = np.repeat(1.0 / n_users, n_users)
                    else:
                        proportions = proportions / proportions.sum()

                    # 转成切分点
                    split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                    # 按比例划分类别 k 的样本索引给各用户
                    for j, idx in enumerate(np.split(idx_k, split_points)):
                        idx_batch[j].extend(idx.tolist())

                min_size = min(len(idx_j) for idx_j in idx_batch)

        

        # 将训练索引存入字典
        for j in range(n_users):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    elif args.dataspilt == 'fixed_m_iid':
        net_dataidx_map = partition_fixed_m_iid(y_train, n_users, m=args.m, seed=rand_seed)


    # -----------------------------
    # 4) 测试集划分（方案 B：近似 IID & 均衡）
    #    不受 Dirichlet / Zipf 控制，只做“按类分配 + array_split”。
    # -----------------------------
    # 初始化每个用户的测试索引列表
    user_test_indices = {j: [] for j in range(n_users)}
    net_dataidx_map_test = {}
    for k in range(K):
        idx_k_test = np.where(y_test == k)[0]
        np.random.shuffle(idx_k_test)

        # 将类别 k 的测试样本近似均匀地切成 n_users 份
        splits = np.array_split(idx_k_test, n_users)

        for j in range(n_users):
            user_test_indices[j].extend(splits[j].tolist())

    # 最后对每个用户的测试索引再打乱一下顺序
    for j in range(n_users):
        idx_j_test = user_test_indices[j]
        np.random.shuffle(idx_j_test)
        net_dataidx_map_test[j] = idx_j_test


    

    # -----------------------------
    # 5) 返回
    # -----------------------------
    return train_dataset, test_dataset, net_dataidx_map, net_dataidx_map_test


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts
