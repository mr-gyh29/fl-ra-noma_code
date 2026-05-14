
import torch
import numpy as np
import os,sys,os.path
from tensorboardX import SummaryWriter
import pickle
from torch import nn
import hashlib
import argparse
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset, SubsetRandomSampler

from models import CNNFemnist,ResNet18,ShuffLeNet
from sampling import *
from option import args_parser

from Server.ServerFedAvg import ServerFedAvg
from Server.ServerFedProx import ServerFedProx
from Server.ServerFedMD import ServerFedMD 
from Server.ServerFedProto import ServerFedProto
from Server.ServerFedHKD import ServerFedHKD
from models import DNN



print(torch.__version__)
torch.cuda.is_available()
np.set_printoptions(threshold=np.inf)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device.type)

args = args_parser()
print(type(args))
args_hash = ''
for k,v in vars(args).items():
    if k == 'eval_only':
        continue
    args_hash += str(k)+str(v)
    
args_hash = hashlib.sha256(args_hash.encode()).hexdigest()




train_dataset, testset, dict_users, dict_users_test = partition_data_fixed_m_noniid_onenine(
    n_users=args.num_clients,
    m=args.m,              # 你自己在 args 里加一个 m
    alpha=args.beta,       # non-IID 强度：越小越非IID
    rand_seed=args.seed,
    dataset=str(args.dataset)
)

# train_dataset,testset, dict_users, dict_users_test = partition_data(args=args, n_users = args.num_clients, n_clients= args.num_clients, zipf= args.zipf, alpha=args.beta,rand_seed = args.seed, dataset=str(args.dataset))

datapilter = ' '
if datapilter == 'non-iid-2class':
    # Repartition train data so each client has exactly two classes and 200-400 samples (random)
    rng = np.random.RandomState(args.seed)

    # collect indices per class from the training dataset
    class_to_indices = {i: [] for i in range(args.num_classes)}
    for idx, (_, label) in enumerate(train_dataset):
        class_to_indices[int(label)].append(idx)

    # keep an immutable copy of original class lists for replacement sampling if needed
    orig_class_indices = {k: v.copy() for k, v in class_to_indices.items()}

    # shuffle per-class pools
    for v in class_to_indices.values():
        rng.shuffle(v)

    new_dict_users = {}
    for client_id in range(args.num_clients):
        # pick two distinct classes for this client
        classes = rng.choice(np.arange(args.num_classes), size=2, replace=False)
        total_samples = rng.randint(200, 401)  # inclusive [200,400]
        # split samples between the two classes (at least 1 and at most total-1)
        n1 = rng.randint(1, total_samples)
        n2 = total_samples - n1

        client_indices = []
        for cls, need in zip(classes, (n1, n2)):
            available = class_to_indices[cls]
            if len(available) >= need:
                # take without replacement from remaining pool
                take = [available.pop() for _ in range(need)]
            else:
                # take all remaining, then sample the shortfall with replacement from original pool
                take = available.copy()
                class_to_indices[cls] = []
                shortfall = need - len(take)
                if len(orig_class_indices[cls]) == 0:
                    raise ValueError(f"No samples available for class {cls} in the dataset.")
                extra = list(rng.choice(orig_class_indices[cls], size=shortfall, replace=True))
                take += extra
            client_indices.extend(take)

        # ensure indices are ints and unique order doesn't matter
        new_dict_users[client_id] = [int(i) for i in client_indices]

    dict_users = new_dict_users


Loaders_train = LocalDataloaders(train_dataset,dict_users,args.batch_size,ShuffleorNot = True,frac=args.part)
Loaders_test = LocalDataloaders(testset,dict_users_test,args.batch_size,ShuffleorNot = True,frac=2*args.part)

# subset_datasets = []
# for loader in Loaders_test:
#     dataset = loader.dataset
#     indices = loader.sampler.indices if hasattr(loader.sampler, 'indices') else list(range(len(dataset)))
#     half_len = int(len(indices)*0.1)
#     selected_indices = np.random.choice(indices, half_len, replace=False)
#     subset_datasets.append(Subset(dataset, selected_indices))

# # 汇总所有子集
# global_test_subset = ConcatDataset(subset_datasets)

# global_loader_test = torch.utils.data.DataLoader(global_test_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
# global_loader_test = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,shuffle=True, num_workers=2)

global_test_method = 'uniform_200'
if global_test_method == 'uniform_200':
    # 对于global_loader_test，我想从十类中各均匀采样200个样本，总共2000个样本
    indices_per_class = {i: [] for i in range(args.num_classes)}
    for idx, (data, label) in enumerate(testset):
        indices_per_class[label].append(idx)
    selected_indices = []
    samples_per_class = 200
    for class_indices in indices_per_class.values():
        selected_indices.extend(np.random.choice(class_indices, samples_per_class, replace=False))
    global_test_subset = Subset(testset, selected_indices)
    global_loader_test = torch.utils.data.DataLoader(global_test_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
else:
    global_loader_test = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,shuffle=True, num_workers=2)


for idx in range(args.num_clients):
    counts = [0]*args.num_classes
    for batch_idx,(X,y) in enumerate(Loaders_train[idx]):
        batch = len(y)
        y = np.array(y)
        for i in range(batch):
            counts[int(y[i])] += 1
    print('Client {} data distribution:'.format(idx))
    print(counts)





logger = SummaryWriter('./logs')
checkpoint_dir = './checkpoint/'+ args.dataset + '/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
print('Checkpoint dir:', checkpoint_dir)




print(args.model)
if args.model == 'CNN':
    # for EMNIST 62 classes
    global_model = CNNFemnist(args, code_length=args.code_len, num_classes = args.num_classes)
    
if args.model == 'resnet18':
    global_model = ResNet18(args, code_length=args.code_len, num_classes = args.num_classes)

if args.model == 'shufflenet':  
    global_model = ShuffLeNet(args, code_length=args.code_len, num_classes = args.num_classes)
if args.model == 'DNN':  
    global_model = DNN(args, code_length=args.code_len, num_classes = args.num_classes)

   
print('# model parameters:', sum(param.numel() for param in global_model.parameters()))
# global_model = nn.DataParallel(global_model)
global_model.to(device)





if args.alg == 'FedAvg':
    server = ServerFedAvg(args,global_model,Loaders_train,Loaders_test,global_loader_test,logger,device)
if args.alg == 'FedProx':
    server = ServerFedProx(args,global_model,Loaders_train,Loaders_test,global_loader_test,logger,device)
if args.alg == 'FedMD':
    server = ServerFedMD(args,global_model,Loaders_train,Loaders_test,global_loader_test,testset,logger,device)
if args.alg == 'FedProto':    
    server = ServerFedProto(args,global_model,Loaders_train,Loaders_test,global_loader_test,logger,device)
if args.alg == 'FedHKD':    
    server = ServerFedHKD(args,global_model,Loaders_train,Loaders_test,global_loader_test,logger,device)


server.Create_Clints()
server.train()

save_path = checkpoint_dir + args_hash + '.pth'
if args.save_model == True:
    server.Save_CheckPoint(save_path)
    print('Model is saved on: ')
    print(save_path)






