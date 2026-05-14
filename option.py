import argparse
import numpy as np
np.random.seed(0)


def args_parser():
    parser = argparse.ArgumentParser()
    num_clients = 10

    # 学习参数
    #Data specifc paremeters
    parser.add_argument('--dataset', default='FMNIST',
                        help='CIFAR10, CIFAR100, SVHN, EMNIST, FMNIST')

    parser.add_argument('--model', default= 'CNN',
                        help='CNN resnet18 shufflenet DNN') 
    
    parser.add_argument('--beta', type=float,default=0.1,
                        help='beta for non-iid distribution')
    parser.add_argument('--part', type=float,default=1,
                        help='percentage of each local data')
    parser.add_argument('--lam', type=float, default=0.04,
                        help='hyper-parameter for loss2')
    parser.add_argument('--num_classes', type=int,default=10,
                        help='number of classes')
    
    #Training specifc parameters
    parser.add_argument('--log_frq', type=int, default=5,
                        help='frequency of logging')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='number of epochs')  
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')        
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=10,
                        help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=False,
                        help='Use lr rate scheduler')
    
    parser.add_argument('--sampling_rate', type=float,default=1,
                        help='frac of local models to update')
    parser.add_argument('--local_ep',type=int, default=5,
                        help='iterations of local updating')
    parser.add_argument('--seed', type=int,default=0,
                        help='random seed for generating datasets')
    parser.add_argument('--code_len', type=int,default=32,
                        help='length of code')
    parser.add_argument('--alg', default='FedAvg',
                        help='FedAvg, FedProx, Moon, FedMD, Fedproto, FedDFKD')
    
    parser.add_argument('--gamma', type=float, default=0.05,
                        help='hyper-parameter for loss3')
    
    parser.add_argument('--std', type=float, default=2,
                        help='std of gaussian noise ')
    
    parser.add_argument('--zipf', type=float,default=0,
                        help='zipf parameter for UAVs') 
    
    parser.add_argument('--temp', type=float,default=0.5,
                        help='temperture for soft prediction')
    
    
    parser.add_argument('--save_model', action="store_true", default= True,
                        help='saved model parameters')
    parser.add_argument('--upload_model', action="store_true", default= True,
                        help='upload parameters')
    parser.add_argument('--eval_only', action="store_true", default=False,help='evaluate the model')

    parser.add_argument('--dataspilt', type=str, default='fixed_m_iid',
                        help='data split method (original, fixed_m_iid)')
    parser.add_argument('--m', type=int, default=200,
                        help='number of samples per client for fixed_m_iid data split')
    

   # 环境参数

    parser.add_argument('--num_clients',  type=int, default=num_clients,
                        help='number of local models')
    parser.add_argument('--num_scatterers', type=int, default=5,
                        help='number of scatterers between client and base station')
    
    
    parser.add_argument('--transmit_power_max', type=float, default=0.2,
                        help='transmit power of client (W)')
    
    
    parser.add_argument('--bandwidth', type=int, default=10e6,
                        help='bandwidth between UAV and base station (Hz)')


    parser.add_argument('--latency_threshold', type=float, default=0.1,
                        help='maximum allowable latency (seconds)')
    
    
    parser.add_argument('--op_method', type=str, default='proposed_with_h_order',
                        help='optimization method for RO and PA (proposed_with_h_order, proposed_with_ph_order)')
    
    parser.add_argument('--NOMA_method', type=str, default='NOMA',
                        help='NOMA or OMA')
    
    parser.add_argument('--LoS_method', type=str, default='not_closed',
                        help='whether consider only LoS path or not (closed, not_closed)')
    
    
    parser.add_argument('--RO_method', type=str, default='proposed_RO',
                        help='Rotatable Antenna orientation method (proposed_RO, random_RO, fixed_RO, none_RO(mi in proposed_RO is zero, mi=0))')
    
    parser.add_argument('--PS_method', type=str, default='proposed_PS',
                        help='power allocation method for clients (proposed_PS, maximum_PS, random_PS, channel_inversion_PS)')
    
    parser.add_argument('--random_not', type=str, default='random',
                        help='whether to use random seed or not (random, fixed)')

    parser.add_argument('--mi', type=int, default=4, help='mi of direction gain')

    parser.add_argument('--phi_max', type=float, default=np.pi*1/10, help='maximum eccentric angle that each RA is allowed to ajust.')

    parser.add_argument('--wavelength', type=float, default=0.1, help='wavelength of carrier frequency (m)')

        # 处理每个样本大约需要 1e3~1e4 CPU cycles，常用取值为 1e3 或 1e4
    parser.add_argument('--CPUcycles_per_sample', type=int, default=80,
                        help='number of CPU cycles required per sample')
    
    # 用户设备的计算能力（每秒可执行的CPU cycles 数），常用范围为 1e8 ~ 1e9
    parser.add_argument('--client_computation_capacity', type=int, default=np.array([1e9]*num_clients), help='computation capacity of client device (CPU cycles per second)')


    

    
    



    args = parser.parse_args()
    return args