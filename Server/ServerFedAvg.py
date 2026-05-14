
from torch.utils.data import Dataset
import torch
import copy
from utils import Accuracy
from Server.ServerBase import Server
from Client.ClientFedAvg import ClientFedAvg
from tqdm import tqdm
import numpy as np
from utils import average_weights
from mem_utils import MemReporter
import time
import cvxpy as cp
import matplotlib.pyplot as plt
from optimization_proposed_with_h_order import overall_optimization_RA_and_PS
# 固定随机种子
import random
random.seed(0)
np.random.seed(0)


class ServerFedAvg(Server):
    def __init__(self, args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device):
        super().__init__(args, global_model,Loader_train,Loaders_local_test,Loader_global_test,logger,device)
       
    
    def Create_Clints(self):
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedAvg(self.args, copy.deepcopy(self.global_model),self.Loaders_train[idx], self.Loaders_local_test[idx], idx=idx, logger=self.logger, code_length = self.args.code_len, num_classes = self.args.num_classes, device=self.device))


    def save_records(self, data_list, count, name="record"):
        if self.args.mi == 0:
            self.args.RO_method = "none_RO"
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{self.args.op_method}/{count+1}_{name}_{self.args.RO_method}_{self.args.PS_method}_{self.args.beta}_{self.args.zipf}_zipf_{self.args.latency_threshold}th_{self.args.transmit_power_max}W_{self.args.mi}mi_{self.args.phi_max*10}_{self.args.bandwidth/1e6}MHz_{self.args.num_clients}K_{self.args.num_scatterers}M_{self.args.m}m_{self.args.random_not}_{self.args.NOMA_method}_{self.args.LoS_method}.txt"
        arr = np.array(data_list)
        if arr.ndim > 2:
            # Fallback to saving as a list of strings if array is not 1D or 2D
            with open(filename, 'w') as f:
                for item in data_list:
                    f.write(f"{item}\n")
        else:
            np.savetxt(filename, arr)

    def print_figure(self, data_list, count, method, name="record", xlabel="x", ylabel="y", title="Title"):
        if self.args.mi == 0:
            self.args.RO_method = "none_RO"
        # 获得当前时间
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.figure()
        plt.plot(data_list, marker='o')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.savefig(f'{self.args.op_method}/{count+1}_{name}_{self.args.RO_method}_{self.args.PS_method}_{self.args.beta}_{self.args.zipf}zipf_{self.args.latency_threshold}th_{self.args.transmit_power_max}W_{self.args.mi}mi_{self.args.phi_max*10}_{self.args.bandwidth/1e6}MHz_{self.args.num_clients}K_{self.args.num_scatterers}M_{self.args.m}m_{self.args.random_not}_{self.args.NOMA_method}_{self.args.LoS_method}_curve.png')
        plt.show()

    def OMA_training_selection(self, h_k, p_k, D_k, model_size=7e5):
        # print("进入！！！！！！！！！！！！！！！！！！！！！！！！！！")
#         h_k = [2.43753423e-07, 1.89795425e-07, 2.35390739e-07, 2.17839205e-07,
#  2.10490063e-07, 2.39206250e-07, 1.79038241e-07, 2.26181363e-07,
#  2.27240641e-07, 1.84104512e-07]
#         print("D_k", D_k)
        S1 = np.zeros(self.args.num_clients)
        power_per_user = self.args.transmit_power_max * np.ones(self.args.num_clients)
        bandwidth_per_user = self.args.bandwidth / self.args.num_clients
        print("Bandwidth per user:", bandwidth_per_user)
        # 计算每个用户的时延
        for k in range(self.args.num_clients):
            # 计算每个用户的传输时延
            transmit_rate = bandwidth_per_user * np.log2(1 + (power_per_user[k] * h_k[k]/(1e-9)))
            transmission_delay = model_size / transmit_rate  # 将数据大小
            # 计算每个用户的计算时延
            computation_delay = (D_k[k] * self.args.CPUcycles_per_sample) / self.args.client_computation_capacity[k]
            total_delay = transmission_delay + computation_delay
            print(f"Client {k} - h_k: {h_k[k]:.23f}, Transmit Rate: {transmit_rate/1e6:.23f} Mbps, Transmission Delay: {transmission_delay:.4f} s, Computation Delay: {computation_delay:.4f} s, Total Delay: {total_delay:.4f} s")
            if total_delay <= self.args.latency_threshold:
                S1[k] = 1
        return S1

    def op_RO_and_PS(self):
        if self.args.op_method == 'proposed_with_h_order':
            S1, h, D_k, p_k, sic_order = overall_optimization_RA_and_PS(self.args)
            # print("优化结果：p_k=", p_k)
            # print("解码顺序：sic_order=", sic_order)
            # 根据h, D_k, p_k, sic_order计算NOMA的传输时延
            # 假定 sic_order 是一个可迭代序列，表示解码顺序（先解码的放前面）。
            # 使用上行 NOMA 的常见 SINR 计算：在解码某用户时，尚未被 SIC 消除的用户（位于该用户之后）的接收功率视为干扰。
            # 根据解码顺序计算每个用户的干扰，记录在一个list里
            # interference_list = [0.0 for _ in range(self.args.num_clients)]
            # sic_order = list(sic_order)
            # for idx in sic_order:
            #     interference = 0.0
            #     for jdx in sic_order:
            #         if sic_order.index(jdx) > sic_order.index(idx):  # j 在 idx 之后解码，视为干扰
            #             interference += p_k[jdx] * (h[jdx] ** 2)
            #     interference_list[idx] = interference

            # # 可选：打印或记录干扰列表，便于调试
            # print("Interference per user:", interference_list)
            
            # for k in range(self.args.num_clients):
            #     sinr_k = (p_k[k] * (h[k] ** 2)) / (interference_list[k] + 1e-9)  # 添加一个小值以避免除以零
            #     transmit_rate = self.args.bandwidth * np.log2(1 + sinr_k)
            #     transmission_delay = D_k[k] / transmit_rate  # 将数据大小 D_k[k] 代入
            #     computation_delay = (D_k[k] * self.args.CPUcycles_per_sample) / self.args.client_computation_capacity[k]
            #     total_delay = transmission_delay + computation_delay
            #     print(f"NOMA Client {k} - h_k: {h[k]:.23f}, SINR: {sinr_k:.23f}, Transmit Rate: {transmit_rate/1e6:.23f} Mbps, Transmission Delay: {transmission_delay:.4f} s, Computation Delay: {computation_delay:.4f} s, Total Delay: {total_delay:.4f} s")
            if self.args.NOMA_method == 'OMA':
                S1 = self.OMA_training_selection(h, p_k, D_k)
        else:
            S1 = np.ones(self.args.num_clients)
            h = np.ones(self.args.num_clients)
        return S1, h


    def train(self):
        # 绘图，跑十次的平均
        print("---------------------------------------------------------------Training---------------------------------------------------------------")
        global_test_count = []
        local_test_count = []
        train_loss_count = []
        h_record = []
        record_loss = 0
        h_each_count = []
        D_k = np.zeros(self.args.num_clients)
        Dd_k = np.zeros(self.args.num_clients)
        reporter = MemReporter()
        start_time = time.time()
        # for k in range(self.args.num_clients):
        #     D_k[k] = self.LocalModels[k].dataset_size
        #     Dd_k[k] = self.LocalModels[k].model_size
        for count in range(10):
            print(f"------------------ Experiment {count+1} ------------------")
            print(self.args)
            # 这里需要对self.global_model进行重新初始化
            global_weights = self.initial_global_model.state_dict()
            global_test_list = []
            train_loss = []
            local_test = []
            h_list = []
            for epoch in tqdm(range(self.args.num_epochs)):
                S1, h = self.op_RO_and_PS()
                # h = np.ones(self.args.num_clients)
                h_list.append(h)
                print("优化完成")
                # S1 = self.pso_client_uav_selection_and_resource_allocation(h_kn, h_n, D_k, seed=epoch)
                selected_clients = np.where(S1 == 1)[0]
                # 从selected_clients中删除5这个客户端（如果存在的话）
                # selected_clients = selected_clients[selected_clients != 5]
                # selected_clients = np.arange(self.args.num_clients)
                print(f"Selected clients in round {epoch}: {selected_clients}")
                test_accuracy = 0
                local_weights, local_losses = [], []
                print(f'\n | Global Training Round : {epoch} |\n')
                m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
                # idxs_users = np.random.choice(range(self.args.num_clients), m, replace=False)
                if len(selected_clients) == 0:
                    print("No clients selected this round, skipping aggregation.")
                    global_test_list.append(self.global_test_accuracy())
                    loss_sum = 0
                    test_sum = 0
                    for i in range(self.args.num_clients):
                        loss = self.LocalModels[i].obtain_loss()
                        acc = self.LocalModels[i].test_accuracy()
                        test_sum += acc
                        loss_sum += loss
                    loss_avg = loss_sum / self.args.num_clients
                    local_test.append(test_sum / self.args.num_clients)
                    train_loss.append(loss_avg)
                    continue
                for idx in selected_clients:
                    if self.args.upload_model == True:
                        self.LocalModels[idx].load_model(global_weights)
                    w, loss = self.LocalModels[idx].update_weights(global_round=epoch)
                    local_losses.append(copy.deepcopy(loss))
                    local_weights.append(copy.deepcopy(w))
                    acc = self.LocalModels[idx].test_accuracy()
                    test_accuracy += acc


                # update global weights
                global_weights = average_weights(local_weights)
                self.global_model.load_state_dict(global_weights)
                loss_avg = sum(local_losses) / len(local_losses)
                train_loss.append(loss_avg)
                global_test_accuracy = self.global_test_accuracy()
                local_test_accuracy = test_accuracy / self.args.num_clients
                
                global_test_list.append(global_test_accuracy)
                local_test.append(local_test_accuracy)
                print("average loss:  ", loss_avg)
                print('average local test accuracy:', local_test_accuracy)
                print('global test accuracy: ', global_test_accuracy)
            h_list_mean_device = np.mean(np.array(h_list), axis=0)
            h_each_count.append(h_list_mean_device)
            h_record.append(h_list)
            global_test_count.append(global_test_list)
            local_test_count.append(local_test)
            train_loss_count.append(train_loss)

        print("---------------------------------------------------------------Results---------------------------------------------------------------")
        h_count_mean = np.mean(np.array(h_each_count), axis=0)
        h_min = np.min(h_count_mean)
        a_sum = 0
        for i in range(len(global_test_count)):
            print(global_test_count[i][0])
            a_sum += global_test_count[i][0]
        print('Average accuracy of the first round: ', a_sum/len(global_test_count))
        global_test_list = np.mean(np.array(global_test_count), axis=0).tolist()
        print('对比', global_test_list[0])
        print("最后五十个round的平均准确率：", np.mean(global_test_list[-50:]))
        print("最高准确率：", np.max(global_test_list))
        print("前五十个最大的准确率的平均值：", np.mean(sorted(global_test_list, reverse=True)[:50]))
        print(" ")
        print("前十个最大的准确率的平均值：", np.mean(sorted(global_test_list, reverse=True)[:10]))
        print("前二十个最大的准确率的平均值：", np.mean(sorted(global_test_list, reverse=True)[:20]))
        print("前三十个最大的准确率的平均值：", np.mean(sorted(global_test_list, reverse=True)[:30]))
        print("前四十个最大的准确率的平均值：", np.mean(sorted(global_test_list, reverse=True)[:40]))
        print("最后十个round的平均准确率：", np.mean(global_test_list[-10:]))
        print("最后二十个round的平均准确率：", np.mean(global_test_list[-20:]))
        print("最后三十个round的平均准确率：", np.mean(global_test_list[-30:]))
        print("最后四十个round的平均准确率：", np.mean(global_test_list[-40:]))
        local_test = np.mean(np.array(local_test_count), axis=0).tolist()
        train_loss = np.mean(np.array(train_loss_count), axis=0).tolist()
        h_rounds = np.mean(np.array(h_record), axis=0).tolist()
        # print("h参数的平均值：", h_count_mean)
        print("h参数的最小值：", h_min)        
        print('Training is completed.')
        # with open('global_test_list.txt', 'w') as f:
        #     for acc in global_test_list:
        #         f.write(f"{acc}\n")
        self.save_records(global_test_list, count, name="global_test_accuracy")
        self.save_records(train_loss, count, name="train_loss")
        self.save_records(h_rounds, count, name="h_record")
        # self.save_records(local_test, count, name="local_test_accuracy")
        print("type of global_test_list:", type(global_test_list))
        # self.print_figure(global_test_list, count, method= self.args.op_method, name="global_test_accuracy", xlabel="Rounds", ylabel="Global Test Accuracy", title="Global Test Accuracy over Rounds")
        # self.print_figure(train_loss, count, method= self.args.op_method, name="train_loss", xlabel="Rounds", ylabel="Training Loss", title="Training Loss over Rounds")
        # self.print_figure(local_test, count, method= self.args.op_method, name="local_test_accuracy", xlabel="Rounds", ylabel="Local Test Accuracy", title="Local Test Accuracy over Rounds")

        end_time = time.time()
        print('running time: {} s '.format(end_time - start_time))
        reporter.report()