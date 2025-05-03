import copy
from collections import OrderedDict

import torch
import numpy as np
from numpy import random

class Server():

    def __init__(self, args, model):
        self.clients_list = np.arange(args.num_clients)
        self.args = args
        self.global_model = copy.deepcopy(model)

    def sample_clients(self):
        """
        Return: array of integers, which corresponds to the indices of sampled deviecs
        """
        sampling_set = np.random.choice(self.args.num_clients, self.args.schedulingsize, replace = False)

        return sampling_set
    
    def broadcast(self, Clients_list, Clients_list_idx = None):
        """
        Input: a list of Client class
        Flow: Set the current global model to sampled clients
        """
        for client_idx in Clients_list_idx:
            with torch.no_grad():
                Clients_list[client_idx].model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))

    def aggregation_ori(self, Clients_list, sampling_set):
        """
        Input: sampling_set: array of integers, which corresponds to the indices of sampled devices and a list of Client class
        Flow: aggregate the updated threholds in the sampling set
        """
        #You can change the weights of clients arbitrarily 
        #For simplicy, we use 1/args.schedulingsize here
    

        weight_dict = OrderedDict()

        weight_difference_dict = OrderedDict()
        for i, client in enumerate(sampling_set):
            local_difference = Clients_list[client].model_difference
            if i == 0:
                for key in local_difference.keys():
                    weight_difference_dict[key] = local_difference[key] * 1/self.args.schedulingsize
            else:
                for key in local_difference.keys():
                    weight_difference_dict[key] += local_difference[key] *1/self.args.schedulingsize

        for key in weight_difference_dict.keys():
            weight_dict[key] = self.global_model.state_dict()[key] + weight_difference_dict[key]
        self.global_model.load_state_dict(weight_dict)

    def aggregation(self, Clients_list, sampling_set):
        """
        精度感知的加权聚合方法：同时考虑量化精度、数据量和计算能力
        """
        weight_dict = OrderedDict()
        weight_difference_dict = OrderedDict()
        
        # 计算每个客户端的综合权重
        client_weights = []
        log_info = []  # 用于记录日志
        
        # 计算总数据量
        total_samples = sum([len(Clients_list[client_idx].tr_loader.dataset) for client_idx in sampling_set])
        
        for client_idx in sampling_set:
            client = Clients_list[client_idx]
            
            # 1. 量化精度权重：高精度客户端获得更高权重
            quant_weight = (client.quant_budget / 32.0) ** 1.5  # 指数1.5增强高精度权重
            
            # 2. 数据量权重：拥有更多数据的客户端获得更高权重
            data_weight = len(client.tr_loader.dataset) / total_samples if total_samples > 0 else 1.0
            
            # 3. 计算能力权重：高计算能力的客户端可能有更高质量的更新
            compute_weight = min(1.0, client.resources.compute_capability / 50.0)
            
            # 组合权重（可以调整各因素的重要性）
            weight = quant_weight * 0.6 + data_weight * 0.3 + compute_weight * 0.1
            client_weights.append(weight)
            
            # 收集日志信息
            log_info.append(f"客户端{client_idx}：量化={client.quant_budget}位, 数据量={len(client.tr_loader.dataset)}, "
                            f"计算能力={client.resources.compute_capability:.1f}, 原始权重={weight:.4f}")
        
        # 归一化权重
        total_weight = sum(client_weights)
        if total_weight > 0:
            client_weights = [w/total_weight for w in client_weights]
        else:
            # 如果所有权重都为0（理论上不应该发生），则使用均匀权重
            client_weights = [1.0/len(sampling_set) for _ in sampling_set]
        
        # 输出调试信息
        print("量化感知的聚合权重分配:")
        for i, client_idx in enumerate(sampling_set):
            print(f"{log_info[i]}, 归一化权重={client_weights[i]:.4f}")
        
        # 使用计算出的权重进行加权聚合
        for i, client_idx in enumerate(sampling_set):
            weight = client_weights[i]
            local_difference = Clients_list[client_idx].model_difference
            
            if i == 0:
                for key in local_difference.keys():
                    weight_difference_dict[key] = local_difference[key] * weight
            else:
                for key in local_difference.keys():
                    weight_difference_dict[key] += local_difference[key] * weight
        
        # 应用聚合后的更新到全局模型
        for key in weight_difference_dict.keys():
            weight_dict[key] = self.global_model.state_dict()[key] + weight_difference_dict[key]
        
        self.global_model.load_state_dict(weight_dict)

    # 接收一个client列表，根据每个client的resources中的功耗，分配合适的量化比特预算
    def allocate_quant_budget(self, Clients_list, Clients_list_idx = None):
        """
        Input: a list of Client class
        Flow: Set the current global model to sampled clients
        """
        # 计算功耗权重
        power_weights = np.zeros(len(Clients_list))
        for client_idx in Clients_list_idx:
            power_weights[client_idx] = Clients_list[client_idx].resources.power_limit

        total_power = np.sum(power_weights)

        for client_idx in Clients_list_idx:
            # 获取每个client的功耗
            power = Clients_list[client_idx].resources.power_limit
            weight = power / total_power
            # 打印显示权重
            print(f"Client {client_idx} power weight: {weight:.4f}")
            # 分配量化比特预算[8, 16, 32]，根据权重进行分配
            if weight < 0.2:
                Clients_list[client_idx].quant_budget = 8
            elif weight < 0.5:
                Clients_list[client_idx].quant_budget = 16
            else:
                Clients_list[client_idx].quant_budget = 32



