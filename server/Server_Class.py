import copy
from collections import OrderedDict

import torch
import numpy as np
from numpy import random

class Server():

    def __init__(self, args, model, logger = None):
        self.clients_list = np.arange(args.num_clients)
        self.args = args
        self.global_model = copy.deepcopy(model)
        self.logger = logger

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
        精度感知的加权聚合方法：根据量化误差动态调整聚合权重
        """
        weight_dict = OrderedDict()
        weight_difference_dict = OrderedDict()
        
        # 从客户端获取量化误差
        quant_errors = [Clients_list[client_idx].quant_error for client_idx in sampling_set]
        
        # 计算基于量化误差的权重: p_i^* = \frac{\frac{1}{1 + q_i}}{\sum_{i \in [n]} \frac{1}{1 + q_i}}
        inverse_error_weights = [1.0 / (1.0 + error) for error in quant_errors]
        sum_inverse_errors = sum(inverse_error_weights)
        
        # 防止除以零
        if sum_inverse_errors < 1e-10:
            error_weights = [1.0 / len(sampling_set)] * len(sampling_set)
        else:
            error_weights = [weight / sum_inverse_errors for weight in inverse_error_weights]
        
        # 综合考虑其他因素（量化精度、数据量、计算能力）
        client_weights = []
        
        # 计算总数据量
        total_samples = sum([len(Clients_list[client_idx].tr_loader.dataset) for client_idx in sampling_set])
        
        for i, client_idx in enumerate(sampling_set):
            client = Clients_list[client_idx]
            
            # 1. 量化精度权重：高精度客户端获得更高权重
            quant_weight = (client.quant_budget / 32.0) ** 1.2
            
            # 2. 数据量权重：拥有更多数据的客户端获得更高权重
            data_weight = len(client.tr_loader.dataset) / total_samples if total_samples > 0 else 1.0
            
            # 3. 量化误差权重（已计算，为error_weights[i]）
            
            # 组合权重（调整各因素的重要性）
            weight = error_weights[i] * 0.8 + data_weight * 0.2
            client_weights.append(weight)
            
            # 收集日志信息
            self.logger.info(f"客户端{client_idx}：量化={client.quant_budget}位, 数据量={len(client.tr_loader.dataset)}, "
                            f"计算能力={client.resources.compute_capability:.1f}, 量化误差={quant_errors[i]:.6f}, "
                            f"原始权重={weight:.4f}")
        
        # 归一化权重
        total_weight = sum(client_weights)
        if total_weight > 0:
            client_weights = [w/total_weight for w in client_weights]
        else:
            # 如果所有权重都为0（理论上不应该发生），则使用均匀权重
            client_weights = [1.0/len(sampling_set) for _ in sampling_set]
        
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
            power_weights[client_idx] = Clients_list[client_idx].resources.training_power

        total_power = np.sum(power_weights)

        for client_idx in Clients_list_idx:
            # 获取每个client的功耗
            power = Clients_list[client_idx].resources.training_power
            weight = power / total_power
            # 打印显示权重
            print(f"Client {client_idx} power weight: {weight:.4f}")
            # 分配量化比特预算[8, 16, 32]，根据权重进行分配
            if weight < (0.85/(len(Clients_list) * self.args.sample_ratio)):
                Clients_list[client_idx].quant_budget = self.args.quant_budget_options[0]  # 8位
            elif weight <= (1/(len(Clients_list) * self.args.sample_ratio)):
                Clients_list[client_idx].quant_budget = self.args.quant_budget_options[1]
            else:
                Clients_list[client_idx].quant_budget = self.args.quant_budget_options[2]

    # 保存当前全局模型
    def save_model(self, save_path):
        """
        Input: save_path: the path to save the model
        Flow: Save the current global model
        """
        torch.save(self.global_model.state_dict(), save_path)
        self.logger.info(f"Global model saved to {save_path}")



