import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
from collections import OrderedDict
import copy
from dataclasses import dataclass, field
import time as py_time

path = os.getcwd() #current path
sys.path.append(os.path.abspath(os.path.join(path, os.pardir))) #import the parent directory

from model import quantization

@dataclass
class ClientResources:
    """
    A dataclass representing the computational and resource capabilities of a simulated client
    in a federated learning environment. This class is designed to model client heterogeneity
    by including various attributes related to processing speed, power levels, memory, and network bandwidth.

    Attributes:
        speed_factor (float): 
            A multiplier representing the client's processing speed relative to a baseline.
            Higher values indicate faster processing.
        
        battery_level (float): 
            The current battery level of the client as a percentage (0.0 to 100.0).
            Used to simulate throttling or resource limitations based on power availability.
        
        bandwidth (float): 
            The network bandwidth available to the client, measured in Mbps.
            Impacts the time required for model upload and download.
        
        dataset_size (int): 
            The size of the local dataset available to the client, measured in the number of samples.
            Larger datasets may increase the time required for local model updates.
        
        CPU_available (bool): 
            A flag indicating whether the client has a CPU available for computation.
            If False, the client cannot perform local updates.
        
        CPU_memory_availability (float): 
            The amount of available memory (in GB) on the client's CPU.
            Determines whether the client can handle memory-intensive computations.
        
        GPU_available (bool): 
            A flag indicating whether the client has a GPU available for computation.
            If True, the client is expected to process faster compared to CPU-only clients.
        
        GPU_memory_availability (float): 
            The amount of available memory (in GB) on the client's GPU.
            Critical for handling large models or datasets during training.
        power_limit (float): 
            The maximum power consumption allowed for the client, measured in watts.
            Impacts the processing speed and resource usage.
    """
    speed_factor: float
    battery_level: float  # 电量水平 (300.0 - 2000.0 J)
    bandwidth: float  # 带宽 (Mbps)
    dataset_size: int
    CPU_available: bool
    CPU_memory_availability: float
    GPU_available: bool
    GPU_memory_availability: float
    # 细化功率属性
    idle_power: float  # 静息功率 (W)
    training_power: float  # 训练功率 (W) 
    transmission_power: float  # 上传功率 (W)
    compute_capability: float = field(init=False) # 计算能力属性
    


    def __post_init__(self):
        # 验证逻辑
        if not (1 <= self.speed_factor):
            raise ValueError("speed_factor must be greater than or equal to 1")
        if not (0 <= self.battery_level ):
            raise ValueError("battery_level must be between 0 and 100")
        if self.bandwidth < 0:
            raise ValueError("bandwidth must be non-negative")
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if not (0 <= self.CPU_memory_availability <= 128):
            raise ValueError("CPU_memory_availability must be between 0 and 128")
        if not (0 <= self.GPU_memory_availability <= 32):
            raise ValueError("GPU_memory_availability must be between 0 and 32")
        if self.idle_power < 0:
            raise ValueError("idle_power must be non-negative")
        if self.training_power < 0:
            raise ValueError("training_power must be non-negative")
        if self.transmission_power < 0:
            raise ValueError("transmission_power must be non-negative")
        
        # 计算 compute_capability
        # 基础计算能力由 speed_factor 决定
        base_capability = self.speed_factor * 25.0  # 基础分值
        
        # GPU 加成
        gpu_bonus = 30.0 if self.GPU_available else 0.0
        
        # 电量状态影响 (电量过低会降低性能)
        battery_factor = min(1.0, self.battery_level / 10000)  # 电量低于20%时开始降低性能
        
        # 功率限制影响 (功率限制越高，计算能力越强)
        power_factor = min(1.5, self.training_power / 5.0)  # 功率限制对计算能力的影响
        
        # 内存影响 (内存过低会限制性能)
        memory_factor = 1.0
        if self.CPU_available:
            memory_factor *= min(1.0, self.CPU_memory_availability / 4.0)  # 4GB为阈值
        if self.GPU_available:
            memory_factor *= min(1.0, self.GPU_memory_availability / 2.0)  # 2GB为阈值
        
        # 综合计算
        self.compute_capability = (base_capability + gpu_bonus) * battery_factor * power_factor * memory_factor
        
        # 确保在合理范围内
        self.compute_capability = max(10.0, min(100.0, self.compute_capability))



    @staticmethod
    def generate_random(dataset_size_range=(500, 2000)):
        """生成随机有效的ClientResources实例"""
        GPU_available = random.choice([True, False])
        
        # 基础功率范围，并确保训练功率 > 传输功率 > 静息功率
        base_power = random.uniform(5, 8)
        training_factor = random.uniform(3.0, 4.0)
        transmission_factor = random.uniform(1.2, 1.8)

        return ClientResources(
            speed_factor=random.uniform(1.0, 2.0),
            battery_level=random.uniform(20000, 30000),
            bandwidth=random.uniform(1, 100),
            dataset_size=random.randint(*dataset_size_range),
            CPU_available=True,  # 确保CPU可用
            CPU_memory_availability=random.uniform(2, 16),
            GPU_available=GPU_available,
            GPU_memory_availability=random.uniform(0, 8) if GPU_available else 0,
            idle_power=random.uniform(0.5, 1.0),  # 静息功率
            training_power=base_power * training_factor,  # 训练功率
            transmission_power=base_power * transmission_factor  # 传输功率
        )


class Client():
    def __init__(self, args, model, loss, client_id, tr_loader, te_loader, device, scheduler = None, resources = None, quant_budget = 16):
        self.args = args
        self.model = model
        self.loss = loss
        self.scheduler = scheduler
        self.client_id = client_id
        self.tr_loader = tr_loader
        self.te_loader = te_loader
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr= self.args.learning_rate, 
                            momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        self.model_difference = OrderedDict()
        self.quant_error = 0.0  # 添加存储量化误差的变量
        self.resources = resources
        self.quant_budget = quant_budget
        self.last_transmission_time = 0.0  # 传输时间记录
        self.has_sufficient_energy = True  # 是否有足够能量参与训练
        self.inactive_rounds = 0  # 连续不活动轮数
    
    def estimate_energy_requirement(self):
        """估算一轮训练所需的总能量消耗（焦耳）"""
        # 估计训练时间
        data_size = len(self.tr_loader.dataset)
        estimated_training_time = 0.01 * data_size / max(1.0, self.resources.compute_capability)
        
        # 估计训练能耗
        training_energy = self.resources.training_power * estimated_training_time
        
        # 估计传输时间
        num_parameters = sum(p.numel() for p in self.model.parameters())
        data_size_bits = num_parameters * self.quant_budget
        bandwidth_bps = self.resources.bandwidth * 1_000_000
        estimated_transmission_time = data_size_bits / (bandwidth_bps * 0.8)  # 假设80%带宽利用率
        
        # 估计传输能耗
        transmission_energy = self.resources.transmission_power * estimated_transmission_time
        
        # 估计等待能耗 (假设平均等待时间是训练时间的1.5倍)
        waiting_time = estimated_training_time * 1.5
        waiting_energy = self.resources.idle_power * waiting_time
        
        # 总能耗估计 (增加20%的安全边际)
        total_energy = (training_energy + transmission_energy + waiting_energy) * 1.2
        
        return total_energy, total_energy
    
    def local_training(self, comm_rounds):
        """执行本地训练并返回能耗、时间和成功状态"""
        # 检查电量是否足够支持训练
        required_energy, estimated_energy = self.estimate_energy_requirement()
        
        if self.resources.battery_level < required_energy:
            self.has_sufficient_energy = False
            self.inactive_rounds += 1
            print(f"警告：客户端 {self.client_id} 电量不足，无法支持训练 (电量: {self.resources.battery_level:.1f}J, 需要: {required_energy:.1f}J)")
            return 0, 0, 0, False  # 返回0能耗、0训练时间、0传输时间和失败标志
    
        # 保存初始模型状态用于后续计算差异
        initial = copy.deepcopy(self.model)
        
        # 记录训练开始时间
        start_time = py_time.time()
        
        # 记录操作和参数数量
        num_parameters = sum(p.numel() for p in self.model.parameters())
        total_data_samples = len(self.tr_loader.dataset)
        
        # 训练循环
        for epoch in range(1, self.args.local_epoch+1):
            for data, label in self.tr_loader:
                data, label = data.to(self.device), label.to(self.device)
                self.model.train()
                output = self.model(data)
                loss_val = self.loss(output, label)

                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
        
        # 计算模型更新差异并量化，同时计算量化误差
        total_squared_diff = 0.0
        total_squared_error = 0.0
        
        for name in self.model.state_dict():
            # 计算原始差异
            original_diff = self.model.state_dict()[name] - initial.state_dict()[name]
            
            # 量化差异
            quantized_diff = self.uniform_quantize(original_diff)
            self.model_difference[name] = quantized_diff
            
            # 计算量化误差
            quant_error = quantized_diff - original_diff
            
            # 计算相对误差: (quantized_x−original_x)²/original_x²
            orig_norm_squared = torch.sum(original_diff ** 2).item()
            error_norm_squared = torch.sum(quant_error ** 2).item()
            
            total_squared_diff += orig_norm_squared
            total_squared_error += error_norm_squared
        
        # 如果全局相对误差值为零（可能是因为更新非常小），设置一个小的默认值
        if total_squared_diff < 1e-10:
            self.quant_error = 0.01
        else:
            # 全局相对误差
            self.quant_error = total_squared_error / total_squared_diff
            
            # 防止相对误差过大造成权重极小
            if self.quant_error > 0.5:
                self.quant_error = 0.5
        
        # 计算训练时间
        training_time = py_time.time() - start_time
        
        # 计算传输时间
        self.last_transmission_time = self.calculate_transmission_time(num_parameters)

        # 计算训练能耗
        training_energy_consumption = self.resources.training_power * training_time
        # training_energy_consumption = self.resources.training_power * training_time
        
        # 传输能耗计算 - 基于传输功率和传输时间
        transmission_energy_consumption = self.resources.transmission_power * self.last_transmission_time
        
        # 总能耗 = 训练能耗 + 传输能耗
        total_energy_consumption = training_energy_consumption + transmission_energy_consumption
        
        # 添加随机波动，模拟真实环境
        energy_variation = random.uniform(0.9, 1.1)
        total_energy_consumption *= energy_variation
        
        # 直接减少电池电量，如果电量小于零，此次训练无效
        if self.resources.battery_level < total_energy_consumption:
            self.has_sufficient_energy = False
            self.inactive_rounds += 1
            print(f"警告：客户端 {self.client_id} 电量不足，无法完成训练 (电量: {self.resources.battery_level:.1f}J, 需要: {total_energy_consumption:.1f}J)")
            self.resources.battery_level = 0
            return 0, 0, 0, False
        
        # 更新电池电量
        self.resources.battery_level = max(0, self.resources.battery_level - total_energy_consumption)
        
        # 更新可用性状态
        self.has_sufficient_energy = self.resources.battery_level > 50.0  # 如果电量低于50J，将不可用
        self.inactive_rounds = 0  # 重置不活动计数
        
        return total_energy_consumption, training_time, self.last_transmission_time, True
    
    def consume_idle_energy(self, idle_time):
        """客户端在等待状态下消耗能量"""
        if self.resources.battery_level <= 0:
            return 0  # 电量已耗尽
        
        # 计算空闲能耗
        idle_energy = self.resources.idle_power * idle_time
        
        # 直接减少电池电量
        self.resources.battery_level = max(0, self.resources.battery_level - idle_energy)
        
        # 更新可用性状态
        self.has_sufficient_energy = self.resources.battery_level > 50.0  # 如果电量低于50J，将不可用
        
        # 如果电量耗尽，增加不活动计数
        if not self.has_sufficient_energy:
            self.inactive_rounds += 1
            
        return idle_energy
    
    def calculate_transmission_time(self, num_parameters):
        """计算模型差异传输时间（秒）"""
        # 计算需要传输的数据量（比特）
        data_size_bits = num_parameters * self.quant_budget
        
        # 根据带宽计算传输时间（秒）
        # 带宽单位是Mbps，需要转换为bps再计算
        bandwidth_bps = self.resources.bandwidth * 1_000_000  # 转换为bps
        
        # 基础传输时间
        transmission_time = data_size_bits / (bandwidth_bps + 1e-6)
        
        # 添加随机延迟因素（网络波动、拥塞等）
        jitter_factor = random.uniform(1.0, 1.5)  # 1.0-1.5倍随机波动
        transmission_time *= jitter_factor
        
        # 带宽利用率（实际上很少能达到理论带宽的100%）
        bandwidth_utilization = random.uniform(0.7, 0.9)  # 70%-90%的带宽利用率
        transmission_time /= bandwidth_utilization
        
        # 返回计算的传输时间（秒）
        return transmission_time
            
    def local_test(self):

        total_acc = 0.0
        num = 0
        self.model.eval()
        std_loss = 0. 
        iteration = 0.
        with torch.no_grad():
            for data, label in self.te_loader:
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                pred = torch.max(output, dim=1)[1]
                te_acc = (pred.cpu().numpy()== label.cpu().numpy()).astype(np.float32).sum()

                total_acc += te_acc
                num += output.shape[0]

                std_loss += self.loss(output, label)
                iteration += 1
        std_acc = total_acc/num*100.
        std_loss /= iteration

        
        return std_acc, std_loss

    def uniform_quantize(self, x):
        if self.quant_budget == 32:
            return x
        elif self.quant_budget == 1:
            return torch.sign(x)
        else:
            m = float(2 ** (self.quant_budget - 1))
            out = torch.round(x * m) / m
            return out
        
    def validation_training(self, max_samples):
        """在有限数据子集上执行1轮训练并在验证集上验证，返回准确率"""
        # 确保有验证集
        if not hasattr(self, 'validation_data') or self.validation_data is None:
            return 0.0, 0.0, 0.0
        
        # 记录开始时间
        start_time = py_time.time()
        
        # 保存当前模型副本
        original_model = copy.deepcopy(self.model)
        
        # 设定使用全精度训练
        original_quant = self.quant_budget
        self.quant_budget = 16
        
        # 准备有限的训练数据
        limited_train_data = []
        limited_train_labels = []
        samples_collected = 0
        
        # 从本地数据集中收集有限样本
        for data, labels in self.tr_loader:
            batch_size = data.shape[0]
            if samples_collected + batch_size > max_samples:
                # 只取需要的部分
                needed = max_samples - samples_collected
                limited_train_data.append(data[:needed])
                limited_train_labels.append(labels[:needed])
                samples_collected += needed
                break
            else:
                limited_train_data.append(data)
                limited_train_labels.append(labels)
                samples_collected += batch_size
                
            if samples_collected >= max_samples:
                break
        
        # 如果没有收集到足够数据，直接返回
        if samples_collected == 0:
            self.quant_budget = original_quant
            return 0.0, 0.0, 0.0
        
        # 合并数据
        train_data = torch.cat(limited_train_data, 0)
        train_labels = torch.cat(limited_train_labels, 0)
        
        # 在有限数据上训练一个epoch
        self.model.train()
        for i in range(0, train_data.shape[0], self.args.batch_size):
            end_idx = min(i + self.args.batch_size, train_data.shape[0])
            data = train_data[i:end_idx].to(self.device)
            labels = train_labels[i:end_idx].to(self.device)
            
            output = self.model(data)
            loss_val = self.loss(output, labels)
            
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()
        
        # 在验证集上评估
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, self.validation_data.shape[0], self.validation_batch_size):
                end_idx = min(i + self.validation_batch_size, self.validation_data.shape[0])
                data = self.validation_data[i:end_idx].to(self.device)
                labels = self.validation_labels[i:end_idx].to(self.device)
                
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # 计算准确率
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        # 计算用时
        validation_time = py_time.time() - start_time
        
        # 计算能耗
        validation_energy = self.resources.training_power * validation_time
        
        # 消耗电池电量
        self.resources.battery_level = max(0, self.resources.battery_level - validation_energy)
        
        # 保存验证精度
        self.validation_accuracy = accuracy
        
        # 恢复原始模型和量化设置
        self.model.load_state_dict(original_model.state_dict())
        self.quant_budget = original_quant
        
        return accuracy, validation_energy, validation_time