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
    battery_level: float
    bandwidth: float
    dataset_size: int
    CPU_available: bool
    CPU_memory_availability: float
    GPU_available: bool
    GPU_memory_availability: float
    power_limit: float  # 新增的功耗限制属性
    compute_capability: float = field(init=False) # 添加计算能力属性
    


    def __post_init__(self):
        # Validation logic
        if not (1 <= self.speed_factor):
            raise ValueError("speed_factor must be greater than or equal to 1")
        if not (1 <= self.battery_level <= 100):
            raise ValueError("battery_level must be between 0 and 100")
        if self.bandwidth < 0:
            raise ValueError("bandwidth must be non-negative")
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if not (0 <= self.CPU_memory_availability <= 128):  # Example: assuming max 128GB
            raise ValueError("CPU_memory_availability must be between 0 and 128")
        if not (0 <= self.GPU_memory_availability <= 32):  # Example: assuming max 32GB
            raise ValueError("GPU_memory_availability must be between 0 and 32")
        if self.power_limit < 0:
            raise ValueError("power_limit must be non-negative")
        
        # 计算 compute_capability
        # 基础计算能力由 speed_factor 决定
        base_capability = self.speed_factor * 25.0  # 基础分值
        
        # GPU 加成
        gpu_bonus = 30.0 if self.GPU_available else 0.0
        
        # 电量状态影响 (电量过低会降低性能)
        battery_factor = min(1.0, self.battery_level / 20.0)  # 电量低于20%时开始降低性能
        
        # 功率限制影响 (功率限制越高，计算能力越强)
        power_factor = min(1.5, self.power_limit / 5.0)  # 功率限制对计算能力的影响
        
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
        """
        Generate random valid ClientResources.
        """

        GPU_available = random.choice([True, False])

        return ClientResources(
            speed_factor=random.uniform(1.0, 2.0),  # Speed factor in range [0.1, 2.0]
            battery_level=random.uniform(1, 100),   # Battery level in range [0, 100]
            bandwidth=random.uniform(1, 100),       # Bandwidth in range [1, 100] Mbps
            dataset_size=random.randint(*dataset_size_range),  # Dataset size
            CPU_available=random.choice([True, False]),        # Random CPU availability
            CPU_memory_availability=random.uniform(0, 128),    # CPU memory in GB
            GPU_available=GPU_available,        # Random GPU availability
            GPU_memory_availability=random.uniform(0, 32) if GPU_available else 0,  # GPU memory if available
            power_limit=random.uniform(0, 10)  # Power limit in range [10, 100] watts
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
        self.resources = resources
        self.quant_budget = quant_budget
    
    def local_training(self, comm_rounds):
        """执行本地训练并返回能耗"""
        # 保存初始模型状态用于后续计算差异
        initial = copy.deepcopy(self.model)
        
        # 记录训练开始时间
        start_time = py_time.time()
        
        # 记录操作和参数数量
        num_parameters = sum(p.numel() for p in self.model.parameters())
        total_data_samples = len(self.tr_loader.dataset)
        total_batches = len(self.tr_loader)
        
        # 原有训练循环
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
        
        # 计算训练时间
        training_time = py_time.time() - start_time
        
        # 计算模型更新差异并量化
        for name in self.model.state_dict():
            foo = self.model.state_dict()[name] - initial.state_dict()[name]
            quantized_foo = self.uniform_quantize(foo)
            self.model_difference[name] = quantized_foo
        
        # 更加精确的能耗计算
        # 基础能耗：与计算资源和训练时间成正比
        base_energy = self.resources.power_limit * training_time
        
        # 量化因子：量化位数与32位全精度的比率，量化位数越低能耗越低
        quant_factor = (self.quant_budget / 32.0) ** 1.5
        
        # 数据量因子：数据集大小对能耗的影响
        data_size = len(self.tr_loader.dataset)
        data_factor = (data_size / 1000.0) ** 0.8  # 使用次线性关系，因为批处理有效率

        # 通信能耗：与模型大小和量化精度有关
        comm_energy = (num_parameters * self.quant_budget / 8) / (self.resources.bandwidth + 1e-6) * 0.01
        
        # 总能耗 = 基础能耗 * 量化因子 * 数据因子 * 效率因子 + 通信能耗
        energy_consumption = base_energy * quant_factor * data_factor + comm_energy
        
        # 添加一些随机波动，模拟真实环境
        energy_variation = random.uniform(0.9, 1.1)
        energy_consumption *= energy_variation
        
        # 记录客户端ID和通信轮次以便于调试
        # print(f"Client {self.client_id}, Round {comm_rounds}: Energy = {energy_consumption:.4f} J")
        
        return energy_consumption
            
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