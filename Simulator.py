import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from torchvision.transforms import ToTensor
from time import time
import numpy as np
import copy
from model import CNN_model
from server import Server_Class
from Split_Data import Non_iid_split
from client import Client_Class
from utils import*
from qmix import qmix_controller

class Simulator():
    def __init__(self, args, logger, local_tr_data_loaders, local_te_data_loaders, device):
        self.args = args
        self.logger = logger
        self.Clients_list = None
        self.Server = None
        self.local_tr_data_loaders = local_tr_data_loaders
        self.local_te_data_loaders = local_te_data_loaders
        self.device = device

        # QMIX参数
        self.use_qmix = args.use_qmix if hasattr(args, 'use_qmix') else False
        self.qmix_controller = None
        
        # 量化预算选项
        args.quant_budget_options = [4, 8, 16]
        
        # 存储当前回合数据
        self.current_state = None
        self.current_obs = None
        self.current_actions = None
        self.prev_global_acc = 0


    def initialization(self, model, client_resources=None):
        """初始化模拟器"""
        loss = nn.CrossEntropyLoss()
        self.Server = Server_Class.Server(self.args, model, self.logger)
        first_batch = next(iter(self.local_tr_data_loaders[0]))
        first_data = first_batch[0]
        self.logger.info(f"数据形状: {first_data.shape}")
        
        # 直接从数据推断通道数
        if hasattr(first_data, 'shape') and len(first_data.shape) > 1:
            actual_channels = first_data.shape[1] if len(first_data.shape) > 3 else 1
            self.logger.info(f"输入通道数: {actual_channels}")
        
        # 检查模型的第一个卷积层通道数(如果存在)
        if hasattr(model, 'conv1') and hasattr(model.conv1, 'in_channels'):
            self.logger.info(f"模型第一层卷积通道数: {model.conv1.in_channels}")
        
        # 创建客户端
        self.Clients_list = []
        for client_id, (tr_loader, te_loader) in enumerate(zip(self.local_tr_data_loaders, self.local_te_data_loaders)):
            # 如果提供了客户端资源列表，使用它；否则随机生成
            resources = client_resources[client_id] if client_resources else Client_Class.ClientResources.generate_random()
            
            client = Client_Class.Client(
                self.args, 
                copy.deepcopy(self.Server.global_model), 
                loss, 
                client_id, 
                tr_loader, 
                te_loader, 
                self.device, 
                scheduler=None, 
                resources=resources
            )
            self.Clients_list.append(client)
        
        # 如果启用QMIX，初始化控制器
        if self.use_qmix:
            self.qmix_controller = qmix_controller.QMIXController(self.args, self.device)
            self.logger.info("QMIX controller initialized")
            
            # 加载模型（如果指定）
            # if hasattr(self.args, 'qmix_model_path') and self.args.qmix_model_path:
            #     self.qmix_controller.load_model(self.args.qmix_model_path)
                

    def get_client_observation(self, client):
        """获取客户端的观察向量 - 现在是5维"""
        # 特征：电池电量、验证集准确率、数据量、上轮训练时间、当前量化精度
        battery = client.resources.battery_level / 20000.0  # 归一化
        # print(f"电池电量: {client.resources.battery_level:.2f}")
        # print(f"电池参数: {battery:.2f}")
        valid_acc = client.validation_accuracy / 100.0 if hasattr(client, 'validation_accuracy') else 0.0  # 新增
        data_size = len(client.tr_loader.dataset) / 1000.0  # 归一化
        train_time = client.last_training_time / 60.0 if hasattr(client, 'last_training_time') else 0.0  # 新增
        quant = client.quant_budget / 32.0  # 归一化
 
        # 确保数值在合理范围内
        battery = max(0.0, min(1.0, battery))
        valid_acc = max(0.0, min(1.0, valid_acc))
        data_size = max(0.0, min(1.0, data_size))
        train_time = max(0.0, min(1.0, train_time))
        quant = max(0.0, min(1.0, quant))
        
        return [battery, valid_acc, data_size, train_time, quant]
    
    def get_global_state(self, global_acc, round_idx):
        """获取全局状态"""
        # 合并所有客户端观察和全局信息
        observations = [self.get_client_observation(client) for client in self.Clients_list]
        flat_observations = [item for sublist in observations for item in sublist]
        
        return np.array(flat_observations)
    
    def calculate_pareto_metrics(self, metrics, clients_involved):
        """计算帕累托前沿和超体积等多目标评估指标"""
        # 归一化指标
        norm_metrics = np.zeros_like(metrics)
        for i in range(metrics.shape[1]):
            col_min = metrics[:, i].min()
            col_max = metrics[:, i].max()
            if col_max - col_min > 1e-8:  # 避免除以零
                norm_metrics[:, i] = (metrics[:, i] - col_min) / (col_max - col_min)
            else:
                norm_metrics[:, i] = 0.5  # 如果所有值相等，设置为0.5
        
        # 计算帕累托前沿比例
        is_efficient = np.ones(len(clients_involved), dtype=bool)
        for i in range(len(clients_involved)):
            if is_efficient[i]:
                # 检查i是否被其他点支配
                for j in range(len(clients_involved)):
                    if i != j and is_efficient[j]:
                        if np.all(norm_metrics[j] <= norm_metrics[i]) and np.any(norm_metrics[j] < norm_metrics[i]):
                            is_efficient[i] = False
                            break
        
        pareto_ratio = np.mean(is_efficient)
        
        # 计算目标平衡性：1 - 标准差
        mean_metrics = norm_metrics.mean(axis=0)
        balance = 1 - np.std(mean_metrics)
        
        # 计算超体积
        try:
            from scipy.spatial import ConvexHull
            if len(norm_metrics) >= 3:  # 至少需要3个点来计算凸包
                try:
                    hull = ConvexHull(norm_metrics)
                    hypervolume = hull.volume
                except:
                    hypervolume = 0.0
            else:
                hypervolume = 0.0
        except ImportError:
            hypervolume = 0.0
         
        return pareto_ratio, balance, hypervolume, is_efficient

    def calculate_reward(self, prev_acc, current_acc, energy_consumption, time_spent, clients_involved):
        """计算奖励函数，引入多目标评估"""
        # 基础奖励组件 - 精度提升
        accuracy_improvement = (current_acc - prev_acc) * 100
        
        # 惩罚组件 - 能耗和时间
        energy_penalty = energy_consumption * 0.01
        time_penalty = time_spent * 0.005
        
        # 构建多维评估指标
        metrics = np.zeros((len(clients_involved), 3))
        for i, client_idx in enumerate(clients_involved):
            client = self.Clients_list[client_idx]
            # 指标1：能耗 (归一化到0-1)
            metrics[i, 0] = min(1.0, client.resources.training_power * time_spent / 100.0)
            # 指标2：时间 (归一化到0-1)
            metrics[i, 1] = min(1.0, time_spent / 60.0)
            # 指标3：模型误差 (1-精度)
            client_acc, _ = client.local_test()
            metrics[i, 2] = 1.0 - (client_acc / 100.0)
        
        # 计算多目标评估指标
        pareto_ratio, balance, hypervolume, is_efficient = self.calculate_pareto_metrics(metrics, clients_involved)
        
        # 多目标奖励组件
        multi_obj_reward = pareto_ratio * 10 + balance * 5 + hypervolume * 20
        
        # 负精度额外惩罚
        neg_penalty = 0
        if accuracy_improvement < 0:
            neg_penalty = 5.0
        
        # 总奖励
        reward = accuracy_improvement - energy_penalty - time_penalty + multi_obj_reward - neg_penalty
        
        # 输出奖励计算详情
        self.logger.info(f"奖励计算明细:")
        self.logger.info(f"  精度变化: {prev_acc:.4f} -> {current_acc:.4f} (提升: {accuracy_improvement:.4f})")
        self.logger.info(f"  能耗惩罚: -{energy_penalty:.4f} (能耗: {energy_consumption:.2f}J)")
        self.logger.info(f"  时间惩罚: -{time_penalty:.4f} (用时: {time_spent:.2f}s)")
        self.logger.info(f"  多目标奖励: +{multi_obj_reward:.4f} (帕累托比: {pareto_ratio:.2f}, 平衡: {balance:.2f}, 超体积: {hypervolume:.4f})")
        if neg_penalty > 0:
            self.logger.info(f"  精度下降额外惩罚: -{neg_penalty:.4f}")
        self.logger.info(f"  总奖励: {reward:.4f}")
        
        return reward

    def select_clients_traditional(self, round_idx):
        """优化的传统客户端选择方法，优先选择电量充足的客户端"""
        sample_ratio = self.args.sample_ratio
        num_selected = max(1, int(self.args.num_clients * sample_ratio))
        
        # 首先检查每个客户端的电量状态
        available_clients = []
        unavailable_clients = []
        
        for i, client in enumerate(self.Clients_list):
            # 检查客户端是否有足够电量
            required_energy, _ = client.estimate_energy_requirement()
            
            if client.resources.battery_level >= required_energy and client.has_sufficient_energy:
                available_clients.append(i)
            else:
                unavailable_clients.append(i)
                # self.logger.info(f"客户端 {i}: 电量不足 ({client.resources.battery_level:.1f}J / 需要: {required_energy:.1f}J), 不可用")
        
        # 检查可用客户端数量
        if len(available_clients) == 0:
            self.logger.warning("没有可用的客户端！所有客户端电量都不足。")
            return []
        
        # 调整选择数量，确保不超过可用客户端数量
        actual_num_selected = min(num_selected, len(available_clients))
        
        if len(available_clients) < num_selected:
            self.logger.warning(f"可用客户端数量({len(available_clients)})少于目标数量({num_selected})!")
        
        # 从可用客户端中随机选择
        sampled_clients = np.random.choice(available_clients, actual_num_selected, replace=False)
        
        # 为选择的客户端分配量化预算
        self.Server.allocate_quant_budget(self.Clients_list, sampled_clients)
        
        # 记录客户端选择结果
        self.logger.info(f"轮次 {round_idx}: 总客户端 {self.args.num_clients}, 可用客户端 {len(available_clients)}, 选择客户端 {len(sampled_clients)}")
        self.logger.info(f"最终选择的客户端: {sampled_clients}")
        
        return sampled_clients
    
    def select_clients_with_qmix(self, round_idx, global_acc):
        """使用QMIX选择客户端和分配量化精度，同时考虑电量情况"""
        # 获取所有客户端的观察
        observations = [self.get_client_observation(client) for client in self.Clients_list]
        
        # 获取全局状态
        global_state = self.get_global_state(global_acc, round_idx)
        
        # 存储当前状态和观察，用于后续训练
        self.current_state = global_state
        self.current_obs = observations
        
        # 重置QMIX控制器的隐藏状态
        self.qmix_controller.reset_hidden_states()
        
        # 选择动作（量化精度）
        actions = self.qmix_controller.select_actions(observations)
        self.current_actions = actions
        
        # 记录每个客户端的观察和动作
        self.logger.info("----- QMIX决策信息 -----")
        self.logger.info(f"当前轮次: {round_idx}, 当前全局精度: {global_acc:.4f}")
        self.logger.info(f"探索率(epsilon): {self.qmix_controller.epsilon:.4f}")
        
        # 显示每个客户端的特征和选择的量化精度
        self.logger.info("客户端状态和QMIX决策:")
        for i, (obs, action) in enumerate(zip(observations, actions)):
            quant_budget = self.args.quant_budget_options[action]
            self.logger.info(f"客户端 {i}: 电量={obs[0]*2000:.1f}J, 训练功率={obs[1]*10:.2f}W, " +
                            f"数据量={obs[2]*1000:.0f}, 选择量化精度={quant_budget}位")
        
        # 分配量化精度
        for i, action in enumerate(actions):
            self.Clients_list[i].quant_budget = self.args.quant_budget_options[action]
        
        # 选择客户端
        sample_ratio = self.args.sample_ratio
        num_selected = max(1, int(self.args.num_clients * sample_ratio))
        
        # 计算每个客户端的价值并检查电量状态
        client_values = []
        available_clients = []  # 电量充足的客户端
        
        for i, client in enumerate(self.Clients_list):
            # 检查客户端是否有足够电量
            required_energy, _ = client.estimate_energy_requirement()
            
            if client.resources.battery_level < required_energy or not client.has_sufficient_energy:
                # 电量不足，设置低价值
                client_values.append(-1)  # 负值确保不会被选中
                self.logger.info(f"客户端 {i}: 电量不足 ({client.resources.battery_level:.1f}J / 需要: {required_energy:.1f}J), 不可用")
            else:
                # 电量充足，计算价值
                quant_budget = self.args.quant_budget_options[actions[i]]
                training_power = client.resources.training_power
                battery_level = client.resources.battery_level
                
                # 价值计算综合考虑: 量化精度、功率、电量水平
                quant_factor = quant_budget / 16.0  
                power_factor = 15.0 / training_power 
                battery_factor = min(1.0, battery_level / 500.0) ** 0.5  # 电量因子，设置500J作为参考点
                
                value = quant_factor * power_factor * battery_factor
                client_values.append(value)
                available_clients.append(i)
            
        # 检查可用客户端数量
        if len(available_clients) == 0:
            self.logger.warning("没有可用的客户端！所有客户端电量都不足。")
            return []
        
        if len(available_clients) < num_selected:
            self.logger.warning(f"可用客户端数量({len(available_clients)})少于目标数量({num_selected})!")
            num_selected = len(available_clients)
        
        # 从可用客户端中选择价值最高的
        sorted_indices = np.argsort(client_values)
        selected_indices = sorted_indices[-num_selected:]
        selected_indices = [idx for idx in selected_indices if client_values[idx] >= 0]  # 过滤掉不可用客户端
        
        # 显示客户端价值和选择结果
        # self.logger.info("客户端价值评估:")
        # for i, value in enumerate(client_values):
        #     selected = "✓" if i in selected_indices else "✗"
        #     self.logger.info(f"客户端 {i}: 价值={value:.4f}, 电量={self.Clients_list[i].resources.battery_level:.1f}%, 选择状态={selected}")
        
        self.logger.info(f"最终选择的客户端: {selected_indices}")
        self.logger.info("------------------------")
        
        return selected_indices

    def FedAvg(self):
        """执行联邦学习过程，增加能量检查和等待能耗计算"""
        best_acc = 0
        acc_history = []
        energy_history = []
        time_history = []  # 添加用时历史记录
        total_energy = 0
        total_time = 0  # 添加总用时统计
        current_global_acc = 0
        
        # 用于检查终止条件的计数器
        low_energy_rounds = 0
        max_low_energy_rounds = 3  # 连续3轮可用客户端不足时终止
        
        for rounds in np.arange(self.args.comm_rounds):
            avg_acc = []
            avg_loss = []
            round_energy_consumption = 0
            self.logger.info("-"*30 + "Epoch start" + "-"*30)
            
            # 存储前一轮的精度用于计算奖励
            prev_global_acc = current_global_acc
            
            # 客户端选择和量化精度分配
            if self.use_qmix:
                # sampled_clients = self.select_clients_with_qmix(rounds, current_global_acc)
                sampled_clients = self.select_clients_with_qmix_new(rounds, current_global_acc)
            else:
                # 使用原始方法，但加入电量检查
                sampled_clients = self.select_clients_traditional(rounds)
                self.Server.allocate_quant_budget(self.Clients_list, sampled_clients)
                # 过滤掉电量不足的客户端
                sampled_clients = [idx for idx in sampled_clients if self.Clients_list[idx].has_sufficient_energy]
            
            # 检查可用客户端数量 - 终止条件
            if len(sampled_clients) < self.args.num_clients * 0.1:  # 如果可用客户端少于10%
                low_energy_rounds += 1
                self.logger.warning(f"警告: 可用客户端数量不足! ({len(sampled_clients)}个客户端可用)")
                
                if low_energy_rounds >= max_low_energy_rounds:
                    self.logger.warning(f"连续{max_low_energy_rounds}轮可用客户端不足，终止训练。")
                    break
            else:
                low_energy_rounds = 0  # 重置计数器
            
            # 如果没有可用客户端，跳过此轮
            if len(sampled_clients) == 0:
                self.logger.warning("没有可用客户端，跳过此轮训练。")
                continue
         
            # 记录选择的客户端和分配的量化精度
            self.logger.info("sampled clients: %s" % (str(sampled_clients)))
            self.logger.info("quant budget: %s" % (str([self.Clients_list[i].quant_budget for i in sampled_clients])))
            self.logger.info("training power: %s" % (str([self.Clients_list[i].resources.training_power for i in sampled_clients])))
            self.logger.info("battery level: %s" % (str([self.Clients_list[i].resources.battery_level for i in sampled_clients])))
            
            # 模型广播
            self.Server.broadcast(self.Clients_list, sampled_clients)
            
            # 测试初始模型
            for client_idx in sampled_clients:
                acc, loss = self.Clients_list[client_idx].local_test()
                avg_acc.append(acc)
                avg_loss.append(loss)
            
            # 本地训练
            train_start_time = time()
            successful_clients = []  # 成功完成训练的客户端
            client_training_times = []  # 记录每个客户端的训练时间
            client_transmission_times = []  # 记录每个客户端的传输时间
            client_total_times = []  # 记录每个客户端的总时间(训练+传输)
            
            for client_idx in sampled_clients:
            # 本地训练并返回能耗、训练时间、传输时间和成功标志
                client_energy, training_time, transmission_time, success = self.Clients_list[client_idx].local_training(rounds)
                
                if success:
                    successful_clients.append(client_idx)
                    client_training_times.append(training_time)
                    client_transmission_times.append(transmission_time)
                    client_total_times.append(training_time + transmission_time)  # 计算客户端总耗时
                    round_energy_consumption += client_energy
                    self.logger.info(f"客户端 {client_idx} 训练用时：{training_time:.4f} s, 传输用时：{transmission_time:.4f} s, "
                                    f"总用时: {training_time + transmission_time:.4f} s, 能耗: {client_energy:.4f} J, "
                                    f"剩余电量: {self.Clients_list[client_idx].resources.battery_level:.1f}J")
                else:
                    self.logger.warning(f"客户端 {client_idx} 能量不足，无法完成训练")
            
            # 计算最长训练时间和传输时间
            max_training_time = max(client_training_times) if client_training_times else 0
            max_transmission_time = max(client_transmission_times) if client_transmission_times else 0
            max_total_time = max(client_total_times) if client_total_times else 0
            
            # 为所有客户端计算等待能耗 (包括非参与客户端)
            idle_energy_consumption = 0
            for client_idx in range(len(self.Clients_list)):
                if client_idx not in successful_clients:  # 对于未参与训练的客户端
                    idle_energy = self.Clients_list[client_idx].consume_idle_energy(max_training_time + max_transmission_time)
                    idle_energy_consumption += idle_energy
                else:  # 对于参与训练的客户端，计算等待其他客户端的能耗
                    client_training_idx = successful_clients.index(client_idx)
                    waiting_time = max_training_time - client_training_times[client_training_idx] + max_transmission_time - client_transmission_times[client_training_idx]
                    if waiting_time > 0:
                        idle_energy = self.Clients_list[client_idx].consume_idle_energy(waiting_time)
                        idle_energy_consumption += idle_energy
            
            # 添加等待能耗到总能耗
            round_energy_consumption += idle_energy_consumption
            self.logger.info(f"等待能耗: {idle_energy_consumption:.4f} J")
            
            # 更新总能耗
            total_energy += round_energy_consumption
            energy_history.append(round_energy_consumption)
            
            begin_time = time()
            # 仅使用成功训练的客户端进行聚合
            if successful_clients:
                if hasattr(self.args, 'aggregation_method') and self.args.aggregation_method == 'original':
                    self.logger.info("使用原始聚合方法 (均等权重)")
                    self.Server.aggregation_ori(self.Clients_list, successful_clients)
                else:
                    self.logger.info("使用加权聚合方法 (基于量化精度)")
                    self.Server.aggregation(self.Clients_list, successful_clients)
                self.logger.info("客户端量化误差统计:")
                quant_errors = [self.Clients_list[client_idx].quant_error for client_idx in successful_clients]
                avg_error = sum(quant_errors) / len(quant_errors) if quant_errors else 0
                max_error = max(quant_errors) if quant_errors else 0
                min_error = min(quant_errors) if quant_errors else 0
                self.logger.info(f"  平均量化误差: {avg_error:.6f}")
                self.logger.info(f"  最大量化误差: {max_error:.6f}")
                self.logger.info(f"  最小量化误差: {min_error:.6f}")
                
                for i, client_idx in enumerate(successful_clients):
                    self.logger.info(f"  客户端 {client_idx}: 量化精度={self.Clients_list[client_idx].quant_budget}位, 量化误差={quant_errors[i]:.6f}")
                    
                # 计算平均精度
                avg_acc_round = np.mean(avg_acc) if avg_acc else prev_global_acc
                current_global_acc = avg_acc_round
                acc_history.append(avg_acc_round)
                
                # 如果使用QMIX，存储经验并训练
                if self.use_qmix and rounds > 0 and self.current_state is not None and successful_clients:
                    # 计算奖励
                    reward = self.calculate_reward(
                        prev_global_acc, current_global_acc, 
                        round_energy_consumption, max_total_time,
                        successful_clients
                    )
                    
                    # 输出奖励详情
                    self.logger.info("----- QMIX奖励信息 -----")
                    self.logger.info(f"精度变化: {prev_global_acc:.4f} -> {current_global_acc:.4f} (变化: {(current_global_acc-prev_global_acc)*100:.2f}%)")
                    self.logger.info(f"能耗: {round_energy_consumption:.2f}, 训练时间: {max_total_time:.2f}秒")
                    self.logger.info(f"计算得到的奖励: {reward:.4f}")
                    
                    # 获取下一个状态和观察
                    next_observations = [self.get_client_observation(client) for client in self.Clients_list]
                    next_state = self.get_global_state(current_global_acc, rounds)
                    
                    # 是否是最后一轮
                    done = 1.0 if (rounds == self.args.comm_rounds - 1 or low_energy_rounds >= max_low_energy_rounds) else 0.0
                    
                    # 存储经验
                    self.qmix_controller.store_transition(
                        self.current_obs, self.current_state, self.current_actions, 
                        reward, next_observations, next_state, done
                    )
                    
                    # 训练QMIX
                    if rounds % 5 == 0:  # 每5轮训练一次
                        loss = self.qmix_controller.train(batch_size=min(32, rounds))
                        if loss is not None:
                            self.logger.info(f"QMIX training - Loss: {loss:.4f}, Epsilon: {self.qmix_controller.epsilon:.4f}")
                    
                    # 保存当前状态和观察用于下一轮
                    self.current_state = next_state
                    self.current_obs = next_observations
            else:
                self.logger.warning("没有客户端成功完成训练，本轮跳过聚合")
                # 保持上一轮的全局精度
                if acc_history:
                    acc_history.append(acc_history[-1])
                else:
                    acc_history.append(0.0)
            
            round_time = max_total_time + time() - begin_time

            # 记录本轮耗时并累加到总耗时
            time_history.append(round_time)
            total_time += round_time

            self.logger.info('round: %d, avg_acc: %.3f, energy: %.2f, time: %.2f' %(
                rounds, current_global_acc, round_energy_consumption, round_time))
            
            # 更新最佳精度
            if current_global_acc > best_acc:
                best_acc = current_global_acc
                # 保存最佳模型
                self.Server.save_model(self.args.model_folder + "/best_model.pt")
                # 保存最佳模型
                if self.use_qmix:
                    self.qmix_controller.save_model(self.args.qmix_model_folder + "/best_qmix_model.pt")
            
            # 能量终止条件：检查所有客户端的电量状态
            available_clients_count = sum(1 for client in self.Clients_list if client.has_sufficient_energy)
            available_ratio = available_clients_count / len(self.Clients_list)
            self.logger.info(f"可用客户端比例: {available_ratio:.2f} ({available_clients_count}/{len(self.Clients_list)})")

            if available_ratio < 0.2:  # 如果可用客户端少于20%
                self.logger.warning(f"可用客户端比例低于20%，考虑提前终止训练。")
                if low_energy_rounds >= max_low_energy_rounds // 2:  # 如果连续低能量轮数达到阈值一半
                    self.logger.warning("系统电量状态不足，提前终止训练。")
                    break
        
        # 最终评估
        final_acc = current_global_acc
        
        self.logger.info(">>>>> Training process finish")
        self.logger.info("Best test accuracy {:.4f}".format(best_acc))  
        self.logger.info("Final test accuracy {:.4f}".format(final_acc))
        self.logger.info("Total energy consumption {:.4f}".format(total_energy))
        self.logger.info("Total time consumption {:.4f}".format(total_time))
        self.logger.info(">>>>> Accuracy history during training")
        self.logger.info(acc_history)
        self.logger.info(">>>>> Energy consumption history during training")
        self.logger.info(energy_history)
        self.logger.info(">>>>> Time consumption history during training")
        self.logger.info(time_history)
        
        # 输出客户端电量状态
        self.logger.info(">>>>> Final battery levels")
        for i, client in enumerate(self.Clients_list):
            self.logger.info(f"Client {i}: {client.resources.battery_level:.1f}J")
        
        self.Server.save_model(self.args.model_folder + "/final_model.pt")
        # 保存最终QMIX模型
        if self.use_qmix:
            self.qmix_controller.save_model(self.args.qmix_model_folder + "/final_qmix_model.pt")
        
        return best_acc, final_acc, acc_history, energy_history, total_energy, time_history, total_time
    
    # ======新方法=====
    def select_clients_with_qmix_new(self, round_idx, global_acc):
        """二阶段客户端选择：先执行验证，再基于QMIX选择参与客户端"""
        # Step 1: 创建和分发验证集
        # if not hasattr(self, 'global_validation_data') or self.global_validation_data is None:
        #     # 从测试数据中创建一个小的验证集
        #     print("创建验证集")
        #     self.create_validation_set()
        self.create_validation_set()
        
        # Step 2: 分发全局模型和验证集到所有客户端
        self.broadcast_model_and_validation(round_idx)
        
        # Step 3: 让所有客户端在小数据集上执行一轮训练并验证
        self.execute_validation_phase(round_idx)
        
        # Step 4: 获取所有客户端的观察向量
        observations = [self.get_client_observation(client) for client in self.Clients_list]
        
        # Step 5: 获取全局状态
        global_state = self.get_global_state(global_acc, round_idx)
        
        # 存储当前状态和观察，用于后续训练
        self.current_state = global_state
        self.current_obs = observations
        
        # 重置QMIX控制器的隐藏状态
        self.qmix_controller.reset_hidden_states()
        
        # Step 6: 使用QMIX选择动作（参与度和量化精度）
        actions = self.qmix_controller.select_actions(observations)
        self.current_actions = actions
        
        # 记录决策日志
        self.logger.info("----- QMIX决策信息 -----")
        self.logger.info(f"当前轮次: {round_idx}, 当前全局精度: {global_acc:.4f}")
        self.logger.info(f"探索率(epsilon): {self.qmix_controller.epsilon:.4f}")
        
        # 显示每个客户端的特征和QMIX决策
        self.logger.info("客户端状态和QMIX决策:")
        for i, (obs, action) in enumerate(zip(observations, actions)):
            action_desc = "不参与" if action == 0 else f"参与, {self.args.quant_budget_options[action-1]}位"
            training_time = "从未训练" if self.Clients_list[i].last_training_time < 0 else f"{self.Clients_list[i].last_training_time:.2f}s"
            self.logger.info(f"客户端 {i}: 电量={obs[0]*20000:.1f}J, 验证精度={obs[1]*100:.2f}%, " +
                            f"数据量={obs[2]*1000:.0f}, 上轮训练时间={training_time}, 决策={action_desc}")
        
        # 根据QMIX动作选择参与客户端
        selected_clients = []
        for i, action in enumerate(actions):
            if action > 0:  # 动作大于0表示参与训练
                # 设置量化精度 (1→8位, 2→16位, 3→32位)
                self.Clients_list[i].quant_budget = self.args.quant_budget_options[action-1]
                
                # 检查客户端电量是否足够
                required_energy, _ = self.Clients_list[i].estimate_energy_requirement()
                if self.Clients_list[i].resources.battery_level >= required_energy and self.Clients_list[i].has_sufficient_energy:
                    selected_clients.append(i)
        
        # Step 7: 客户端补充选择机制
        required_clients = max(1, int(self.args.num_clients * self.args.sample_ratio))
        
        if len(selected_clients) < required_clients:
            self.logger.info(f"QMIX选择的客户端数量({len(selected_clients)})小于要求({required_clients})，执行补充选择")
            
            # 计算未选中客户端的价值
            available_clients = []
            client_values = {}
            
            for i, client in enumerate(self.Clients_list):
                if i not in selected_clients:  # 只考虑未被选中的客户端
                    # 检查电量是否足够
                    required_energy, _ = client.estimate_energy_requirement()
                    
                    if client.resources.battery_level >= required_energy and client.has_sufficient_energy:
                        # 计算价值：验证准确率/(电量消耗 * 训练时间因子)
                        validation_acc = client.validation_accuracy if hasattr(client, 'validation_accuracy') else 0.0
                        power_factor = client.resources.training_power / 10.0
                        
                        # 使用历史训练时间作为因子(如果有)
                        time_factor = 1.0
                        if client.last_training_time > 0:
                            time_factor = min(2.0, client.last_training_time / 20.0)  # 归一化，但限制影响
                        
                        # 避免除以零
                        combined_factor = max(0.1, power_factor * time_factor)
                        value = validation_acc / combined_factor
                        
                        client_values[i] = value
                        available_clients.append(i)
            
            # 按价值排序并选择
            sorted_clients = sorted(available_clients, key=lambda i: client_values.get(i, 0), reverse=True)
            needed_clients = required_clients - len(selected_clients)
            additional_clients = sorted_clients[:needed_clients]
            
            # 为补充选择的客户端设置默认量化精度（16位）
            for i in additional_clients:
                self.Clients_list[i].quant_budget = 16  # 默认使用16位量化
                selected_clients.append(i)
                
            self.logger.info(f"补充选择了 {len(additional_clients)} 个客户端: {additional_clients}")
        
        self.logger.info(f"最终选择的客户端: {selected_clients}, 总计 {len(selected_clients)} 个")
        self.logger.info("------------------------")
        
        return selected_clients

    def create_validation_set(self):
        """创建一个通用的验证集，用于所有客户端，确保分配不同数据给不同客户端"""
        # 从测试数据中采样一部分作为验证集
        all_test_data = []
        all_test_labels = []
        
        # 收集所有测试数据
        for loader in self.local_te_data_loaders:
            for data, labels in loader:
                # 仅采样部分数据
                if len(all_test_data) * data.shape[0] >= 5000:  # 限制总样本量
                    break
                all_test_data.append(data)
                all_test_labels.append(labels)
        
        # 合并数据
        if all_test_data:
            self.global_validation_data = torch.cat(all_test_data, 0)
            self.global_validation_labels = torch.cat(all_test_labels, 0)
            
            # 创建验证集大小和批次大小属性
            self.validation_size = min(200, len(self.global_validation_data))
            self.validation_batch_size = 32
            
            print(f"创建了大小为 {self.validation_size} 的公共验证集")
        else:
            self.global_validation_data = None
            self.global_validation_labels = None
            print("警告：无法创建验证集")

    def broadcast_model_and_validation(self, round_idx):
        """向所有客户端分发全局模型和验证集，确保每个客户端得到不同的验证子集"""
        self.logger.info("正在向所有客户端分发全局模型和验证集...")
        
        # 确保验证集存在
        if not hasattr(self, 'global_validation_data') or self.global_validation_data is None:
            self.create_validation_set()
            
        if self.global_validation_data is None:
            return
        
        # 向所有客户端分发全局模型
        for client in self.Clients_list:
            # 复制全局模型
            with torch.no_grad():
                client.model.load_state_dict(copy.deepcopy(self.Server.global_model.state_dict()))
                
            # 为每个客户端提供略微不同的验证集
            # 通过使用客户端ID作为随机种子，确保每个客户端获得不同但一致的验证子集
            torch.manual_seed(client.client_id + round_idx)  # 添加轮次索引确保每轮有变化
            indices = torch.randperm(len(self.global_validation_data))[:self.validation_size]
            
            # 分配验证数据和标签
            client.validation_data = self.global_validation_data[indices]
            client.validation_labels = self.global_validation_labels[indices]
            client.validation_batch_size = self.validation_batch_size

    def execute_validation_phase(self, round_idx):
        """执行验证阶段：每个客户端在小数据集上训练并验证"""
        self.logger.info("正在执行验证阶段...")
        
        # 设定最大训练样本数B
        max_samples = self.args.validation_sample_size if hasattr(self.args, 'validation_sample_size') else 1000
        
        for i, client in enumerate(self.Clients_list):
            # 跳过电量不足的客户端
            if not client.has_sufficient_energy:
                self.logger.info(f"客户端 {i} 电量不足，跳过验证阶段")
                client.validation_accuracy = 0.0
                continue
            # 确保客户端有验证数据
            if not hasattr(client, 'validation_data') or client.validation_data is None:
                self.logger.warning(f"客户端 {i} 没有验证数据，跳过验证")
                continue
                
            # 在有限数据集上训练一个epoch，然后验证
            accuracy, validation_energy, validation_time = client.validation_training(max_samples)
            client.validation_accuracy = accuracy
            self.logger.info(f"客户端 {i} 验证结果: 准确率={accuracy:.2f}%, 能耗={validation_energy:.2f}J, 用时={validation_time:.2f}s")