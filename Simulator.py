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
        self.Clients_list = None
        self.Server = None
        self.local_tr_data_loaders = local_tr_data_loaders
        self.local_te_data_loaders = local_te_data_loaders
        self.device = device

        # QMIX参数
        self.use_qmix = args.use_qmix if hasattr(args, 'use_qmix') else False
        self.qmix_controller = None
        
        # 量化预算选项
        args.quant_budget_options = [1, 2, 4, 8, 16, 32]
        
        # 存储当前回合数据
        self.current_state = None
        self.current_obs = None
        self.current_actions = None
        self.prev_global_acc = 0


    def initialization(self, model):

        """初始化模拟器"""
        loss = nn.CrossEntropyLoss()
        self.Server = Server_Class.Server(self.args, model)
        
        # 创建客户端
        self.Clients_list = [Client_Class.Client(
                        self.args, 
                        copy.deepcopy(self.Server.global_model), 
                        loss, 
                        client_id, 
                        tr_loader, 
                        te_loader, 
                        self.device, 
                        scheduler=None, 
                        resources=Client_Class.ClientResources.generate_random()
                    ) for (client_id, (tr_loader, te_loader)) in enumerate(zip(self.local_tr_data_loaders, self.local_te_data_loaders))]
        
        # 如果启用QMIX，初始化控制器
        if self.use_qmix:
            self.qmix_controller = qmix_controller.QMIXController(self.args, self.device)
            self.logger.info("QMIX controller initialized")
            
            # 加载模型（如果指定）
            if hasattr(self.args, 'qmix_model_path') and self.args.qmix_model_path:
                self.qmix_controller.load_model(self.args.qmix_model_path)

    def get_client_observation(self, client):
        """获取客户端的观察向量"""
        # 特征：功率限制、计算能力、数据量、当前量化精度
        power = client.resources.power_limit / 10.0  # 归一化
        compute = client.resources.compute_capability / 100.0  # 归一化
        data_size = len(client.tr_loader.dataset) / 1000.0  # 归一化
        quant = client.quant_budget / 32.0  # 归一化
        
        return [power, compute, data_size, quant]
    
    def get_global_state(self, global_acc, round_idx):
        """获取全局状态"""
        # 合并所有客户端观察和全局信息
        observations = [self.get_client_observation(client) for client in self.Clients_list]
        flat_observations = [item for sublist in observations for item in sublist]
        
        return np.array(flat_observations)
    
    def calculate_reward(self, prev_acc, current_acc, energy_consumption, time_spent):
        """计算奖励函数"""
        # 奖励 = 精度提升 - 能耗惩罚
        accuracy_improvement = (current_acc - prev_acc) * 100  # 精度提升（放大100倍）
        energy_penalty = energy_consumption * 0.01  # 能耗惩罚
        time_penalty = time_spent * 0.005  # 时间惩罚
        
        reward = accuracy_improvement - energy_penalty - time_penalty
        
        # 如果精度提升为负，增加惩罚
        neg_penalty = 0
        if accuracy_improvement < 0:
            neg_penalty = 5.0
            reward -= neg_penalty
        
        # 输出奖励计算详情
        self.logger.info(f"奖励计算明细:")
        self.logger.info(f"  精度提升奖励: {accuracy_improvement:.4f}")
        self.logger.info(f"  能耗惩罚: -{energy_penalty:.4f}")
        self.logger.info(f"  时间惩罚: -{time_penalty:.4f}")
        if neg_penalty > 0:
            self.logger.info(f"  精度下降额外惩罚: -{neg_penalty:.4f}")
        self.logger.info(f"  总奖励: {reward:.4f}")
        
        return reward

    def select_clients_with_qmix(self, round_idx, global_acc):
        """使用QMIX选择客户端和分配量化精度"""
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
            self.logger.info(f"客户端 {i}: 功率={obs[0]*10:.2f}W, 计算能力={obs[1]*100:.1f}, " + 
                            f"数据量={obs[2]*1000:.0f}, 选择量化精度={quant_budget}位")
    
        
        # 分配量化精度
        for i, action in enumerate(actions):
            self.Clients_list[i].quant_budget = self.args.quant_budget_options[action]
        
        # 选择客户端
        # 规则：
        # 1. 根据动作值（量化精度值）计算客户端价值
        # 2. 根据价值选择部分客户端
        sample_ratio = self.args.sample_ratio
        num_selected = max(1, int(self.args.num_clients * sample_ratio))
        
        # 计算每个客户端的价值
        # 价值 = 1/量化精度（较低精度更省能源）* 1/功率限制（较低功率更省能源）
        client_values = []
        for i, client in enumerate(self.Clients_list):
            quant_budget = self.args.quant_budget_options[actions[i]]
            power_limit = client.resources.power_limit
            value = (1.0 / quant_budget) * (10.0 / power_limit)
            client_values.append(value)
        
        # 选择价值最高的客户端
        selected_indices = np.argsort(client_values)[-num_selected:]
        selected_indices = selected_indices.tolist()
    
        # 显示客户端价值和选择结果
        self.logger.info("客户端价值评估:")
        for i, value in enumerate(client_values):
            selected = "✓" if i in selected_indices else "✗"
            self.logger.info(f"客户端 {i}: 价值={value:.4f}, 选择状态={selected}")
        
        self.logger.info(f"最终选择的客户端: {selected_indices}")
        self.logger.info("------------------------")
    
        return selected_indices

    def FedAvg(self):
        """执行联邦学习过程"""
        best_acc = 0
        acc_history = []
        current_global_acc = 0

        for rounds in np.arange(self.args.comm_rounds):
            begin_time = time()
            avg_acc =[]
            avg_loss =[]
            energy_consumption = 0
            self.logger.info("-"*30 + "Epoch start" + "-"*30)

            # 存储前一轮的精度用于计算奖励
            prev_global_acc = current_global_acc
            
            # 客户端选择和量化精度分配
            if self.use_qmix:
                sampled_clients = self.select_clients_with_qmix(rounds, current_global_acc)
            else:
                # 使用原始方法
                sampled_clients = self.Server.sample_clients()
                self.Server.allocate_quant_budget(self.Clients_list, sampled_clients)
            
            # 记录选择的客户端和分配的量化精度
            self.logger.info("sampled clients: %s" % (str(sampled_clients)))
            self.logger.info("quant budget: %s" % (str([self.Clients_list[i].quant_budget for i in sampled_clients])))
            self.logger.info("resources power: %s" % (str([self.Clients_list[i].resources.power_limit for i in sampled_clients])))
            
            # 模型广播
            self.Server.broadcast(self.Clients_list, sampled_clients)

            # 测试初始模型
            for client_idx in sampled_clients:
                acc, loss = self.Clients_list[client_idx].local_test()
                avg_acc.append(acc)
                avg_loss.append(loss)

            # 本地训练
            train_start_time = time()
            for client_idx in sampled_clients:
                # 本地训练并返回能耗
                client_energy = self.Clients_list[client_idx].local_training(rounds)
                energy_consumption += client_energy
            train_time = time() - train_start_time
            
            # 模型聚合
            self.Server.aggregation(self.Clients_list, sampled_clients)
            
            # 计算平均精度
            avg_acc_round = np.mean(avg_acc)
            current_global_acc = avg_acc_round
            acc_history.append(avg_acc_round)
            
            # 如果使用QMIX，存储经验并训练
            if self.use_qmix and rounds > 0 and self.current_state is not None:
                # 计算奖励
                reward = self.calculate_reward(prev_global_acc, current_global_acc, 
                                               energy_consumption, train_time)
                # 输出奖励详情
                self.logger.info("----- QMIX奖励信息 -----")
                self.logger.info(f"精度变化: {prev_global_acc:.4f} -> {current_global_acc:.4f} (变化: {(current_global_acc-prev_global_acc)*100:.2f}%)")
                self.logger.info(f"能耗: {energy_consumption:.2f}, 训练时间: {train_time:.2f}秒")
                self.logger.info(f"计算得到的奖励: {reward:.4f}")

                # 获取下一个状态和观察
                next_observations = [self.get_client_observation(client) for client in self.Clients_list]
                next_state = self.get_global_state(current_global_acc, rounds)
                
                # 是否是最后一轮
                done = 1.0 if rounds == self.args.comm_rounds - 1 else 0.0
                
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
            
            round_time = time() - begin_time
            self.logger.info('round: %d, avg_acc: %.3f, energy: %.2f, time: %.2f' %(
                rounds, avg_acc_round, energy_consumption, round_time))
            
            # 更新最佳精度
            if avg_acc_round > best_acc:
                best_acc = avg_acc_round
                # 保存最佳模型
                if self.use_qmix:
                    self.qmix_controller.save_model("./models/qmix_best.pt")

        # 最终评估
        self.Server.broadcast(self.Clients_list, range(0, self.args.num_clients))
        final_acc = []
        for client_idx, client in enumerate(self.Clients_list):
            acc, loss = client.local_test()
            final_acc.append(acc)
            self.logger.info('client_id: %d, final acc: %.3f' %(client_idx, acc))
        final_avg_acc = np.mean(final_acc)
        
        self.logger.info(">>>>> Training process finish")
        self.logger.info("Best test accuracy {:.4f}".format(best_acc))  
        self.logger.info("Final test accuracy {:.4f}".format(final_avg_acc))
        self.logger.info(">>>>> Accuracy history during training")
        self.logger.info(acc_history)
        
        # 保存最终QMIX模型
        if self.use_qmix:
            self.qmix_controller.save_model("./models/qmix_final.pt")
        
        return best_acc, final_avg_acc, acc_history

