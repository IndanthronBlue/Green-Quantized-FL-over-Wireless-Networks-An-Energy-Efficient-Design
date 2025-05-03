import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from train_argument import parser, print_args
import random
import copy
from time import time
from model import CNN_model
from utils import *
from Simulator import Simulator
from Split_Data import Non_iid_split, data_stats
from client import Client_Class

def run_comprehensive_comparison(args, logger, tr_dataset, te_dataset, num_classes, device):
    """运行全面比较：QMIX vs 非QMIX，原始聚合 vs 加权聚合"""
    logger.info("=" * 60)
    logger.info(" " * 15 + "启动QMIX与聚合方法全面比较实验")
    logger.info("=" * 60)
    
    # 创建比较结果存储目录
    comparison_folder = os.path.join(args.log_root, 'aggregation_comparison_comprehensive')
    makedirs(comparison_folder)
    
    # 预先生成客户端资源和数据集，确保所有实验使用相同的初始条件
    logger.info("预生成客户端资源和数据集，确保四种配置使用相同的初始条件")
    
    # 1. 预生成客户端资源配置
    client_resources = []
    for i in range(args.num_clients):
        client_resources.append(Client_Class.ClientResources.generate_random())
    
    # 2. 预生成数据集划分
    Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split(
        num_classes, args.num_clients, tr_dataset, te_dataset, args.alpha)
    
    # 确保批次大小合适
    client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, num_classes, args.num_clients)
    client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, num_classes, args.num_clients)
    
    while 1 in np.remainder(client_total_samples, args.batch_size) or 1 in np.remainder(client_total_te_samples, args.batch_size):
        Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split(
            num_classes, args.num_clients, tr_dataset, te_dataset, args.alpha)
        client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, num_classes, args.num_clients)
        client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, num_classes, args.num_clients)
    
    common_datasets = (Non_iid_tr_datasets, Non_iid_te_datasets)
    
    logger.info(f"已生成 {args.num_clients} 个客户端资源配置和数据集划分")
    
    # 存储四种配置的结果
    results = {}
    
    # 运行四种不同配置
    # 1. 不使用QMIX + 原始聚合
    args_copy = copy.deepcopy(args)
    args_copy.use_qmix = False
    args_copy.aggregation_method = 'original'
    logger.info("\n" + "=" * 60)
    logger.info(" " * 15 + "配置1: 不使用QMIX + 原始聚合")
    logger.info("=" * 60)
    results['no_qmix_original'] = run_with_aggregation_method(
        args_copy, logger, tr_dataset, te_dataset, num_classes, device, 'original', 
        client_resources, common_datasets)
    
    # 2. 不使用QMIX + 加权聚合
    args_copy = copy.deepcopy(args)
    args_copy.use_qmix = False
    args_copy.aggregation_method = 'weighted'
    logger.info("\n" + "=" * 60)
    logger.info(" " * 15 + "配置2: 不使用QMIX + 加权聚合")
    logger.info("=" * 60)
    results['no_qmix_weighted'] = run_with_aggregation_method(
        args_copy, logger, tr_dataset, te_dataset, num_classes, device, 'weighted', 
        client_resources, common_datasets)
    
    # 3. 使用QMIX + 原始聚合
    args_copy = copy.deepcopy(args)
    args_copy.use_qmix = True
    args_copy.aggregation_method = 'original'
    logger.info("\n" + "=" * 60)
    logger.info(" " * 15 + "配置3: 使用QMIX + 原始聚合")
    logger.info("=" * 60)
    results['qmix_original'] = run_with_aggregation_method(
        args_copy, logger, tr_dataset, te_dataset, num_classes, device, 'original', 
        client_resources, common_datasets)
    
    # 4. 使用QMIX + 加权聚合
    args_copy = copy.deepcopy(args)
    args_copy.use_qmix = True
    args_copy.aggregation_method = 'weighted'
    logger.info("\n" + "=" * 60)
    logger.info(" " * 15 + "配置4: 使用QMIX + 加权聚合")
    logger.info("=" * 60)
    results['qmix_weighted'] = run_with_aggregation_method(
        args_copy, logger, tr_dataset, te_dataset, num_classes, device, 'weighted', 
        client_resources, common_datasets)
    
    # 绘制对比图并生成比较报告
    generate_comparison_report(results, comparison_folder, logger)
    
    return results


def generate_comparison_report(results, comparison_folder, logger):
    """生成四种配置的比较报告和可视化"""
    config_names = {
        'no_qmix_original': '无QMIX-原始聚合',
        'no_qmix_weighted': '无QMIX-加权聚合',
        'qmix_original': 'QMIX-原始聚合',
        'qmix_weighted': 'QMIX-加权聚合'
    }
    
    # 提取数据
    acc_histories = {}
    energy_histories = {}
    best_accs = {}
    final_accs = {}
    total_energies = {}
    
    for config, result in results.items():
        best_acc, final_acc, acc_history, energy_history, total_energy = result
        acc_histories[config] = acc_history
        energy_histories[config] = energy_history
        best_accs[config] = best_acc
        final_accs[config] = final_acc
        total_energies[config] = total_energy
        
        # 保存原始数据
        np.save(os.path.join(comparison_folder, f'{config}_acc_history.npy'), np.array(acc_history))
        np.save(os.path.join(comparison_folder, f'{config}_energy_history.npy'), np.array(energy_history))
    
    # 绘制精度比较图
    plt.figure(figsize=(12, 8))
    rounds = range(1, len(next(iter(acc_histories.values()))) + 1)
    for config, history in acc_histories.items():
        plt.plot(rounds, history, label=config_names[config])
    plt.title('四种配置精度对比')
    plt.xlabel('通信轮次')
    plt.ylabel('精度 (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(comparison_folder, 'comprehensive_comparison_accuracy.png'))
    
    # 绘制能耗比较图
    plt.figure(figsize=(12, 8))
    for config, history in energy_histories.items():
        plt.plot(rounds, history, label=config_names[config])
    plt.title('四种配置能耗对比')
    plt.xlabel('通信轮次')
    plt.ylabel('能耗 (J)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(comparison_folder, 'comprehensive_comparison_energy.png'))
    
    # 绘制累积能耗比较图
    plt.figure(figsize=(12, 8))
    for config, history in energy_histories.items():
        plt.plot(rounds, np.cumsum(history), label=config_names[config])
    plt.title('四种配置累积能耗对比')
    plt.xlabel('通信轮次')
    plt.ylabel('累积能耗 (J)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(comparison_folder, 'comprehensive_comparison_cumulative_energy.png'))
    
    # 生成比较报告
    logger.info("\n" + "="*80)
    logger.info(" "*30 + "四种配置比较结果")
    logger.info("="*80)
    
    # 表头
    logger.info(f"{'指标':<20} | {'无QMIX-原始':<15} | {'无QMIX-加权':<15} | {'QMIX-原始':<15} | {'QMIX-加权':<15}")
    logger.info("-"*80)
    
    # 精度比较
    logger.info(f"{'最终精度 (%)':<20} | {final_accs['no_qmix_original']:<15.2f} | {final_accs['no_qmix_weighted']:<15.2f} | {final_accs['qmix_original']:<15.2f} | {final_accs['qmix_weighted']:<15.2f}")
    logger.info(f"{'最佳精度 (%)':<20} | {best_accs['no_qmix_original']:<15.2f} | {best_accs['no_qmix_weighted']:<15.2f} | {best_accs['qmix_original']:<15.2f} | {best_accs['qmix_weighted']:<15.2f}")
    
    # 能耗比较
    logger.info(f"{'总能耗 (J)':<20} | {total_energies['no_qmix_original']:<15.2f} | {total_energies['no_qmix_weighted']:<15.2f} | {total_energies['qmix_original']:<15.2f} | {total_energies['qmix_weighted']:<15.2f}")
    
    # 计算各配置达到90%最终精度的轮次
    convergence_rounds = {}
    convergence_energies = {}
    
    for config in results.keys():
        threshold = 0.9 * final_accs[config]
        conv_round = next((i+1 for i, acc in enumerate(acc_histories[config]) if acc >= threshold), len(acc_histories[config]))
        convergence_rounds[config] = conv_round
        convergence_energies[config] = sum(energy_histories[config][:conv_round])
    
    logger.info(f"{'收敛轮次':<20} | {convergence_rounds['no_qmix_original']:<15d} | {convergence_rounds['no_qmix_weighted']:<15d} | {convergence_rounds['qmix_original']:<15d} | {convergence_rounds['qmix_weighted']:<15d}")
    logger.info(f"{'收敛能耗 (J)':<20} | {convergence_energies['no_qmix_original']:<15.2f} | {convergence_energies['no_qmix_weighted']:<15.2f} | {convergence_energies['qmix_original']:<15.2f} | {convergence_energies['qmix_weighted']:<15.2f}")
    
    # 计算能效比(达到相同精度所需的能耗)
    efficiency_threshold = min(best_accs.values()) * 0.95  # 使用所有配置中最低最佳精度的95%作为比较点
    efficiency_rounds = {}
    efficiency_energies = {}
    
    for config in results.keys():
        eff_round = next((i for i, acc in enumerate(acc_histories[config]) if acc >= efficiency_threshold), len(acc_histories[config])-1)
        efficiency_rounds[config] = eff_round + 1  # +1 转换为轮次
        efficiency_energies[config] = sum(energy_histories[config][:eff_round+1])
    
    logger.info(f"{'能效比(J/精度)':<20} | {efficiency_energies['no_qmix_original']:<15.2f} | {efficiency_energies['no_qmix_weighted']:<15.2f} | {efficiency_energies['qmix_original']:<15.2f} | {efficiency_energies['qmix_weighted']:<15.2f}")
    
    # QMIX与聚合方法的影响分析
    logger.info("\n" + "="*80)
    logger.info(" "*30 + "QMIX与聚合方法影响分析")
    logger.info("="*80)
    
    # QMIX对原始聚合的影响
    qmix_impact_original_acc = final_accs['qmix_original'] - final_accs['no_qmix_original']
    qmix_impact_original_energy = total_energies['no_qmix_original'] - total_energies['qmix_original']
    qmix_impact_original_acc_pct = (qmix_impact_original_acc / final_accs['no_qmix_original']) * 100
    qmix_impact_original_energy_pct = (qmix_impact_original_energy / total_energies['no_qmix_original']) * 100
    
    # QMIX对加权聚合的影响
    qmix_impact_weighted_acc = final_accs['qmix_weighted'] - final_accs['no_qmix_weighted']
    qmix_impact_weighted_energy = total_energies['no_qmix_weighted'] - total_energies['qmix_weighted']
    qmix_impact_weighted_acc_pct = (qmix_impact_weighted_acc / final_accs['no_qmix_weighted']) * 100
    qmix_impact_weighted_energy_pct = (qmix_impact_weighted_energy / total_energies['no_qmix_weighted']) * 100
    
    logger.info("QMIX对原始聚合的影响:")
    logger.info(f"  精度: {qmix_impact_original_acc:+.2f} ({qmix_impact_original_acc_pct:+.2f}%)")
    logger.info(f"  能耗: {qmix_impact_original_energy:+.2f} ({qmix_impact_original_energy_pct:+.2f}%)")
    
    logger.info("QMIX对加权聚合的影响:")
    logger.info(f"  精度: {qmix_impact_weighted_acc:+.2f} ({qmix_impact_weighted_acc_pct:+.2f}%)")
    logger.info(f"  能耗: {qmix_impact_weighted_energy:+.2f} ({qmix_impact_weighted_energy_pct:+.2f}%)")
    
    # 加权聚合的影响分析(不使用QMIX)
    weighted_impact_no_qmix_acc = final_accs['no_qmix_weighted'] - final_accs['no_qmix_original']
    weighted_impact_no_qmix_energy = total_energies['no_qmix_original'] - total_energies['no_qmix_weighted']
    weighted_impact_no_qmix_acc_pct = (weighted_impact_no_qmix_acc / final_accs['no_qmix_original']) * 100
    weighted_impact_no_qmix_energy_pct = (weighted_impact_no_qmix_energy / total_energies['no_qmix_original']) * 100
    
    # 加权聚合的影响分析(使用QMIX)
    weighted_impact_qmix_acc = final_accs['qmix_weighted'] - final_accs['qmix_original']
    weighted_impact_qmix_energy = total_energies['qmix_original'] - total_energies['qmix_weighted']
    weighted_impact_qmix_acc_pct = (weighted_impact_qmix_acc / final_accs['qmix_original']) * 100
    weighted_impact_qmix_energy_pct = (weighted_impact_qmix_energy / total_energies['qmix_original']) * 100
    
    logger.info("\n加权聚合对非QMIX系统的影响:")
    logger.info(f"  精度: {weighted_impact_no_qmix_acc:+.2f} ({weighted_impact_no_qmix_acc_pct:+.2f}%)")
    logger.info(f"  能耗: {weighted_impact_no_qmix_energy:+.2f} ({weighted_impact_no_qmix_energy_pct:+.2f}%)")
    
    logger.info("加权聚合对QMIX系统的影响:")
    logger.info(f"  精度: {weighted_impact_qmix_acc:+.2f} ({weighted_impact_qmix_acc_pct:+.2f}%)")
    logger.info(f"  能耗: {weighted_impact_qmix_energy:+.2f} ({weighted_impact_qmix_energy_pct:+.2f}%)")
    
    # 最终结论
    best_config = max(results.keys(), key=lambda k: final_accs[k])
    most_efficient = min(results.keys(), key=lambda k: total_energies[k])
    
    logger.info("\n" + "="*80)
    logger.info(f"最佳精度配置: {config_names[best_config]} (精度: {final_accs[best_config]:.2f}%)")
    logger.info(f"最低能耗配置: {config_names[most_efficient]} (能耗: {total_energies[most_efficient]:.2f}J)")
    
    best_overall = max(results.keys(), key=lambda k: final_accs[k]/total_energies[k])
    logger.info(f"综合最优配置: {config_names[best_overall]} (精度/能耗比: {final_accs[best_overall]/total_energies[best_overall]:.4f})")
    logger.info("="*80)

def run_with_aggregation_method(args, logger, tr_dataset, te_dataset, num_classes, device, method_name, client_resources=None, common_datasets=None):
    """使用指定的聚合方法运行训练"""
    # 创建本地副本
    args_copy = copy.deepcopy(args)
    args_copy.aggregation_method = method_name
    
    # 记录日志
    logger.info(f"开始使用 {method_name} 聚合方法进行训练")
    
    # 数据分割 - 如果提供了预先生成的数据集则使用它，否则重新分割
    if common_datasets is not None:
        Non_iid_tr_datasets, Non_iid_te_datasets = common_datasets
        logger.info("使用预先生成的通用数据集")
    else:
        Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split(
            num_classes, args_copy.num_clients, tr_dataset, te_dataset, args_copy.alpha)
    
    local_tr_data_loaders = [DataLoader(dataset, num_workers=0,
                                      batch_size=args_copy.batch_size, 
                                      shuffle=True)
                for dataset in Non_iid_tr_datasets]
    local_te_data_loaders = [DataLoader(dataset, num_workers=0,
                                      batch_size=args_copy.batch_size, 
                                      shuffle=True)
                for dataset in Non_iid_te_datasets]
    
    # 确保批次大小合适
    client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, num_classes, args_copy.num_clients)
    client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, num_classes, args_copy.num_clients)
    
    while 1 in np.remainder(client_total_samples, args_copy.batch_size) or 1 in np.remainder(client_total_te_samples, args_copy.batch_size):
        Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split(
            num_classes, args_copy.num_clients, tr_dataset, te_dataset, args_copy.alpha)
        client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, num_classes, args_copy.num_clients)
        client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, num_classes, args_copy.num_clients)
    
    # 初始化模型
    if args_copy.model == "Base_CNN":
        model = CNN_model.Base_CNN(n_bit=args_copy.n_bit).to(device)
    
    # 训练
    trainer = Simulator(args_copy, logger, local_tr_data_loaders, local_te_data_loaders, device)
    trainer.initialization(copy.deepcopy(model), client_resources)
    best_acc, final_acc, acc_history, energy_per_round, total_energy = trainer.FedAvg()
    
    # 记录结果
    logger.info(f"使用 {method_name} 聚合方法训练完成")
    logger.info(f"最佳精度: {best_acc:.4f}")
    logger.info(f"最终精度: {final_acc:.4f}")
    logger.info(f"总能耗: {total_energy:.4f}")
    
    return best_acc, final_acc, acc_history, energy_per_round, total_energy

def main(args):
    save_folder = args.affix
    
    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)
    qmix_model_folder = os.path.join(args.model_root, 'qmix')
    comparison_folder = os.path.join(args.log_root, 'aggregation_comparison')

    makedirs(log_folder)
    makedirs(model_folder)
    makedirs(comparison_folder)

    if args.use_qmix:
        makedirs(qmix_model_folder)
        setattr(args, 'qmix_model_folder', qmix_model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    if not hasattr(args, 'sample_ratio'):
        setattr(args, 'sample_ratio', args.schedulingsize / args.num_clients)

    logger = create_logger(log_folder, 'train', 'info')
    print_args(args, logger)

    if args.use_qmix:
        logger.info("QMIX enabled - will use reinforcement learning for client scheduling and quantization")
        if args.qmix_model_path:
            logger.info(f"Loading pre-trained QMIX model from: {args.qmix_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
       
    if args.dataset == 'mnist':
        tr_dataset = torchvision.datasets.MNIST(args.data_root, 
                                      train=True, 
                                      transform=torchvision.transforms.ToTensor(), 
                                      download=True)
        te_dataset = torchvision.datasets.MNIST(args.data_root, 
                                      train=False, 
                                      transform=torchvision.transforms.ToTensor(), 
                                      download=True)
    
    # 如果是全面比较模式，运行四种配置的实验
    if args.comparison_mode == 'comprehensive':
        logger.info("启动QMIX与聚合方法全面比较模式")
        run_comprehensive_comparison(args, logger, tr_dataset, te_dataset, num_classes, device)
        
    # 其他模式保持不变
    elif args.aggregation_method == 'compare':
        logger.info("启动聚合方法比较模式")
        
        # 为了公平比较，预先生成客户端资源和数据集分割
        logger.info("预生成客户端资源和数据集，确保两种方法使用相同的初始条件")
        
        # 1. 预生成客户端资源配置
        client_resources = []
        for i in range(args.num_clients):
            client_resources.append(Client_Class.ClientResources.generate_random())
        
        # 2. 预生成数据集划分
        Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split(
            num_classes, args.num_clients, tr_dataset, te_dataset, args.alpha)
        
        client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, num_classes, args.num_clients)
        client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, num_classes, args.num_clients)
        
        # 确保批次大小合适
        while 1 in np.remainder(client_total_samples, args.batch_size) or 1 in np.remainder(client_total_te_samples, args.batch_size):
            Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split(
                num_classes, args.num_clients, tr_dataset, te_dataset, args.alpha)
            client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, num_classes, args.num_clients)
            client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, num_classes, args.num_clients)
        
        common_datasets = (Non_iid_tr_datasets, Non_iid_te_datasets)
        
        logger.info(f"已生成 {args.num_clients} 个客户端资源配置和数据集划分")
        
        # 运行原始聚合方法
        orig_best_acc, orig_final_acc, orig_acc_history, orig_energy_history, orig_total_energy = run_with_aggregation_method(
            args, logger, tr_dataset, te_dataset, num_classes, device, 'original', 
            client_resources, common_datasets)
        
        # 运行加权聚合方法
        weighted_best_acc, weighted_final_acc, weighted_acc_history, weighted_energy_history, weighted_total_energy = run_with_aggregation_method(
            args, logger, tr_dataset, te_dataset, num_classes, device, 'weighted', 
            client_resources, common_datasets)
        
        # 保存比较结果
        if args.save_comparison_plot:
            # 精度比较图
            plt.figure(figsize=(12, 8))
            rounds = range(1, len(orig_acc_history) + 1)
            plt.plot(rounds, orig_acc_history, 'b-', label='Original Aggregation')
            plt.plot(rounds, weighted_acc_history, 'r-', label='Weighted Aggregation')
            plt.title('Aggregation Methods - Accuracy Comparison')
            plt.xlabel('Communication Rounds')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(comparison_folder, 'aggregation_comparison_accuracy.png'))
            
            # 能耗比较图
            plt.figure(figsize=(12, 8))
            plt.plot(rounds, orig_energy_history, 'b-', label='Original Aggregation')
            plt.plot(rounds, weighted_energy_history, 'r-', label='Weighted Aggregation')
            plt.title('Aggregation Methods - Energy Consumption Comparison')
            plt.xlabel('Communication Rounds')
            plt.ylabel('Energy Consumption (J)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(comparison_folder, 'aggregation_comparison_energy.png'))
            
            # 累积能耗比较图
            plt.figure(figsize=(12, 8))
            orig_cumulative_energy = np.cumsum(orig_energy_history)
            weighted_cumulative_energy = np.cumsum(weighted_energy_history)
            plt.plot(rounds, orig_cumulative_energy, 'b-', label='Original Aggregation')
            plt.plot(rounds, weighted_cumulative_energy, 'r-', label='Weighted Aggregation')
            plt.title('Aggregation Methods - Cumulative Energy Consumption')
            plt.xlabel('Communication Rounds')
            plt.ylabel('Cumulative Energy (J)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(comparison_folder, 'aggregation_comparison_cumulative_energy.png'))
            
            logger.info(f"比较结果图表已保存至 {comparison_folder} 目录")
            
            # 保存原始数据
            np.save(os.path.join(comparison_folder, 'original_acc_history.npy'), np.array(orig_acc_history))
            np.save(os.path.join(comparison_folder, 'weighted_acc_history.npy'), np.array(weighted_acc_history))
            np.save(os.path.join(comparison_folder, 'original_energy_history.npy'), np.array(orig_energy_history))
            np.save(os.path.join(comparison_folder, 'weighted_energy_history.npy'), np.array(weighted_energy_history))
        
        # 打印比较摘要
        logger.info("\n" + "="*60)
        logger.info(" "*20 + "聚合方法比较结果")
        logger.info("="*60)
        logger.info(f"{'指标':<15} | {'原始聚合':<12} | {'加权聚合':<12} | {'改进':<12} | {'改进率 (%)':<12}")
        logger.info("-"*60)
        
        # 精度比较
        acc_improvement = weighted_final_acc - orig_final_acc
        acc_improvement_pct = (acc_improvement / orig_final_acc) * 100 if orig_final_acc > 0 else float('inf')
        logger.info(f"{'最终精度 (%)':<15} | {orig_final_acc:<12.2f} | {weighted_final_acc:<12.2f} | {acc_improvement:+<12.2f} | {acc_improvement_pct:+<12.2f}")
        
        best_improvement = weighted_best_acc - orig_best_acc
        best_improvement_pct = (best_improvement / orig_best_acc) * 100 if orig_best_acc > 0 else float('inf')
        logger.info(f"{'最佳精度 (%)':<15} | {orig_best_acc:<12.2f} | {weighted_best_acc:<12.2f} | {best_improvement:+<12.2f} | {best_improvement_pct:+<12.2f}")
        
        # 能耗比较
        energy_improvement = orig_total_energy - weighted_total_energy
        energy_improvement_pct = (energy_improvement / orig_total_energy) * 100 if orig_total_energy > 0 else float('inf')
        logger.info(f"{'总能耗 (J)':<15} | {orig_total_energy:<12.2f} | {weighted_total_energy:<12.2f} | {energy_improvement:+<12.2f} | {energy_improvement_pct:+<12.2f}")
        
        # 计算收敛速度比较（达到90%最终精度的轮次）
        orig_threshold = 0.9 * orig_final_acc
        weighted_threshold = 0.9 * weighted_final_acc
        
        orig_convergence = next((i+1 for i, acc in enumerate(orig_acc_history) if acc >= orig_threshold), len(orig_acc_history))
        weighted_convergence = next((i+1 for i, acc in enumerate(weighted_acc_history) if acc >= weighted_threshold), len(weighted_acc_history))
        
        conv_improvement = orig_convergence - weighted_convergence
        conv_improvement_pct = (conv_improvement / orig_convergence) * 100 if orig_convergence > 0 else float('inf')
        logger.info(f"{'收敛轮次':<15} | {orig_convergence:<12d} | {weighted_convergence:<12d} | {conv_improvement:+<12d} | {conv_improvement_pct:+<12.2f}")
        
        # 收敛时的能耗
        orig_convergence_energy = np.sum(orig_energy_history[:orig_convergence])
        weighted_convergence_energy = np.sum(weighted_energy_history[:weighted_convergence])
        convergence_energy_improvement = orig_convergence_energy - weighted_convergence_energy
        convergence_energy_pct = (convergence_energy_improvement / orig_convergence_energy) * 100 if orig_convergence_energy > 0 else float('inf')
        logger.info(f"{'收敛能耗 (J)':<15} | {orig_convergence_energy:<12.2f} | {weighted_convergence_energy:<12.2f} | {convergence_energy_improvement:+<12.2f} | {convergence_energy_pct:+<12.2f}")
        
        # 效率指标：达到相同精度所需的能耗
        efficiency_threshold = min(orig_best_acc, weighted_best_acc) * 0.95  # 使用95%的较低最佳精度作为效率比较点
        orig_efficiency_round = next((i for i, acc in enumerate(orig_acc_history) if acc >= efficiency_threshold), len(orig_acc_history))
        weighted_efficiency_round = next((i for i, acc in enumerate(weighted_acc_history) if acc >= efficiency_threshold), len(weighted_acc_history))
        
        orig_efficiency_energy = np.sum(orig_energy_history[:orig_efficiency_round+1])
        weighted_efficiency_energy = np.sum(weighted_energy_history[:weighted_efficiency_round+1])
        
        efficiency_improvement = orig_efficiency_energy - weighted_efficiency_energy
        efficiency_pct = (efficiency_improvement / orig_efficiency_energy) * 100 if orig_efficiency_energy > 0 else float('inf')
        logger.info(f"{'能效比 (J/精度)':<15} | {orig_efficiency_energy:<12.2f} | {weighted_efficiency_energy:<12.2f} | {efficiency_improvement:+<12.2f} | {efficiency_pct:+<12.2f}")
        
        logger.info("="*60)
        
    else:
        # 常规模式，运行单一聚合方法
        # 此部分保持不变
        Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split(
            num_classes, args.num_clients, tr_dataset, te_dataset, args.alpha)
        
        local_tr_data_loaders = [DataLoader(dataset, num_workers=0,
                                          batch_size=args.batch_size, 
                                          shuffle=True)
                    for dataset in Non_iid_tr_datasets]
        local_te_data_loaders = [DataLoader(dataset, num_workers=0,
                                          batch_size=args.batch_size, 
                                          shuffle=True)
                    for dataset in Non_iid_te_datasets]

        client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, num_classes, args.num_clients)
        client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, num_classes, args.num_clients)

        while 1 in np.remainder(client_total_samples, args.batch_size) or 1 in np.remainder(client_total_te_samples, args.batch_size):
            Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split(
                num_classes, args.num_clients, tr_dataset, te_dataset, args.alpha)
            client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, num_classes, args.num_clients)
            client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, num_classes, args.num_clients)    
        
        if args.model == "Base_CNN":
            model = CNN_model.Base_CNN(n_bit=args.n_bit).to(device)

        trainer = Simulator(args, logger, local_tr_data_loaders, local_te_data_loaders, device)
        trainer.initialization(copy.deepcopy(model))
        best_acc, final_acc, acc_history, energy_history, total_energy = trainer.FedAvg()
        
        logger.info(f"Training completed with best accuracy: {best_acc:.4f}")
        logger.info(f"Final accuracy: {final_acc:.4f}")
        logger.info(f"Total energy consumption: {total_energy:.4f}")
        
        if args.use_qmix:
            np.save(os.path.join(qmix_model_folder, 'acc_history.npy'), np.array(acc_history))
            np.save(os.path.join(qmix_model_folder, 'energy_history.npy'), np.array(energy_history))
            logger.info(f"Accuracy and energy history saved to {qmix_model_folder}")

if __name__ == '__main__':
    args = parser()
    print_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    random.seed(args.seed)
    main(args)