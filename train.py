import os
import logging
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision

from train_argument import parser, print_args

import random
import copy

from time import time
from model import CNN_model
from utils import *
from Simulator import Simulator
from Split_Data import Non_iid_split, data_stats

def main(args):
    save_folder = args.affix
    
    log_folder = os.path.join(args.log_root, save_folder) #return a new path 
    model_folder = os.path.join(args.model_root, save_folder)
    # 添加QMIX模型保存路径
    qmix_model_folder = os.path.join(args.model_root, 'qmix')

    makedirs(log_folder)
    makedirs(model_folder)

    # 如果使用QMIX，创建QMIX模型目录
    if args.use_qmix:
        makedirs(qmix_model_folder)
        setattr(args, 'qmix_model_folder', qmix_model_folder)

    setattr(args, 'log_folder', log_folder) #setattr(obj, var, val) assign object attribute to its value, just like args.'log_folder' = log_folder
    setattr(args, 'model_folder', model_folder)

    # 设置客户端采样率参数，用于QMIX控制器
    if not hasattr(args, 'sample_ratio'):
        setattr(args, 'sample_ratio', args.schedulingsize / args.num_clients)

    logger = create_logger(log_folder, 'train', 'info')
    print_args(args, logger) #It prints arguments

    # 如果使用QMIX，记录相关信息
    if args.use_qmix:
        logger.info("QMIX enabled - will use reinforcement learning for client scheduling and quantization")
        if args.qmix_model_path:
            logger.info(f"Loading pre-trained QMIX model from: {args.qmix_model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
       
    if args.dataset =='mnist':
        tr_dataset = torchvision.datasets.MNIST(args.data_root, 
                                        train=True, 
                                        transform=torchvision.transforms.ToTensor(), 
                                        download=True)

        # evaluation during training
        te_dataset = torchvision.datasets.MNIST(args.data_root, 
                                        train=False, 
                                        transform=torchvision.transforms.ToTensor(), 
                                        download=True)    
        
       
    Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split(
            num_classes, args.num_clients, tr_dataset, te_dataset, args.alpha)
    
    local_tr_data_loaders = [DataLoader(dataset, num_workers = 0,
                                        batch_size = args.batch_size, 
                                        shuffle = True)
                    for dataset in Non_iid_tr_datasets]
    local_te_data_loaders = [DataLoader(dataset, num_workers = 0,
                                        batch_size = args.batch_size, 
                                        shuffle = True)
                    for dataset in Non_iid_te_datasets]

    client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, num_classes, args.num_clients)
    client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, num_classes, args.num_clients)

    while 1 in np.remainder(client_total_samples, args.batch_size) or 1 in np.remainder(client_total_te_samples, args.batch_size): #There should be more than one sample in a batch
            Non_iid_tr_datasets, Non_iid_te_datasets = Non_iid_split(
            num_classes, args.num_clients, tr_dataset, te_dataset, args.alpha)
            client_data_counts, client_total_samples = data_stats(Non_iid_tr_datasets, num_classes, args.num_clients)
            client_te_data_counts, client_total_te_samples = data_stats(Non_iid_te_datasets, num_classes, args.num_clients)    
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "Base_CNN":
        model = CNN_model.Base_CNN(n_bit= args.n_bit).to(device)

    trainer = Simulator(args, logger, local_tr_data_loaders, local_te_data_loaders, device)
    trainer.initialization(copy.deepcopy(model))
    # 启动训练，并获取训练结果
    best_acc, final_acc, acc_history = trainer.FedAvg()
    
    # 记录训练结果
    logger.info(f"Training completed with best accuracy: {best_acc:.4f}")
    logger.info(f"Final accuracy: {final_acc:.4f}")
    
    # 如果使用QMIX，保存结果比较
    if args.use_qmix:
        # 保存训练曲线数据
        np.save(os.path.join(qmix_model_folder, 'acc_history.npy'), np.array(acc_history))
        logger.info(f"Accuracy history saved to {os.path.join(qmix_model_folder, 'acc_history.npy')}")

if __name__ == '__main__':
    args = parser()
    print_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    random.seed(args.seed)
    main(args)
