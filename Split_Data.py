from torch.utils.data import Dataset
import numpy as np
import torch
import random

class Non_iid(Dataset):
    def __init__(self, x, y):
        # 修改这里以处理通道维度
        if len(x.shape) == 4:  # 如果是 [batch, height, width, channels]
            # 将通道维度从最后一维移到第二维
            self.x_data = x.permute(0, 3, 1, 2).to(torch.float32)
        elif len(x.shape) == 3:  # 如果是 [batch, height, width]
            self.x_data = x.unsqueeze(1).to(torch.float32)  # 添加通道维度
        else:
            self.x_data = x.unsqueeze(1).to(torch.float32) if len(x.shape) == 1 else x.to(torch.float32)
        self.y_data = y.to(torch.int64)
        self.cuda_available = torch.cuda.is_available()
    
    #Return the number of data
    def __len__(self):
        return len(self.x_data)
    
    #Sampling
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        if self.cuda_available:
            return x.cuda(), y.cuda()
        else:
            return x, y


def data_stats(non_iid_datasets, num_classes, num_clients):
    client_data_counts = {client:{} for client in range(num_clients)}
    client_total_samples = []
    for client, data in enumerate(non_iid_datasets):
        total_sample = 0
        for label in range(num_classes):
            idx_label = len(np.where(data.y_data == label)[0])
            label_sum = np.sum(idx_label)
            client_data_counts[client][label] = label_sum
            total_sample += label_sum
        client_total_samples.append(total_sample)

    return client_data_counts, client_total_samples

def Non_iid_split(num_classes, num_clients, tr_datasets, te_datasets, alpha):
    """
    Input: num_classes, num_clients, datasets, alpha
    Output: Dataset classes of the number of num_clients 
    """
    # 确保 targets 属性存在并且是 numpy 数组
    if hasattr(tr_datasets, 'labels') and not hasattr(tr_datasets, 'targets'):
        tr_datasets.targets = tr_datasets.labels
    if hasattr(te_datasets, 'labels') and not hasattr(te_datasets, 'targets'):
        te_datasets.targets = te_datasets.labels
    
    # 如果 targets 是列表，转换为 numpy 数组
    if isinstance(tr_datasets.targets, list):
        tr_datasets.targets = np.array(tr_datasets.targets)
    if isinstance(te_datasets.targets, list):
        te_datasets.targets = np.array(te_datasets.targets)
    
    # 如果 targets 是 tensor，转换为 numpy 数组
    if torch.is_tensor(tr_datasets.targets):
        tr_datasets.targets = tr_datasets.targets.numpy()
    if torch.is_tensor(te_datasets.targets):
        te_datasets.targets = te_datasets.targets.numpy()

    tr_idx_batch = [[] for _ in range(num_clients)]
    tr_data_index_map = {}
    te_idx_batch = [[] for _ in range(num_clients)]
    te_data_index_map = {}

    #for each class in the training/test dataset
    for label in range(num_classes):
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients)) #It generates dirichichlet random variable with alpha over num_clients

        # 使用 numpy 布尔索引而不是 np.where
        tr_idx_label = np.arange(len(tr_datasets.targets))[tr_datasets.targets == label]
        np.random.shuffle(tr_idx_label)
        tr_proportions = (np.cumsum(proportions) * len(tr_idx_label)).astype(int)[:-1]

        tr_idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                         zip(tr_idx_batch, np.split(tr_idx_label, tr_proportions))]
        
        te_idx_label = np.arange(len(te_datasets.targets))[te_datasets.targets == label]
        np.random.shuffle(te_idx_label)
        te_proportions = (np.cumsum(proportions) * len(te_idx_label)).astype(int)[:-1]

        te_idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                         zip(te_idx_batch, np.split(te_idx_label, te_proportions))]
        
    for client in range(num_clients):
        np.random.shuffle(tr_idx_batch[client])
        tr_data_index_map[client] = tr_idx_batch[client]
        te_data_index_map[client] = te_idx_batch[client]

    # 确保 data 属性存在
    if not hasattr(tr_datasets, 'data') and hasattr(tr_datasets, 'images'):
        tr_datasets.data = tr_datasets.images
    if not hasattr(te_datasets, 'data') and hasattr(te_datasets, 'images'):
        te_datasets.data = te_datasets.images

    Non_iid_tr_datasets = []
    Non_iid_te_datasets = []

    for client in range(num_clients):
        # 检查是否为 tensor 格式，如果不是转换为 tensor
        if not torch.is_tensor(tr_datasets.data):
            tr_x_data = torch.tensor(tr_datasets.data[tr_data_index_map[client]])
        else:
            tr_x_data = tr_datasets.data[tr_data_index_map[client]]
        
        if not torch.is_tensor(tr_datasets.targets):
            tr_y_data = torch.tensor(tr_datasets.targets[tr_data_index_map[client]])
        else:
            tr_y_data = tr_datasets.targets[tr_data_index_map[client]]
        
        Non_iid_tr_datasets.append(Non_iid(tr_x_data, tr_y_data))

        if not torch.is_tensor(te_datasets.data):
            te_x_data = torch.tensor(te_datasets.data[te_data_index_map[client]])
        else:
            te_x_data = te_datasets.data[te_data_index_map[client]]
        
        if not torch.is_tensor(te_datasets.targets):
            te_y_data = torch.tensor(te_datasets.targets[te_data_index_map[client]])
        else:
            te_y_data = te_datasets.targets[te_data_index_map[client]]
        
        Non_iid_te_datasets.append(Non_iid(te_x_data, te_y_data))

    return Non_iid_tr_datasets, Non_iid_te_datasets