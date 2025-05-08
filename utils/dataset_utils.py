import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class TinyImageNet(Dataset):
    """Tiny ImageNet数据集加载器"""
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
        # 路径设置
        if train:
            self.image_dir = os.path.join(root, 'train')
        else:
            self.image_dir = os.path.join(root, 'val')
        
        self.images = []
        self.targets = []
        
        # 加载类别映射
        wnids_path = os.path.join(root, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            self.class_names = [x.strip() for x in f]
        
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        # 加载图像和标签
        if train:
            for class_name in self.class_names:
                class_dir = os.path.join(self.image_dir, class_name, 'images')
                for img_file in os.listdir(class_dir):
                    if img_file.endswith('.JPEG'):
                        self.images.append(os.path.join(class_dir, img_file))
                        self.targets.append(self.class_to_idx[class_name])
        else:
            val_annotations_path = os.path.join(self.image_dir, 'val_annotations.txt')
            with open(val_annotations_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    img_file = parts[0]
                    class_name = parts[1]
                    if class_name in self.class_names:  # 确保类别有效
                        self.images.append(os.path.join(self.image_dir, 'images', img_file))
                        self.targets.append(self.class_to_idx[class_name])
    
    def __getitem__(self, index):
        img_path = self.images[index]
        target = self.targets[index]
        
        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        return len(self.images)


class ShakespeareDataset(Dataset):
    """Shakespeare文本数据集加载器"""
    def __init__(self, file_path, seq_length=100, train=True, split_ratio=0.8):
        # 加载文本
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        
        # 创建字符到索引的映射
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # 编码文本
        self.encoded_text = np.array([self.char_to_idx[ch] for ch in self.text])
        
        # 分割训练集和测试集
        data_size = len(self.encoded_text) - seq_length
        train_size = int(data_size * split_ratio)
        
        if train:
            self.data_indices = np.arange(train_size)
        else:
            self.data_indices = np.arange(train_size, data_size)
        
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data_indices)
    
    def __getitem__(self, idx):
        # 获取索引位置
        pos = self.data_indices[idx]
        
        # 提取序列和标签（预测下一个字符）
        x = torch.from_numpy(self.encoded_text[pos:pos+self.seq_length]).long()
        y = torch.from_numpy(self.encoded_text[pos+1:pos+self.seq_length+1]).long()
        
        return x, y
    
    def get_vocab_size(self):
        return self.vocab_size


def get_dataset(dataset_name, args):
    """获取指定的数据集"""
    if dataset_name == 'mnist':
        # MNIST数据集
        transform = transforms.ToTensor()
        tr_dataset = torchvision.datasets.MNIST(args.data_root, 
                                          train=True, 
                                          transform=transform, 
                                          download=True)
        te_dataset = torchvision.datasets.MNIST(args.data_root, 
                                          train=False, 
                                          transform=transform, 
                                          download=True)
        num_classes = 10
        in_channels = 1
    
    elif dataset_name == 'cifar10':
        # CIFAR-10数据集
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        
        tr_dataset = torchvision.datasets.CIFAR10(args.data_root, 
                                            train=True, 
                                            transform=transform_train, 
                                            download=True)
        te_dataset = torchvision.datasets.CIFAR10(args.data_root, 
                                            train=False, 
                                            transform=transform_test, 
                                            download=True)
        num_classes = 10
        in_channels = 3
    
    elif dataset_name == 'cifar100':
        # CIFAR-100数据集
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        tr_dataset = torchvision.datasets.CIFAR100(args.data_root, 
                                             train=True, 
                                             transform=transform_train, 
                                             download=True)
        te_dataset = torchvision.datasets.CIFAR100(args.data_root, 
                                             train=False, 
                                             transform=transform_test, 
                                             download=True)
        num_classes = 100
        in_channels = 3
    
    elif dataset_name == 'tiny-imagenet':
        # Tiny-ImageNet数据集
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        tiny_imagenet_path = os.path.join(args.data_root, 'tiny-imagenet-200')
        if not os.path.exists(tiny_imagenet_path):
            raise FileNotFoundError(f"Tiny-ImageNet 数据集未找到。请下载并解压到 {tiny_imagenet_path}")
        
        tr_dataset = TinyImageNet(tiny_imagenet_path, 
                                  train=True, 
                                  transform=transform_train)
        te_dataset = TinyImageNet(tiny_imagenet_path, 
                                  train=False, 
                                  transform=transform_test)
        num_classes = 200
        in_channels = 3
    
    elif dataset_name == 'shakespeare':
        # Shakespeare文本数据集
        shakespeare_path = os.path.join(args.data_root, 'shakespeare', 'shakespeare.txt')
        if not os.path.exists(shakespeare_path):
            os.makedirs(os.path.dirname(shakespeare_path), exist_ok=True)
            # 提供一个简单的错误信息，指导用户下载数据集
            raise FileNotFoundError(f"Shakespeare 数据集未找到，请下载并放置在 {shakespeare_path}")
        
        seq_length = 100  # 序列长度为100个字符
        tr_dataset = ShakespeareDataset(shakespeare_path, seq_length=seq_length, train=True)
        te_dataset = ShakespeareDataset(shakespeare_path, seq_length=seq_length, train=False)
        
        num_classes = tr_dataset.get_vocab_size()  # 词汇表大小作为类别数
        in_channels = None  # 文本数据没有通道数概念
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
        
    # 确保所有数据集都有 targets 属性
    if hasattr(tr_dataset, 'labels') and not hasattr(tr_dataset, 'targets'):
        tr_dataset.targets = tr_dataset.labels
    if hasattr(te_dataset, 'labels') and not hasattr(te_dataset, 'targets'):
        te_dataset.targets = te_dataset.labels
    
    # 确保 targets 是 numpy 数组
    if isinstance(tr_dataset.targets, list):
        tr_dataset.targets = np.array(tr_dataset.targets)
    if isinstance(te_dataset.targets, list):
        te_dataset.targets = np.array(te_dataset.targets)
    
    # 确保 data 属性存在
    if not hasattr(tr_dataset, 'data') and hasattr(tr_dataset, 'images'):
        tr_dataset.data = tr_dataset.images
    if not hasattr(te_dataset, 'data') and hasattr(te_dataset, 'images'):
        te_dataset.data = te_dataset.images
        
    return tr_dataset, te_dataset, num_classes, in_channels