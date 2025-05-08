import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantization import *

# 基本模块定义，用于ResNet
class BasicBlock_Q(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, n_bit, stride=1):
        super(BasicBlock_Q, self).__init__()
        self.n_bit = n_bit
        self.activation = activation_quantize_fn(self.n_bit)
        
        self.conv1 = Conv2d_Q(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_Q(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 设置量化级别
        self.conv1.set_quantization_level(self.n_bit)
        self.conv2.set_quantization_level(self.n_bit)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                Conv2d_Q(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
            # 设置量化级别
            self.shortcut[0].set_quantization_level(self.n_bit)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet_Q(nn.Module):
    def __init__(self, block, num_blocks, n_bit=16, num_classes=10):
        super(ResNet_Q, self).__init__()
        self.n_bit = n_bit
        self.activation = activation_quantize_fn(self.n_bit)
        self.in_planes = 64
        
        self.conv1 = Conv2d_Q(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = Linear_Q(512 * block.expansion, num_classes)
        
        # 设置量化级别
        self.conv1.set_quantization_level(self.n_bit)
        self.linear.set_quantization_level(self.n_bit)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.n_bit, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# 定义量化版本ResNet模型
def ResNet18_Q(n_bit=16, num_classes=10):
    return ResNet_Q(BasicBlock_Q, [2, 2, 2, 2], n_bit, num_classes)

def ResNet34_Q(n_bit=16, num_classes=10):
    return ResNet_Q(BasicBlock_Q, [3, 4, 6, 3], n_bit, num_classes)


# MobileNet量化版本
class MobileNetBlock_Q(nn.Module):
    def __init__(self, in_planes, out_planes, n_bit, stride=1):
        super(MobileNetBlock_Q, self).__init__()
        self.n_bit = n_bit
        self.activation = activation_quantize_fn(self.n_bit)
        
        # 深度卷积
        self.conv1 = Conv2d_Q(in_planes, in_planes, kernel_size=3, stride=stride, 
                              padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        
        # 逐点卷积
        self.conv2 = Conv2d_Q(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        # 设置量化级别
        self.conv1.set_quantization_level(self.n_bit)
        self.conv2.set_quantization_level(self.n_bit)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        return out


class MobileNet_Q(nn.Module):
    # (128,2) 表示通道数为128，步长为2
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    
    def __init__(self, n_bit=16, num_classes=10):
        super(MobileNet_Q, self).__init__()
        self.n_bit = n_bit
        self.activation = activation_quantize_fn(self.n_bit)
        
        # 根据输入数据集调整初始卷积尺寸
        self.conv1 = Conv2d_Q(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = Linear_Q(1024, num_classes)
        
        # 设置量化级别
        self.conv1.set_quantization_level(self.n_bit)
        self.linear.set_quantization_level(self.n_bit)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(MobileNetBlock_Q(in_planes, out_planes, self.n_bit, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# 非量化版本的模型 (使用标准PyTorch层)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# 定义非量化ResNet模型
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


class MobileNetBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(MobileNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, 
                              padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(MobileNetBlock(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# 为Shakespeare数据集添加LSTM语言模型
class LSTM_Q(nn.Module):
    def __init__(self, vocab_size, n_bit=16, embedding_dim=8, hidden_dim=256, num_layers=2, dropout=0.2):
        super(LSTM_Q, self).__init__()
        self.n_bit = n_bit
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        
        # 量化激活函数
        self.activation = activation_quantize_fn(self.n_bit)
        
        # 全连接层
        self.fc = Linear_Q(hidden_dim, vocab_size)
        self.fc.set_quantization_level(self.n_bit)
        
    def forward(self, x):
        # x维度 [batch, seq_len]
        batch_size = x.size(0)
        
        # 嵌入
        embed = self.embedding(x)  # [batch, seq_len, embedding_dim]
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(embed, (h0, c0))  # [batch, seq_len, hidden_dim]
        
        # 量化激活
        lstm_out = self.activation(lstm_out)
        
        # 全连接层
        output = self.fc(lstm_out)  # [batch, seq_len, vocab_size]
        
        return output


class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim=8, hidden_dim=256, num_layers=2, dropout=0.2):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        embed = self.embedding(x)  # [batch, seq_len, embedding_dim]
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        lstm_out, _ = self.lstm(embed, (h0, c0))  # [batch, seq_len, hidden_dim]
        
        output = self.fc(lstm_out)  # [batch, seq_len, vocab_size]
        
        return output