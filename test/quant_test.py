import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
from collections import OrderedDict
import copy
from dataclasses import dataclass, field

def uniform_quantize(quant_budget, x):
    if quant_budget == 32:
        return x
    elif quant_budget == 1:
        return torch.sign(x)
    else:
        m = float(2 ** (quant_budget - 1))
        out = torch.round(x * m) / m
        return out
    
# 反量化，将经过m位的tensor反量化，参考以下代码
# local_difference = {
#                 key: value / (2 ** quantization_bits) 
#                 for key, value in quantized_difference.items()
#             }
def dequantize(x, quant_budget):
    m = float(2 ** (quant_budget - 1))
    # 反量化公式：将量化后的张量乘以 m，得到整数部分，再除以 m 得到浮点值
    dequantized_x = torch.round(x * m) / m
    return dequantized_x

# 原始张量
x = torch.tensor([0.6, -0.3, 1.2])

# 量化位数
quant_budget = 8

# 量化
quantized_x = uniform_quantize(quant_budget, x)
print("量化后的张量:", quantized_x)
# 反量化
dequantized_x = dequantize(quantized_x, quant_budget)
print("反量化后的张量:", dequantized_x)
# 验证反量化是否正确
assert torch.allclose(x, dequantized_x, atol=1e-5), "反量化不正确"
