"""
自适应梯度优化器封装
Adam, AdamW, RAdam等
"""

import torch
import torch.optim as optim
from typing import Optional


def create_adam(params, learning_rate=0.001, weight_decay=0.0, betas=(0.9, 0.999)):
    """
    创建Adam优化器
    
    Args:
        params: 模型参数
        learning_rate: 学习率
        weight_decay: 权重衰减（L2正则化）
        betas: Adam的动量参数
    """
    return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, betas=betas)


def create_adamw(params, learning_rate=0.001, weight_decay=0.01, betas=(0.9, 0.999)):
    """
    创建AdamW优化器（解耦权重衰减）
    
    Args:
        params: 模型参数
        learning_rate: 学习率
        weight_decay: 权重衰减（推荐0.01）
        betas: Adam的动量参数
    """
    return optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, betas=betas)


def create_radam(params, learning_rate=0.001, weight_decay=0.0, betas=(0.9, 0.999)):
    """
    创建RAdam优化器（Rectified Adam）
    
    注意：如果torch中没有RAdam，使用低学习率的Adam作为替代
    
    Args:
        params: 模型参数
        learning_rate: 学习率（RAdam通常用更低的学习率）
        weight_decay: 权重衰减
        betas: Adam的动量参数
    """
    try:
        # 尝试导入RAdam（需要安装pytorch-optimizer包）
        from torch_optimizer import RAdam
        return RAdam(params, lr=learning_rate, weight_decay=weight_decay, betas=betas)
    except ImportError:
        # 如果没有RAdam，使用低学习率的Adam作为替代
        # RAdam的核心优势是更稳定的训练，低学习率Adam也能达到类似效果
        return optim.Adam(params, lr=learning_rate * 0.5, weight_decay=weight_decay, betas=betas)


def create_sgd(params, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
    """
    创建SGD优化器（带动量）
    
    Args:
        params: 模型参数
        learning_rate: 学习率
        momentum: 动量系数
        weight_decay: 权重衰减
    """
    return optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


# 便捷函数：根据名称创建优化器
def create_optimizer_by_name(name: str, params, learning_rate=0.001, **kwargs):
    """
    根据名称创建优化器
    
    Args:
        name: 优化器名称 ('adam', 'adamw', 'radam', 'sgd')
        params: 模型参数
        learning_rate: 学习率
        **kwargs: 其他参数
    """
    optimizers = {
        'adam': create_adam,
        'adamw': create_adamw,
        'radam': create_radam,
        'sgd': create_sgd
    }
    
    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
    
    return optimizers[name.lower()](params, learning_rate=learning_rate, **kwargs)

