"""
基础PINN框架
包含通用的PINN训练流程和损失函数
支持可配置的网络架构（标准MLP、MFFM、R-gPINN等）
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import os
import json
from utils.metrics import compute_errors

class MLP(nn.Module):
    """标准多层感知机网络"""
    
    def __init__(self, layers, activation='tanh'):
        """
        Args:
            layers: 列表，例如 [2, 50, 50, 1] 表示输入2维，两个隐藏层各50个神经元，输出1维
            activation: 激活函数 'tanh', 'relu', 'sin'
        """
        super().__init__()
        self.depth = len(layers) - 1
        
        # 选择激活函数（必须是nn.Module）
        activation_func = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sin': nn.SiLU(),  # SiLU作为sin的替代
            'gelu': nn.GELU()
        }.get(activation, nn.Tanh())
        
        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            layer_list.append(activation_func)
        
        layer_list.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*layer_list)
        
        # Xavier初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


class BasePINN(nn.Module):
    """PINN基类，提供通用的训练流程
    
    支持可配置的网络架构：
    - 'standard': 标准MLP
    - 'mffm': 多尺度傅里叶特征映射（用于高波数Helmholtz）
    - 'r_gpinn': 梯度增强残差PINN（用于反问题）
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 网络架构选择
        network_type = config.get('network_type', 'standard')
        layers = config.get('layers', [2, 50, 50, 50, 1])
        activation = config.get('activation', 'tanh')
        
        # 根据类型创建网络
        if network_type == 'standard':
            self.net = MLP(layers, activation=activation)
        elif network_type == 'mffm':
            # 使用多尺度傅里叶特征映射
            from .network_architectures import MFFM_PINN
            self.net = MFFM_PINN(
                input_dim=layers[0],
                fourier_features=config.get('fourier_features', 32),
                hidden_layers=layers[1:-1],
                fourier_scale=config.get('fourier_scale', 3.0),
                learnable_freq=config.get('learnable_freq', True),
                use_adaptive_activation=config.get('use_adaptive_activation', False)
            )
        elif network_type == 'r_gpinn':
            # 使用梯度增强残差PINN
            from .network_architectures import R_gPINN
            self.net = R_gPINN(
                input_dim=layers[0],
                hidden_dim=layers[1],
                num_blocks=config.get('num_residual_blocks', 3),
                output_dim=layers[-1],
                activation=activation
            )
        else:
            self.net = MLP(layers, activation=activation)
        
        self.net.to(self.device)
        
        # 优化器配置
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.0)
        
        # 损失权重
        self.lambda_pde = config.get('lambda_pde', 1.0)
        self.lambda_bc = config.get('lambda_bc', 1.0)
        self.lambda_data = config.get('lambda_data', 1.0)
        
        # 训练历史
        self.train_loss_history = []
        self.test_loss_history = []
        self.pde_loss_history = []  # PDE损失历史
        self.bc_loss_history = []  # 边界损失历史
        self.data_loss_history = []  # 数据损失历史
        self.l2_error_history = []  # L2误差历史

    def forward(self, x):
        """前向传播"""
        return self.net(x)

    def pde_loss(self, x, u_pred):
        """
        PDE残差损失
        子类需要重写此方法以定义具体的PDE
        """
        raise NotImplementedError("Subclass must implement pde_loss")

    def boundary_loss(self, x_bc, u_bc_true):
        """
        边界条件损失
        """
        u_bc_pred = self.forward(x_bc)
        return torch.mean((u_bc_pred - u_bc_true) ** 2)

    def data_loss(self, x_data, u_data_true):
        """
        数据拟合损失（用于反问题）
        """
        u_data_pred = self.forward(x_data)
        return torch.mean((u_data_pred - u_data_true) ** 2)

    def compute_loss_components(self, batch_dict: Dict) -> Dict:
        """
        计算各个损失分量
        返回: 包含各个损失分量的字典
        """
        loss_components = {}

        # PDE残差损失
        if 'x_interior' in batch_dict:
            x_pde = batch_dict['x_interior']
            u_pde_pred = self.forward(x_pde)
            loss_components['pde_loss'] = self.pde_loss(x_pde, u_pde_pred)

        # 边界条件损失
        if 'x_boundary' in batch_dict and 'u_boundary' in batch_dict:
            loss_components['bc_loss'] = self.boundary_loss(
                batch_dict['x_boundary'],
                batch_dict['u_boundary']
            )

        # 数据拟合损失（用于反问题）
        if 'x_data' in batch_dict and 'u_data' in batch_dict:
            loss_components['data_loss'] = self.data_loss(
                batch_dict['x_data'],
                batch_dict['u_data']
            )

        return loss_components
    
    def compute_total_loss_improved(self, batch_dict: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失和各个损失分量
        返回: (总损失, 损失分量字典)
        """
        loss_components = self.compute_loss_components(batch_dict)

        total_loss = torch.tensor(0.0, device=self.device)  # 确保是张量
        if 'pde_loss' in loss_components:
            total_loss += self.lambda_pde * loss_components['pde_loss']
        if 'bc_loss' in loss_components:
            total_loss += self.lambda_bc * loss_components['bc_loss']
        if 'data_loss' in loss_components:
            total_loss += self.lambda_data * loss_components['data_loss']

        return total_loss, loss_components

    def compute_total_loss(self, batch_dict: Dict) -> torch.Tensor:
        """
        计算总损失
        batch_dict包含: x_interior, x_boundary, x_data等
        """
        loss = 0.0

        # PDE残差损失
        if 'x_interior' in batch_dict:
            x_pde = batch_dict['x_interior']
            u_pde_pred = self.forward(x_pde)
            loss += self.lambda_pde * self.pde_loss(x_pde, u_pde_pred)

        # 边界条件损失
        if 'x_boundary' in batch_dict and 'u_boundary' in batch_dict:
            loss += self.lambda_bc * self.boundary_loss(
                batch_dict['x_boundary'],
                batch_dict['u_boundary']
            )

        # 数据拟合损失（用于反问题）
        if 'x_data' in batch_dict and 'u_data' in batch_dict:
            loss += self.lambda_data * self.data_loss(
                batch_dict['x_data'],
                batch_dict['u_data']
            )

        return loss

    def train_step(self, batch_dict: Dict, optimizer) -> float:
        """单步训练"""
        optimizer.zero_grad()
        loss = self.compute_total_loss(batch_dict)
        loss.backward()
        optimizer.step()
        return loss.item()

    def train_step_improved(self, batch_dict: Dict, optimizer) -> Tuple[float, Dict]:
        """单步训练，返回总损失和损失分量"""
        optimizer.zero_grad()
        total_loss, loss_components = self.compute_total_loss_improved(batch_dict)
        total_loss.backward()
        optimizer.step()
        return total_loss.item(), loss_components
    
    def evaluate(self, x_test: np.ndarray, u_true: np.ndarray) -> Dict:
        """
        评估模型性能
        """
        x_test_tensor = torch.FloatTensor(x_test).to(self.device)
        
        with torch.no_grad():
            u_pred = self.forward(x_test_tensor).cpu().numpy().flatten()

        # 调用 compute_errors 计算所有指标
        errors = compute_errors(u_pred, u_true)
        errors['u_pred'] = u_pred  # 如果还想返回预测值

        return errors
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Dict):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': optimizer_state,
            'config': self.config,
            'train_loss_history': self.train_loss_history,
            'test_loss_history': self.test_loss_history
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.train_loss_history = checkpoint.get('train_loss_history', [])
        self.test_loss_history = checkpoint.get('test_loss_history', [])
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch']


def create_batches(X: np.ndarray, batch_size: int, shuffle=True) -> list:
    """创建批量数据"""
    n = len(X)
    indices = np.arange(n)
    
    if shuffle:
        np.random.shuffle(indices)
    
    batches = []
    for i in range(0, n, batch_size):
        batch_indices = indices[i:min(i+batch_size, n)]
        
        batches.append(X[batch_indices])
    
    return batches


def create_dataloader(X_interior, X_boundary=None, u_boundary=None, 
                     X_data=None, u_data=None, batch_size=128):
    """
    创建数据加载器
    返回一个批次的batch_dict
    """
    n_interior = len(X_interior)
    
    # 创建批量索引
    indices_interior = np.arange(n_interior)
    np.random.shuffle(indices_interior)
    
    batches = []
    for i in range(0, n_interior, batch_size):
        batch_idx = indices_interior[i:min(i+batch_size, n_interior)]
        
        batch_dict = {
            'x_interior': torch.FloatTensor(X_interior[batch_idx])
        }
        
        if X_boundary is not None:
            # 边界点一起加入（或者也可以随机采样）
            n_bc_sample = min(batch_size // 4, len(X_boundary))
            bc_indices = np.random.choice(len(X_boundary), n_bc_sample, replace=False)
            batch_dict['x_boundary'] = torch.FloatTensor(X_boundary[bc_indices])
            if u_boundary is not None:
                batch_dict['u_boundary'] = torch.FloatTensor(u_boundary[bc_indices])
        
        if X_data is not None and u_data is not None:
            # 数据点（用于反问题）
            batch_dict['x_data'] = torch.FloatTensor(X_data)
            batch_dict['u_data'] = torch.FloatTensor(u_data)
        
        batches.append(batch_dict)
    
    return batches
