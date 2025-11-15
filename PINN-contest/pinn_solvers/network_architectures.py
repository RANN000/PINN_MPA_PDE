"""
先进网络架构
基于文献调研实现多种创新网络结构
"""

import torch
import torch.nn as nn
import numpy as np


class FourierFeatureMapping(nn.Module):
    """
    多尺度傅里叶特征映射（文献3）
    用于提升高频表达能力，解决频谱偏差问题
    """
    
    def __init__(self, input_dim=2, num_features=64, scale=10.0, learnable=True):
        """
        Args:
            input_dim: 输入维度（通常是2D空间坐标）
            num_features: 傅里叶特征数量
            scale: 傅里叶基的尺度（频率）
            learnable: 是否让频率可学习
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        
        # 初始化频率矩阵 B（采样自正态分布）
        if learnable:
            self.B = nn.Parameter(torch.randn(num_features, input_dim) * scale)
        else:
            self.register_buffer('B', torch.randn(num_features, input_dim) * scale)
    
    def forward(self, x):
        """
        将输入映射到傅里叶特征空间
        x: (batch, input_dim)
        output: (batch, 2*num_features) - sin和cos特征拼接
        """
        # 计算 Bx
        Bx = torch.matmul(2 * np.pi * x, self.B.t())  # (batch, num_features)
        
        # 生成sin和cos特征
        sin_features = torch.sin(Bx)
        cos_features = torch.cos(Bx)
        
        # 拼接
        features = torch.cat([sin_features, cos_features], dim=1)  # (batch, 2*num_features)
        return features


class AdaptiveSinActivation(nn.Module):
    """
    自适应正弦激活函数（文献4）
    sin(w0 * x)，其中w0可学习
    """
    
    def __init__(self, initial_freq=1.0):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(initial_freq))
    
    def forward(self, x):
        return torch.sin(self.w0 * x)


class ResidualUnit(nn.Module):
    """
    残差单元（文献6 - R-gPINN）
    Pre-activation residual block
    """
    
    def __init__(self, dim, activation='tanh'):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sin':
            self.activation = AdaptiveSinActivation()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return out + residual  # 残差连接


class MFFM_PINN(nn.Module):
    """
    多尺度傅里叶特征映射PINN（文献3）
    适用于高频Helmholtz问题
    """
    
    def __init__(self, input_dim=2, fourier_features=64, hidden_layers=[50, 50, 50], 
                 fourier_scale=10.0, learnable_freq=True, use_adaptive_activation=False):
        super().__init__()
        
        # 傅里叶特征映射
        self.fourier_map = FourierFeatureMapping(
            input_dim=input_dim,
            num_features=fourier_features,
            scale=fourier_scale,
            learnable=learnable_freq
        )
        
        # 网络主体
        feature_dim = 2 * fourier_features  # sin + cos
        layers = []
        
        layers.append(nn.Linear(feature_dim, hidden_layers[0]))
        layers.append(AdaptiveSinActivation() if use_adaptive_activation else nn.Tanh())
        
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(AdaptiveSinActivation() if use_adaptive_activation else nn.Tanh())
        
        layers.append(nn.Linear(hidden_layers[-1], 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # 先通过傅里叶特征映射
        features = self.fourier_map(x)
        # 再通过网络
        return self.net(features)


class R_gPINN(nn.Module):
    """
    梯度增强残差PINN（文献6）
    适用于变系数方程和反问题
    """
    
    def __init__(self, input_dim=2, hidden_dim=50, num_blocks=3, output_dim=1, activation='tanh'):
        super().__init__()
        
        # 输入层
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualUnit(hidden_dim, activation=activation) 
            for _ in range(num_blocks)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 激活函数
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sin':
            self.activation = AdaptiveSinActivation()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        out = self.activation(self.input_layer(x))
        for block in self.residual_blocks:
            out = block(out)
        return self.output_layer(out)


class DualNetworkPINN(nn.Module):
    """
    双网络架构（文献5, 6）
    一个网络学习u(x,y)，另一个学习参数λ(x,y)
    适用于反问题
    """
    
    def __init__(self, u_net, lambda_net):
        super().__init__()
        self.u_net = u_net  # 学习解u(x,y)
        self.lambda_net = lambda_net  # 学习参数λ(x,y)
    
    def forward(self, x):
        u = self.u_net(x)
        lambda_val = self.lambda_net(x)
        return u, lambda_val
    
    def forward_u(self, x):
        """只输出u"""
        return self.u_net(x)
    
    def forward_lambda(self, x):
        """只输出λ"""
        return self.lambda_net(x)


# 便捷函数：创建不同类型的网络
def create_network(network_type='standard', **kwargs):
    """
    创建不同类型的网络
    
    Args:
        network_type: 'standard', 'mffm', 'r_gpinn', 'adaptive_sin'
        **kwargs: 网络参数
    """
    input_dim = kwargs.get('input_dim', 2)
    hidden_layers = kwargs.get('hidden_layers', [50, 50, 50])
    
    if network_type == 'standard':
        # 标准MLP
        from .base_pinn import MLP
        layers = [input_dim] + hidden_layers + [1]
        return MLP(layers, activation=kwargs.get('activation', 'tanh'))
    
    elif network_type == 'mffm':
        # 多尺度傅里叶特征映射
        return MFFM_PINN(
            input_dim=input_dim,
            fourier_features=kwargs.get('fourier_features', 64),
            hidden_layers=hidden_layers,
            fourier_scale=kwargs.get('fourier_scale', 10.0),
            learnable_freq=kwargs.get('learnable_freq', True),
            use_adaptive_activation=kwargs.get('use_adaptive_activation', False)
        )
    
    elif network_type == 'r_gpinn':
        # 梯度增强残差PINN
        return R_gPINN(
            input_dim=input_dim,
            hidden_dim=hidden_layers[0],
            num_blocks=kwargs.get('num_blocks', 3),
            output_dim=1,
            activation=kwargs.get('activation', 'tanh')
        )
    
    elif network_type == 'adaptive_sin':
        # 自适应正弦激活函数的MLP
        from .base_pinn import MLP
        # 注意：这需要修改MLP以支持自适应激活
        layers = [input_dim] + hidden_layers + [1]
        # 简化版本：使用固定sin激活
        return MLP(layers, activation='tanh')  # 临时，需要完整实现
    
    else:
        raise ValueError(f"Unknown network type: {network_type}")

