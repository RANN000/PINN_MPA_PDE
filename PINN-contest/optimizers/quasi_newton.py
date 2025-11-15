"""
拟牛顿优化器（L-BFGS系列）
"""

import torch
import torch.optim as optim
from typing import Optional, Callable


def create_lbfgs(params, learning_rate=0.01, max_iter=20, 
                 history_size=100, line_search_fn=None):
    """
    创建L-BFGS优化器
    
    L-BFGS适合：
    - 低维到中维问题
    - 需要精确最小化
    - 收敛速度快（但每次迭代计算量大）
    
    Args:
        params: 模型参数
        learning_rate: 学习率（L-BFGS通常需要较小的学习率）
        max_iter: 每次优化的最大迭代数
        history_size: 保存的历史梯度数量
        line_search_fn: 线搜索函数 ('strong_wolfe' 或 None)
    """
    return optim.LBFGS(
        params,
        lr=learning_rate,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn=line_search_fn
    )


class LBFGSTrainer:
    """
    L-BFGS训练器封装
    
    L-BFGS需要特殊的训练循环，因为closure函数的需求
    """
    
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def train_step(self, batch_dict, closure_fn: Callable):
        """
        单步训练（L-BFGS特殊处理）
        
        Args:
            batch_dict: 批次数据
            closure_fn: closure函数，返回损失值
        """
        # 确保数据在正确设备上
        for key in batch_dict:
            if torch.is_tensor(batch_dict[key]):
                batch_dict[key] = batch_dict[key].to(self.device)
        
        def closure():
            self.optimizer.zero_grad()
            loss = closure_fn(batch_dict)
            loss.backward()
            return loss
        
        loss = self.optimizer.step(closure)
        return loss.item() if isinstance(loss, torch.Tensor) else loss
    
    @staticmethod
    def create_closure_fn(model, loss_fn):
        """
        创建closure函数
        
        Args:
            model: PINN模型
            loss_fn: 损失函数
        """
        def closure(batch_dict):
            return loss_fn(batch_dict)
        return closure

