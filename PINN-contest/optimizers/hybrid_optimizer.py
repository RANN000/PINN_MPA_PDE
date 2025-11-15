"""
混合优化策略
先使用Adam快速下降，再用L-BFGS精调
"""

import torch
import torch.optim as optim
from typing import Tuple, Optional
from .adaptive_optimizers import create_adam
from .quasi_newton import create_lbfgs


class HybridOptimizer:
    """
    混合优化器：Adam + L-BFGS
    
    策略：
    1. 先用Adam快速下降到局部最优附近（warmup阶段）
    2. 再用L-BFGS精确优化（fine-tuning阶段）
    """
    
    def __init__(self, params, adam_lr=0.001, lbfgs_lr=0.01, 
                 switch_epoch=1000, lbfgs_max_iter=20):
        """
        Args:
            params: 模型参数
            adam_lr: Adam阶段的学习率
            lbfgs_lr: L-BFGS阶段的学习率
            switch_epoch: 切换点（多少epoch后切换到L-BFGS）
            lbfgs_max_iter: L-BFGS每次迭代的最大步数
        """
        self.params = params
        self.adam_lr = adam_lr
        self.lbfgs_lr = lbfgs_lr
        self.switch_epoch = switch_epoch
        self.lbfgs_max_iter = lbfgs_max_iter
        
        # 第一阶段：Adam
        self.optimizer_adam = create_adam(params, learning_rate=adam_lr)
        
        # 第二阶段：L-BFGS（延迟创建）
        self.optimizer_lbfgs = None
        self.current_optimizer = self.optimizer_adam
        self.current_stage = 'adam'
        
    def zero_grad(self):
        """清零梯度"""
        self.current_optimizer.zero_grad()
    
    def step(self, closure=None):
        """
        执行一步优化
        
        Args:
            closure: L-BFGS需要的closure函数
        """
        if self.current_stage == 'adam':
            self.current_optimizer.step()
        else:
            # L-BFGS需要closure
            if closure is None:
                raise ValueError("L-BFGS requires a closure function")
            self.current_optimizer.step(closure)
    
    def switch_to_lbfgs(self):
        """切换到L-BFGS优化器"""
        if self.current_stage == 'adam':
            print(f"\n切换到L-BFGS优化器（第{self.switch_epoch}轮）...")
            self.optimizer_lbfgs = create_lbfgs(
                self.params, 
                learning_rate=self.lbfgs_lr,
                max_iter=self.lbfgs_max_iter
            )
            self.current_optimizer = self.optimizer_lbfgs
            self.current_stage = 'lbfgs'
    
    def should_switch(self, epoch: int) -> bool:
        """判断是否应该切换到L-BFGS"""
        return epoch >= self.switch_epoch and self.current_stage == 'adam'


def create_hybrid_optimizer(params, adam_lr=0.001, lbfgs_lr=0.01, 
                           switch_epoch=1000, lbfgs_max_iter=20):
    """
    便捷函数：创建混合优化器
    
    Args:
        params: 模型参数
        adam_lr: Adam阶段学习率
        lbfgs_lr: L-BFGS阶段学习率
        switch_epoch: 切换轮数
        lbfgs_max_iter: L-BFGS最大迭代数
    """
    return HybridOptimizer(
        params=params,
        adam_lr=adam_lr,
        lbfgs_lr=lbfgs_lr,
        switch_epoch=switch_epoch,
        lbfgs_max_iter=lbfgs_max_iter
    )


# 使用示例
def example_usage():
    """使用示例"""
    import torch.nn as nn
    
    # 创建模型
    model = nn.Sequential(nn.Linear(2, 50), nn.Tanh(), nn.Linear(50, 1))
    
    # 创建混合优化器
    optimizer = create_hybrid_optimizer(
        model.parameters(),
        adam_lr=0.001,
        lbfgs_lr=0.01,
        switch_epoch=1000
    )
    
    # 训练循环
    for epoch in range(2000):
        optimizer.zero_grad()
        # ... 计算损失和反向传播 ...
        # loss.backward()
        
        if optimizer.should_switch(epoch):
            optimizer.switch_to_lbfgs()
        
        # 对于L-BFGS需要closure
        if optimizer.current_stage == 'lbfgs':
            def closure():
                optimizer.zero_grad()
                # 重新计算损失
                # loss = compute_loss(...)
                # loss.backward()
                return loss  # 返回损失值
            
            optimizer.step(closure)
        else:
            optimizer.step()

