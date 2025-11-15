"""
优化器选择器
基于MPA对齐分数选择最优优化器
"""

import torch
import torch.optim as optim
from typing import Dict, Tuple
import numpy as np


class OptimizerSelector:
    """基于MPA的优化器选择器"""
    
    def __init__(self):
        self.decision_rules = self._initialize_rules()
    
    def _initialize_rules(self):
        """初始化决策规则"""
        return {
            'adam': lambda gdc, ts, lsm: gdc > 0.7 and lsm > 0.6,
            'adamw': lambda gdc, ts, lsm: gdc > 0.6 and ts < 0.6,
            'radam': lambda gdc, ts, lsm: ts < 0.5,
            'lbfgs': lambda gdc, ts, lsm: gdc < 0.4,
            'hybrid': lambda gdc, ts, lsm: gdc < 0.5 and lsm < 0.5
        }
    
    def select_optimizer(self, gdc: float, ts: float, lsm: float, 
                        model_params, learning_rate=0.001) -> Tuple[optim.Optimizer, str]:
        """
        根据MPA分数选择优化器
        
        Args:
            gdc: GDC分数
            ts: TS分数
            lsm: LSM分数
            model_params: 模型参数
            learning_rate: 学习率
        
        Returns:
            (优化器, 优化器名称)的元组
        """
        # 检查每个规则
        for opt_name, rule in self.decision_rules.items():
            if rule(gdc, ts, lsm):
                return self._create_optimizer(opt_name, model_params, learning_rate, gdc, ts, lsm)
        
        # 默认使用Adam
        return self._create_optimizer('adam', model_params, learning_rate, gdc, ts, lsm)
    
    def _create_optimizer(self, opt_name: str, model_params, 
                         learning_rate: float, gdc: float, ts: float, lsm: float):
        """
        创建优化器
        
        Args:
            opt_name: 优化器名称
            model_params: 模型参数
            learning_rate: 基础学习率
            gdc, ts, lsm: MPA分数（用于调整学习率）
        """

        # 不更改学习率
        adjusted_lr = learning_rate
        # # 根据分数调整学习率
        # adjusted_lr = self._adjust_learning_rate(learning_rate, gdc, ts, lsm, opt_name)

        if opt_name == 'adam':
            return optim.Adam(model_params, lr=adjusted_lr), 'Adam'
        
        elif opt_name == 'adamw':
            weight_decay = 0.01  # 添加权重衰减
            return optim.AdamW(model_params, lr=adjusted_lr, weight_decay=weight_decay), 'AdamW'
        
        elif opt_name == 'radam':
            # RAdam: 更稳定的Adam变体
            return optim.Adam(model_params, lr=adjusted_lr
                                               # * 0.5
                              ), 'RAdam(Adam+lowLR)'
        
        elif opt_name == 'lbfgs':
            # L-BFGS对于病态问题更有效
            return optim.LBFGS(model_params, lr=adjusted_lr, max_iter=20), 'L-BFGS'
        
        elif opt_name == 'hybrid':
            # 混合策略：先用Adam，后续可用L-BFGS精调
            return optim.Adam(model_params, lr=adjusted_lr), 'Hybrid(Adam)'
        
        else:
            return optim.Adam(model_params, lr=adjusted_lr), 'Adam'
    
    def _adjust_learning_rate(self, base_lr: float, gdc: float, ts: float, 
                             lsm: float, opt_name: str) -> float:
        """
        根据MPA分数调整学习率
        
        策略：
        - 高GDC: 可以用较大学习率
        - 低TS: 降低学习率以稳定训练
        - 低LSM: 降低学习率避免跳出
        """
        adjustment = 1.0
        
        # 根据GDC调整
        if gdc > 0.7:
            adjustment *= 1.2
        elif gdc < 0.4:
            adjustment *= 0.5
        
        # 根据TS调整
        if ts < 0.5:
            adjustment *= 0.8
        
        # 根据LSM调整
        if lsm < 0.5:
            adjustment *= 0.8
        
        # L-BFGS通常需要更小的学习率
        if opt_name == 'lbfgs':
            adjustment *= 0.1
        
        return base_lr * adjustment


def select_optimizer_by_mpa(gdc: float, ts: float, lsm: float,
                           model_params, learning_rate=0.001) -> Tuple[optim.Optimizer, str]:
    """
    便捷函数：根据MPA分数选择优化器
    
    Args:
        gdc: GDC分数
        ts: TS分数
        lsm: LSM分数
        model_params: 模型参数
        learning_rate: 学习率
    
    Returns:
        (优化器, 优化器名称)
    """
    selector = OptimizerSelector()
    return selector.select_optimizer(gdc, ts, lsm, model_params, learning_rate)


def print_optimizer_selection(gdc: float, ts: float, lsm: float, 
                              opt_name: str, adjusted_lr: float):
    """打印优化器选择信息"""
    print("\n" + "="*60)
    print("MPA优化器选择")
    print("="*60)
    print(f"GDC (梯度友好度): {gdc:.3f}")
    print(f"TS (轨迹平滑度):  {ts:.3f}")
    print(f"LSM (局部平滑度): {lsm:.3f}")
    print(f"推荐优化器: {opt_name}")
    print(f"调整后学习率: {adjusted_lr:.6f}")
    print("="*60)

