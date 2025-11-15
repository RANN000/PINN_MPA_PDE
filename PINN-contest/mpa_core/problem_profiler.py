"""
问题特征分析器
为不同子任务提供预定义的特征分析
"""

import numpy as np
from typing import Dict


class ProblemProfiler:
    """问题特征分析器"""
    
    def __init__(self):
        pass
    
    def profile_helmholtz(self, k: float) -> Dict[str, float]:
        """
        分析Helmholtz问题的特征
        
        Args:
            k: 波数
        
        Returns:
            预测的MPA分数
        """
        # 波数越高，问题越困难
        if k <= 10:
            # 低波数：相对简单
            gdc = 0.75
            ts = 0.70
            lsm = 0.72
        elif k <= 100:
            # 中等波数
            gdc = 0.55
            ts = 0.60
            lsm = 0.58
        else:
            # 高波数：非常困难
            gdc = 0.35
            ts = 0.45
            lsm = 0.40
        
        return {
            'GDC': gdc,
            'TS': ts,
            'LSM': lsm,
            'difficulty': 'low' if k <= 10 else ('medium' if k <= 100 else 'high')
        }
    
    def profile_poisson_inverse(self) -> Dict[str, float]:
        """
        分析Poisson反问题的特征
        
        反问题通常：
        - 梯度可能不够友好（Hessian条件数高）
        - 优化轨迹可能震荡（数据拟合项与PDE项的权衡）
        - 局部平滑度中等（可能存在多解）
        """
        return {
            'GDC': 0.50,
            'TS': 0.55,
            'LSM': 0.52,
            'difficulty': 'medium'
        }


def profile_task1() -> Dict[str, float]:
    """子任务1特征"""
    profiler = ProblemProfiler()
    return profiler.profile_helmholtz(k=4)


def profile_task2(k: int) -> Dict[str, float]:
    """子任务2特征"""
    profiler = ProblemProfiler()
    return profiler.profile_helmholtz(k=k)


def profile_task3() -> Dict[str, float]:
    """子任务3特征"""
    profiler = ProblemProfiler()
    return profiler.profile_poisson_inverse()

