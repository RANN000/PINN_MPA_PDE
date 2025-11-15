"""
评估指标计算
"""

import numpy as np


def compute_errors(u_pred, u_true):
    """
    计算各种误差指标
    
    Args:
        u_pred: 预测值
        u_true: 真实值
        
    Returns:
        dict: 包含各种误差的字典
    """
    # 确保是numpy数组
    u_pred = np.array(u_pred).flatten()
    u_true = np.array(u_true).flatten()
    
    # L2误差
    l2_error = np.sqrt(np.mean((u_pred - u_true) ** 2))
    
    # 相对L2误差
    norm_true = np.sqrt(np.mean(u_true ** 2))
    relative_l2_error = l2_error / (norm_true + 1e-10)
    
    # 最大误差
    max_error = np.max(np.abs(u_pred - u_true))
    
    # 相对最大误差
    max_abs_true = np.max(np.abs(u_true))
    relative_max_error = max_error / (max_abs_true + 1e-10)
    
    # 平均绝对误差
    mae = np.mean(np.abs(u_pred - u_true))
    
    # 相对平均绝对误差
    mean_abs_true = np.mean(np.abs(u_true))
    relative_mae = mae / (mean_abs_true + 1e-10)
    
    return {
        'l2_error': l2_error,
        'relative_l2_error': relative_l2_error,
        'max_error': max_error,
        'relative_max_error': relative_max_error,
        'mae': mae,
        'relative_mae': relative_mae
    }


def compute_relative_error(u_pred, u_true):
    """计算相对误差（快捷函数）"""
    errors = compute_errors(u_pred, u_true)
    return errors['relative_l2_error']


def print_error_summary(errors):
    """打印误差摘要"""
    print("\n评估结果:")
    print(f"   L2误差: {errors['l2_error']:.6e}")
    print(f"   相对L2误差: {errors['relative_l2_error']:.6e}")
    print(f"   最大误差: {errors['max_error']:.6e}")
    print(f"   相对最大误差: {errors['relative_max_error']:.6e}")
    print(f"   平均绝对误差: {errors['mae']:.6e}")
    print(f"   相对平均绝对误差: {errors['relative_mae']:.6e}")

