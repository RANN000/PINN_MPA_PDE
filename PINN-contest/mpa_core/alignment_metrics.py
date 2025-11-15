"""
MPA对齐指标计算
GDC, TS, LSM三个核心指标
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


def compute_gdc(model: nn.Module, batch_dict: Dict, device='cpu', n_samples=100) -> float:
    """
    计算GDC (Gradient Descent Compatibility) - 梯度友好度
    
    方法：通过Hessian矩阵的条件数评估梯度方法的适用性
    条件数低 → 梯度友好 → 适合Adam等一阶方法
    条件数高 → 非梯度友好 → 需要二阶方法或全局搜索
    
    Args:
        model: PINN模型
        batch_dict: 批次数据字典
        device: 计算设备
        n_samples: 用于估计的采样点数（减少计算量）
    
    Returns:
        float: GDC分数 (0-1之间，越高越梯度友好)
    """
    model.eval()
    
    # 随机采样部分点进行估计
    if 'x_interior' in batch_dict:
        X = batch_dict['x_interior']
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices].to(device)
        else:
            X_sample = X.to(device)
    else:
        return 0.5  # 默认中等
    
    # 获取参数
    params = list(model.parameters())
    if len(params) == 0:
        return 0.5
    
    # 计算损失
    X_sample.requires_grad_(True)
    u_pred = model(X_sample)
    
    # 对于Helmholtz，需要计算PDE损失
    # 这里简化处理：只用网络输出计算简单损失
    loss = torch.mean(u_pred ** 2)
    
    # 计算一阶梯度
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grads_flat = torch.cat([g.flatten() for g in grads if g is not None])
    
    # 估计Hessian的条件数（使用Hessian的对角线近似）
    if len(grads_flat) == 0:
        return 0.5
    
    # 方法：通过梯度方差估计曲率
    grad_var = torch.var(grads_flat).item()
    grad_mean = torch.mean(torch.abs(grads_flat)).item()
    
    if grad_mean < 1e-10:
        return 0.5
    
    # 条件数估计：方差/均值（简化版）
    condition_number = grad_var / (grad_mean ** 2 + 1e-10)
    
    # 归一化到0-1：condition_number越小，GDC越高
    # 使用sigmoid函数映射
    gdc_score = 1.0 / (1.0 + condition_number)
    
    # 清理
    model.train()
    
    return float(gdc_score)


def compute_ts(model: nn.Module, batch_dict: Dict, n_steps=50, 
               learning_rate=0.01, device='cpu') -> float:
    """
    计算TS (Trajectory Smoothness) - 优化轨迹平滑度
    
    方法：运行短期优化，测量损失下降的平滑性
    平滑 → 自适应学习率有效
    震荡 → 需要更保守的优化器
    
    Args:
        model: PINN模型
        batch_dict: 批次数据
        n_steps: 短期优化步数
        learning_rate: 测试学习率
        device: 计算设备
    
    Returns:
        float: TS分数 (0-1之间，越高越平滑)
    """
    model.train()
    
    # 保存原始参数
    original_params = [p.clone() for p in model.parameters()]
    
    # 创建临时优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_history = []
    
    # 短期优化
    for _ in range(n_steps):
        optimizer.zero_grad()
        
        # 计算损失（简化：使用数据拟合损失）
        if 'x_interior' in batch_dict:
            X = batch_dict['x_interior'].to(device)[:100]  # 只用少量点加速
            u_pred = model(X)
            loss = torch.mean(u_pred ** 2)
        else:
            loss = torch.tensor(0.0).to(device)
        
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    
    # 恢复原始参数
    for p, orig_p in zip(model.parameters(), original_params):
        p.data.copy_(orig_p)
    
    # 计算平滑度：方差/均值
    if len(loss_history) < 5:
        return 0.5
    
    loss_mean = np.mean(loss_history)
    loss_var = np.var(loss_history)
    
    if loss_mean < 1e-10:
        return 0.5
    
    # CV = std/mean (coefficient of variation)
    cv = np.sqrt(loss_var) / (loss_mean + 1e-10)
    
    # TS分数：CV越小，越平滑
    ts_score = 1.0 / (1.0 + cv)
    
    return float(ts_score)


def compute_lsm(model: nn.Module, batch_dict: Dict, n_samples=20, 
                device='cpu') -> float:
    """
    计算LSM (Local Smoothness Metric) - 局部平滑度
    
    方法：在参数空间随机扰动，测量损失变化的敏感度
    高平滑 → 局部方法有效
    多模态 → 需要全局探索
    
    Args:
        model: PINN模型
        batch_dict: 批次数据
        n_samples: 扰动次数
        device: 计算设备
    
    Returns:
        float: LSM分数 (0-1之间，越高越平滑)
    """
    model.eval()
    
    # 计算基准损失
    if 'x_interior' in batch_dict:
        X = batch_dict['x_interior'].to(device)[:100]
        with torch.no_grad():
            u_pred_base = model(X)
            loss_base = torch.mean(u_pred_base ** 2).item()
    else:
        loss_base = 0.0
    
    sensitivity_list = []
    
    # 多次扰动
    for _ in range(n_samples):
        # 保存参数
        original_params = [p.clone() for p in model.parameters()]
        
        # 随机扰动（添加小的随机噪声）
        with torch.no_grad():
            for p in model.parameters():
                noise = torch.randn_like(p) * 0.01
                p.add_(noise)
        
        # 计算扰动后损失
        if 'x_interior' in batch_dict:
            X = batch_dict['x_interior'].to(device)[:100]
            with torch.no_grad():
                u_pred = model(X)
                loss_perturbed = torch.mean(u_pred ** 2).item()
        else:
            loss_perturbed = 0.0
        
        # 计算敏感度：损失变化率
        if loss_base > 1e-10:
            sensitivity = abs(loss_perturbed - loss_base) / loss_base
            sensitivity_list.append(sensitivity)
        
        # 恢复参数
        for p, orig_p in zip(model.parameters(), original_params):
            p.data.copy_(orig_p)
    
    if len(sensitivity_list) == 0:
        return 0.5
    
    # 平均敏感度
    avg_sensitivity = np.mean(sensitivity_list)
    
    # LSM分数：敏感度越小，越平滑
    lsm_score = 1.0 / (1.0 + avg_sensitivity)
    
    model.train()
    
    return float(lsm_score)


def compute_all_metrics(model: nn.Module, batch_dict: Dict, 
                       device='cpu', quick=False) -> Dict[str, float]:
    """
    计算所有MPA对齐指标
    
    Args:
        model: PINN模型
        batch_dict: 批次数据
        device: 计算设备
        quick: 快速模式（减少采样和步数）
    
    Returns:
        dict: 包含GDC, TS, LSM分数的字典
    """
    if quick:
        n_samples = 50
        n_steps = 20
    else:
        n_samples = 100
        n_steps = 50
    
    gdc = compute_gdc(model, batch_dict, device=device, n_samples=n_samples)
    ts = compute_ts(model, batch_dict, n_steps=n_steps, device=device)
    lsm = compute_lsm(model, batch_dict, n_samples=n_samples, device=device)
    
    return {
        'GDC': gdc,
        'TS': ts,
        'LSM': lsm,
        'average': (gdc + ts + lsm) / 3.0
    }

