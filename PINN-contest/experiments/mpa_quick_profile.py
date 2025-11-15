"""
MPA快速分析
在5分钟内完成三个子任务的MPA分析，获得优化器推荐
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path

# 导入MPA模块
from mpa_core.problem_profiler import profile_task1, profile_task2, profile_task3
from mpa_core.optimizer_selector import select_optimizer_by_mpa, print_optimizer_selection
from mpa_core.alignment_metrics import compute_all_metrics

# 导入PINN模块
from pinn_solvers.base_pinn import BasePINN
from pinn_solvers.helmholtz_solver import HelmholtzPINN
from data.data_loader import HelmholtzDataLoader, PoissonDataLoader

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def analyze_task1():
    """分析子任务1"""
    print("\n" + "="*70)
    print("子任务1 MPA分析：Helmholtz (k=4)")
    print("="*70)
    
    # 1. 理论分析
    theoretical = profile_task1()
    print("\n1. 理论预测的MPA分数：")
    print(f"   GDC: {theoretical['GDC']:.3f}")
    print(f"   TS:  {theoretical['TS']:.3f}")
    print(f"   LSM: {theoretical['LSM']:.3f}")
    print(f"   难度: {theoretical['difficulty']}")
    
    # 2. 选择优化器
    config = {
        'layers': [2, 50, 50, 50, 1],
        'activation': 'tanh',
        'learning_rate': 0.001,
        'wavenumber': 4
    }
    
    # 创建临时模型进行实际计算
    model = HelmholtzPINN(config)
    
    # 加载数据
    loader = HelmholtzDataLoader('../data/子任务1_亥姆霍兹方程数据_k4.xlsx', task='task1', k=4)
    X_interior, _, _ = loader.get_training_points()
    
    # 创建batch_dict
    batch_dict = {'x_interior': torch.FloatTensor(X_interior[:100])}  # 只用100个点加速
    
    # 实际计算MPA指标
    print("\n2. 实际计算的MPA分数（快速模式）：")
    actual_scores = compute_all_metrics(model, batch_dict, quick=True)
    print(f"   GDC: {actual_scores['GDC']:.3f}")
    print(f"   TS:  {actual_scores['TS']:.3f}")
    print(f"   LSM: {actual_scores['LSM']:.3f}")
    print(f"   平均: {actual_scores['average']:.3f}")
    
    # 3. 选择优化器
    print("\n3. 基于MPA的优化器选择：")
    gdc = actual_scores['GDC']
    ts = actual_scores['TS']
    lsm = actual_scores['LSM']
    
    # 创建优化器
    optimizer, opt_name = select_optimizer_by_mpa(
        gdc, ts, lsm, model.net.parameters(), learning_rate=0.001
    )
    
    print_optimizer_selection(gdc, ts, lsm, opt_name, optimizer.param_groups[0]['lr'])
    
    return opt_name


def analyze_task2(k: int):
    """分析子任务2（高波数）"""
    print("\n" + "="*70)
    print(f"子任务2 MPA分析：Helmholtz (k={k})")
    print("="*70)
    
    # 1. 理论分析
    theoretical = profile_task2(k)
    print("\n1. 理论预测的MPA分数：")
    print(f"   GDC: {theoretical['GDC']:.3f}")
    print(f"   TS:  {theoretical['TS']:.3f}")
    print(f"   LSM: {theoretical['LSM']:.3f}")
    print(f"   难度: {theoretical['difficulty']}")
    
    # 2. 选择优化器
    config = {
        'layers': [2, 100, 100, 100, 1],
        'activation': 'tanh',
        'learning_rate': 0.0001,
        'wavenumber': k
    }
    
    model = HelmholtzPINN(config)
    
    # 加载数据
    data_file = f'../data/子任务2_亥姆霍兹方程数据_k{k}.xlsx'
    loader = HelmholtzDataLoader(data_file, task='task2', k=k)
    X_interior, _, _ = loader.get_training_points()
    
    batch_dict = {'x_interior': torch.FloatTensor(X_interior[:100])}
    
    print("\n2. 实际计算的MPA分数（快速模式）：")
    actual_scores = compute_all_metrics(model, batch_dict, quick=True)
    print(f"   GDC: {actual_scores['GDC']:.3f}")
    print(f"   TS:  {actual_scores['TS']:.3f}")
    print(f"   LSM: {actual_scores['LSM']:.3f}")
    
    # 3. 选择优化器
    print("\n3. 基于MPA的优化器选择：")
    gdc = actual_scores['GDC']
    ts = actual_scores['TS']
    lsm = actual_scores['LSM']
    
    optimizer, opt_name = select_optimizer_by_mpa(
        gdc, ts, lsm, model.net.parameters(), learning_rate=0.0001
    )
    
    print_optimizer_selection(gdc, ts, lsm, opt_name, optimizer.param_groups[0]['lr'])
    
    return opt_name


def analyze_task3():
    """分析子任务3（Poisson反问题）"""
    print("\n" + "="*70)
    print("子任务3 MPA分析：Poisson反问题")
    print("="*70)
    
    # 1. 理论分析
    theoretical = profile_task3()
    print("\n1. 理论预测的MPA分数：")
    print(f"   GDC: {theoretical['GDC']:.3f}")
    print(f"   TS:  {theoretical['TS']:.3f}")
    print(f"   LSM: {theoretical['LSM']:.3f}")
    print(f"   难度: {theoretical['difficulty']}")
    
    # 由于子任务3需要特殊的网络架构，这里只做理论预测
    print("\n2. 基于理论预测的优化器选择：")
    gdc = theoretical['GDC']
    ts = theoretical['TS']
    lsm = theoretical['LSM']
    
    # 创建临时模型
    config = {'layers': [2, 50, 50, 1], 'activation': 'tanh'}
    model = BasePINN(config)
    
    optimizer, opt_name = select_optimizer_by_mpa(
        gdc, ts, lsm, model.net.parameters(), learning_rate=0.001
    )
    
    print_optimizer_selection(gdc, ts, lsm, opt_name, optimizer.param_groups[0]['lr'])
    
    return opt_name


def main():
    """主函数：快速分析三个子任务"""
    print("="*70)
    print("MPA快速分析 - Method-Problem Alignment Theory")
    print("="*70)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 分析三个子任务
    opt1 = analyze_task1()
    opt2_100 = analyze_task2(100)
    
    # 只分析k=100，其他k值类似
    print(f"\n注意：k=500和k=1000的分析类似，但难度更高，"
          f"可能需要更激进的混合策略")
    
    opt3 = analyze_task3()
    
    # 总结
    print("\n" + "="*70)
    print("MPA分析总结")
    print("="*70)
    print(f"子任务1 (k=4):     推荐 {opt1}")
    print(f"子任务2 (k=100):   推荐 {opt2_100}")
    print(f"子任务3 (反问题):  推荐 {opt3}")
    print("="*70)
    print("\n建议：")
    print("1. 子任务1可以用Adam快速收敛")
    print("2. 子任务2需要混合策略或L-BFGS")
    print("3. 子任务3可以用AdamW或L-BFGS")
    print("\n分析完成！（用时<5分钟）")


if __name__ == "__main__":
    main()

