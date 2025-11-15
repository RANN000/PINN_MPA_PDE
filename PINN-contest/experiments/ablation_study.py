"""
消融实验（Ablation Study）
验证MPA理论的有效性和各创新点的贡献
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import json

from data.data_loader import HelmholtzDataLoader
from pinn_solvers.helmholtz_solver import HelmholtzPINN
from mpa_core.optimizer_selector import select_optimizer_by_mpa
from mpa_core.alignment_metrics import compute_all_metrics
from optimizers import create_adam, create_adamw, create_lbfgs, create_hybrid_optimizer


class AblationExperiment:
    """消融实验管理器"""
    
    def __init__(self, task: str = 'task1', save_dir: str = './results/ablation'):
        self.task = task
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def run_experiment(self, config: Dict, experiment_name: str):
        """运行单个实验"""
        print(f"\n{'='*60}")
        print(f"实验: {experiment_name}")
        print(f"{'='*60}")
        
        # 加载数据
        if self.task == 'task1':
            loader = HelmholtzDataLoader('data/子任务1_亥姆霍兹方程数据_k4.xlsx', task='task1', k=4)
        else:
            raise ValueError(f"Task {self.task} not implemented")
        
        X_train, _, boundary_mask = loader.get_training_points()
        X_interior = X_train[~boundary_mask]
        X_boundary = X_train[boundary_mask]
        X_test, _ = loader.get_test_points()
        u_test_true = np.sin(np.pi * X_test[:, 0]) * np.sin(3 * np.pi * X_test[:, 1])
        
        # 创建模型
        model = HelmholtzPINN(config)
        
        # 创建优化器
        optimizer_config = config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        lr = optimizer_config.get('lr', 0.001)
        
        if optimizer_type == 'adam':
            optimizer = create_adam(model.net.parameters(), learning_rate=lr)
        elif optimizer_type == 'adamw':
            optimizer = create_adamw(model.net.parameters(), learning_rate=lr)
        elif optimizer_type == 'lbfgs':
            optimizer = create_lbfgs(model.net.parameters(), learning_rate=lr)
        elif optimizer_type == 'mpa_recommended':
            # 使用MPA推荐的优化器
            batch_dict = {'x_interior': torch.FloatTensor(X_interior[:100])}
            mpa_scores = compute_all_metrics(model, batch_dict, quick=True)
            optimizer, opt_name = select_optimizer_by_mpa(
                mpa_scores['GDC'], mpa_scores['TS'], mpa_scores['LSM'],
                model.net.parameters(), learning_rate=lr
            )
            print(f"MPA推荐优化器: {opt_name}")
        else:
            optimizer = create_adam(model.net.parameters(), learning_rate=lr)
        
        # 训练（简化版，只训练少量epoch用于对比）
        n_epochs = config.get('n_epochs', 500)
        batch_size = config.get('batch_size', 256)
        
        from pinn_solvers.base_pinn import create_dataloader
        u_boundary = np.zeros(len(X_boundary))
        dataloader = create_dataloader(X_interior, X_boundary, u_boundary, batch_size=batch_size)
        
        train_losses = []
        test_errors = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch_dict in dataloader:
                for key in batch_dict:
                    if torch.is_tensor(batch_dict[key]):
                        batch_dict[key] = batch_dict[key].to(model.device)
                
                loss = model.train_step(batch_dict, optimizer)
                epoch_loss += loss
            
            train_losses.append(epoch_loss / len(dataloader))
            
            # 每100个epoch评估一次
            if (epoch + 1) % 100 == 0 or epoch == 0:
                X_test_tensor = torch.FloatTensor(X_test).to(model.device)
                with torch.no_grad():
                    u_test_pred = model.forward(X_test_tensor).cpu().numpy().flatten()
                
                test_error = np.sqrt(np.mean((u_test_pred - u_test_true) ** 2))
                relative_error = test_error / (np.sqrt(np.mean(u_test_true ** 2)) + 1e-10)
                test_errors.append(relative_error)
        
        # 最终评估
        X_test_tensor = torch.FloatTensor(X_test).to(model.device)
        with torch.no_grad():
            u_test_pred = model.forward(X_test_tensor).cpu().numpy().flatten()
        
        final_relative_error = np.sqrt(np.mean((u_test_pred - u_test_true) ** 2)) / \
                              (np.sqrt(np.mean(u_test_true ** 2)) + 1e-10)
        
        # 记录结果
        result = {
            'experiment': experiment_name,
            'config': config,
            'final_relative_error': float(final_relative_error),
            'final_loss': float(train_losses[-1]),
            'train_losses': train_losses,
            'test_errors': test_errors
        }
        
        self.results.append(result)
        
        print(f"最终相对误差: {final_relative_error:.6e}")
        print(f"最终损失: {train_losses[-1]:.6e}")
        
        return result
    
    def save_results(self):
        """保存结果"""
        # 保存为JSON
        results_file = self.save_dir / 'ablation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 保存为CSV（表格形式）
        df_data = []
        for r in self.results:
            df_data.append({
                'Experiment': r['experiment'],
                'Final_Relative_Error': r['final_relative_error'],
                'Final_Loss': r['final_loss'],
                'Optimizer': r['config'].get('optimizer', {}).get('type', 'adam'),
                'Learning_Rate': r['config'].get('optimizer', {}).get('lr', 0.001),
                'Layers': str(r['config'].get('layers', []))
            })
        
        df = pd.DataFrame(df_data)
        csv_file = self.save_dir / 'ablation_results.csv'
        df.to_csv(csv_file, index=False)
        
        print(f"\n结果已保存到:")
        print(f"  - {results_file}")
        print(f"  - {csv_file}")
        
        return df
    
    def plot_comparison(self):
        """绘制对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 相对误差对比（柱状图）
        ax1 = axes[0, 0]
        experiments = [r['experiment'] for r in self.results]
        errors = [r['final_relative_error'] for r in self.results]
        bars = ax1.bar(range(len(experiments)), errors)
        ax1.set_xticks(range(len(experiments)))
        ax1.set_xticklabels(experiments, rotation=45, ha='right')
        ax1.set_ylabel('相对L2误差')
        ax1.set_title('最终相对误差对比')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, error) in enumerate(zip(bars, errors)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{error:.2e}', ha='center', va='bottom', fontsize=8)
        
        # 2. 训练损失曲线
        ax2 = axes[0, 1]
        for r in self.results:
            ax2.plot(r['train_losses'], label=r['experiment'], alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Train Loss')
        ax2.set_title('训练损失对比')
        ax2.set_yscale('log')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. 测试误差曲线
        ax3 = axes[1, 0]
        for r in self.results:
            epochs = list(range(0, len(r['train_losses']), 100)) + [len(r['train_losses'])-1]
            test_errors = r['test_errors']
            ax3.plot(epochs[:len(test_errors)], test_errors, 
                    marker='o', label=r['experiment'], markersize=3, alpha=0.7)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Test Relative Error')
        ax3.set_title('测试误差变化')
        ax3.set_yscale('log')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. 性能提升百分比
        ax4 = axes[1, 1]
        if len(self.results) > 1:
            baseline_error = self.results[0]['final_relative_error']
            improvements = []
            labels = []
            for r in self.results[1:]:
                improvement = (baseline_error - r['final_relative_error']) / baseline_error * 100
                improvements.append(improvement)
                labels.append(r['experiment'])
            
            colors = ['green' if x > 0 else 'red' for x in improvements]
            bars = ax4.barh(labels, improvements, color=colors)
            ax4.set_xlabel('性能提升 (%)')
            ax4.set_title(f'相对Baseline的性能提升')
            ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'ablation_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  图表已保存到: {self.save_dir / 'ablation_comparison.png'}")
        plt.close()


def run_task1_ablation():
    """运行子任务1的消融实验"""
    print("="*70)
    print("消融实验：子任务1 (Helmholtz k=4)")
    print("="*70)
    
    exp = AblationExperiment(task='task1', save_dir='./results/ablation/task1')
    
    base_config = {
        'layers': [2, 50, 50, 50, 1],
        'activation': 'tanh',
        'learning_rate': 0.001,
        'wavenumber': 4,
        'lambda_pde': 1.0,
        'lambda_bc': 1.0,
        'n_epochs': 500,  # 消融实验用较少的epoch
        'batch_size': 256
    }
    
    # 实验1: Baseline (Adam, lr=0.001)
    print("\n实验1: Baseline")
    config1 = base_config.copy()
    config1['optimizer'] = {'type': 'adam', 'lr': 0.001}
    exp.run_experiment(config1, 'Baseline (Adam, lr=0.001)')
    
    # 实验2: MPA推荐的优化器
    print("\n实验2: MPA推荐优化器")
    config2 = base_config.copy()
    config2['optimizer'] = {'type': 'mpa_recommended', 'lr': 0.001}
    exp.run_experiment(config2, 'MPA推荐优化器')
    
    # 实验3: 低学习率Adam
    print("\n实验3: 低学习率")
    config3 = base_config.copy()
    config3['optimizer'] = {'type': 'adam', 'lr': 0.0002}
    exp.run_experiment(config3, 'Adam (lr=0.0002)')
    
    # 实验4: AdamW
    print("\n实验4: AdamW")
    config4 = base_config.copy()
    config4['optimizer'] = {'type': 'adamw', 'lr': 0.001}
    exp.run_experiment(config4, 'AdamW (lr=0.001)')
    
    # 实验5: 更深网络
    print("\n实验5: 更深网络")
    config5 = base_config.copy()
    config5['layers'] = [2, 100, 100, 100, 100, 1]
    config5['optimizer'] = {'type': 'adam', 'lr': 0.001}
    exp.run_experiment(config5, '更深网络 (4层100神经元)')
    
    # 保存结果并绘图
    df = exp.save_results()
    exp.plot_comparison()
    
    print("\n" + "="*70)
    print("消融实验完成！")
    print("="*70)
    print("\n结果摘要:")
    print(df.to_string(index=False))
    
    return df


def run_mpa_effectiveness_study():
    """验证MPA有效性的专项实验"""
    print("\n" + "="*70)
    print("MPA有效性验证实验")
    print("="*70)
    
    # 加载数据
    loader = HelmholtzDataLoader('data/子任务1_亥姆霍兹方程数据_k4.xlsx', task='task1', k=4)
    X_train, _, boundary_mask = loader.get_training_points()
    X_interior = X_train[~boundary_mask]
    X_boundary = X_train[boundary_mask]
    
    # 创建模型
    config = {
        'layers': [2, 50, 50, 50, 1],
        'activation': 'tanh',
        'learning_rate': 0.001,
        'wavenumber': 4
    }
    
    model = HelmholtzPINN(config)
    batch_dict = {'x_interior': torch.FloatTensor(X_interior[:100])}
    
    # 计算MPA分数
    print("\n计算MPA对齐指标...")
    mpa_scores = compute_all_metrics(model, batch_dict, quick=True)
    
    print(f"\nMPA分析结果:")
    print(f"  GDC (梯度友好度): {mpa_scores['GDC']:.3f}")
    print(f"  TS (轨迹平滑度):  {mpa_scores['TS']:.3f}")
    print(f"  LSM (局部平滑度): {mpa_scores['LSM']:.3f}")
    print(f"  平均分数: {mpa_scores['average']:.3f}")
    
    # 选择优化器
    optimizer, opt_name = select_optimizer_by_mpa(
        mpa_scores['GDC'], mpa_scores['TS'], mpa_scores['LSM'],
        model.net.parameters(), learning_rate=0.001
    )
    
    print(f"\nMPA推荐: {opt_name}")
    print(f"调整后学习率: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 分析推荐理由
    print("\n推荐理由:")
    gdc, ts, lsm = mpa_scores['GDC'], mpa_scores['TS'], mpa_scores['LSM']
    
    if gdc < 0.4:
        print(f"  - GDC低 ({gdc:.3f}): 梯度不友好，建议使用混合策略或L-BFGS")
    elif gdc > 0.7:
        print(f"  - GDC高 ({gdc:.3f}): 梯度友好，适合Adam等一阶方法")
    
    if ts < 0.5:
        print(f"  - TS低 ({ts:.3f}): 轨迹不平滑，建议降低学习率或使用RAdam")
    elif ts > 0.7:
        print(f"  - TS高 ({ts:.3f}): 轨迹平滑，自适应学习率有效")
    
    if lsm < 0.5:
        print(f"  - LSM低 ({lsm:.3f}): 局部不平滑，可能存在多模态")
    elif lsm > 0.7:
        print(f"  - LSM高 ({lsm:.3f}): 局部平滑，局部优化方法有效")
    
    return mpa_scores, opt_name


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 运行MPA有效性验证
    mpa_scores, opt_name = run_mpa_effectiveness_study()
    
    # 运行消融实验（可选，需要较长时间）
    print("\n是否运行完整消融实验？这将需要一些时间...")
    print("（可以稍后单独运行: python experiments/ablation_study.py --full）")
    
    import sys
    if '--full' in sys.argv:
        run_task1_ablation()

