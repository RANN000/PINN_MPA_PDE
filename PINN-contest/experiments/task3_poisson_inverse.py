"""
子任务3：Poisson反问题 - 参数识别
给定200个观测点，识别参数λ
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   # 使用黑体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

from pathlib import Path

from data.data_loader import PoissonDataLoader
from pinn_solvers.poisson_inverse_solver import PoissonInversePINN, train_poisson_inverse_pinn
from utils.metrics import compute_errors, print_error_summary

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def main():
    print("="*70)
    print("子任务3：Poisson方程参数识别反问题")
    print("="*70)
    
    # 1. 加载观测数据
    print("\n1. 加载数据...")
    loader = PoissonDataLoader('../data/子任务3数据.xlsx')
    obs_data = loader.load_data()
    
    X_data = obs_data['X']  # 200个观测点的坐标
    u_data = obs_data['u_obs']  # 200个观测点的解值
    
    print(f"   观测点数: {len(X_data)}")
    print(f"   u的数值范围: [{u_data.min():.4f}, {u_data.max():.4f}]")
    
    # 2. 生成PDE残差计算的配点
    print("\n2. 生成配点...")
    X_interior = loader.generate_collocation_points(n_interior=2000, n_boundary=400)
    
    # 边界点（边界条件u=0）
    n_bc = 400
    X_boundary = []
    n_bc_per_edge = n_bc // 4
    
    # 四条边采样
    X_boundary.append(np.column_stack([np.random.uniform(-1, 1, n_bc_per_edge), -1 + np.zeros(n_bc_per_edge)]))
    X_boundary.append(np.column_stack([np.random.uniform(-1, 1, n_bc_per_edge), 1 + np.zeros(n_bc_per_edge)]))
    X_boundary.append(np.column_stack([-1 + np.zeros(n_bc_per_edge), np.random.uniform(-1, 1, n_bc_per_edge)]))
    X_boundary.append(np.column_stack([1 + np.zeros(n_bc_per_edge), np.random.uniform(-1, 1, n_bc_per_edge)]))
    X_boundary = np.vstack(X_boundary)
    
    print(f"   内部配点: {len(X_interior)}")
    print(f"   边界点: {len(X_boundary)}")
    
    # 3. 创建模型
    print("\n3. 创建模型...")
    config = {
        'network_type': 'r_gpinn',
        'layers_u': [2, 50, 50, 50, 1],
        'layers_lambda': [2, 50, 50, 1],
        'activation': 'tanh',
        'learning_rate': 0.001,
        'weight_decay': 0.0,
        'lambda_type': 'spatial',  # 标量参数（不变） / 空间参数（随空间变化）
        'lambda_pde': 0.1,
        'lambda_data': 50.0,  # 数据拟合项权重更大
        'lambda_bc': 10,
        'reg_lambda': 0.001
    }
    
    model = PoissonInversePINN(config).to(device)
    n_params_u = sum(p.numel() for p in model.net.parameters())
    print(f"   u网络参数: {n_params_u}")
    if model.lambda_param is not None:
        print(f"   λ参数: 标量")
    
    # 4. 训练模型
    print("\n4. 训练模型...")
    results_dir = Path('../results/task3_poisson_inverse')
    
    model = train_poisson_inverse_pinn(
        model=model,
        X_interior=X_interior,
        X_boundary=X_boundary,
        X_data=X_data,
        u_data=u_data,
        n_epochs=2000,
        batch_size=128,
        log_every=100,
        save_dir=str(results_dir)
    )
    
    # 5. 评估结果
    print("\n5. 评估模型...")
    
    # 在观测点上评估
    X_data_tensor = torch.FloatTensor(X_data).to(model.device)
    with torch.no_grad():
        u_pred = model.forward(X_data_tensor).cpu().numpy().flatten()
    
    # 计算误差
    eval_result = compute_errors(u_pred, u_data)
    print_error_summary(eval_result)
    
    # 获取识别的参数
    lambda_identified = model.get_lambda_value()
    print(f"\n识别的参数 λ = {lambda_identified:.6f}")
    print(f"（注意：真实λ值需要根据题目确定，这里用于验证）")
    
    # 6. 可视化
    print("\n6. 可视化结果...")
    visualize_results(X_data, u_data, u_pred, model.train_loss_history, results_dir)
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)
    
    return lambda_identified, eval_result


def visualize_results(X_data, u_true, u_pred, train_loss, save_dir):
    # """可视化结果"""
    # fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    #
    # x = X_data[:, 0]
    # y = X_data[:, 1]
    #
    # # 1. 真实u的分布
    # ax1 = axes[0, 0]
    # scatter1 = ax1.scatter(x, y, c=u_true, cmap='coolwarm', s=20)
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.set_title('观测数据 u(x,y)')
    # plt.colorbar(scatter1, ax=ax1)
    #
    # # 2. 预测u的分布
    # ax2 = axes[0, 1]
    # scatter2 = ax2.scatter(x, y, c=u_pred, cmap='coolwarm', s=20)
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_title('预测解 u_pred(x,y)')
    # plt.colorbar(scatter2, ax=ax2)
    #
    # # 3. 误差分布
    # ax3 = axes[1, 0]
    # error = np.abs(u_pred - u_true)
    # scatter3 = ax3.scatter(x, y, c=error, cmap='hot', s=20)
    # ax3.set_xlabel('x')
    # ax3.set_ylabel('y')
    # ax3.set_title('误差分布 |u_pred - u_true|')
    # plt.colorbar(scatter3, ax=ax3)
    #
    # # 4. 训练损失
    # ax4 = axes[1, 1]
    # ax4.plot(train_loss)
    # ax4.set_xlabel('Epoch')
    # ax4.set_ylabel('Loss')
    # ax4.set_title('训练损失')
    # ax4.set_yscale('log')

    """可视化结果"""

    # 创建2x3的布局
    fig = plt.figure(figsize=(15, 10))

    # 提取坐标
    x = X_data[:, 0]
    y = X_data[:, 1]

    # 1. 真实解
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter1 = ax1.scatter(x, y, u_true.flatten(), c=u_true.flatten(), cmap='coolwarm')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    ax1.set_title('True Solution')
    plt.colorbar(scatter1, ax=ax1)

    # 2. 预测解
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    scatter2 = ax2.scatter(x, y, u_pred.flatten(), c=u_pred.flatten(), cmap='coolwarm')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
    ax2.set_title('Predicted Solution')
    plt.colorbar(scatter2, ax=ax2)

    # 3. 误差分布
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    error = np.abs(u_pred.flatten() - u_true.flatten())
    scatter3 = ax3.scatter(x, y, error, c=error, cmap='hot')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('|Error|')
    ax3.set_title('Error Distribution')
    plt.colorbar(scatter3, ax=ax3)

    # 4. 学习曲线
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(train_loss, label='Train Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training Loss')
    ax4.legend()
    ax4.set_yscale('log')


    # 6. 散点图：真实 vs 预测
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.scatter(u_true, u_pred, alpha=0.5)
    min_val = min(u_true.min(), u_pred.min())
    max_val = max(u_true.max(), u_pred.max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    ax6.set_xlabel('True')
    ax6.set_ylabel('Predicted')
    ax6.set_title('True vs Predicted')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'poisson_inverse_results.png', dpi=150)
    print(f"   结果已保存到: {save_dir / 'poisson_inverse_results.png'}")
    plt.close()


if __name__ == "__main__":
    main()

