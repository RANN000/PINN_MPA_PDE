"""
子任务1：Helmholtz方程求解 (k=4) - r-gpinn实现
使用 r-gpinn 网络
"""

import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 导入自定义模块
from data.data_loader import HelmholtzDataLoader
from pinn_solvers.helmholtz_solver import HelmholtzPINN, train_helmholtz_pinn

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def main():
    print("=" * 70)
    print("子任务1：Helmholtz方程求解 (k=4) - rgpinn")
    print("=" * 70)

    # 1. 加载数据
    print("\n1. 加载数据...")
    loader = HelmholtzDataLoader('../data/子任务1_亥姆霍兹方程数据_k4.xlsx', task='task1', k=4)

    # 获取训练数据
    X_train, q_train, boundary_mask = loader.get_training_points()
    X_interior = X_train[~boundary_mask]
    X_boundary = X_train[boundary_mask]

    print(f"   内部点: {len(X_interior)}")
    print(f"   边界点: {len(X_boundary)}")

    # 获取测试数据
    X_test, _ = loader.get_test_points()

    # 生成真实解用于验证
    # 解析解: u(x,y) 约等于 -1.35 * sin(πx)sin(3πy)
    a1, a2, k = 1, 3, 4
    c = - ((a1 * np.pi) ** 2 + (a2 * np.pi) ** 2 + k ** 2) / ((a1 * np.pi) ** 2 + (a2 * np.pi) ** 2 - k ** 2)
    u_test_true = c * np.sin(np.pi * X_test[:, 0]) * np.sin(3 * np.pi * X_test[:, 1])

    print(f"   测试点: {len(X_test)}")

    # 2. 创建模型
    print("\n2. 创建模型...")
    config = {
        'network_type': 'r_gpinn',
        'layers': [2, 40, 40, 40, 1],
        'activation': 'tanh',
        'learning_rate': 0.001,
        'weight_decay': 0.0,
        'wavenumber': 4,
        'lambda_pde': 0.5,
        'lambda_bc': 50
    }

    model = HelmholtzPINN(config)

    # 统计参数数量
    n_params = sum(p.numel() for p in model.net.parameters())
    print(f"   网络参数数量: {n_params}")

    # 3. 训练模型
    print("\n3. 训练模型...")
    results_dir = Path('../results/task1_r_gpinn')

    model = train_helmholtz_pinn(
        model=model,
        X_interior=X_interior,
        X_boundary=X_boundary,
        X_test=X_test,
        u_test_true=u_test_true,
        n_epochs=2000,
        batch_size=256,
        log_every=100,
        save_dir=str(results_dir)
    )

    # 4. 最终评估
    print("\n4. 评估模型...")
    eval_result = model.evaluate(X_test, u_test_true)

    print(f"\n评估结果:")
    print(f"   L2误差: {eval_result['l2_error']:.6e}")
    print(f"   相对L2误差: {eval_result['relative_l2_error']:.6e}")
    print(f"   最大误差: {eval_result['max_error']:.6e}")
    print(f"   相对最大误差: {eval_result['relative_max_error']:.6e}")

    # 5. 可视化结果
    print("\n5. 可视化结果...")
    visualize_results(X_test, u_test_true, eval_result['u_pred'],
                      model.train_loss_history, model.test_loss_history,
                      results_dir)

    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)


def visualize_results(X_test, u_true, u_pred, train_loss, test_loss, save_dir):
    """可视化结果"""

    # 创建2x3的布局
    fig = plt.figure(figsize=(15, 10))

    # 提取坐标
    x = X_test[:, 0]
    y = X_test[:, 1]

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

    # 5. 测试误差曲线
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(test_loss, label='Test Relative L2 Error')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Relative L2 Error')
    ax5.set_title('est Error')
    ax5.legend()
    ax5.set_yscale('log')

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
    plt.savefig(save_dir / 'r_gpinn_results.png', dpi=150)
    print(f"   结果已保存到: {save_dir / 'r_gpinn_results.png'}")
    plt.close()


if __name__ == "__main__":
    main()

