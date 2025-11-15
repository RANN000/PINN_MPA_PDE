"""
子任务 1 & 2 ：Helmholtz方程求解 (k=n) - any网络实现
可使用 baseline、mffm、r-gpinn 网络 + 参数网格搜索
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    print("子任务 1 & 2 ：Helmholtz方程求解 (k=n) - any + 网格搜索")
    print("=" * 70)

    # 1. 加载数据选择文件，可改写
    print("\n1. 加载数据...")
    # 选择文件
    path = "../data/子任务2_亥姆霍兹方程数据_k1000.xlsx"
    # 选择子任务
    task = "task2"
    # 选择波数
    k = 1000
    # 选择网络类型: standard, mffm, r_gpinn
    network = 'standard'

    loader = HelmholtzDataLoader(path, task=task, k=k)
    X_train, q_train, boundary_mask = loader.get_training_points()
    X_interior = X_train[~boundary_mask]
    X_boundary = X_train[boundary_mask]
    X_test, _ = loader.get_test_points()
    print(f"   内部点: {len(X_interior)}")
    print(f"   边界点: {len(X_boundary)}")
    print(f"   测试点: {len(X_test)}")

    # 生成真实解
    a1, a2, k_val = 1, 3, loader.k
    c = - ((a1 * np.pi) ** 2 + (a2 * np.pi) ** 2 + k_val ** 2) / ((a1 * np.pi) ** 2 + (a2 * np.pi) ** 2 - k_val ** 2)
    u_test_true = c * np.sin(np.pi * X_test[:, 0]) * np.sin(3 * np.pi * X_test[:, 1])

    # ----------------- 参数矩阵 -----------------
    # ---------------可添加修改参数----------------
    widths = [40]
    depths = [3]
    lrs = [0.001]
    lambda_pdes = [0.5]
    lambda_bcs = [50]

    param_grid = [(w, d, lr, lp, lb) for w in widths for d in depths for lr in lrs for lp in lambda_pdes for lb in
                  lambda_bcs]

    all_results = []

    # 2. 网格搜索训练，遍历参数矩阵
    for i, (width, depth, lr, lambda_pde, lambda_bc) in enumerate(param_grid, 1):
        print("\n" + "=" * 50)
        print(
            f"参数组合 {i}/{len(param_grid)}: width={width}, depth={depth}, lr={lr}, lambda_pde={lambda_pde}, lambda_bc={lambda_bc}")

        layers = [2] + [width] * depth + [1]
        config = {
            # 可手动选择测试网络类型
            'network_type': network,
            'layers': layers,
            'activation': 'tanh',
            'learning_rate': lr,
            'weight_decay': 0.0,
            'wavenumber': k_val,
            'lambda_pde': lambda_pde,
            'lambda_bc': lambda_bc
        }

        model = HelmholtzPINN(config)
        results_dir = Path(f'../results/{task}_{network}/width{width}_depth{depth}_lr{lr}_pde{lambda_pde}_bc{lambda_bc}')
        results_dir.mkdir(exist_ok=True, parents=True)

        model = train_helmholtz_pinn(
            model=model,
            X_interior=X_interior,
            X_boundary=X_boundary,
            X_test=X_test,
            u_test_true=u_test_true,
            n_epochs=2000,  # 网格搜索时先用小 epoch
            batch_size=256,
            log_every=100,
            save_dir=str(results_dir)
        )
        # 评估
        eval_result = model.evaluate(X_test, u_test_true)
        print(f"   L2误差: {eval_result['l2_error']:.6e}")
        print(f"   相对L2误差: {eval_result['relative_l2_error']:.6f}")
        print(f"   最大误差: {eval_result['max_error']:.6e}")
        print(f"   相对最大误差: {eval_result['relative_max_error']:.6e}")

        all_results.append({
            'width': width,
            'depth': depth,
            'lr': lr,
            'lambda_pde': lambda_pde,
            'lambda_bc': lambda_bc,
            'relative_l2_error': eval_result['relative_l2_error'],
            'max_error': eval_result['max_error']
        })

        # 生成每组组合的可视化结果
        visualize_results(X_test, u_test_true, eval_result['u_pred'],
                          model.train_loss_history, model.test_loss_history,
                          results_dir, k_val, network)

    # 3. 保存汇总表
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f'../results/{task}_{network}/grid_search_results.csv', index=False)
    print(f"\n网格搜索完成，结果已保存到 ../results/{task}_{network}/grid_search_results.csv")


def visualize_results(X_test, u_true, u_pred, train_loss, test_loss, save_dir, k, network):
    """可视化结果"""
    fig = plt.figure(figsize=(15, 10))
    x, y = X_test[:, 0], X_test[:, 1]

    # 真实解
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter1 = ax1.scatter(x, y, u_true.flatten(), c=u_true.flatten(), cmap='coolwarm')
    ax1.set_title('True Solution');
    plt.colorbar(scatter1, ax=ax1)

    # 预测解
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    scatter2 = ax2.scatter(x, y, u_pred.flatten(), c=u_pred.flatten(), cmap='coolwarm')
    ax2.set_title('Predicted Solution');
    plt.colorbar(scatter2, ax=ax2)

    # 误差
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    error = np.abs(u_pred.flatten() - u_true.flatten())
    scatter3 = ax3.scatter(x, y, error, c=error, cmap='hot')
    ax3.set_title('Error Distribution');
    plt.colorbar(scatter3, ax=ax3)

    # 学习曲线
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(train_loss, label='Train Loss')
    ax4.set_yscale('log');
    ax4.set_title('Training Loss');
    ax4.legend()

    # 测试误差曲线
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(test_loss, label='Test Relative L2 Error')
    ax5.set_yscale('log');
    ax5.set_title('Test Error');
    ax5.legend()

    # 散点图: 真实 vs 预测
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.scatter(u_true, u_pred, alpha=0.5)
    min_val, max_val = min(u_true.min(), u_pred.min()), max(u_true.max(), u_pred.max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax6.set_title('True vs Predicted')

    plt.tight_layout()
    plt.savefig(save_dir / f'{network}_results_k{k}.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
