"""
子任务2：Helmholtz方程求解 (k=100,500,1000) - mffm实现
使用mffm网络
"""

import sys
import os

from mpa_core import select_optimizer_by_mpa

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
    print("子任务2：Helmholtz方程求解 (k=100,500,1000) - mffm")
    print("=" * 70)

    # 1. 加载数据
    print("\n1. 加载数据...")
    path = '../data/子任务2_亥姆霍兹方程数据_k100.xlsx'
    loader = HelmholtzDataLoader(path, task='task2', k=100)

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
    a1, a2, k = 1, 3, loader.k
    c = - ((a1 * np.pi) ** 2 + (a2 * np.pi) ** 2 + k ** 2) / ((a1 * np.pi) ** 2 + (a2 * np.pi) ** 2 - k ** 2)
    u_test_true = c * np.sin(np.pi * X_test[:, 0]) * np.sin(3 * np.pi * X_test[:, 1])

    print(f"   测试点: {len(X_test)}")

    # 2. 创建模型
    print("\n2. 创建模型...")
    config = {
        'layers': [2, 40, 40, 40, 1],
        'activation': 'tanh',
        'learning_rate': 0.001,
        'weight_decay': 0.0,
        'wavenumber': loader.k, #根据加载的数据不同而改变
        'lambda_pde': 50,
        'lambda_bc': 0.5
    }

    model = HelmholtzPINN(config).to(device)

    # 统计参数数量
    n_params = sum(p.numel() for p in model.net.parameters())
    print(f"   网络参数数量: {n_params}")

    # MPA分析（快速模式）
    print("   运行MPA快速分析...")
    from mpa_core.alignment_metrics import compute_all_metrics
    batch_dict = {'x_interior': torch.FloatTensor(X_interior[:100]).to(device)}
    mpa_scores = compute_all_metrics(model, batch_dict, device=device, quick=True)
    print(f"   MPA分数: GDC={mpa_scores['GDC']:.3f}, TS={mpa_scores['TS']:.3f}, LSM={mpa_scores['LSM']:.3f}")

    # 选择优化器
    optimizer, opt_name = select_optimizer_by_mpa(
        mpa_scores['GDC'],
        mpa_scores['TS'],
        mpa_scores['LSM'],
        model.net.parameters(),
        learning_rate=config['learning_rate']
    )

    print(f"   MPA推荐优化器: {opt_name}")
    print(f"   调整后学习率: {optimizer.param_groups[0]['lr']:.6f}")


    # 3. 训练模型
    print("\n3. 训练模型...")
    results_dir = Path('../results/task2_mffm_mpa')

    import time
    start_time = time.time()

    model = train_helmholtz_pinn(
        model=model,
        X_interior=X_interior,
        X_boundary=X_boundary,
        X_test=X_test,
        u_test_true=u_test_true,
        optimizer=optimizer,
        n_epochs=2000,
        batch_size=256,
        log_every=100,
        save_dir=str(results_dir)
    )

    elapsed = time.time() - start_time
    print(f"\n训练用时: {elapsed / 60:.1f} 分钟")
    print(f"\n训练结果已保存至\'results/task2_mffm_mpa\'")

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
                      results_dir,loader.k)

    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)


def visualize_results(X_test, u_true, u_pred, train_loss, test_loss, save_dir ,k):
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
    ax5.set_title('Test Error')
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
    plt.savefig(save_dir / f'mffm_results_k{k}.png', dpi=150)
    print(f"   结果已保存到: {save_dir / f'mffm_results_k{k}.png'}")
    plt.close()


def train_with_custom_optimizer(model, X_interior, X_boundary,
                                X_test, u_test_true,
                                optimizer,
                                n_epochs=2000, batch_size=128,
                                log_every=500, save_dir='../results'):
    """使用自定义优化器训练"""
    import os
    os.makedirs(save_dir, exist_ok=True)

    u_boundary = np.zeros(len(X_boundary))

    from pinn_solvers.base_pinn import create_dataloader
    dataloader = create_dataloader(X_interior, X_boundary, u_boundary, batch_size=batch_size)

    print(f"\n开始训练 (优化器: 自定义)")
    print(f"{'=' * 60}")

    best_test_loss = float('inf')

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for batch_dict in dataloader:
            for key in batch_dict:
                if torch.is_tensor(batch_dict[key]):
                    batch_dict[key] = batch_dict[key].to(model.device)

            loss = model.train_step(batch_dict, optimizer)
            epoch_loss += loss

        avg_loss = epoch_loss / len(dataloader)
        model.train_loss_history.append(avg_loss)

        if (epoch + 1) % log_every == 0 or epoch == 0:
            X_test_tensor = torch.FloatTensor(X_test).to(model.device)
            with torch.no_grad():
                u_test_pred = model.forward(X_test_tensor).cpu().numpy().flatten()

            test_loss = np.sqrt(np.mean((u_test_pred - u_test_true) ** 2))
            relative_error = test_loss / (np.sqrt(np.mean(u_test_true ** 2)) + 1e-10)

            model.test_loss_history.append(relative_error)

            print(f"Epoch {epoch + 1}/{n_epochs} | "
                  f"Train Loss: {avg_loss:.6e} | "
                  f"Test Rel L2: {relative_error:.6e}")

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                model.save_checkpoint(
                    os.path.join(save_dir, f'best_model_mpa.pth'),
                    epoch, optimizer.state_dict()
                )
        else:
            model.test_loss_history.append(model.test_loss_history[-1] if model.test_loss_history else 1.0)

    print(f"\n训练完成！最佳测试误差: {best_test_loss:.6e}")
    return model


if __name__ == "__main__":
    main()

