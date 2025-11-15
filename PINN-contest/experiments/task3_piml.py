"""
子任务3：Poisson反问题 - 参数识别
给定200个观测点，识别参数k
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

from pinn_solvers.poisson_inverse_piml_solver import (
    stage1_train_forward_model,
    stage2_train_lambda_net,
    evaluate_models,
    load_observation_data,
    visualize_results)


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
    print("子任务3：PIML两阶段Poisson反问题求解（修复版）")
    print("=" * 70)

    # 配置模型参数
    config = {
        'hidden_layers': [128, 128, 64, 32],
        'lr_stage1': 0.0008,
        'lr_stage2': 0.0005,
        'epochs_stage1': 2000,
        'epochs_stage2': 3000,
        'batch_size_stage1': 128,
        'batch_size_stage2': 1024,
        'n_boundary_per_edge': 100,
        'n_collocation': 8000,
        'lambda_reg': 0.001,
        'weight_decay': 0.00001,
        'results_dir': '../results/task3_piml'
    }

    # 1. 加载数据
    data_path = '../data/子任务3数据.xlsx'
    X_obs, u_obs = load_observation_data(data_path)

    # 2. 阶段1：训练前向模型
    u_model, losses_stage1 = stage1_train_forward_model(X_obs, u_obs, config)

    # 3. 阶段2：训练参数网络
    lambda_net, losses_stage2 = stage2_train_lambda_net(u_model, config)

    # 4. 评估模型
    results = evaluate_models(u_model, lambda_net, X_obs, u_obs, config)
    visualize_results(u_model, lambda_net, X_obs, u_obs, results['u_pred'],
                      config, losses_stage1, losses_stage2)

    # 5. 保存模型
    torch.save({
        'u_model_state': u_model.state_dict(),
        'lambda_net_state': lambda_net.state_dict(),
        'config': config
    }, Path(config['results_dir']) / 'models.pth')

    print("\n" + "=" * 70)
    print("PIML两阶段求解完成！")
    print("=" * 70)

    # 打印训练总结
    print(f"\n训练总结:")
    print(f"  阶段1最终损失: {losses_stage1[-1]:.6f}")
    print(f"  阶段2最终损失: {losses_stage2[-1]:.6f}")
    print(f"  预测MSE: {results['mse']:.6e}")
    print(f"  相对L2误差: {results['relative_l2']:.6e}")

    return u_model, lambda_net, results

if __name__ == "__main__":
    main()