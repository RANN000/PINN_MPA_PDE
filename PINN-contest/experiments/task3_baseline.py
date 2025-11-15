"""
子任务3：标准PINN方法训练评估
修复了评估阶段的PDE残差计算问题
"""

import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class StandardPoissonPINN(nn.Module):
    """标准PINN方法：端到端学习u和k"""

    def __init__(self, hidden_layers=[128, 128, 64, 32], activation='tanh'):
        super().__init__()

        # u网络
        layers_u = [2] + hidden_layers + [1]
        self.u_net = self._create_mlp(layers_u, activation)

        # k网络
        layers_k = [2] + hidden_layers + [1]
        self.k_net = self._create_mlp(layers_k, activation)

        self._initialize_weights()

    def _create_mlp(self, layers, activation):
        """创建MLP网络"""
        activation_fn = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'silu': nn.SiLU()
        }.get(activation, nn.Tanh())

        net_layers = []
        for i in range(len(layers) - 1):
            net_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                net_layers.append(activation_fn)

        return nn.Sequential(*net_layers)

    def _initialize_weights(self):
        """初始化权重"""
        for net in [self.u_net, self.k_net]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """前向传播：返回u和k"""
        u = self.u_net(x)
        k_raw = self.k_net(x)
        k = F.softplus(k_raw) + 1e-8  # 确保k>0
        return u, k


def f_source(x):
    """
    源项函数 f(x,y)
    """
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    pi = torch.tensor(torch.pi, device=x.device)

    f = (pi ** 2 / 2.0) * (1 + x_coord ** 2 + y_coord ** 2) * torch.sin(pi * x_coord / 2.0) * torch.cos(
        pi * y_coord / 2.0) \
        - pi * x_coord * torch.cos(pi * x_coord / 2.0) * torch.cos(pi * y_coord / 2.0) \
        + pi * y_coord * torch.sin(pi * x_coord / 2.0) * torch.sin(pi * y_coord / 2.0)

    return f


def safe_pde_residual_standard(model, x):
    """
    安全的PDE残差计算，用于评估阶段
    避免计算图构建问题
    """
    # 临时启用梯度计算
    with torch.enable_grad():
        x = x.clone().requires_grad_(True)

        # 计算 u(x) 和 k(x)
        u, k = model(x)

        # 计算 ∇u
        u_grad = torch.autograd.grad(
            outputs=u, inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        u_x, u_y = u_grad[:, 0:1], u_grad[:, 1:2]

        # 计算通量 k∇u
        flux_x = k * u_x
        flux_y = k * u_y

        # 计算散度 ∇·(k∇u)
        flux_x_grad = torch.autograd.grad(
            outputs=flux_x, inputs=x,
            grad_outputs=torch.ones_like(flux_x),
            create_graph=True, retain_graph=True
        )[0][:, 0:1]

        flux_y_grad = torch.autograd.grad(
            outputs=flux_y, inputs=x,
            grad_outputs=torch.ones_like(flux_y),
            create_graph=True, retain_graph=True
        )[0][:, 1:2]

        divergence = flux_x_grad + flux_y_grad

        # 计算源项
        f = f_source(x)

        # PDE残差
        residual = -divergence - f

        return residual


def pde_residual_standard(model, x):
    """
    计算PDE残差: R = -∇·(k∇u) - f
    用于训练阶段
    """
    x = x.clone().requires_grad_(True)

    # 计算 u(x) 和 k(x)
    u, k = model(x)

    # 计算 ∇u
    u_grad = torch.autograd.grad(
        outputs=u, inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    u_x, u_y = u_grad[:, 0:1], u_grad[:, 1:2]

    # 计算通量 k∇u
    flux_x = k * u_x
    flux_y = k * u_y

    # 计算散度 ∇·(k∇u)
    flux_x_grad = torch.autograd.grad(
        outputs=flux_x, inputs=x,
        grad_outputs=torch.ones_like(flux_x),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]

    flux_y_grad = torch.autograd.grad(
        outputs=flux_y, inputs=x,
        grad_outputs=torch.ones_like(flux_y),
        create_graph=True, retain_graph=True
    )[0][:, 1:2]

    divergence = flux_x_grad + flux_y_grad

    # 计算源项
    f = f_source(x)

    # PDE残差
    residual = -divergence - f
    return residual


def generate_boundary_points(n_per_edge=100):
    """生成边界点"""
    n = n_per_edge
    x1 = np.random.uniform(-1, 1, n)
    x2 = np.random.uniform(-1, 1, n)

    b1 = np.column_stack([x1, -1 * np.ones(n)])
    b2 = np.column_stack([x1, 1 * np.ones(n)])
    b3 = np.column_stack([-1 * np.ones(n), x2])
    b4 = np.column_stack([1 * np.ones(n), x2])

    return np.vstack([b1, b2, b3, b4]).astype(np.float32)


def generate_interior_points(n_interior=2000):
    """生成内部配点"""
    return np.random.uniform(-1, 1, (n_interior, 2)).astype(np.float32)


def load_observation_data(data_path):
    """加载观测数据"""
    df = pd.read_excel(data_path)
    print(f"    成功加载数据: {data_path}")
    print(f"   列名: {df.columns.tolist()}, 形状: {df.shape}")

    if {'xi', 'yi', 'u(xi,yi)'}.issubset(df.columns):
        X = df[['xi', 'yi']].values.astype(np.float32)
        u = df['u(xi,yi)'].values.astype(np.float32).reshape(-1, 1)
    else:
        X = df.iloc[:, :2].values.astype(np.float32)
        u = df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)

    print(f"   样本数: {len(X)}, u 范围: [{u.min():.4f}, {u.max():.4f}]")
    return X, u


def train_standard_pinn(X_obs, u_obs, config):
    """训练标准PINN模型"""
    print("\n" + "=" * 50)
    print("训练标准PINN模型")
    print("=" * 50)

    # 创建模型
    model = StandardPoissonPINN(
        hidden_layers=config['hidden_layers'],
        activation='tanh'
    ).to(device)

    # 准备训练数据
    X_boundary = generate_boundary_points(n_per_edge=config['n_boundary_per_edge'])
    u_boundary = np.zeros((len(X_boundary), 1), dtype=np.float32)

    # 合并观测数据和边界数据
    X_train = np.vstack([X_obs, X_boundary])
    u_train = np.vstack([u_obs, u_boundary])

    # 生成配点
    X_collocation = generate_interior_points(n_interior=config['n_collocation'])

    print(f"   观测点: {len(X_obs)}, 边界点: {len(X_boundary)}, 配点: {len(X_collocation)}")

    # 转换为tensor
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    u_train_tensor = torch.FloatTensor(u_train).to(device)
    X_collocation_tensor = torch.FloatTensor(X_collocation).to(device)

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr_standard'],
        weight_decay=config.get('weight_decay', 1e-5)
    )

    # 训练循环
    model.train()
    train_losses = []
    data_losses = []
    pde_losses = []
    bc_losses = []

    start_time = time.time()

    for epoch in range(config['epochs_standard']):
        # 随机打乱
        indices = torch.randperm(len(X_train_tensor))

        total_loss = 0.0
        total_data_loss = 0.0
        total_pde_loss = 0.0
        total_bc_loss = 0.0

        for i in range(0, len(X_train_tensor), config['batch_size_standard']):
            batch_indices = indices[i:i + config['batch_size_standard']]
            X_batch = X_train_tensor[batch_indices]
            u_batch = u_train_tensor[batch_indices]

            # 判断批次中的边界点（u=0）和观测点（u≠0）
            is_boundary = (u_batch == 0).all(dim=1)
            is_data = ~is_boundary

            optimizer.zero_grad()

            # 前向传播
            u_pred, k_pred = model(X_batch)

            # 数据损失（观测点）
            data_loss = torch.tensor(0.0).to(device)
            if is_data.any():
                data_loss = F.mse_loss(u_pred[is_data], u_batch[is_data])

            # 边界损失（边界点）
            bc_loss = torch.tensor(0.0).to(device)
            if is_boundary.any():
                bc_loss = F.mse_loss(u_pred[is_boundary], u_batch[is_boundary])

            # PDE残差损失（使用配点）
            pde_res = pde_residual_standard(model, X_collocation_tensor[:len(X_batch)])
            pde_loss = torch.mean(pde_res ** 2)

            # 正则化损失（防止k过大）
            reg_loss = config['lambda_reg'] * torch.mean(k_pred ** 2)

            # 总损失
            loss = (config['lambda_data'] * data_loss +
                    config['lambda_bc'] * bc_loss +
                    config['lambda_pde'] * pde_loss +
                    reg_loss)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_indices)
            total_data_loss += data_loss.item() * len(batch_indices)
            total_pde_loss += pde_loss.item() * len(batch_indices)
            total_bc_loss += bc_loss.item() * len(batch_indices)

        # 记录损失
        avg_loss = total_loss / len(X_train_tensor)
        avg_data_loss = total_data_loss / len(X_train_tensor)
        avg_pde_loss = total_pde_loss / len(X_train_tensor)
        avg_bc_loss = total_bc_loss / len(X_train_tensor)

        train_losses.append(avg_loss)
        data_losses.append(avg_data_loss)
        pde_losses.append(avg_pde_loss)
        bc_losses.append(avg_bc_loss)

        if (epoch + 1) % max(1, config['epochs_standard'] // 10) == 0:
            # 监控k的平均值
            with torch.no_grad():
                k_sample = model.k_net(X_collocation_tensor[:100])
                k_sample = F.softplus(k_sample) + 1e-8
                mean_k = k_sample.mean().item()

            print(f"   Epoch {epoch + 1}/{config['epochs_standard']} | "
                  f"Total Loss: {avg_loss:.6f} | "
                  f"Data Loss: {avg_data_loss:.6f} | "
                  f"PDE Loss: {avg_pde_loss:.6f} | "
                  f"Mean k: {mean_k:.4f}")

    training_time = time.time() - start_time
    print(f"    标准PINN训练完成，用时: {training_time / 60:.1f} 分钟")

    return model, {
        'train_losses': train_losses,
        'data_losses': data_losses,
        'pde_losses': pde_losses,
        'bc_losses': bc_losses,
        'training_time': training_time
    }


def evaluate_standard_model(model, X_obs, u_obs, config):
    """评估标准PINN模型"""
    print("\n" + "=" * 50)
    print("评估标准PINN模型")
    print("=" * 50)

    model.eval()

    # 转换为tensor
    X_tensor = torch.FloatTensor(X_obs).to(device)
    u_tensor = torch.FloatTensor(u_obs).to(device)

    with torch.no_grad():
        # 预测u值
        u_pred, k_pred = model(X_tensor)
        u_pred = u_pred.cpu().numpy().flatten()
        k_pred = k_pred.cpu().numpy().flatten()

        # 计算u的预测误差
        u_true = u_obs.flatten()
        mse = np.mean((u_pred - u_true) ** 2)
        l2_error = np.sqrt(mse)
        relative_l2 = l2_error / (np.sqrt(np.mean(u_true ** 2)) + 1e-8)
        mae = np.mean(np.abs(u_pred - u_true))

    # 计算PDE残差（使用安全函数）
    X_collocation = generate_interior_points(n_interior=1000)
    X_collocation_tensor = torch.FloatTensor(X_collocation).to(device)
    pde_res = safe_pde_residual_standard(model, X_collocation_tensor)
    pde_res_norm = torch.mean(pde_res ** 2).item()

    print(f"预测性能:")
    print(f"  MSE: {mse:.6e}")
    print(f"  L2误差: {l2_error:.6e}")
    print(f"  相对L2误差: {relative_l2:.6f}")
    print(f"  MAE: {mae:.6e}")
    print(f"  PDE残差范数: {pde_res_norm:.6e}")
    print(f"  k值范围: [{k_pred.min():.4f}, {k_pred.max():.4f}]")
    print(f"  平均k值: {k_pred.mean():.4f}")

    return {
        'u_pred': u_pred,
        'k_pred': k_pred,
        'mse': mse,
        'l2_error': l2_error,
        'relative_l2': relative_l2,
        'mae': mae,
        'pde_res_norm': pde_res_norm,
        'k_range': [k_pred.min(), k_pred.max()],
        'k_mean': k_pred.mean()
    }


def visualize_standard_results(model, X_obs, u_obs, eval_results, losses, config):
    """可视化标准PINN结果"""
    save_dir = Path(config['results_dir']) / 'standard_pinn'
    save_dir.mkdir(parents=True, exist_ok=True)

    # 生成网格用于可视化
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X_grid, Y_grid = np.meshgrid(x, y)
    points = np.column_stack([X_grid.ravel(), Y_grid.ravel()]).astype(np.float32)

    with torch.no_grad():
        points_tensor = torch.FloatTensor(points).to(device)
        u_grid, k_grid = model(points_tensor)
        u_grid = u_grid.cpu().numpy().reshape(100, 100)
        k_grid = k_grid.cpu().numpy().reshape(100, 100)

    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 真实u值（观测点）
    u_true = u_obs.flatten()
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(X_obs[:, 0], X_obs[:, 1], c=u_true, cmap='coolwarm', s=30)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('True u (Observation Points)')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1)

    # 2. 预测u值（观测点）
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(X_obs[:, 0], X_obs[:, 1], c=eval_results['u_pred'], cmap='coolwarm', s=30)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Predicted u (Observation Points)')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2)

    # 3. 误差分布
    ax3 = axes[0, 2]
    error = np.abs(eval_results['u_pred'] - u_true)
    sc3 = ax3.scatter(X_obs[:, 0], X_obs[:, 1], c=error, cmap='hot', s=30)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Prediction Error |u_pred - u_true|')
    ax3.set_aspect('equal')
    plt.colorbar(sc3, ax=ax3)

    # 4. 识别参数k(x,y)
    ax4 = axes[1, 0]
    im4 = ax4.imshow(k_grid, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis', aspect='auto')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Identified Parameter k(x,y)')
    plt.colorbar(im4, ax=ax4)

    # 5. 学习曲线
    ax5 = axes[1, 1]
    ax5.plot(losses['train_losses'], 'b-', linewidth=1.5, label='Total Loss')
    ax5.plot(losses['data_losses'], 'g-', linewidth=1.5, label='Data Loss')
    ax5.plot(losses['pde_losses'], 'r-', linewidth=1.5, label='PDE Loss')
    ax5.plot(losses['bc_losses'], 'orange', linewidth=1.5, label='BC Loss')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.set_title('Training Loss History')
    ax5.legend()
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)

    # 6. 性能指标
    ax6 = axes[1, 2]
    ax6.axis('off')

    metrics_text = f"""
    Performance Metrics:
    MSE: {eval_results['mse']:.4e}
    L2 Error: {eval_results['l2_error']:.4e}
    Relative L2: {eval_results['relative_l2']:.4f}
    MAE: {eval_results['mae']:.4e}
    PDE Residual: {eval_results['pde_res_norm']:.4e}

    Training Info:
    Epochs: {config['epochs_standard']}
    Training Time: {losses['training_time'] / 60:.1f} min
    Final Loss: {losses['train_losses'][-1]:.4e}

    k Statistics:
    Range: [{eval_results['k_range'][0]:.4f}, {eval_results['k_range'][1]:.4f}]
    Mean: {eval_results['k_mean']:.4f}
    """

    ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_dir / 'standard_pinn_results.png', dpi=150, bbox_inches='tight')
    print(f"结果已保存到: {save_dir / 'standard_pinn_results.png'}")
    plt.close()

    # 保存数据
    np.save(save_dir / 'k_field_standard.npy', k_grid)
    np.save(save_dir / 'u_pred_standard.npy', eval_results['u_pred'])

    return save_dir


def main():
    print("=" * 70)
    print("子任务3：标准PINN方法训练评估")
    print("=" * 70)

    # 配置参数
    config = {
        'hidden_layers': [128, 128, 64, 32],
        'lr_standard': 0.001,
        'epochs_standard': 3000,
        'batch_size_standard': 128,
        'n_boundary_per_edge': 100,
        'n_collocation': 8000,
        'lambda_pde': 1.0,
        'lambda_data': 1.0,
        'lambda_bc': 50,
        'lambda_reg': 0.001,
        'weight_decay': 1e-5,
        'results_dir': '../results/task3_standard_pinn'
    }

    # 1. 加载数据
    data_path = '../data/子任务3数据.xlsx'
    X_obs, u_obs = load_observation_data(data_path)

    # 2. 训练标准PINN模型
    model, losses = train_standard_pinn(X_obs, u_obs, config)

    # 3. 评估模型
    eval_results = evaluate_standard_model(model, X_obs, u_obs, config)

    # 4. 可视化结果
    save_dir = visualize_standard_results(model, X_obs, u_obs, eval_results, losses, config)

    # 5. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'losses': losses,
        'eval_results': eval_results
    }, save_dir / 'standard_pinn_model.pth')

    print(f"\n模型已保存到: {save_dir / 'standard_pinn_model.pth'}")

    print("\n" + "=" * 70)
    print("标准PINN训练评估完成！")
    print("=" * 70)

    return model, eval_results, losses


if __name__ == "__main__":
    model, eval_results, losses = main()