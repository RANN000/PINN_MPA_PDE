"""
Poisson反问题求解器（子任务3）
使用 piml 方法求解
"""

import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
from pathlib import Path
import pandas as pd

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class MLP(nn.Module):
    """
    标准多层感知机
    用于后续的前向模型 / 参数学习模型
    """

    def __init__(self, layers, activation='tanh'):
        super().__init__()
        self.layers = nn.ModuleList()

        # 激活函数
        act_dict = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'silu': nn.SiLU()
        }
        activation_fn = act_dict.get(activation, nn.Tanh())

        # 构建网络层
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:  # 隐藏层添加激活函数
                self.layers.append(activation_fn)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ForwardUModel(nn.Module):
    """前向模型：学习 u(x,y)"""

    def __init__(self, hidden_layers=[64, 64, 32]):
        super().__init__()
        layers = [2] + hidden_layers + [1]
        self.net = MLP(layers, activation='tanh')

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class LambdaNet(nn.Module):
    """参数网络：学习 k(x,y)"""

    def __init__(self, hidden_layers=[64, 64, 32]):
        super().__init__()
        layers = [2] + hidden_layers + [1]
        self.net = MLP(layers, activation='tanh')
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 使用softplus确保k>0
        k_raw = self.net(x)
        return F.softplus(k_raw) + 1e-8


def f_source(x):
    """
    源项函数 f(x,y)，根据题目公式
    输入: x (N, 2) tensor
    输出: f (N, 1) tensor
    """
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    pi = torch.tensor(torch.pi, device=x.device)

    f = (pi ** 2 / 2.0) * (1 + x_coord ** 2 + y_coord ** 2) * torch.sin(pi * x_coord / 2.0) * torch.cos(
        pi * y_coord / 2.0) \
        - pi * x_coord * torch.cos(pi * x_coord / 2.0) * torch.cos(pi * y_coord / 2.0) \
        + pi * y_coord * torch.sin(pi * x_coord / 2.0) * torch.sin(pi * y_coord / 2.0)

    return f


def pde_residual(u_model, lambda_net, x):
    """
    计算PDE残差: R = -∇·(k∇u) - f
    输入: x (N, 2) tensor, requires_grad=True
    输出: residual (N, 1) tensor
    """
    # 确保梯度计算
    x = x.clone().requires_grad_(True)

    # 计算 u(x)
    u = u_model(x)

    # 计算 ∇u
    u_grad = torch.autograd.grad(
        outputs=u, inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    u_x, u_y = u_grad[:, 0:1], u_grad[:, 1:2]

    # 计算 k(x)
    k = lambda_net(x)

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
    """
    生成边界点
    用于残差计算
    """
    n = n_per_edge
    x1 = np.random.uniform(-1, 1, n)
    x2 = np.random.uniform(-1, 1, n)

    b1 = np.column_stack([x1, -1 * np.ones(n)])  # 下边界
    b2 = np.column_stack([x1, 1 * np.ones(n)])  # 上边界
    b3 = np.column_stack([-1 * np.ones(n), x2])  # 左边界
    b4 = np.column_stack([1 * np.ones(n), x2])  # 右边界

    return np.vstack([b1, b2, b3, b4]).astype(np.float32)


def generate_interior_points(n_interior=2000):
    """
    生成内部配点
    用于残差计算
    """
    return np.random.uniform(-1, 1, (n_interior, 2)).astype(np.float32)


def stage1_train_forward_model(X_obs, u_obs, config):
    """阶段1：训练前向模型 u(x,y)"""
    print("\n" + "=" * 50)
    print("阶段1：训练前向模型 u(x,y)")
    print("=" * 50)

    # 创建模型
    u_model = ForwardUModel(hidden_layers=config['hidden_layers']).to(device)
    optimizer = optim.Adam(u_model.parameters(), lr=config['lr_stage1'], weight_decay=config['weight_decay'])

    # 准备训练数据，生成边界数据（观测数据 + 边界数据）
    X_boundary = generate_boundary_points(n_per_edge=config['n_boundary_per_edge'])
    u_boundary = np.zeros((len(X_boundary), 1), dtype=np.float32)

    # 合并数据
    X_train = np.vstack([X_obs, X_boundary])
    u_train = np.vstack([u_obs, u_boundary])

    print(f"   观测点: {len(X_obs)}, 边界点: {len(X_boundary)}, 总计: {len(X_train)}")

    # 转换为tensor
    X_tensor = torch.FloatTensor(X_train).to(device)
    u_tensor = torch.FloatTensor(u_train).to(device)

    # 训练循环
    u_model.train()
    losses = []

    for epoch in range(config['epochs_stage1']):
        # 随机打乱
        indices = torch.randperm(len(X_tensor))

        total_loss = 0.0
        for i in range(0, len(X_tensor), config['batch_size_stage1']):
            batch_indices = indices[i:i + config['batch_size_stage1']]
            X_batch = X_tensor[batch_indices]
            u_batch = u_tensor[batch_indices]

            optimizer.zero_grad()
            u_pred = u_model(X_batch)
            loss = F.mse_loss(u_pred, u_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_indices)

        avg_loss = total_loss / len(X_tensor)
        losses.append(avg_loss)

        if (epoch + 1) % max(1, config['epochs_stage1'] // 10) == 0:
            print(f"   Epoch {epoch + 1}/{config['epochs_stage1']} | Loss: {avg_loss:.6f}")

    print("✅ 前向模型训练完成")
    return u_model, losses


def stage2_train_lambda_net(u_model, config):
    """阶段2：训练参数网络 k(x,y)"""
    print("\n" + "=" * 50)
    print("阶段2：训练参数网络 k(x,y)")
    print("=" * 50)

    # 冻结前向模型
    u_model.eval()
    for param in u_model.parameters():
        param.requires_grad = False

    # 创建参数网络
    lambda_net = LambdaNet(hidden_layers=config['hidden_layers']).to(device)
    optimizer = optim.Adam(lambda_net.parameters(), lr=config['lr_stage2'])

    # 生成配点
    X_collocation = generate_interior_points(n_interior=config['n_collocation'])
    X_collocation_tensor = torch.FloatTensor(X_collocation).to(device)

    print(f"   配点数: {len(X_collocation)}")

    # 训练循环
    lambda_net.train()
    losses = []

    for epoch in range(config['epochs_stage2']):
        # 随机打乱
        indices = torch.randperm(len(X_collocation_tensor))

        total_loss = 0.0
        for i in range(0, len(X_collocation_tensor), config['batch_size_stage2']):
            batch_indices = indices[i:i + config['batch_size_stage2']]
            X_batch = X_collocation_tensor[batch_indices]

            optimizer.zero_grad()

            # 计算PDE残差
            residual = pde_residual(u_model, lambda_net, X_batch)
            pde_loss = torch.mean(residual ** 2)

            # 正则化项
            k_values = lambda_net(X_batch)
            reg_loss = config['lambda_reg'] * torch.mean(k_values ** 2)

            # 总损失
            loss = pde_loss + reg_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_indices)

        avg_loss = total_loss / len(X_collocation_tensor)
        losses.append(avg_loss)

        if (epoch + 1) % max(1, config['epochs_stage2'] // 10) == 0:
            # 监控k的平均值
            with torch.no_grad():
                k_sample = lambda_net(X_collocation_tensor[:100])
                mean_k = k_sample.mean().item()
            print(f"   Epoch {epoch + 1}/{config['epochs_stage2']} | Loss: {avg_loss:.6f} | Mean k: {mean_k:.4f}")

    print("✅ 参数网络训练完成")
    return lambda_net, losses


def safe_pde_residual(u_model, lambda_net, x):
    """
    安全的PDE残差计算，用于评估阶段
    避免计算图构建问题
    """
    # 临时启用梯度计算
    with torch.enable_grad():
        x = x.clone().requires_grad_(True)

        # 计算 u(x)
        u = u_model(x)

        # 计算 ∇u
        u_grad = torch.autograd.grad(
            outputs=u, inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        u_x, u_y = u_grad[:, 0:1], u_grad[:, 1:2]

        # 计算 k(x)
        k = lambda_net(x)

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


def evaluate_models(u_model, lambda_net, X_obs, u_obs, config):
    """评估模型性能"""
    print("\n" + "=" * 50)
    print("模型评估")
    print("=" * 50)

    u_model.eval()
    lambda_net.eval()

    with torch.no_grad():
        # 在观测点上评估
        X_obs_tensor = torch.FloatTensor(X_obs).to(device)
        u_pred = u_model(X_obs_tensor).cpu().numpy().flatten()
        u_true = u_obs.flatten()

        # 计算误差
        mse = np.mean((u_pred - u_true) ** 2)
        l2_error = np.sqrt(mse)
        relative_l2 = l2_error / (np.sqrt(np.mean(u_true ** 2)) + 1e-8)
        mae = np.mean(np.abs(u_pred - u_true))

        print(f"   MSE: {mse:.6e}")
        print(f"   L2误差: {l2_error:.6e}")
        print(f"   相对L2误差: {relative_l2:.6e}")
        print(f"   MAE: {mae:.6e}")

    # 计算PDE残差（使用安全版本）
    try:
        X_test = generate_interior_points(n_interior=1000)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        residual = safe_pde_residual(u_model, lambda_net, X_test_tensor)
        pde_residual_norm = torch.mean(residual ** 2).item()
        print(f"   PDE残差范数: {pde_residual_norm:.6e}")
    except Exception as e:
        print(f"   PDE残差计算失败: {e}")
        pde_residual_norm = float('nan')


    return {
        'mse': mse,
        'l2_error': l2_error,
        'relative_l2': relative_l2,
        'mae': mae,
        'pde_residual': pde_residual_norm,
        'u_pred': u_pred
    }


def visualize_results(u_model, lambda_net, X_obs, u_obs, u_pred, config,
                      losses_stage1=None, losses_stage2=None):
    """可视化结果"""
    save_dir = Path(config['results_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # 生成网格用于可视化
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X_grid, Y_grid = np.meshgrid(x, y)
    points = np.column_stack([X_grid.ravel(), Y_grid.ravel()]).astype(np.float32)

    with torch.no_grad():
        points_tensor = torch.FloatTensor(points).to(device)
        u_grid = u_model(points_tensor).cpu().numpy().reshape(100, 100)
        k_grid = lambda_net(points_tensor).cpu().numpy().reshape(100, 100)

    u_true = u_obs.flatten()

    # 创建3x3的图形布局
    fig = plt.figure(figsize=(18, 15))

    # 1. 真实解 - 3D散点图 (位置: 1)
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    scatter1 = ax1.scatter(X_obs[:, 0], X_obs[:, 1], u_true.flatten(),
                           c=u_true.flatten(), cmap='coolwarm', s=20)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    ax1.set_title('True Solution (Observation Points)')
    plt.colorbar(scatter1, ax=ax1)

    # 2. 预测解 - 3D散点图 (位置: 2)
    ax2 = fig.add_subplot(3, 3, 2, projection='3d')
    scatter2 = ax2.scatter(X_obs[:, 0], X_obs[:, 1], u_pred.flatten(),
                           c=u_pred.flatten(), cmap='coolwarm', s=20)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
    ax2.set_title('Predicted Solution (Observation Points)')
    plt.colorbar(scatter2, ax=ax2)

    # 3. 识别参数k(x,y) (位置: 3)
    ax3 = fig.add_subplot(3, 3, 3)
    im3 = ax3.imshow(k_grid, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis', aspect='auto')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Identified Parameter k(x,y)')
    plt.colorbar(im3, ax=ax3)

    # 4. 误差分布 (位置: 4)
    ax4 = fig.add_subplot(3, 3, 4)
    error = np.abs(u_pred - u_true)
    sc4 = ax4.scatter(X_obs[:, 0], X_obs[:, 1], c=error, cmap='hot', s=30)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Prediction Error |u_pred - u_true|')
    ax4.set_aspect('equal')
    plt.colorbar(sc4, ax=ax4)

    # 5. 真实vs预测散点图 (位置: 5)
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.scatter(u_true, u_pred, alpha=0.6)
    min_val = min(u_true.min(), u_pred.min())
    max_val = max(u_true.max(), u_pred.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax5.set_xlabel('True u')
    ax5.set_ylabel('Predicted u')
    ax5.set_title('True vs Predicted')
    ax5.grid(True, alpha=0.3)

    # 6. k值分布直方图 (位置: 6)
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.hist(k_grid.ravel(), bins=50, alpha=0.7, color='green', density=True)
    ax6.set_xlabel('k value')
    ax6.set_ylabel('Density')
    ax6.set_title('Distribution of k values')
    ax6.grid(True, alpha=0.3)

    # 7. 阶段1学习曲线 (位置: 7)
    ax7 = fig.add_subplot(3, 3, 7)
    if losses_stage1 is not None:
        ax7.plot(losses_stage1, 'b-', linewidth=1.5, label='Training Loss')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Loss')
        ax7.set_title('Stage 1: Forward Model Training')
        ax7.legend()
        ax7.set_yscale('log')
        ax7.grid(True, alpha=0.3)

        # 标记最终损失
        final_loss = losses_stage1[-1]
        ax7.plot(len(losses_stage1) - 1, final_loss, 'ro', markersize=6,
                 label=f'Final: {final_loss:.4f}')
        ax7.legend()

    # 8. 阶段2学习曲线 (位置: 8)
    ax8 = fig.add_subplot(3, 3, 8)
    if losses_stage2 is not None:
        ax8.plot(losses_stage2, 'r-', linewidth=1.5, label='Training Loss')
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Loss')
        ax8.set_title('Stage 2: Parameter Network Training')
        ax8.legend()
        ax8.set_yscale('log')
        ax8.grid(True, alpha=0.3)

        # 标记最终损失
        final_loss = losses_stage2[-1]
        ax8.plot(len(losses_stage2) - 1, final_loss, 'ro', markersize=6,
                 label=f'Final: {final_loss:.4f}')
        ax8.legend()

    # 9. 性能指标总结 (位置: 9)
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')  # 关闭坐标轴

    # 计算评估指标
    mse = np.mean((u_pred - u_true) ** 2)
    l2_error = np.sqrt(mse)
    relative_l2 = l2_error / (np.sqrt(np.mean(u_true ** 2)) + 1e-8)
    mae = np.mean(np.abs(u_pred - u_true))

    # 在图上显示关键指标
    metrics_text = f"""
        Performance Metrics:
        MSE: {mse:.4e}
        L2 Error: {l2_error:.4e}
        Relative L2: {relative_l2:.4f}
        MAE: {mae:.4e}

        Training Info:
        Stage 1 Epochs: {len(losses_stage1) if losses_stage1 else 0}
        Stage 2 Epochs: {len(losses_stage2) if losses_stage2 else 0}
        Final Stage1 Loss: {losses_stage1[-1] if losses_stage1 else 0:.4e}
        Final Stage2 Loss: {losses_stage2[-1] if losses_stage2 else 0:.4e}
    """

    ax9.text(0.1, 0.9, metrics_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_dir / 'piml_mpa_results.png', dpi=150, bbox_inches='tight')
    print(f"✅ 结果已保存到: {save_dir / 'piml_mpa_results.png'}")
    plt.close()

    # 保存数据
    np.save(save_dir / 'k_field.npy', k_grid)
    np.save(save_dir / 'u_pred.npy', u_pred)


def load_observation_data(data_path):
    """
    加载观测数据
    """
    df = pd.read_excel(data_path)
    print(f"✅ 成功加载数据: {data_path}")
    print(f"   列名: {df.columns.tolist()}, 形状: {df.shape}")

    if {'xi', 'yi', 'u(xi,yi)'}.issubset(df.columns):
        X = df[['xi', 'yi']].values.astype(np.float32)
        u = df['u(xi,yi)'].values.astype(np.float32).reshape(-1, 1)
    else:
        X = df.iloc[:, :2].values.astype(np.float32)
        u = df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)

    print(f"   样本数: {len(X)}, u 范围: [{u.min():.4f}, {u.max():.4f}]")
    return X, u
