"""
Helmholtz方程求解器（子任务1&2）
求解：-Δu - k²u = q 在 [-1,1]×[-1,1] 上，边界条件 u=0
"""

import torch
import torch.nn as nn
import numpy as np
from pinn_solvers.base_pinn import BasePINN, create_dataloader
from typing import Dict
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HelmholtzPINN(BasePINN):
    """Helmholtz方程的PINN求解器，支持 standard / mffm 网络"""
    
    def __init__(self, config: Dict):
        # 根据波数自动选择网络类型
        k = config.get('wavenumber', 4)
        self.k = k
        if 'network_type' not in config:
            config['network_type'] = 'mffm' if k >= 100 else 'standard'

        #选择好网络类型后在父类中自动配置
        super().__init__(config)
        self.to(device)
        print(f"创建 HelmholtzPINN: k={self.k}, 网络类型={config['network_type']}")
        
    def pde_loss(self, x, u_pred):
        """
        计算PDE残差: -Δu - k²u - q = 0
        
        需要计算：
        1. u_xx (x的二阶偏导)
        2. u_yy (y的二阶偏导)
        3. Δu = u_xx + u_yy
        """
        # 开启梯度追踪
        x.requires_grad_(True)
        u_pred = self.forward(x)
        
        # 计算一阶偏导
        u_x = torch.autograd.grad(
            outputs=u_pred, inputs=x,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True
        )[0][:, 0:1]
        
        u_y = torch.autograd.grad(
            outputs=u_pred, inputs=x,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True
        )[0][:, 1:2]
        
        # 计算二阶偏导 (u_xx, u_yy)
        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0][:, 0:1]
        
        u_yy = torch.autograd.grad(
            outputs=u_y, inputs=x,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True
        )[0][:, 1:2]
        
        # 拉普拉斯算子
        laplacian = u_xx + u_yy
        
        # 计算源项 q(x,y)
        # q(x,y) = -(a₁π)²sin(a₁πx)sin(a₂πy) - (a₂π)²sin(a₁πx)sin(a₂πy) - k²sin(a₁πx)sin(a₂πy)
        a1, a2 = 1, 3
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        
        sin_a1x = torch.sin(a1 * np.pi * x_coord)
        sin_a2y = torch.sin(a2 * np.pi * y_coord)
        
        q = -((a1 * np.pi) ** 2 + (a2 * np.pi) ** 2 + self.k ** 2) * sin_a1x * sin_a2y
        
        # PDE残差
        pde_residual = -laplacian - self.k ** 2 * u_pred - q
        
        # 返回MSE损失
        return torch.mean(pde_residual ** 2)


def train_helmholtz_pinn(model: HelmholtzPINN, X_interior, X_boundary, 
                         X_test, u_test_true,
                         optimizer=None,
                         n_epochs=5000, batch_size=128,
                         log_every=500, save_dir='./results'):
    """
    训练Helmholtz PINN
    
    Args:
        model: HelmholtzPINN模型
        X_interior: 内部点 (n_interior, 2)
        X_boundary: 边界点 (n_boundary, 2)
        X_test: 测试点 (n_test, 2)
        u_test_true: 真实解（测试点）(n_test,)
        n_epochs: 训练轮数
        batch_size: 批次大小
        log_every: 每多少轮打印一次
        save_dir: 保存目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 边界条件：u=0
    u_boundary = np.zeros(len(X_boundary))
    
    # 创建数据加载器
    dataloader = create_dataloader(X_interior, X_boundary, u_boundary, batch_size=batch_size)
    
    # 优化器
    if optimizer is None:
        from torch.optim import Adam
        optimizer = Adam(model.net.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)

    best_test_loss = float('inf')

    # 训练循环
    print(f"\n开始训练Helmholtz PINN (k={model.k})，优化器: 自定义")
    print(f"内部点: {len(X_interior)}, 边界点: {len(X_boundary)}")
    print(f"{'='*60}")

    # # ======== 训练前 forward 输出 ========
    # # 将测试点转换为 tensor
    # X_test_tensor = torch.FloatTensor(X_test).to(model.device)
    # with torch.no_grad():
    #     u_test_pred = model.forward(X_test_tensor)
    #     print("训练前 forward (前5个样本):", u_test_pred[:5])
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        # 遍历批次
        for batch_dict in dataloader:
            # 转移到设备
            for key in batch_dict:
                if torch.is_tensor(batch_dict[key]):
                    batch_dict[key] = batch_dict[key].to(model.device)
            
            # 训练一步
            loss = model.train_step(batch_dict, optimizer)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(dataloader)
        model.train_loss_history.append(avg_loss)
        
        # 测试
        if (epoch + 1) % log_every == 0 or epoch == 0:
            # 将测试点转换为tensor
            X_test_tensor = torch.FloatTensor(X_test).to(model.device)
            with torch.no_grad():
                u_test_pred = model.forward(X_test_tensor).cpu().numpy().flatten()

            # # 训练中 ==========
            # print(f"\nEpoch {epoch + 1} forward (前5个样本):", u_test_pred[:5])
            # print(f"u_pred max/min: {np.max(u_test_pred):.6f}/{np.min(u_test_pred):.6f}")

            test_loss = np.sqrt(np.mean((u_test_pred - u_test_true) ** 2))
            relative_error = test_loss / (np.sqrt(np.mean(u_test_true ** 2)) + 1e-10)
            
            model.test_loss_history.append(relative_error)
            
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss: {avg_loss:.6e} | "
                  f"Test Rel L2: {relative_error:.6e}")
            
            # 保存最佳模型
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                model.save_checkpoint(
                    os.path.join(save_dir, f'best_model_k{model.k}.pth'),
                    epoch, optimizer.state_dict()
                )
        else:
            # 即使不测试也记录损失
            model.test_loss_history.append(model.test_loss_history[-1] if model.test_loss_history else 1.0)

    # # ======== 训练后 forward 输出 ========
    # with torch.no_grad():
    #     u_test_pred = model.forward(X_test_tensor)
    #     print("训练后 forward (前5个样本):", u_test_pred[:5])

    print(f"\n训练完成！最佳测试误差: {best_test_loss:.6e}")
    return model


if __name__ == "__main__":
    # 测试代码
    print("测试Helmholtz求解器...")
    
    # 加载数据
    import sys
    sys.path.append('.')
    from data.data_loader import HelmholtzDataLoader
    
    loader = HelmholtzDataLoader('../data/子任务2_亥姆霍兹方程数据_k100.xlsx', task='task2', k=100)
    X_interior, _, _ = loader.get_training_points()
    X_test, _ = loader.get_test_points()
    
    print(f"测试数据加载成功: 内部点={len(X_interior)}, 测试点={len(X_test)}")
    
    # 创建模型
    config = {
        'network_type': 'r_gpinn',
        'layers': [2, 50, 50, 50, 1],
        'activation': 'tanh',
        'learning_rate': 0.001,
        'wavenumber': 100,
        'lambda_pde': 1.0,
        'lambda_bc': 1.0
    }

    #创建模型时输出波数以及选择的网络类型
    model = HelmholtzPINN(config)

    print(f"模型创建成功，参数数量: {sum(p.numel() for p in model.net.parameters())}")

