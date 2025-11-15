"""
Helmholtz方程求解器（子任务1&2）
求解：-Δu - k²u = q 在 [-1,1]×[-1,1] 上，边界条件 u=0
"""

import torch
import torch.nn as nn
import numpy as np
from pinn_solvers.base_pinn import BasePINN, create_dataloader
from typing import Dict, Tuple, List
from pathlib import Path
import matplotlib.pyplot as plt
import os

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


class LearningRateScheduler:
    """学习率退火调度器"""

    def __init__(self, optimizer, scheduler_type='cosine', initial_lr=0.001,
                 min_lr=1e-6, T_max=1000, patience=100, decay_rate=0.5):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.T_max = T_max
        self.patience = patience
        self.decay_rate = decay_rate
        self.no_improvement_count = 0
        self.best_loss = float('inf')

        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=min_lr)
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=patience, factor=decay_rate,
                min_lr=min_lr)
        elif scheduler_type == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=decay_rate)
        else:
            self.scheduler = None

    def step(self, loss=None):
        """更新学习率"""
        if self.scheduler_type == 'plateau' and loss is not None:
            self.scheduler.step(loss)
        elif self.scheduler is not None:
            self.scheduler.step()

    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


def train_helmholtz_pinn_improved(
        model: HelmholtzPINN,
        X_interior: np.ndarray,
        X_boundary: np.ndarray,
        X_test: np.ndarray,
        u_test_true: np.ndarray,
        optimizer=None,
        n_epochs: int = 5000,
        batch_size: int = 128,
        log_every: int = 100,
        save_dir: str = '../results',
        initial_checkpoint: str = None,
        use_lr_scheduler: bool = True,
        scheduler_type: str = 'plateau',
        early_stopping_patience: int = 500,
        min_delta: float = 1e-6
) -> Tuple[HelmholtzPINN, Dict]:
    """
    改进的Helmholtz PINN训练函数

    Args:
        model: HelmholtzPINN模型
        X_interior: 内部点 (n_interior, 2)
        X_boundary: 边界点 (n_boundary, 2)
        X_test: 测试点 (n_test, 2)
        u_test_true: 真实解（测试点）(n_test,)
        initial_checkpoint: 初始检查点路径，用于继续训练
        use_lr_scheduler: 是否使用学习率退火
        scheduler_type: 退火类型 ['plateau', 'cosine', 'exponential']
        early_stopping_patience: 早停耐心值
        min_delta: 最小改进阈值

    Returns:
        model: 训练好的模型
        training_info: 训练信息字典
    """

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 边界条件：u=0
    u_boundary = np.zeros(len(X_boundary))

    # 从检查点继续训练
    start_epoch = 0
    if initial_checkpoint and os.path.exists(initial_checkpoint):
        start_epoch = model.load_checkpoint(initial_checkpoint)
        print(f"从检查点继续训练: {initial_checkpoint}, 起始轮次: {start_epoch}")

    # 创建数据加载器
    dataloader = create_dataloader(X_interior, X_boundary, u_boundary, batch_size=batch_size)

    # 优化器
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.net.parameters(),
            lr=model.learning_rate,
            weight_decay=model.weight_decay
        )

    # 学习率调度器
    lr_scheduler = None
    if use_lr_scheduler:
        lr_scheduler = LearningRateScheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            initial_lr=model.learning_rate,
            min_lr=1e-6,
            T_max=n_epochs,
            patience=100,
            decay_rate=0.5
        )

    # 早停和最佳模型跟踪
    best_l2_error = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # 训练循环
    print(f"\n开始训练Helmholtz PINN (k={model.k})")
    print(f"内部点: {len(X_interior)}, 边界点: {len(X_boundary)}")
    print(f"使用学习率调度器: {use_lr_scheduler} ({scheduler_type})")
    print(f"早停耐心: {early_stopping_patience}")
    print(f"{'=' * 60}")

    for epoch in range(start_epoch, n_epochs):
        model.net.train()
        epoch_total_loss = 0.0
        epoch_pde_loss = 0.0
        epoch_bc_loss = 0.0

        # 遍历批次
        for batch_dict in dataloader:
            # 转移到设备
            for key in batch_dict:
                if torch.is_tensor(batch_dict[key]):
                    batch_dict[key] = batch_dict[key].to(model.device)

            # 训练一步，获取损失分量
            total_loss, loss_components = model.train_step_improved(batch_dict, optimizer)
            epoch_total_loss += total_loss

            # 记录损失分量
            if 'pde_loss' in loss_components:
                epoch_pde_loss += loss_components['pde_loss'].item()
            if 'bc_loss' in loss_components:
                epoch_bc_loss += loss_components['bc_loss'].item()

        # 计算平均损失
        num_batches = len(dataloader)
        avg_total_loss = epoch_total_loss / num_batches
        avg_pde_loss = epoch_pde_loss / num_batches
        avg_bc_loss = epoch_bc_loss / num_batches

        # 记录训练损失
        model.train_loss_history.append(avg_total_loss)
        model.pde_loss_history.append(avg_pde_loss)
        model.bc_loss_history.append(avg_bc_loss)

        # 定期评估和日志
        if (epoch + 1) % log_every == 0 or epoch == start_epoch:
            model.net.eval()

            # 计算测试误差
            X_test_tensor = torch.FloatTensor(X_test).to(model.device)
            with torch.no_grad():
                u_test_pred = model.forward(X_test_tensor).cpu().numpy().flatten()

            l2_error = np.sqrt(np.mean((u_test_pred - u_test_true) ** 2))
            relative_l2_error = l2_error / (np.sqrt(np.mean(u_test_true ** 2)) + 1e-10)

            model.test_loss_history.append(relative_l2_error)
            model.l2_error_history.append(l2_error)

            # 获取当前学习率
            current_lr = lr_scheduler.get_lr() if lr_scheduler else model.learning_rate

            print(f"Epoch {epoch + 1}/{n_epochs} | "
                  f"Total: {avg_total_loss:.3e} | "
                  f"PDE: {avg_pde_loss:.3e} | "
                  f"BC: {avg_bc_loss:.3e} | "
                  f"L2: {l2_error:.3e} | "
                  f"Rel L2: {relative_l2_error:.3e} | "
                  f"LR: {current_lr:.2e}")

            # 检查是否为最佳模型
            if l2_error < best_l2_error - min_delta:
                best_l2_error = l2_error
                best_epoch = epoch
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.net.state_dict().copy(),
                    'optimizer_state_dict': optimizer.state_dict().copy(),
                    'train_loss_history': model.train_loss_history.copy(),
                    'test_loss_history': model.test_loss_history.copy(),
                    'pde_loss_history': model.pde_loss_history.copy(),
                    'bc_loss_history': model.bc_loss_history.copy(),
                    'l2_error_history': model.l2_error_history.copy()
                }

                # 保存最佳模型检查点
                model.save_checkpoint(
                    os.path.join(save_dir, f'best_model_k{model.k}.pth'),
                    epoch, optimizer.state_dict()
                )
            else:
                patience_counter += 1

            # 学习率调度
            if lr_scheduler:
                if scheduler_type == 'plateau':
                    lr_scheduler.step(l2_error)
                else:
                    lr_scheduler.step()

        # 早停检查
        if patience_counter >= early_stopping_patience:
            print(f"\n早停在 epoch {epoch + 1}, 连续 {patience_counter} 轮未改进")
            print(f"最佳 L2 误差: {best_l2_error:.6e} (epoch {best_epoch + 1})")
            break

    # 恢复最佳模型
    if best_model_state is not None:
        model.net.load_state_dict(best_model_state['model_state_dict'])
        model.train_loss_history = best_model_state['train_loss_history']
        model.test_loss_history = best_model_state['test_loss_history']
        model.pde_loss_history = best_model_state['pde_loss_history']
        model.bc_loss_history = best_model_state['bc_loss_history']
        model.l2_error_history = best_model_state['l2_error_history']
        print(f"已恢复到最佳模型 (epoch {best_epoch + 1})")

    # 绘制损失曲线
    plot_training_losses(model, save_dir)

    training_info = {
        'best_epoch': best_epoch,
        'best_l2_error': best_l2_error,
        'final_epoch': epoch
    }

    return model, training_info


def plot_training_losses(model: HelmholtzPINN, save_dir: str):
    """绘制训练损失曲线"""
    plt.figure(figsize=(15, 10))

    # 总损失
    plt.subplot(2, 3, 1)
    plt.semilogy(model.train_loss_history)
    plt.title('Total Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # PDE损失
    plt.subplot(2, 3, 2)
    plt.semilogy(model.pde_loss_history)
    plt.title('PDE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # 边界损失
    plt.subplot(2, 3, 3)
    plt.semilogy(model.bc_loss_history)
    plt.title('Boundary Condition Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # 测试相对L2误差
    plt.subplot(2, 3, 4)
    plt.semilogy(model.test_loss_history)
    plt.title('Test Relative L2 Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)

    # L2误差
    plt.subplot(2, 3, 5)
    plt.semilogy(model.l2_error_history)
    plt.title('Test L2 Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)

    # 损失分量对比
    plt.subplot(2, 3, 6)
    epochs = range(len(model.train_loss_history))
    plt.semilogy(epochs, model.pde_loss_history, label='PDE Loss')
    plt.semilogy(epochs, model.bc_loss_history, label='BC Loss')
    plt.semilogy(epochs, model.train_loss_history, label='Total Loss', linestyle='--')
    plt.title('Loss Components Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_losses_k{model.k}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"训练损失曲线已保存至: {os.path.join(save_dir, f'training_losses_k{model.k}.png')}")


def continue_training_from_checkpoint(
        checkpoint_path: str,
        config: Dict,
        X_interior: np.ndarray,
        X_boundary: np.ndarray,
        X_test: np.ndarray,
        u_test_true: np.ndarray,
        additional_epochs: int = 2000,
        **train_kwargs
        ) -> Tuple[HelmholtzPINN, Dict]:
    """
    从检查点继续训练

    Args:
        checkpoint_path: 检查点路径
        config: 模型配置
        additional_epochs: 额外训练的轮数
        **train_kwargs: 传递给train_helmholtz_pinn_improved的其他参数
    """

    # 创建新模型
    model = HelmholtzPINN(config)

    # 设置总训练轮数为当前轮次 + 额外轮次
    if 'n_epochs' not in train_kwargs:
        train_kwargs['n_epochs'] = additional_epochs

    # 从检查点继续训练
    model, training_info = train_helmholtz_pinn_improved(
        model=model,
        X_interior=X_interior,
        X_boundary=X_boundary,
        X_test=X_test,
        u_test_true=u_test_true,
        initial_checkpoint=checkpoint_path,
        **train_kwargs
    )

    return model, training_info


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
        'network_type': 'mffm',
        'layers': [2, 40, 40, 40, 1],
        'activation': 'tanh',
        'learning_rate': 0.001,
        'wavenumber': 100,
        'lambda_pde': 0.5,
        'lambda_bc': 50
    }

    #创建模型时输出波数以及选择的网络类型
    model = HelmholtzPINN(config)

    print(f"模型创建成功，参数数量: {sum(p.numel() for p in model.net.parameters())}")

