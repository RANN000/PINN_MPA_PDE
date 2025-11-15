"""
Poisson反问题求解器（子任务3）
求解：-Δu = λf 在 [-1,1]×[-1,1] 上
任务：给定200个观测点，识别参数λ
"""

import torch
import torch.nn as nn
import numpy as np
from pinn_solvers.base_pinn import BasePINN, create_dataloader
from typing import Dict


class PoissonInversePINN(BasePINN):
    """Poisson反问题的PINN求解器（双网络架构）"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典
                - layers_u: u网络的层数 [2, 50, 50, 1]
                - layers_lambda: lambda网络的层数 [2, 20, 20, 1] 或标量
                - activation: 激活函数
                - lambda_pde: PDE损失权重
                - lambda_data: 数据拟合损失权重
                - reg_lambda: lambda的正则化系数
        """
        # 先调用父类，但不传layers（我们将自定义）
        config_custom = config.copy()
        config_custom['layers'] = config.get('layers_u', [2, 50, 50, 50, 1])
        
        super().__init__(config_custom)
        print(f'network_type: {config.get("network_type")}')
        
        # lambda网络（学习参数λ）
        layers_lambda = config.get('layers_lambda', [2, 20, 20, 1])
        
        # 如果lambda是标量（常数），使用nn.Parameter
        # 如果是空间变化的，使用网络
        self.lambda_type = config.get('lambda_type', 'scalar')  # 'scalar' or 'spatial'
        
        if self.lambda_type == 'scalar':
            # 标量参数
            self.lambda_param = nn.Parameter(torch.tensor(1.0))
            self.lambda_net = None
        else:
            # 空间变化的参数（使用网络）
            from pinn_solvers.base_pinn import MLP
            self.lambda_net = MLP(layers_lambda, activation=config.get('activation', 'tanh'))
            self.lambda_net.to(self.device)
            self.lambda_param = None
        
        self.reg_lambda = config.get('reg_lambda', 0.01)
        
    def get_lambda(self, x):
        """获取参数k(x,y)的值"""
        if self.lambda_type == 'scalar':
            return self.lambda_param
        else:
            k_raw = self.lambda_net(x)
            return torch.nn.functional.softplus(k_raw)
    
    def pde_loss(self, x, u_pred):
        """
         计算PDE残差: -∇·(k∇u) - f = 0
        正确形式：-∇·(k(x,y)∇u(x,y)) = f(x,y)
        """

        x.requires_grad_(True)
        u_pred = self.forward(x)

        # 计算u的梯度 ∇u
        u_grad = torch.autograd.grad(
            outputs=u_pred, inputs=x,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True, retain_graph=True
        )[0]

        u_x, u_y = u_grad[:, 0:1], u_grad[:, 1:2]

        # 获取k(x,y)
        k = self.get_lambda(x)

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

        # 计算源项 f(x,y)
        # 根据题目，源项应该是给定的函数
        # 这里需要根据实际题目补充源项的具体形式
        pi = torch.pi
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        f = (pi ** 2 / 2) * (1 + x_coord ** 2 + y_coord ** 2) * torch.sin(pi * x_coord / 2) * torch.cos(pi * y_coord / 2) \
            - pi * x_coord * torch.cos(pi * x_coord / 2) * torch.cos(pi * y_coord / 2) \
            + pi * y_coord * torch.sin(pi * x_coord / 2) * torch.sin(pi * y_coord / 2)

        # PDE残差：-∇·(k∇u) - f = 0
        pde_residual = -divergence - f

        # MSE损失
        return torch.mean(pde_residual ** 2)
    
    def compute_total_loss(self, batch_dict: Dict) -> torch.Tensor:
        """计算总损失（包括正则化项和梯度增强项）"""
        loss = super().compute_total_loss(batch_dict)
        
        # Lambda正则化项（L2正则化，防止过拟合）
        if self.lambda_type == 'scalar':
            lambda_reg = self.reg_lambda * (self.lambda_param - 1.0) ** 2
        else:
            # 对于空间变化的lambda，计算L2正则化
            if 'x_interior' in batch_dict:
                x_sample = batch_dict['x_interior'][:10]
            else:
                x_sample = torch.randn(10, 2).to(self.device)
            lambda_vals = self.lambda_net(x_sample)
            lambda_reg = self.reg_lambda * torch.mean(lambda_vals ** 2)
        
        loss += lambda_reg
        
        # 梯度增强正则化（文献6创新点）
        # 约束lambda的空间梯度，保证平滑性
        if self.lambda_type != 'scalar' and 'x_interior' in batch_dict:
            x_grad = batch_dict['x_interior'][:100].clone()  # 采样部分点计算梯度
            x_grad.requires_grad_(True)
            lambda_grad = self.lambda_net(x_grad)
            
            # 计算lambda的梯度
            lambda_x = torch.autograd.grad(
                outputs=lambda_grad, inputs=x_grad,
                grad_outputs=torch.ones_like(lambda_grad),
                create_graph=True,
                retain_graph=True
            )[0]
            
            # 梯度正则化：||∇λ||²
            gradient_reg = self.reg_lambda * 0.1 * torch.mean(lambda_x ** 2)
            loss += gradient_reg
        
        return loss
    
    def get_lambda_value(self) -> float:
        """获取参数λ的值（用于评估）"""
        if self.lambda_type == 'scalar':
            return self.lambda_param.item()
        else:
            # 返回平均值
            x_sample = torch.randn(100, 2).to(self.device)
            with torch.no_grad():
                lambda_vals = self.lambda_net(x_sample).cpu().numpy()
            return float(np.mean(lambda_vals))


def train_poisson_inverse_pinn(model: PoissonInversePINN, 
                               X_interior, X_boundary, X_data, u_data,
                               n_epochs=3000, batch_size=128,
                               log_every=100, save_dir='./results'):
    """
    训练Poisson反问题PINN
    
    Args:
        model: PoissonInversePINN模型
        X_interior: 内部配点
        X_boundary: 边界点
        X_data: 观测数据点 (200个)
        u_data: 观测数据值
        n_epochs: 训练轮数
        batch_size: 批次大小
        log_every: 每多少轮打印一次
        save_dir: 保存目录
    """
    import os
    from torch.optim import Adam
    os.makedirs(save_dir, exist_ok=True)
    
    # 边界条件：u=0（假设）
    u_boundary = np.zeros(len(X_boundary))
    
    # 创建数据加载器
    dataloader = create_dataloader(
        X_interior, X_boundary, u_boundary, 
        X_data, u_data, batch_size=batch_size
    )
    
    # 优化器
    optimizer = Adam(model.net.parameters(), lr=model.learning_rate)
    
    # 如果lambda是网络，也添加其参数
    if model.lambda_net is not None:
        optimizer = Adam(
            list(model.net.parameters()) + list(model.lambda_net.parameters()),
            lr=model.learning_rate
        )
    elif model.lambda_param is not None:
        optimizer = Adam(
            list(model.net.parameters()) + [model.lambda_param],
            lr=model.learning_rate
        )
    
    print(f"\n开始训练Poisson反问题PINN")
    print(f"配点: {len(X_interior)}, 边界点: {len(X_boundary)}, 观测点: {len(X_data)}")
    print(f"{'='*60}")
    
    best_loss = float('inf')
    
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
            lambda_val = model.get_lambda_value()
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Loss: {avg_loss:.6e} | "
                  f"λ: {lambda_val:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                model.save_checkpoint(
                    os.path.join(save_dir, 'best_model_poisson_inverse.pth'),
                    epoch, optimizer.state_dict()
                )
    
    print(f"\n训练完成！最佳损失: {best_loss:.6e}")
    lambda_final = model.get_lambda_value()
    print(f"识别的参数 λ: {lambda_final:.6f}")
    
    return model


if __name__ == "__main__":
    # 测试代码
    print("测试Poisson反问题求解器...")
    
    config = {
        'network_type': 'r_gpinn',
        'layers_u': [2, 50, 50, 50, 1],
        'layers_lambda': [2, 20, 1],
        'activation': 'tanh',
        'learning_rate': 0.001,
        'lambda_type': 'spatial',  # 标量参数
        'lambda_pde': 1.0,
        'lambda_data': 10.0,
        'reg_lambda': 0.01
    }
    
    model = PoissonInversePINN(config)
    print(f"模型创建成功，参数数量: {sum(p.numel() for p in model.net.parameters())}")
    print(f"Lambda类型: {model.lambda_type}")

