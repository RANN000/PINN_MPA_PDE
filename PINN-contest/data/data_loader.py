"""
Data Loader for PINN Contest
支持加载三个子任务的数据：Helmholtz k=4, 高波数Helmholtz, Poisson反问题
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List


class HelmholtzDataLoader:
    """Helmholtz方程数据加载器（子任务1&2）"""
    
    def __init__(self, data_path: str, task: str = "task1", k: int = 4):
        """
        Args:
            data_path: 数据文件路径
            task: "task1" (k=4) 或 "task2" (高波数)
            k: 波数（4, 100, 500, 1000）
        """
        self.data_path = Path(data_path)
        self.task = task
        self.k = k
        self.data = {}
        
    def load_excel_data(self) -> Dict[str, pd.DataFrame]:
        """加载Excel文件中的多个sheet"""
        xls = pd.ExcelFile(self.data_path)
        
        data_dict = {}
        for sheet_name in xls.sheet_names:
            data_dict[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
            print(f"Loaded {sheet_name}: {len(data_dict[sheet_name])} points")
        
        self.data = data_dict
        return data_dict
    
    def get_training_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取训练点：内部点、边界点、以及对应的源项"""
        if not self.data:
            self.load_excel_data()
        
        # 合并内部点和边界点
        interior = self.data.get('内部点', pd.DataFrame())
        boundary = self.data.get('边界点', pd.DataFrame())
        
        # 合并所有训练点
        train_points = pd.concat([interior, boundary], ignore_index=True)
        
        X = train_points[['x', 'y']].values  # 坐标
        q = train_points['q'].values  # 源项
        
        # 标签（边界点u=0，内部点无标签）
        labels = np.zeros(len(train_points))
        boundary_mask = np.zeros(len(train_points), dtype=bool)
        boundary_mask[len(interior):] = True
        
        return X, q, boundary_mask
    
    def get_test_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取测试点"""
        if not self.data:
            self.load_excel_data()
        
        test_grid = self.data.get('测试网格', pd.DataFrame())
        
        if test_grid.empty:
            # 如果测试网格不存在，从训练数据汇总中取验证集
            summary = self.data.get('训练数据汇总', pd.DataFrame())
            test_size = min(500, len(summary) // 5)
            test_grid = summary.sample(test_size, random_state=42)
        
        X_test = test_grid[['x', 'y']].values
        q_test = test_grid['q'].values if 'q' in test_grid.columns else np.zeros(len(X_test))
        
        return X_test, q_test


class PoissonDataLoader:
    """Poisson反问题数据加载器（子任务3）"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load_data(self) -> Dict[str, np.ndarray]:
        """加载观测数据"""
        df = pd.read_excel(self.data_path)
        
        print(f"   文件列名: {df.columns.tolist()}")
        print(f"   Shape: {df.shape}")
        
        # 根据实际列名加载数据
        col_names = df.columns.tolist()
        
        # 子任务3的列名: ['xi', 'yi', 'u(xi,yi)']
        if 'xi' in col_names and 'yi' in col_names:
            X = df[['xi', 'yi']].values
            u_obs = df['u(xi,yi)'].values
        elif 'x' in col_names and 'y' in col_names:
            X = df[['x', 'y']].values
            u_obs = df.get('u', df.iloc[:, -1]).values
        else:
            # 默认前两列是x,y，最后一列是u
            X = df.iloc[:, :2].values
            u_obs = df.iloc[:, -1].values if df.shape[1] > 2 else np.zeros(len(df))
        
        return {
            'X': X,
            'u_obs': u_obs
        }
    
    def generate_collocation_points(self, n_interior: int = 2000, 
                                    n_boundary: int = 400) -> np.ndarray:
        """生成用于PDE残差计算的配点"""
        # 在[-1, 1] x [-1, 1]域内采样
        X_interior = np.random.uniform(-1, 1, (n_interior, 2))
        
        # 边界点采样
        n_bc_per_edge = n_boundary // 4
        X_boundary = []
        
        # 四条边
        X_boundary.append(np.column_stack([np.random.uniform(-1, 1, n_bc_per_edge), -1 + np.zeros(n_bc_per_edge)]))  # y=-1
        X_boundary.append(np.column_stack([np.random.uniform(-1, 1, n_bc_per_edge), 1 + np.zeros(n_bc_per_edge)]))   # y=1
        X_boundary.append(np.column_stack([-1 + np.zeros(n_bc_per_edge), np.random.uniform(-1, 1, n_bc_per_edge)]))  # x=-1
        X_boundary.append(np.column_stack([1 + np.zeros(n_bc_per_edge), np.random.uniform(-1, 1, n_bc_per_edge)]))    # x=1
        
        X_boundary = np.vstack(X_boundary)
        
        return np.vstack([X_interior, X_boundary])


def visualize_data_distribution(data_loader: HelmholtzDataLoader):
    """可视化数据分布"""
    X, q, boundary_mask = data_loader.get_training_points()
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 散点图
    axes[0].scatter(X[~boundary_mask, 0], X[~boundary_mask, 1], 
                    c='blue', s=5, alpha=0.5, label='Interior')
    axes[0].scatter(X[boundary_mask, 0], X[boundary_mask, 1], 
                    c='red', s=5, alpha=0.5, label='Boundary')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Data Point Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 源项分布
    axes[1].scatter(X[:, 0], X[:, 1], c=q, cmap='coolwarm', s=5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Source Term q(x,y)')
    plt.colorbar(axes[1].collections[0], ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('数据分布_可视化.png', dpi=150)
    print("Data visualization saved to data/数据分布_可视化.png")
    plt.close()


if __name__ == "__main__":
    # 测试数据加载
    print("=" * 50)
    print("Testing Data Loader")
    print("=" * 50)
    
    # 子任务1测试
    print("\n1. Task 1 (Helmholtz k=4):")
    loader1 = HelmholtzDataLoader('子任务1_亥姆霍兹方程数据_k4.xlsx', task='task1', k=4)
    data1 = loader1.load_excel_data()
    
    X_train, q_train, boundary_mask = loader1.get_training_points()
    print(f"   Training points: {len(X_train)} (interior: {np.sum(~boundary_mask)}, boundary: {np.sum(boundary_mask)})")
    
    X_test, q_test = loader1.get_test_points()
    print(f"   Test points: {len(X_test)}")
    
    # 可视化
    visualize_data_distribution(loader1)
    
    print("\n2. Task 2 (High Wavenumber k=100):")
    loader2 = HelmholtzDataLoader('子任务2_亥姆霍兹方程数据_k100.xlsx', task='task2', k=100)
    data2 = loader2.load_excel_data()
    X_train2, _, _ = loader2.get_training_points()
    print(f"   Training points: {len(X_train2)}")
    
    print("\n3. Task 3 (Poisson Inverse):")
    loader3 = PoissonDataLoader('子任务3数据.xlsx')
    data3 = loader3.load_data()
    print(f"   Observation points: {len(data3['X'])}")
    
    print("\n" + "=" * 50)
    print("Data loading test completed!")
    print("=" * 50)

