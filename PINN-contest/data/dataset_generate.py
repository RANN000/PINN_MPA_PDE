import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class HelmholtzDataGenerator:
    def __init__(self, a1=1, a2=3, k=4):
        """
        初始化亥姆霍兹方程参数
        Args:
            a1, a2: 源项参数
            k: 波数
        """
        self.a1 = a1
        self.a2 = a2
        self.k = k

    def source_term(self, x, y):
        """
        计算亥姆霍兹方程的源项 q(x,y)
        按照题目给出的公式
        """
        term1 = -(self.a1 * np.pi) ** 2 * np.sin(self.a1 * np.pi * x) * np.sin(self.a2 * np.pi * y)
        term2 = -(self.a2 * np.pi) ** 2 * np.sin(self.a1 * np.pi * x) * np.sin(self.a2 * np.pi * y)
        term3 = -self.k ** 2 * np.sin(self.a1 * np.pi * x) * np.sin(self.a2 * np.pi * y)
        return term1 + term2 + term3

    def generate_interior_points(self, n_points=1000, seed=42):
        """
        生成内部训练点
        Args:
            n_points: 内部点数量
            seed: 随机种子
        """
        np.random.seed(seed)
        x = np.random.uniform(-1, 1, n_points)
        y = np.random.uniform(-1, 1, n_points)
        q = self.source_term(x, y)

        interior_data = pd.DataFrame({
            'x': x,
            'y': y,
            'q': q
        })
        return interior_data

    def generate_boundary_points(self, n_points_per_edge=100, seed=42):
        """
        生成边界训练点
        Args:
            n_points_per_edge: 每条边的点数
            seed: 随机种子
        """
        np.random.seed(seed)

        # 四条边界：x=-1, x=1, y=-1, y=1
        boundary_points = []

        # 左边界 x = -1
        y_left = np.random.uniform(-1, 1, n_points_per_edge)
        x_left = np.full_like(y_left, -1)
        boundary_points.append((x_left, y_left))

        # 右边界 x = 1
        y_right = np.random.uniform(-1, 1, n_points_per_edge)
        x_right = np.full_like(y_right, 1)
        boundary_points.append((x_right, y_right))

        # 下边界 y = -1
        x_bottom = np.random.uniform(-1, 1, n_points_per_edge)
        y_bottom = np.full_like(x_bottom, -1)
        boundary_points.append((x_bottom, y_bottom))

        # 上边界 y = 1
        x_top = np.random.uniform(-1, 1, n_points_per_edge)
        y_top = np.full_like(x_top, 1)
        boundary_points.append((x_top, y_top))

        # 合并所有边界点
        x_boundary = np.concatenate([points[0] for points in boundary_points])
        y_boundary = np.concatenate([points[1] for points in boundary_points])
        q_boundary = self.source_term(x_boundary, y_boundary)

        boundary_data = pd.DataFrame({
            'x': x_boundary,
            'y': y_boundary,
            'q': q_boundary
        })
        return boundary_data

    def generate_test_grid(self, n_grid=50):
        """
        生成测试网格点（用于验证和可视化）
        Args:
            n_grid: 网格点数（每边）
        """
        x = np.linspace(-1, 1, n_grid)
        y = np.linspace(-1, 1, n_grid)
        X, Y = np.meshgrid(x, y)

        test_data = pd.DataFrame({
            'x': X.flatten(),
            'y': Y.flatten(),
            'q': self.source_term(X.flatten(), Y.flatten())
        })
        return test_data

    def save_to_excel(self, filename, n_interior=2000, n_boundary_per_edge=100, n_test_grid=50):
        """
        将所有数据保存到Excel文件的不同sheet中
        """
        print(f"生成亥姆霍兹方程数据: a1={self.a1}, a2={self.a2}, k={self.k}")

        # 生成数据
        interior_data = self.generate_interior_points(n_interior)
        boundary_data = self.generate_boundary_points(n_boundary_per_edge)
        test_data = self.generate_test_grid(n_test_grid)

        # 创建Excel写入器
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 保存到不同sheet
            interior_data.to_excel(writer, sheet_name='内部点', index=False)
            boundary_data.to_excel(writer, sheet_name='边界点', index=False)
            test_data.to_excel(writer, sheet_name='测试网格', index=False)

            # 创建汇总sheet
            summary_data = pd.concat([interior_data, boundary_data], ignore_index=True)
            summary_data.to_excel(writer, sheet_name='训练数据汇总', index=False)

        print(f"数据已保存到: {filename}")
        print(f"  - 内部点: {len(interior_data)}")
        print(f"  - 边界点: {len(boundary_data)}")
        print(f"  - 测试网格点: {len(test_data)}")
        print(f"  - 训练数据总计: {len(summary_data)}")

        return interior_data, boundary_data, test_data


def generate_subtask1_excel():
    """生成子任务1 Excel文件 (k=4)"""
    generator = HelmholtzDataGenerator(a1=1, a2=3, k=4)
    filename = '子任务1_亥姆霍兹方程数据_k4.xlsx'
    interior, boundary, test = generator.save_to_excel(
        filename=filename,
        n_interior=2000,
        n_boundary_per_edge=100,
        n_test_grid=50
    )

    # 生成数据分布图
    plot_data_distribution(interior, boundary, test, '子任务1数据分布 (k=4)')

    return interior, boundary, test


def generate_subtask2_excel():
    """生成子任务2 Excel文件 (高波数情况)"""
    high_k_values = [100, 500, 1000]
    results = {}

    for k in high_k_values:
        print(f"\n{'=' * 50}")
        print(f"生成 k={k} 的数据")
        print(f"{'=' * 50}")

        generator = HelmholtzDataGenerator(a1=1, a2=3, k=k)
        filename = f'子任务2_亥姆霍兹方程数据_k{k}.xlsx'
        interior, boundary, test = generator.save_to_excel(
            filename=filename,
            n_interior=3000,  # 高波数需要更多点
            n_boundary_per_edge=150,
            n_test_grid=50
        )

        results[k] = (interior, boundary, test)

        # 生成数据分布图
        plot_data_distribution(interior, boundary, test, f'子任务2数据分布 (k={k})')

    return results


def plot_data_distribution(interior, boundary, test, title):
    """绘制数据分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 内部点分布
    scatter1 = axes[0, 0].scatter(interior['x'], interior['y'],
                                  c=interior['q'], cmap='viridis',
                                  s=10, alpha=0.6)
    plt.colorbar(scatter1, ax=axes[0, 0])
    axes[0, 0].set_title(f'{title} - 内部点分布')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    axes[0, 0].grid(True, alpha=0.3)

    # 边界点分布
    scatter2 = axes[0, 1].scatter(boundary['x'], boundary['y'],
                                  c=boundary['q'], cmap='viridis',
                                  s=20, alpha=0.8)
    plt.colorbar(scatter2, ax=axes[0, 1])
    axes[0, 1].set_title(f'{title} - 边界点分布')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_aspect('equal')
    axes[0, 1].grid(True, alpha=0.3)

    # 测试网格分布
    n_grid = int(np.sqrt(len(test)))
    if n_grid * n_grid == len(test):
        X = test['x'].values.reshape(n_grid, n_grid)
        Y = test['y'].values.reshape(n_grid, n_grid)
        Z = test['q'].values.reshape(n_grid, n_grid)

        contour = axes[1, 0].contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(contour, ax=axes[1, 0])
        axes[1, 0].set_title(f'{title} - 测试网格源项分布')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        axes[1, 0].set_aspect('equal')

    # 源项值分布直方图
    all_q = pd.concat([interior['q'], boundary['q']])
    axes[1, 1].hist(all_q, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].set_title(f'{title} - 源项值分布')
    axes[1, 1].set_xlabel('q(x,y)')
    axes[1, 1].set_ylabel('密度')
    axes[1, 1].grid(True, alpha=0.3)

    # 添加统计信息文本框
    stats_text = f"""统计信息:
内部点: {len(interior)}
边界点: {len(boundary)}
测试点: {len(test)}
源项范围: [{all_q.min():.2f}, {all_q.max():.2f}]
源项均值: {all_q.mean():.4f}
源项标准差: {all_q.std():.4f}"""

    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_readme_file():
    """创建说明文件"""
    readme_content = """

# 亥姆霍兹方程数据集说明

## 文件结构

### 子任务1文件
- `子任务1_亥姆霍兹方程数据_k4.xlsx`: 包含4个sheet
  - `内部点`: 2000个内部训练点
  - `边界点`: 400个边界训练点  
  - `测试网格`: 2500个均匀网格点（用于验证）
  - `训练数据汇总`: 所有训练点合并（2400个=内部点+边界点）

### 子任务2文件
- `子任务2_亥姆霍兹方程数据_k100.xlsx`: 波数k=100的数据
- `子任务2_亥姆霍兹方程数据_k500.xlsx`: 波数k=500的数据  
- `子任务2_亥姆霍兹方程数据_k1000.xlsx`: 波数k=1000的数据
每个文件包含相同的sheet结构，但点数更多（内部点3000，边界点600）

### 子任务3文件
 -`子任务3数据`由官方提供

### 最终提交文件
- `test`文件中填写由模型预测出的结果

## 数据列说明

所有数据文件包含以下列：
- `x`: x坐标，范围[-1, 1]
- `y`: y坐标，范围[-1, 1]  
- `q`: 源项值，根据亥姆霍兹方程源项公式计算

## 亥姆霍兹方程

方程形式：
-Δu(x,y) - k²u(x,y) = q(x,y),  (x,y) ∈ [-1,1]×[-1,1]
u(x,y) = 0,  (x,y) ∈ ∂Ω

源项公式：
q(x,y) = -(a₁π)²sin(a₁πx)sin(a₂πy) - (a₂π)²sin(a₁πx)sin(a₂πy) - k²sin(a₁πx)sin(a₂πy)

参数：
- a₁ = 1, a₂ = 3
- k: 波数（子任务1: k=4, 子任务2: k=100,500,1000）

## 使用方法

1. **训练数据**: 使用`内部点`和`边界点`sheet中的数据
2. **验证数据**: 使用`测试网格`sheet中的数据  
3. **边界条件**: 所有边界点满足u=0

## 生成代码

数据由`dataset.py`生成，确保可复现性。

"""

    with open('数据集说明.txt', 'w', encoding='utf-8') as f:
        f.write(readme_content)


if __name__ == "__main__":
    print("开始生成亥姆霍兹方程Excel数据集...")

    # 创建输出目录
    if not os.path.exists('data'):
        os.makedirs('data')

    os.chdir('data')

    # 生成子任务1数据
    print("\n" + "=" * 60)
    print("生成子任务1数据 (k=4)")
    print("=" * 60)
    subtask1_data = generate_subtask1_excel()

    # 生成子任务2数据
    print("\n" + "=" * 60)
    print("生成子任务2数据 (高波数)")
    print("=" * 60)
    subtask2_data = generate_subtask2_excel()

    # 创建说明文件
    create_readme_file()

    print("\n所有数据生成完成！")
    print("\n生成的文件:")
    print("1. 子任务1:")
    print("   - 子任务1_亥姆霍兹方程数据_k4.xlsx")
    print("2. 子任务2:")
    print("   - 子任务2_亥姆霍兹方程数据_k100.xlsx")
    print("   - 子任务2_亥姆霍兹方程数据_k500.xlsx")
    print("   - 子任务2_亥姆霍兹方程数据_k1000.xlsx")
    print("3. 数据分布图: 多个PNG文件")
    print("4. 数据集说明.txt")

    # 显示文件大小信息
    print("\n文件大小:")
    for file in os.listdir('.'):
        if file.endswith('.xlsx'):
            size = os.path.getsize(file) / 1024  # KB
            print(f"  - {file}: {size:.1f} KB")