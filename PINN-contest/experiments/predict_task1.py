"""
子任务1预测脚本 - 使用BasePINN架构
"""

import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# 导入必要的类
from pinn_solvers.base_pinn import BasePINN, MLP


class HelmholtzPINNForPrediction(BasePINN):
    """用于预测的Helmholtz PINN简化版"""

    def __init__(self, config):
        super().__init__(config)
        self.k = config.get('wavenumber', 4)

    def pde_loss(self, x, u_pred):
        """占位方法，预测时不需要"""
        return torch.tensor(0.0)


def main():
    # 配置路径
    model_path = '../results/task1_mpa_recommended/best_model_mpa.pth'
    test_file_path = '../data/test.xlsx'
    output_path = '../results/predictions/子任务1预测结果.xlsx'

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 重新创建模型配置（必须与训练时一致）
    config = {
        'layers': [2, 40, 40, 40, 1],
        'activation': 'tanh',
        'learning_rate': 0.001,
        'wavenumber': 4,
        'lambda_pde': 0.5,
        'lambda_bc': 50,
        'network_type': 'standard'  # 使用标准网络
    }

    # 创建模型实例
    model = HelmholtzPINNForPrediction(config)

    print("尝试加载模型...")

    # 尝试多种加载方式
    try:
        # 方法1: 使用weights_only=False
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("检查点键:", list(checkpoint.keys()))

        # 根据检查点结构加载
        if 'model_state_dict' in checkpoint:
            model.net.load_state_dict(checkpoint['model_state_dict'])
            print("从model_state_dict加载成功")
        elif 'state_dict' in checkpoint:
            model.net.load_state_dict(checkpoint['state_dict'])
            print("从state_dict加载成功")
        else:
            # 假设检查点直接是状态字典
            model.net.load_state_dict(checkpoint)
            print("从直接状态字典加载成功")

    except Exception as e:
        print(f"加载失败: {e}")
        print("无法加载模型，请检查模型文件路径和格式")
        return

    model.eval()
    print("模型加载完成并设置为评估模式")

    # 加载test.xlsx中的子任务1数据
    try:
        df_task1 = pd.read_excel(test_file_path, sheet_name='子任务1')
        print(f"加载子任务1数据: {len(df_task1)} 行")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return

    # 提取坐标
    X_pred = df_task1[['xi', 'yi']].values.astype(np.float32)
    X_tensor = torch.FloatTensor(X_pred).to(device)

    print(f"开始预测 {len(X_pred)} 个点...")

    # 预测u值
    with torch.no_grad():
        u_pred = model.forward(X_tensor).cpu().numpy().flatten()

    print(f"预测完成，u值形状: {u_pred.shape}")

    # 将预测结果添加到DataFrame
    df_task1['u(xi,yi)'] = u_pred

    # 同时加载子任务3数据（保持原样）
    try:
        df_task3 = pd.read_excel(test_file_path, sheet_name='子任务3')
        print(f"加载子任务3数据: {len(df_task3)} 行")
    except:
        print("无法加载子任务3数据，将创建空表")
        df_task3 = pd.DataFrame(columns=['xi', 'yi', 'k(xi,yi)'])

    # 保存完整的test.xlsx
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_task1.to_excel(writer, sheet_name='子任务1', index=False)
            df_task3.to_excel(writer, sheet_name='子任务3', index=False)

        print(f"    预测完成！")
        print(f"   结果文件: {output_path}")
    except Exception as e:
        print(f"    保存文件失败: {e}")
        # 尝试保存为CSV作为备选
        csv_path = output_path.replace('.xlsx', '.csv')
        df_task1.to_csv(csv_path, index=False)
        print(f"   备选结果文件: {csv_path}")

    # 显示统计信息
    print(f"\n预测结果统计:")
    print(f"   u值范围: [{u_pred.min():.6f}, {u_pred.max():.6f}]")
    print(f"   u值均值: {u_pred.mean():.6f}")
    print(f"   u值标准差: {u_pred.std():.6f}")

    # 显示前10行结果
    print("\n前10行预测结果:")
    print(df_task1[['xi', 'yi', 'u(xi,yi)']].head(10))

    # 验证预测的合理性
    print(f"\n预测结果验证:")
    expected_range = [-1.5, 1.5]  # Helmholtz方程的预期范围
    actual_range = [u_pred.min(), u_pred.max()]

    if actual_range[0] >= expected_range[0] and actual_range[1] <= expected_range[1]:
        print("u值范围合理")
    else:
        print(f"u值范围异常，预期 {expected_range}，实际 {[round(x, 3) for x in actual_range]}")




if __name__ == "__main__":
    main()