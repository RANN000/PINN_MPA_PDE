"""
子任务3预测脚本
使用训练好的模型预测k值
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


# 从原代码复制模型定义
class MLP(torch.nn.Module):
    def __init__(self, layers, activation='tanh'):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        act_dict = {
            'tanh': torch.nn.Tanh(),
            'relu': torch.nn.ReLU(),
            'silu': torch.nn.SiLU()
        }
        activation_fn = act_dict.get(activation, torch.nn.Tanh())
        for i in range(len(layers) - 1):
            self.layers.append(torch.nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(activation_fn)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LambdaNet(torch.nn.Module):
    def __init__(self, hidden_layers=[128, 128, 64, 32]):
        super().__init__()
        layers = [2] + hidden_layers + [1]
        self.net = MLP(layers, activation='tanh')
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        k_raw = self.net(x)
        return torch.nn.functional.softplus(k_raw) + 1e-8


def main():
    # 配置路径
    model_path = '../results/task3_piml_mpa/models.pth'
    data_path = '../data/test.xlsx'
    output_path = '../results/predictions/子任务3预测结果.xlsx'

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print("加载训练好的模型...")
    checkpoint = torch.load(model_path, map_location=device)

    # 创建模型实例
    lambda_net = LambdaNet(hidden_layers=checkpoint['config']['hidden_layers']).to(device)
    lambda_net.load_state_dict(checkpoint['lambda_net_state'])
    lambda_net.eval()

    print("模型加载完成")

    # 加载子任务3数据
    df = pd.read_excel(data_path, sheet_name='子任务3')
    print(f"加载子任务3数据: {len(df)} 行")

    # 提取坐标并预测
    X_pred = df[['xi', 'yi']].values.astype(np.float32)
    X_tensor = torch.FloatTensor(X_pred).to(device)

    with torch.no_grad():
        k_pred = lambda_net(X_tensor).cpu().numpy()

    # 保存结果
    df['k(xi,yi)'] = k_pred.flatten()
    df.to_excel(output_path, index=False)

    print(f"    预测完成！")
    print(f"   预测点数: {len(df)}")
    print(f"   k值统计:")
    print(f"     最小值: {k_pred.min():.6f}")
    print(f"     最大值: {k_pred.max():.6f}")
    print(f"     平均值: {k_pred.mean():.6f}")
    print(f"     中位数: {np.median(k_pred):.6f}")
    print(f"   结果文件: {output_path}")


    # 显示前10行结果
    print("\n前10行预测结果:")
    print(df.head(10))


if __name__ == "__main__":
    main()