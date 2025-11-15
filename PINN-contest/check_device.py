"""
设备检查脚本
检查CUDA是否可用，并测试性能
"""

import torch
import numpy as np
import time

print("="*60)
print("设备检查")
print("="*60)

# 检查CUDA
print(f"\n1. PyTorch版本: {torch.__version__}")
print(f"2. CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"3. GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"4. CUDA版本: {torch.version.cuda}")
    print(f"5. GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = torch.device('cuda')
else:
    print("3. 使用CPU")
    device = torch.device('cpu')

# 简单性能测试
print("\n6. 性能测试（简单矩阵运算）...")
size = 1000
A = torch.randn(size, size).to(device)
B = torch.randn(size, size).to(device)

# CPU预热
_ = A @ B

start = time.time()
for _ in range(10):
    C = A @ B
torch.cuda.synchronize() if torch.cuda.is_available() else None
end = time.time()

print(f"   10次矩阵乘法({size}x{size}): {end-start:.4f} 秒")

print("\n" + "="*60)
print("建议：")
if torch.cuda.is_available():
    print("✅ 可以使用GPU加速训练！")
    print("   对于PINN训练，GPU可以加速5-10倍")
else:
    print("⚠️  当前使用CPU，训练会较慢")
    print("   如果有NVIDIA GPU，建议安装CUDA版PyTorch")
print("="*60)

