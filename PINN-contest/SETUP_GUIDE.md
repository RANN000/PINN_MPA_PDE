# 环境设置指南

## 当前状态

✅ **代码已完成的部分：**
- `data/data_loader.py` - 数据加载器，支持三个子任务
- `pinn_solvers/base_pinn.py` - 基础PINN框架
- `pinn_solvers/helmholtz_solver.py` - Helmholtz求解器
- `experiments/task1_helmholtz_baseline.py` - 子任务1实验脚本

## 下一步：安装PyTorch

您需要自己在conda环境中安装PyTorch。推荐方式：

### 方式1：使用conda安装PyTorch (推荐)
```bash
conda create -n pinn python=3.10
conda activate pinn

# CPU版本（如果不需要GPU）
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 或GPU版本（如果有CUDA）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 方式2：使用pip安装
```bash
pip install torch torchvision pandas openpyxl matplotlib numpy scipy scikit-learn tqdm seaborn
```

## 测试安装

激活环境后运行：
```bash
cd PINN-contest
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

## 运行实验

环境准备完成后，运行子任务1的baseline：

```bash
python experiments/task1_helmholtz_baseline.py
```

这将：
1. 加载k=4的Helmholtz数据
2. 训练标准PINN（MLP 50-50-50）
3. 在测试集上评估
4. 生成可视化结果到 `results/task1_baseline/`

## 预期输出

训练过程将显示：
- 每100个epoch输出一次损失和测试误差
- 最终生成6个子图的可视化结果
- 保存最佳模型到 `results/task1_baseline/best_model_k4.pth`

预期相对误差：< 1% （对于简单的k=4问题）

## 注意事项

1. **首夜运行时间**：子任务1 baseline约需10-20分钟（取决于CPU）
2. **Checkpoint**：训练会每100 epoch保存一次最佳模型
3. **结果位置**：所有结果保存在 `results/` 目录下

## 下一步工作

运行完子任务1 baseline后，我们可以继续：

1. **实现MPA模块**：添加优化器选择和问题分析
2. **子任务2**：实现高波数Helmholtz求解（傅里叶特征）
3. **子任务3**：实现Poisson反问题
4. **在test.xlsx上生成预测**

