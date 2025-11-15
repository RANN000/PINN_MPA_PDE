# PINN Contest - MPA-Guided Solution

基于**方法-问题对齐（MPA）理论**的PINN求解框架，用于竞赛问题：深度学习方法求解亥姆霍兹方程和泊松方程。

## 🎯 项目目标

1. **高精度求解**：在三个子任务上达到优异的预测精度
2. **方法论创新**：通过MPA理论指导优化器选择，避免盲目试错
3. **完整流程**：从数据加载到结果预测的完整pipeline

## 📁 代码结构

```
PINN-contest/
├── data/                          # 数据管理
│   ├── data_loader.py            # 统一数据加载接口
│   ├── *.xlsx                     # 训练和测试数据
│   └── test.xlsx                  # 最终提交文件
│
├── pinn_solvers/                  # PINN求解器
│   ├── base_pinn.py              # 基础PINN框架
│   ├── helmholtz_solver.py       # Helmholtz求解器
│   └── poisson_inverse_solver.py # Poisson反问题求解器（待实现）
│
├── mpa_core/                      # MPA理论核心（待实现）
│   ├── alignment_metrics.py      # 对齐指标计算
│   ├── problem_profiler.py       # 问题特征分析
│   └── optimizer_selector.py     # 优化器选择器
│
├── experiments/                   # 实验脚本
│   ├── task1_helmholtz_baseline.py  # 子任务1基线实验
│   ├── task2_high_wavenumber.py     # 子任务2实验（待实现）
│   └── task3_poisson_inverse.py      # 子任务3实验（待实现）
│
├── results/                       # 实验结果
│   ├── models/                    # 训练好的模型
│   ├── predictions/               # 预测结果
│   └── logs/                      # 训练日志
│
└── requirements.txt               # 依赖包
```

## 🚀 快速开始

### 环境设置

```bash
# 创建并激活conda环境
conda create -n pinn python=3.10
conda activate pinn

# 安装PyTorch（CPU版本）
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 安装其他依赖
pip install pandas openpyxl matplotlib numpy scipy scikit-learn tqdm seaborn
```

### 运行子任务1 Baseline

```bash
python experiments/task1_helmholtz_baseline.py
```

## 📊 子任务描述

### 子任务1：Helmholtz方程求解（k=4）
- 方程：$-\\Delta u - k^2 u = q$，边界条件 $u|_{\\partial\\Omega} = 0$
- 域：$[-1,1] \\times [-1,1]$
- 源项：$q(x,y) = -((\\pi)^2 + (3\\pi)^2 + k^2)\\sin(\\pi x)\\sin(3\\pi y)$
- 解析解：$u(x,y) = \\sin(\\pi x)\\sin(3\\pi y)$

### 子任务2：高波数Helmholtz求解
- 同子任务1，但波数更高（k=100, 500, 1000）
- 挑战：高频振荡难以捕捉
- 创新点：傅里叶特征、多尺度网络

### 子任务3：Poisson反问题
- 方程：$-\\Delta u = \\lambda f$
- 任务：给定200个观测点，反演参数$\\lambda$
- 创新点：双网络架构、正则化策略

## 🧪 MPA理论应用

### 对齐指标

1. **GDC (Gradient Descent Compatibility)**：梯度友好度
   - 计算Hessian条件数，评估梯度方法的适用性

2. **TS (Trajectory Smoothness)**：优化轨迹平滑度
   - 分析短期优化过程中损失的方差

3. **LSM (Local Smoothness Metric)**：局部平滑度
   - 测量参数扰动对损失的影响

### 优化器选择策略

基于MPA分数自动选择最优优化器：
- 高GDC + 高LSM → Adam（快速收敛）
- 低GDC → L-BFGS或混合策略
- 低TS → RAdam（更稳定）

## 📈 实验计划

### Step 1: Baseline实现 ✅
- [x] 数据加载器
- [x] 基础PINN框架
- [x] Helmholtz求解器
- [ ] 子任务1训练和验证

### Step 2: MPA模块集成（待实现）
- [ ] 对齐指标计算
- [ ] 问题特征分析
- [ ] 优化器选择器
- [ ] MPA有效性验证

### Step 3: 针对性创新（待实现）
- [ ] 子任务2：高波数技巧（傅里叶特征、多尺度）
- [ ] 子任务3：反问题架构（双网络、正则化）
- [ ] 消融实验

### Step 4: 预测与整理（待实现）
- [ ] 生成test.xlsx预测结果
- [ ] 可视化分析
- [ ] 撰写说明文档

## 🎓 创新点

1. **MPA方法论**：用对齐理论指导优化器选择，而非经验选择
2. **高波数求解**：傅里叶特征嵌入和多尺度网络架构
3. **反问题设计**：双网络架构同时学习解和参数
4. **完整实验流程**：从baseline到MPA优化，展示性能提升路径

## 📝 注意事项

- 确保`conda activate pinn`激活环境后再运行
- 训练过程较长，建议使用checkpoint保存模型
- 提交前确保在test.xlsx中生成了完整的预测结果

## 📄 许可证

本项目仅用于竞赛提交。
