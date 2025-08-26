# 🚀 Milano消融实验 - 快速开始指南

## ✅ 环境状态

**已完成的设置：**
- ✅ 消融实验文件夹创建完成
- ✅ 四种消融模型架构实现完成
- ✅ 基线模型兼容性验证通过
- ✅ 所有模型架构测试通过
- ✅ GPU环境配置正常 (NVIDIA GeForce RTX 2080 Ti)

**特征配置：**
- 输入特征数：13个最佳特征
- 序列长度：8
- 数据分割：70%-15%-15%（训练-验证-测试）

## 🧪 消融实验模型

| 模型 | 参数量 | 架构描述 |
|------|--------|----------|
| 单向GRU | 19,393 | `GRU(64) → Dense(64) → Output(1)` |
| 双向GRU | 38,657 | `BiGRU(64*2) → Dense(64) → Output(1)` |
| 单向GRU+LSTM | 29,889 | `GRU(64) → LSTM(32) → Dense(64) → Output(1)` |
| 纯LSTM | 24,449 | `LSTM(64) → Dense(64) → Output(1)` |
| 基线模型 | 53,249 | `BiGRU(64*2) → LSTM(32) → Dense(64) → Output(1)` |

## 🚀 运行实验

### 方法1：一键运行所有实验（推荐）

```bash
cd ablation_study
python run_all_experiments.py
```

这将：
- 自动运行所有5个模型的训练和评估
- 生成完整的对比报告和可视化图表
- 保存所有结果到 `results/` 文件夹

### 方法2：运行单个实验

```bash
cd ablation_study
python ablation_experiment.py
```

### 方法3：快速模型测试

```bash
cd ablation_study
python test_models.py
```

## 📊 预期结果

实验完成后，您将在 `ablation_study/results/` 文件夹中获得：

1. **对比图表** (`ablation_study_comparison.png`)
   - R²分数对比
   - RMSE对比
   - 参数量和训练时间对比
   - MAE对比

2. **数据文件**
   - `ablation_study_comparison.csv` - 对比数据表
   - `ablation_study_summary.json` - 完整结果汇总
   - `*_model.pth` - 各模型权重文件
   - `*_predictions.csv` - 各模型预测结果

## ⏱️ 预计实验时间

- **快速测试**：< 1分钟
- **完整消融实验**：约 15-30 分钟（取决于数据量）

## 🎯 分析目标

通过消融实验，您可以分析：
1. **双向vs单向**：双向GRU是否比单向GRU性能更好？
2. **GRU vs LSTM**：哪种序列建模方法更适合网络流量预测？
3. **模型组合**：GRU+LSTM组合是否比单一架构更好？
4. **效率权衡**：参数量与性能的最佳平衡点在哪里？

## 🔧 自定义实验

如需修改实验参数，编辑 `run_all_experiments.py` 中的配置：

```python
def create_experiment_config():
    return {
        'sequence_length': 8,      # 序列长度
        'max_grids': 300,          # 最大网格数
        'epochs': 30,              # 训练轮数
        'batch_size': 32,          # 批次大小
    }
```

## ⚡ 立即开始

```bash
# 确保在正确目录
pwd  # 应该显示: /root/autodl-tmp/xiaorong0802

# 进入消融实验目录
cd ablation_study

# 一键运行所有实验
python run_all_experiments.py
```

**准备好了吗？让我们开始消融实验！** 🚀