# Milano网络流量预测 - 消融实验

基于基线模型 `optimal_features_training/milano_optimal_features_corrected.py` 的消融实验套件，用于比较不同序列建模架构的性能。

## 🎯 实验目标

通过消融实验分析不同模型组件对Milano网络流量预测性能的影响：

1. **单向GRU** - 使用单向GRU进行序列建模
2. **双向GRU** - 使用双向GRU进行序列建模  
3. **单向GRU+LSTM** - 单向GRU与LSTM的级联组合
4. **纯LSTM** - 只使用LSTM进行序列建模
5. **基线模型** - 双向GRU+LSTM组合（原始架构）

## 📁 文件结构

```
ablation_study/
├── models.py                   # 消融实验模型架构定义
├── ablation_experiment.py      # 主要的消融实验脚本
├── run_all_experiments.py      # 一键运行所有实验
├── test_models.py              # 模型快速测试脚本
├── README.md                   # 本文件
└── results/                    # 实验结果目录（自动创建）
    ├── ablation_study_comparison.png       # 对比图表
    ├── ablation_study_comparison.csv       # 对比数据表
    ├── ablation_study_summary.json        # 汇总结果
    ├── experiment_config.json             # 实验配置
    ├── *_model.pth                        # 各模型权重文件
    ├── *_predictions.csv                  # 各模型预测结果
    └── *_detailed_results.png             # 各模型详细结果图
```

## 🚀 快速开始

### 1. 环境要求

确保已安装以下依赖：
```bash
torch>=1.9.0
numpy
pandas
scikit-learn
matplotlib
seaborn
```

### 2. 运行完整消融实验

```bash
# 进入消融实验目录
cd ablation_study

# 一键运行所有实验（推荐）
python run_all_experiments.py

# 或者运行单个实验脚本
python ablation_experiment.py
```

### 3. 测试模型架构

```bash
# 快速测试所有模型架构是否正常
python test_models.py
```

## 📊 实验配置

默认实验配置：
- **序列长度**: 8
- **最大网格数**: 300（与基线模型一致）
- **训练轮数**: 30
- **批次大小**: 32
- **优化器**: Adam (lr=0.001)
- **数据分割**: 70%-15%-15%（训练-验证-测试）

## 🏗️ 模型架构对比

| 模型 | 架构描述 | 参数量估计 |
|------|----------|------------|
| 单向GRU | `GRU(64) → Dense(64) → Output(1)` | ~13K |
| 双向GRU | `BiGRU(64*2) → Dense(64) → Output(1)` | ~25K |
| 单向GRU+LSTM | `GRU(64) → LSTM(32) → Dense(64) → Output(1)` | ~22K |
| 纯LSTM | `LSTM(64) → Dense(64) → Output(1)` | ~20K |
| 基线模型 | `BiGRU(64*2) → LSTM(32) → Dense(64) → Output(1)` | ~34K |

## 📈 评估指标

每个模型都会在验证集和测试集上评估以下指标：
- **RMSE** (Root Mean Square Error)
- **R²** (R-squared Score) 
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **训练时间**
- **参数量**

## 📊 结果可视化

实验完成后将生成以下可视化：

1. **整体对比图** (`ablation_study_comparison.png`)
   - R²分数对比
   - RMSE对比  
   - 参数量和训练时间对比
   - MAE对比

2. **详细结果图** (每个模型的 `*_detailed_results.png`)
   - 验证集预测vs真实值散点图
   - 测试集预测vs真实值散点图

## 📄 结果文件说明

- `ablation_study_comparison.csv`: 所有模型的性能对比表格
- `ablation_study_summary.json`: 完整的实验结果汇总（JSON格式）
- `experiment_config.json`: 实验配置记录
- `*_model.pth`: 各模型的训练权重
- `*_predictions.csv`: 各模型的详细预测结果

## 🔧 自定义实验

如需自定义实验参数，可修改 `run_all_experiments.py` 中的配置：

```python
def create_experiment_config():
    return {
        'sequence_length': 8,      # 序列长度
        'max_grids': 300,          # 最大网格数
        'epochs': 30,              # 训练轮数
        'batch_size': 32,          # 批次大小
        'models_to_test': [        # 选择要测试的模型
            'unidirectional_gru',
 
            'unidirectional_gru_lstm',
            'pure_lstm',
            'baseline'
        ]
    }
```

## 🧪 实验分析建议

1. **性能对比**: 重点关注R²分数，这是回归任务的主要指标
2. **效率分析**: 比较参数量与性能的权衡，寻找高效模型
3. **稳定性**: 观察验证集和测试集性能的一致性
4. **复杂度**: 分析模型复杂度与训练时间的关系

## ⚠️ 注意事项

1. 确保基线模型文件 `optimal_features_training/milano_optimal_features_corrected.py` 存在
2. 确保数据文件路径正确：`/root/autodl-tmp/xiaorong0802/data/cleaned_dataset_30/cleaned-sms-call-internet-mi-2013-11-01.csv`
3. 实验需要较长时间（约10-30分钟），建议在稳定的环境中运行
4. 如果GPU内存不足，可以减小 `batch_size` 或 `max_grids`

## 🤝 扩展实验

可以基于此框架扩展更多消融实验：
- 不同的激活函数
- 不同的dropout比率
- 不同的序列长度
- 不同的特征组合
- 不同的优化器和学习率

## 📞 问题反馈

如遇到问题，请检查：
1. 依赖包是否完整安装
2. 数据文件路径是否正确
3. GPU内存是否充足
4. Python版本兼容性