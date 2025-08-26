#!/usr/bin/env python3
"""
Milano网络流量预测 - 数据量对比实验（修复版）
==================================

基于基线模型研究不同训练数据量对模型性能的影响
回答"最佳输入数据量是多少"这个研究问题
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import json
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径以导入基线模型代码
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入基线模型的数据处理类
try:
    from optimal_features_training.milano_optimal_features_corrected import CorrectedOptimalFeaturesMilanoPredictor
    print("✅ 成功导入基线模型")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保运行环境包含必要的依赖")
    sys.exit(1)

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# GPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据路径
DATA_PATH = 'data/cleaned_dataset_30/cleaned-sms-call-internet-mi-2013-11-01.csv'

class BaselineModel(nn.Module):
    """基线模型: 双向GRU + LSTM (与消融实验保持一致)"""
    def __init__(self, input_size, sequence_length):
        super(BaselineModel, self).__init__()
        self.model_name = "基线模型(双向GRU+LSTM)"
        
        # 双向GRU层
        self.gru = nn.GRU(input_size, 64, batch_first=True, bidirectional=True, dropout=0.2)
        
        # LSTM层
        self.lstm = nn.LSTM(128, 32, batch_first=True, dropout=0.2)
        
        # 全连接层
        self.dense = nn.Linear(32, 64)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        # 双向GRU处理
        x, _ = self.gru(x)
        
        # LSTM处理
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后时间步
        
        # 全连接层
        x = torch.relu(self.dense(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


class DataScalingExperiment:
    """数据量对比实验类"""
    
    def __init__(self, sequence_length=8):
        """
        初始化数据量对比实验
        
        Args:
            sequence_length: 序列长度
        """
        self.sequence_length = sequence_length
        self.results = {}
        
    def load_and_prepare_data_with_grids(self, max_grids):
        """
        加载和准备数据（指定网格数量）
        
        Args:
            max_grids: 最大网格数量
            
        Returns:
            tuple: (X, y) 序列数据
        """
        print(f"📥 加载数据 (最大网格数: {max_grids})...")
        
        # 使用基线模型的数据处理器
        data_processor = CorrectedOptimalFeaturesMilanoPredictor(
            sequence_length=self.sequence_length, 
            max_grids=max_grids
        )
        
        # 加载数据
        data, selected_grids = data_processor.load_and_preprocess_data()
        
        # 创建序列数据
        features = data_processor.optimal_features
        X, y = data_processor.create_sequences(features)
        
        if X is None or len(X) < 100:
            print(f"⚠️  网格数 {max_grids} 的数据量不足，跳过...")
            return None, None
        
        print(f"   数据形状: X={X.shape}, y={y.shape}")
        print(f"   实际网格数: {len(selected_grids)}")
        print(f"   样本总数: {len(X):,}")
        
        return X, y
    
    def train_single_model(self, X, y, data_size_name, epochs=30, batch_size=32):
        """
        训练单个模型
        
        Args:
            X: 输入特征
            y: 目标值
            data_size_name: 数据规模名称
            epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            dict: 训练和评估结果
        """
        print(f"\n🚀 开始训练: {data_size_name}")
        print(f"   样本数量: {len(X):,}")
        
        # 数据分割 (70%-15%-15%)
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        print(f"   训练集: {len(X_train):,}, 验证集: {len(X_val):,}, 测试集: {len(X_test):,}")
        
        # 转换为Tensor
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 创建模型
        model = BaselineModel(X.shape[2], self.sequence_length).to(device)
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   参数量: {total_params:,}")
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 训练模型
        model.train()
        start_time = time.time()
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_train_loss = epoch_loss / batch_count
            train_losses.append(avg_train_loss)
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_tensor)
                val_loss = criterion(val_pred.squeeze(), y_val_tensor).item()
                val_losses.append(val_loss)
            model.train()
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型状态
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   早停于第 {epoch+1} 轮")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.6f}, 验证损失: {val_loss:.6f}")
        
        training_time = time.time() - start_time
        print(f"   训练完成，耗时: {training_time:.1f}秒")
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        # 最终评估
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor).cpu().numpy().squeeze()
            test_pred = model(X_test_tensor).cpu().numpy().squeeze()
        
        # 逆变换到原始尺度
        y_val_orig = np.expm1(y_val) - 1.0
        y_test_orig = np.expm1(y_test) - 1.0
        val_pred_orig = np.expm1(val_pred) - 1.0
        test_pred_orig = np.expm1(test_pred) - 1.0
        
        # 计算指标
        val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred_orig))
        test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred_orig))
        val_r2 = r2_score(y_val_orig, val_pred_orig)
        test_r2 = r2_score(y_test_orig, test_pred_orig)
        val_mae = mean_absolute_error(y_val_orig, val_pred_orig)
        test_mae = mean_absolute_error(y_test_orig, test_pred_orig)
        
        # 计算MAPE
        val_mape = np.mean(np.abs((y_val_orig - val_pred_orig) / np.maximum(y_val_orig, 1e-8))) * 100
        test_mape = np.mean(np.abs((y_test_orig - test_pred_orig) / np.maximum(y_test_orig, 1e-8))) * 100
        
        # 结果字典
        results = {
            'data_size_name': data_size_name,
            'sample_count': len(X),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'total_params': total_params,
            'training_time': training_time,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs_trained': len(train_losses),
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mae': val_mae,
            'val_mape': val_mape,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_mape': test_mape,
            'y_val_true': y_val_orig,
            'y_val_pred': val_pred_orig,
            'y_test_true': y_test_orig,
            'y_test_pred': test_pred_orig
        }
        
        print(f"   验证集 - RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}, MAE: {val_mae:.4f}")
        print(f"   测试集 - RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, MAE: {test_mae:.4f}")
        
        return results
    
    def run_data_scaling_experiment(self, grid_sizes=None, epochs=30, batch_size=32):
        """
        运行完整的数据量对比实验
        
        Args:
            grid_sizes: 要测试的网格数量列表
            epochs: 训练轮数
            batch_size: 批次大小
        """
        if grid_sizes is None:
            # 默认测试不同的网格数量（对应不同的数据量）
            grid_sizes = [50, 100, 150, 200, 250, 300, 400, 500, 600, 800]
        
        print("📊 开始数据量对比实验")
        print("=" * 80)
        print(f"实验配置:")
        print(f"  序列长度: {self.sequence_length}")
        print(f"  训练轮数: {epochs}")
        print(f"  批次大小: {batch_size}")
        print(f"  测试网格数量: {grid_sizes}")
        print("=" * 80)
        
        # 逐个测试不同数据量
        for i, max_grids in enumerate(grid_sizes, 1):
            print(f"\n📊 实验 {i}/{len(grid_sizes)}: 最大网格数 = {max_grids}")
            print("-" * 60)
            
            try:
                # 加载数据
                X, y = self.load_and_prepare_data_with_grids(max_grids)
                
                if X is None:
                    continue
                
                # 训练模型
                data_size_name = f"网格数_{max_grids}"
                results = self.train_single_model(
                    X, y, data_size_name,
                    epochs=epochs,
                    batch_size=batch_size
                )
                
                # 存储结果
                self.results[max_grids] = results
                print(f"✅ {data_size_name} 实验完成")
                
                # 计算每个样本的平均训练时间
                time_per_sample = results['training_time'] / results['train_samples']
                print(f"   平均训练时间: {time_per_sample*1000:.2f} 毫秒/样本")
                
            except Exception as e:
                print(f"❌ 网格数 {max_grids} 实验失败: {e}")
                continue
        
        print(f"\n🎯 数据量对比实验完成! 成功完成 {len(self.results)} 个实验")
        return self.results
    
    def create_comparison_report(self):
        """创建对比报告"""
        if not self.results:
            print("❌ 没有实验结果可供对比")
            return None
        
        print("\n📊 创建数据量对比报告...")
        
        # 创建对比表格
        comparison_data = []
        for max_grids, results in self.results.items():
            comparison_data.append({
                '网格数': max_grids,
                '样本总数': results['sample_count'],
                '训练样本数': results['train_samples'],
                '训练时间(秒)': results['training_time'],
                '训练轮数': results['epochs_trained'],
                '时间/样本(ms)': (results['training_time'] / results['train_samples']) * 1000,
                '验证_RMSE': results['val_rmse'],
                '验证_R²': results['val_r2'],
                '验证_MAE': results['val_mae'],
                '验证_MAPE': results['val_mape'],
                '测试_RMSE': results['test_rmse'],
                '测试_R²': results['test_r2'],
                '测试_MAE': results['test_mae'],
                '测试_MAPE': results['test_mape']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 按网格数排序
        comparison_df = comparison_df.sort_values('网格数')
        
        return comparison_df
    
    def create_visualizations(self, save_dir='data_scaling_results'):
        """创建可视化图表（英文版）"""
        if not self.results:
            print("❌ 没有实验结果可供可视化")
            return
        
        print("📊 创建可视化图表...")
        
        # 创建结果目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置英文字体
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 准备数据
        grid_numbers = sorted(self.results.keys())
        sample_counts = [self.results[g]['sample_count'] for g in grid_numbers]
        train_samples = [self.results[g]['train_samples'] for g in grid_numbers]
        
        val_r2 = [self.results[g]['val_r2'] for g in grid_numbers]
        test_r2 = [self.results[g]['test_r2'] for g in grid_numbers]
        val_rmse = [self.results[g]['val_rmse'] for g in grid_numbers]
        test_rmse = [self.results[g]['test_rmse'] for g in grid_numbers]
        
        training_times = [self.results[g]['training_time'] for g in grid_numbers]
        time_per_sample = [(self.results[g]['training_time'] / self.results[g]['train_samples']) * 1000 
                          for g in grid_numbers]
        
        # 1. 主要性能指标随数据量变化
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Scaling Experiment Results', fontsize=16, fontweight='bold')
        
        # R²随数据量变化
        axes[0, 0].plot(sample_counts, val_r2, 'o-', label='Validation', alpha=0.8, linewidth=2, markersize=6)
        axes[0, 0].plot(sample_counts, test_r2, 's-', label='Test', alpha=0.8, linewidth=2, markersize=6)
        axes[0, 0].set_title('R² Score vs Data Size')
        axes[0, 0].set_xlabel('Total Samples')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE随数据量变化
        axes[0, 1].plot(sample_counts, val_rmse, 'o-', label='Validation', alpha=0.8, linewidth=2, markersize=6)
        axes[0, 1].plot(sample_counts, test_rmse, 's-', label='Test', alpha=0.8, linewidth=2, markersize=6)
        axes[0, 1].set_title('RMSE vs Data Size')
        axes[0, 1].set_xlabel('Total Samples')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 训练时间随数据量变化
        axes[1, 0].plot(train_samples, training_times, 'o-', color='green', alpha=0.8, linewidth=2, markersize=6)
        axes[1, 0].set_title('Training Time vs Data Size')
        axes[1, 0].set_xlabel('Training Samples')
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 训练效率（时间/样本）
        axes[1, 1].plot(train_samples, time_per_sample, 'o-', color='red', alpha=0.8, linewidth=2, markersize=6)
        axes[1, 1].set_title('Training Efficiency (time/sample)')
        axes[1, 1].set_xlabel('Training Samples')
        axes[1, 1].set_ylabel('Time per Sample (ms)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存主要对比图
        main_comparison_path = os.path.join(save_dir, 'data_scaling_comparison.png')
        plt.savefig(main_comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 主要对比图保存至: {main_comparison_path}")
        
        # 2. 性能改进分析图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Performance Improvement Analysis', fontsize=14, fontweight='bold')
        
        # R²改进曲线
        if len(val_r2) > 1:
            r2_improvement = [(val_r2[i] - val_r2[0]) for i in range(len(val_r2))]
            axes[0].plot(sample_counts, r2_improvement, 'o-', color='blue', alpha=0.8, linewidth=2, markersize=6)
            axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0].set_title('R² Improvement Relative to Minimum Data')
            axes[0].set_xlabel('Total Samples')
            axes[0].set_ylabel('R² Improvement')
            axes[0].grid(True, alpha=0.3)
        
        # 边际收益分析
        if len(val_r2) > 1:
            marginal_gains = [0] + [val_r2[i] - val_r2[i-1] for i in range(1, len(val_r2))]
            axes[1].bar(range(len(sample_counts)), marginal_gains, alpha=0.7, color='orange')
            axes[1].set_title('R² Marginal Gains')
            axes[1].set_xlabel('Experiment Index')
            axes[1].set_ylabel('R² Marginal Improvement')
            axes[1].set_xticks(range(len(sample_counts)))
            axes[1].set_xticklabels([f'{sc//1000}K' for sc in sample_counts], rotation=45)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存改进分析图
        improvement_path = os.path.join(save_dir, 'performance_improvement_analysis.png')
        plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 改进分析图保存至: {improvement_path}")
        
        print(f"📊 所有可视化图表已保存至: {save_dir}/")
    
    def save_results(self, save_dir='data_scaling_results'):
        """保存实验结果"""
        if not self.results:
            print("❌ 没有实验结果可保存")
            return
        
        print("💾 保存实验结果...")
        
        # 创建结果目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 保存对比报告
        comparison_df = self.create_comparison_report()
        if comparison_df is not None:
            comparison_path = os.path.join(save_dir, 'data_scaling_comparison.csv')
            comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
            print(f"📊 对比报告: {comparison_path}")
        
        # 2. 保存详细结果
        summary_data = {
            'experiment_info': {
                'name': 'milano_data_scaling_experiment',
                'description': 'Milano网络流量预测数据量对比实验',
                'datetime': datetime.now().isoformat(),
                'sequence_length': self.sequence_length,
                'device': str(device)
            },
            'grid_sizes_tested': list(self.results.keys()),
            'results_summary': {}
        }
        
        # 3. 保存每个实验的详细结果
        for grid_size, results in self.results.items():
            # 保存预测结果
            predictions_df = pd.DataFrame({
                'val_true': results['y_val_true'],
                'val_pred': results['y_val_pred'],
                'val_error': results['y_val_pred'] - results['y_val_true'],
                'test_true': results['y_test_true'][:len(results['y_val_true'])],
                'test_pred': results['y_test_pred'][:len(results['y_val_true'])],
                'test_error': (results['y_test_pred'] - results['y_test_true'])[:len(results['y_val_true'])]
            })
            pred_path = os.path.join(save_dir, f'grid_{grid_size}_predictions.csv')
            predictions_df.to_csv(pred_path, index=False)
            
            # 添加到汇总数据
            summary_data['results_summary'][f'grid_{grid_size}'] = {
                'grid_size': grid_size,
                'sample_count': int(results['sample_count']),
                'train_samples': int(results['train_samples']),
                'training_time': float(results['training_time']),
                'epochs_trained': int(results['epochs_trained']),
                'validation_metrics': {
                    'rmse': float(results['val_rmse']),
                    'r2': float(results['val_r2']),
                    'mae': float(results['val_mae']),
                    'mape': float(results['val_mape'])
                },
                'test_metrics': {
                    'rmse': float(results['test_rmse']),
                    'r2': float(results['test_r2']),
                    'mae': float(results['test_mae']),
                    'mape': float(results['test_mape'])
                }
            }
        
        # 4. 保存汇总JSON
        summary_path = os.path.join(save_dir, 'data_scaling_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"📈 汇总结果: {summary_path}")
        
        print(f"💾 所有结果已保存至: {save_dir}/")
        
        return save_dir


def print_final_summary(results, comparison_df, save_dir, total_time):
    """打印最终总结"""
    print("\n" + "=" * 100)
    print("🎉 数据量对比实验完成!")
    print("=" * 100)
    print(f"⏱️  总实验时间: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    
    if comparison_df is not None and len(comparison_df) > 0:
        print("📊 实验结果汇总:")
        print("-" * 80)
        
        # 显示关键指标
        print("关键发现:")
        
        # 最佳性能
        best_r2_idx = comparison_df['验证_R²'].idxmax()
        best_result = comparison_df.iloc[best_r2_idx]
        print(f"🏆 最佳验证R²: {best_result['验证_R²']:.4f} (网格数: {best_result['网格数']}, 样本数: {best_result['样本总数']:,})")
        
        # 性能趋势
        first_r2 = comparison_df.iloc[0]['验证_R²']
        last_r2 = comparison_df.iloc[-1]['验证_R²']
        r2_improvement = last_r2 - first_r2
        print(f"📈 R²总体改进: {r2_improvement:.4f} ({first_r2:.4f} → {last_r2:.4f})")
        
        # 效率分析
        min_time_per_sample = comparison_df['时间/样本(ms)'].min()
        max_time_per_sample = comparison_df['时间/样本(ms)'].max()
        print(f"⏱️  训练效率范围: {min_time_per_sample:.2f} - {max_time_per_sample:.2f} 毫秒/样本")
        
        # 样本数量范围
        min_samples = comparison_df['样本总数'].min()
        max_samples = comparison_df['样本总数'].max()
        print(f"📊 样本数量范围: {min_samples:,} - {max_samples:,}")
        
        print(f"\n💡 关键洞察:")
        
        # 找到性能饱和点
        r2_scores = comparison_df['验证_R²'].values
        if len(r2_scores) > 3:
            # 计算改进的边际效益
            marginal_gains = [r2_scores[i] - r2_scores[i-1] for i in range(1, len(r2_scores))]
            avg_marginal_gain = np.mean(marginal_gains)
            
            # 找到边际收益低于平均值的点
            saturation_points = [i for i, gain in enumerate(marginal_gains) if gain < avg_marginal_gain * 0.5]
            if saturation_points:
                saturation_idx = saturation_points[0] + 1  # +1因为marginal_gains比原数组少1个元素
                saturation_result = comparison_df.iloc[saturation_idx]
                print(f"   🎯 性能饱和点: 约 {saturation_result['样本总数']:,} 样本 (网格数: {saturation_result['网格数']})")
                print(f"      在此点R²: {saturation_result['验证_R²']:.4f}")
        
        # 效率建议
        efficiency_scores = comparison_df['验证_R²'] / comparison_df['时间/样本(ms)']
        best_efficiency_idx = efficiency_scores.idxmax()
        efficient_result = comparison_df.iloc[best_efficiency_idx]
        print(f"   ⚡ 最高效配置: {efficient_result['样本总数']:,} 样本 (网格数: {efficient_result['网格数']})")
        print(f"      R²: {efficient_result['验证_R²']:.4f}, 效率: {efficiency_scores.iloc[best_efficiency_idx]:.6f}")
    
    print(f"\n📁 结果文件:")
    print(f"   📊 对比图表: {save_dir}/data_scaling_comparison.png")
    print(f"   📈 改进分析: {save_dir}/performance_improvement_analysis.png")
    print(f"   📋 对比数据: {save_dir}/data_scaling_comparison.csv")
    print(f"   💾 详细结果: {save_dir}/data_scaling_summary.json")
    
    print("=" * 100)


def main():
    """主函数"""
    print("📊 Milano网络流量预测 - 数据量对比实验")
    print("=" * 80)
    print("研究问题: 最佳输入数据量是多少？")
    print("实验方法: 使用不同数量的网格来控制数据量，观察模型性能变化")
    print("=" * 80)
    
    # 创建实验实例
    experiment = DataScalingExperiment(sequence_length=8)
    
    # 定义要测试的网格数量（对应不同数据量）
    grid_sizes = [50, 100, 150, 200, 250, 300, 400, 500, 600, 800]
    
    print(f"将测试的网格数量: {grid_sizes}")
    print("预计每个实验需要 3-5 分钟...")
    
    # 询问用户是否继续
    response = input("\n是否开始实验? (y/n): ").lower().strip()
    if response not in ['y', 'yes', '是', '']:
        print("❌ 实验已取消")
        return
    
    # 运行实验
    print(f"\n🚀 开始数据量对比实验...")
    total_start_time = time.time()
    
    try:
        results = experiment.run_data_scaling_experiment(
            grid_sizes=grid_sizes,
            epochs=30,
            batch_size=32
        )
        
        if not results:
            print("❌ 实验失败!")
            return
        
        total_time = time.time() - total_start_time
        
        # 生成报告和可视化
        comparison_df = experiment.create_comparison_report()
        experiment.create_visualizations()
        save_dir = experiment.save_results()
        
        # 打印最终总结
        print_final_summary(results, comparison_df, save_dir, total_time)
        
    except KeyboardInterrupt:
        print("\n⚠️  实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
