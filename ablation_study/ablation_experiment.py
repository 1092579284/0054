#!/usr/bin/env python3
"""
Milano网络流量预测 - 消融实验主脚本
==================================

基于基线模型进行消融实验，测试不同模型架构的性能：
1. 单向GRU
2. 单向GRU+LSTM
3. 纯LSTM
4. 基线模型(双向GRU+LSTM)
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
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入基线模型的数据处理类和消融实验模型
try:
    from optimal_features_training.milano_optimal_features_corrected import CorrectedOptimalFeaturesMilanoPredictor
    from ablation_study.models import get_model_by_name, get_all_model_info
    print("✅ 成功导入基线模型和消融实验模型")
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


class AblationExperiment:
    """消融实验类"""
    
    def __init__(self, sequence_length=8, max_grids=300):
        """
        初始化消融实验
        
        Args:
            sequence_length: 序列长度
            max_grids: 最大网格数量
        """
        self.sequence_length = sequence_length
        self.max_grids = max_grids
        
        # 使用基线模型的数据处理器
        self.data_processor = CorrectedOptimalFeaturesMilanoPredictor(
            sequence_length=sequence_length, 
            max_grids=max_grids
        )
        
        self.results = {}
        
    def load_and_prepare_data(self):
        """加载和准备数据"""
        print("📥 加载和预处理数据...")
        
        # 使用基线模型的数据加载方法
        data, selected_grids = self.data_processor.load_and_preprocess_data()
        
        # 创建序列数据
        features = self.data_processor.optimal_features
        X, y = self.data_processor.create_sequences(features)
        
        if X is None or len(X) < 100:
            raise ValueError("数据序列创建失败或数据量不足")
        
        print(f"序列数据形状: X={X.shape}, y={y.shape}")
        
        # 数据分割 (70%-15%-15%)
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        self.X_train = X[:train_size]
        self.y_train = y[:train_size]
        self.X_val = X[train_size:train_size + val_size]
        self.y_val = y[train_size:train_size + val_size]
        self.X_test = X[train_size + val_size:]
        self.y_test = y[train_size + val_size:]
        
        print(f"数据分割: 训练{len(self.X_train)}, 验证{len(self.X_val)}, 测试{len(self.X_test)}")
        
        # 转换为Tensor
        self.X_train_tensor = torch.FloatTensor(self.X_train).to(device)
        self.y_train_tensor = torch.FloatTensor(self.y_train).to(device)
        self.X_val_tensor = torch.FloatTensor(self.X_val).to(device)
        self.y_val_tensor = torch.FloatTensor(self.y_val).to(device)
        self.X_test_tensor = torch.FloatTensor(self.X_test).to(device)
        self.y_test_tensor = torch.FloatTensor(self.y_test).to(device)
        
        self.input_size = X.shape[2]
        print(f"输入特征维度: {self.input_size}")
        
    def train_single_model(self, model_key, epochs=30, batch_size=32, learning_rate=0.001):
        """
        训练单个模型
        
        Args:
            model_key: 模型键名
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            
        Returns:
            dict: 训练和评估结果
        """
        model_info = get_all_model_info()[model_key]
        print(f"\n🚀 开始训练: {model_info['name']}")
        print(f"   架构: {model_info['architecture']}")
        
        # 创建模型
        model = get_model_by_name(model_key, self.input_size, self.sequence_length).to(device)
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   参数量: {total_params:,}")
        
        # 创建数据加载器
        train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 训练模型
        model.train()
        start_time = time.time()
        train_losses = []
        
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
            
            avg_loss = epoch_loss / batch_count
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        training_time = time.time() - start_time
        print(f"   训练完成，耗时: {training_time:.1f}秒")
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            val_pred = model(self.X_val_tensor).cpu().numpy().squeeze()
            test_pred = model(self.X_test_tensor).cpu().numpy().squeeze()
        
        # 逆变换到原始尺度
        y_val_orig = np.expm1(self.y_val) - 1.0
        y_test_orig = np.expm1(self.y_test) - 1.0
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
            'model_key': model_key,
            'model_name': model_info['name'],
            'model_architecture': model_info['architecture'],
            'total_params': total_params,
            'training_time': training_time,
            'train_losses': train_losses,
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
            'y_test_pred': test_pred_orig,
            'model_state_dict': model.state_dict()
        }
        
        print(f"   验证集 - RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
        print(f"   测试集 - RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        
        return results
    
    def run_ablation_study(self, models_to_test=None, epochs=30, batch_size=32):
        """
        运行完整的消融实验
        
        Args:
            models_to_test: 要测试的模型列表，None表示测试所有模型
            epochs: 训练轮数
            batch_size: 批次大小
        """
        if models_to_test is None:
            models_to_test = [
                'unidirectional_gru',
                'unidirectional_gru_lstm',
                'pure_lstm',
                'baseline'
            ]
        
        print("🧪 开始消融实验")
        print("=" * 80)
        print(f"实验配置:")
        print(f"  序列长度: {self.sequence_length}")
        print(f"  最大网格: {self.max_grids}")
        print(f"  训练轮数: {epochs}")
        print(f"  批次大小: {batch_size}")
        print(f"  测试模型: {len(models_to_test)} 个")
        print("=" * 80)
        
        # 加载数据
        self.load_and_prepare_data()
        
        # 逐个训练模型
        for i, model_key in enumerate(models_to_test, 1):
            print(f"\n📊 实验 {i}/{len(models_to_test)}: {model_key}")
            print("-" * 50)
            
            try:
                results = self.train_single_model(
                    model_key=model_key,
                    epochs=epochs,
                    batch_size=batch_size
                )
                self.results[model_key] = results
                print(f"✅ {results['model_name']} 训练完成")
                
            except Exception as e:
                print(f"❌ {model_key} 训练失败: {e}")
                continue
        
        print(f"\n🎯 消融实验完成! 成功训练 {len(self.results)} 个模型")
        return self.results
    
    def create_comparison_report(self):
        """创建对比报告"""
        if not self.results:
            print("❌ 没有实验结果可供对比")
            return None
        
        print("\n📊 创建消融实验对比报告...")
        
        # 创建对比表格
        comparison_data = []
        for model_key, results in self.results.items():
            comparison_data.append({
                '模型': results['model_name'],
                '架构': results['model_architecture'],
                '参数量': results['total_params'],
                '训练时间(秒)': results['training_time'],
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
        
        # 按验证集R²排序
        comparison_df = comparison_df.sort_values('验证_R²', ascending=False)
        
        return comparison_df
    
    def create_visualizations(self, save_dir='ablation_study/results'):
        """创建可视化图表"""
        if not self.results:
            print("❌ 没有实验结果可供可视化")
            return
        
        print("📊 创建可视化图表...")
        
        # 创建结果目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 模型性能对比图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🧪 消融实验结果对比', fontsize=16, fontweight='bold')
        
        model_names = [self.results[key]['model_name'] for key in self.results.keys()]
        
        # R²对比
        val_r2 = [self.results[key]['val_r2'] for key in self.results.keys()]
        test_r2 = [self.results[key]['test_r2'] for key in self.results.keys()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, val_r2, width, label='验证集', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_r2, width, label='测试集', alpha=0.8)
        axes[0, 0].set_title('R² 分数对比')
        axes[0, 0].set_ylabel('R² 分数')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE对比
        val_rmse = [self.results[key]['val_rmse'] for key in self.results.keys()]
        test_rmse = [self.results[key]['test_rmse'] for key in self.results.keys()]
        
        axes[0, 1].bar(x - width/2, val_rmse, width, label='验证集', alpha=0.8)
        axes[0, 1].bar(x + width/2, test_rmse, width, label='测试集', alpha=0.8)
        axes[0, 1].set_title('RMSE 对比')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 参数量和训练时间对比
        params = [self.results[key]['total_params'] for key in self.results.keys()]
        train_times = [self.results[key]['training_time'] for key in self.results.keys()]
        
        ax2 = axes[1, 0]
        ax2_twin = ax2.twinx()
        
        bars1 = ax2.bar(x - width/2, params, width, label='参数量', alpha=0.8, color='skyblue')
        bars2 = ax2_twin.bar(x + width/2, train_times, width, label='训练时间(秒)', alpha=0.8, color='lightcoral')
        
        ax2.set_title('模型复杂度对比')
        ax2.set_ylabel('参数量', color='skyblue')
        ax2_twin.set_ylabel('训练时间(秒)', color='lightcoral')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 添加图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # MAE对比
        val_mae = [self.results[key]['val_mae'] for key in self.results.keys()]
        test_mae = [self.results[key]['test_mae'] for key in self.results.keys()]
        
        axes[1, 1].bar(x - width/2, val_mae, width, label='验证集', alpha=0.8)
        axes[1, 1].bar(x + width/2, test_mae, width, label='测试集', alpha=0.8)
        axes[1, 1].set_title('MAE 对比')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存对比图
        comparison_path = os.path.join(save_dir, 'ablation_study_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 对比图保存至: {comparison_path}")
        
        # 2. 为每个模型创建详细的预测图
        for model_key, results in self.results.items():
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'{results["model_name"]} - 预测结果详情', fontsize=14, fontweight='bold')
            
            # 验证集预测 vs 真实值
            y_val_true = results['y_val_true']
            y_val_pred = results['y_val_pred']
            
            axes[0].scatter(y_val_true, y_val_pred, alpha=0.5, s=1)
            min_val = min(y_val_true.min(), y_val_pred.min())
            max_val = max(y_val_true.max(), y_val_pred.max())
            axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[0].set_title(f'验证集 (R² = {results["val_r2"]:.4f})')
            axes[0].set_xlabel('真实值')
            axes[0].set_ylabel('预测值')
            axes[0].grid(True, alpha=0.3)
            
            # 测试集预测 vs 真实值
            y_test_true = results['y_test_true']
            y_test_pred = results['y_test_pred']
            
            axes[1].scatter(y_test_true, y_test_pred, alpha=0.5, s=1)
            min_val = min(y_test_true.min(), y_test_pred.min())
            max_val = max(y_test_true.max(), y_test_pred.max())
            axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[1].set_title(f'测试集 (R² = {results["test_r2"]:.4f})')
            axes[1].set_xlabel('真实值')
            axes[1].set_ylabel('预测值')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存每个模型的详细图
            model_path = os.path.join(save_dir, f'{model_key}_detailed_results.png')
            plt.savefig(model_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 所有可视化图表已保存至: {save_dir}/")
    
    def save_results(self, save_dir='ablation_study/results'):
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
            comparison_path = os.path.join(save_dir, 'ablation_study_comparison.csv')
            comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
            print(f"📊 对比报告: {comparison_path}")
        
        # 2. 保存详细结果
        summary_data = {
            'experiment_info': {
                'name': 'milano_ablation_study',
                'description': 'Milano网络流量预测消融实验',
                'datetime': datetime.now().isoformat(),
                'sequence_length': self.sequence_length,
                'max_grids': self.max_grids,
                'device': str(device)
            },
            'models_tested': list(self.results.keys()),
            'results_summary': {}
        }
        
        # 3. 保存每个模型的详细结果
        for model_key, results in self.results.items():
            # 保存模型状态
            model_path = os.path.join(save_dir, f'{model_key}_model.pth')
            torch.save(results['model_state_dict'], model_path)
            
            # 保存预测结果
            predictions_df = pd.DataFrame({
                'val_true': results['y_val_true'],
                'val_pred': results['y_val_pred'],
                'val_error': results['y_val_pred'] - results['y_val_true'],
                'test_true': results['y_test_true'][:len(results['y_val_true'])],
                'test_pred': results['y_test_pred'][:len(results['y_val_true'])],
                'test_error': (results['y_test_pred'] - results['y_test_true'])[:len(results['y_val_true'])]
            })
            pred_path = os.path.join(save_dir, f'{model_key}_predictions.csv')
            predictions_df.to_csv(pred_path, index=False)
            
            # 添加到汇总数据
            summary_data['results_summary'][model_key] = {
                'model_name': results['model_name'],
                'architecture': results['model_architecture'],
                'total_params': int(results['total_params']),
                'training_time': float(results['training_time']),
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
        summary_path = os.path.join(save_dir, 'ablation_study_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"📈 汇总结果: {summary_path}")
        
        print(f"💾 所有结果已保存至: {save_dir}/")
        
        return save_dir


def main():
    """主函数"""
    print("🧪 Milano网络流量预测 - 消融实验")
    print("=" * 80)
    
    # 创建实验实例
    experiment = AblationExperiment(sequence_length=8, max_grids=300)
    
    # 运行消融实验
    results = experiment.run_ablation_study(epochs=30, batch_size=32)
    
    if not results:
        print("❌ 实验失败，没有成功训练的模型")
        return
    
    # 创建对比报告
    comparison_df = experiment.create_comparison_report()
    if comparison_df is not None:
        print("\n📊 消融实验结果汇总:")
        print("=" * 100)
        print(comparison_df.round(4).to_string(index=False))
        
        # 找出最佳模型
        best_model_idx = comparison_df['验证_R²'].idxmax()
        best_model = comparison_df.iloc[best_model_idx]
        print(f"\n🏆 最佳模型: {best_model['模型']}")
        print(f"   验证集R²: {best_model['验证_R²']:.4f}")
        print(f"   测试集R²: {best_model['测试_R²']:.4f}")
        print(f"   参数量: {best_model['参数量']:,}")
        print(f"   训练时间: {best_model['训练时间(秒)']:.1f}秒")
    
    # 创建可视化
    experiment.create_visualizations()
    
    # 保存结果
    save_dir = experiment.save_results()
    
    print(f"\n🎯 消融实验完成!")
    print(f"📁 所有结果保存在: {save_dir}")
    print(f"🔍 查看对比图: {save_dir}/ablation_study_comparison.png")


if __name__ == "__main__":
    main()