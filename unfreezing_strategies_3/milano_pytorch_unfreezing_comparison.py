#!/usr/bin/env python3
"""
Milano Network Traffic Prediction - PyTorch Unfreezing Strategies Comparison
===========================================================================

使用PyTorch实现的迁移学习解冻策略对比实验
包含基线模型（从头训练）用于对比
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import json
import math
import copy
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# GPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 文件路径
PRETRAINED_MODEL_PATH = 'optimal_features_training/results_corrected/corrected_optimal_features_model.pth'
DATA_PATH = 'data/cleaned_dataset_30/cleaned-sms-call-internet-mi-2013-11-01.csv'

class FastTrainingModel(nn.Module):
    """与预训练模型完全相同的架构"""
    def __init__(self, input_size, sequence_length):
        super(FastTrainingModel, self).__init__()
        
        # 原始架构（与预训练模型保持一致）
        self.gru = nn.GRU(input_size, 64, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm = nn.LSTM(128, 32, batch_first=True, dropout=0.2)
        self.dense = nn.Linear(32, 64)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        x, _ = self.gru(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后时间步
        x = F.relu(self.dense(x))
        x = self.dropout(x)
        x = self.output(x)
        return x
    
    def get_layer_groups(self):
        """获取可以分组解冻的层"""
        return {
            'feature_extractor': [self.gru],  # GRU作为特征提取器
            'sequence_processor': [self.lstm],  # LSTM作为序列处理器
            'classifier': [self.dense, self.dropout, self.output]  # 分类器层
        }

class UnfreezingStrategy:
    """解冻策略基类"""
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.unfrozen_params_history = []
    
    def apply_strategy(self, model, epoch, total_epochs, **kwargs):
        """应用解冻策略"""
        raise NotImplementedError
    
    def get_trainable_ratio(self, model):
        """计算可训练参数比例"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return trainable_params / total_params if total_params > 0 else 0

class BaselineStrategy(UnfreezingStrategy):
    """基线策略：从头训练（不使用预训练模型）"""
    def __init__(self):
        super().__init__("Baseline (From Scratch)", "Train from scratch without pre-trained weights")
    
    def apply_strategy(self, model, epoch, total_epochs, **kwargs):
        # 所有参数都可训练
        for param in model.parameters():
            param.requires_grad = True

class NoUnfreezingStrategy(UnfreezingStrategy):
    """策略1: 不解冻 - 只训练分类器"""
    def __init__(self):
        super().__init__("No Unfreezing", "Only train the classifier, freeze feature extractor")
    
    def apply_strategy(self, model, epoch, total_epochs, **kwargs):
        # 冻结特征提取器和序列处理器
        for param in model.gru.parameters():
            param.requires_grad = False
        for param in model.lstm.parameters():
            param.requires_grad = False
        
        # 只解冻分类器
        for param in model.dense.parameters():
            param.requires_grad = True
        for param in model.output.parameters():
            param.requires_grad = True

class FullUnfreezingStrategy(UnfreezingStrategy):
    """策略2: 全量解冻"""
    def __init__(self):
        super().__init__("Full Unfreezing", "Unfreeze all layers from the beginning")
    
    def apply_strategy(self, model, epoch, total_epochs, **kwargs):
        for param in model.parameters():
            param.requires_grad = True

class LinearProgressiveStrategy(UnfreezingStrategy):
    """策略3: 线性渐进解冻"""
    def __init__(self, start_epoch=5, end_epoch=15):
        super().__init__("Linear Progressive", f"Linear unfreezing from epoch {start_epoch} to {end_epoch}")
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
    
    def apply_strategy(self, model, epoch, total_epochs, **kwargs):
        if epoch < self.start_epoch:
            # 只训练分类器
            for param in model.gru.parameters():
                param.requires_grad = False
            for param in model.lstm.parameters():
                param.requires_grad = False
            for param in model.dense.parameters():
                param.requires_grad = True
            for param in model.output.parameters():
                param.requires_grad = True
        elif epoch >= self.end_epoch:
            # 全部解冻
            for param in model.parameters():
                param.requires_grad = True
        else:
            # 渐进解冻
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            
            # 分类器始终可训练
            for param in model.dense.parameters():
                param.requires_grad = True
            for param in model.output.parameters():
                param.requires_grad = True
            
            if progress >= 0.5:
                # 解冻序列处理器
                for param in model.lstm.parameters():
                    param.requires_grad = True
            else:
                for param in model.lstm.parameters():
                    param.requires_grad = False
            
            if progress >= 1.0:
                # 解冻特征提取器
                for param in model.gru.parameters():
                    param.requires_grad = True
            else:
                for param in model.gru.parameters():
                    param.requires_grad = False

class ExponentialProgressiveStrategy(UnfreezingStrategy):
    """策略4: 指数渐进解冻"""
    def __init__(self, start_epoch=3, decay_rate=0.3):
        super().__init__("Exponential Progressive", f"Exponential unfreezing starting at epoch {start_epoch}")
        self.start_epoch = start_epoch
        self.decay_rate = decay_rate
    
    def apply_strategy(self, model, epoch, total_epochs, **kwargs):
        if epoch < self.start_epoch:
            # 只训练分类器
            for param in model.gru.parameters():
                param.requires_grad = False
            for param in model.lstm.parameters():
                param.requires_grad = False
            for param in model.dense.parameters():
                param.requires_grad = True
            for param in model.output.parameters():
                param.requires_grad = True
        else:
            # 指数进度
            progress = 1 - math.exp(-self.decay_rate * (epoch - self.start_epoch))
            
            # 分类器始终可训练
            for param in model.dense.parameters():
                param.requires_grad = True
            for param in model.output.parameters():
                param.requires_grad = True
            
            if progress >= 0.3:
                # 解冻序列处理器
                for param in model.lstm.parameters():
                    param.requires_grad = True
            else:
                for param in model.lstm.parameters():
                    param.requires_grad = False
            
            if progress >= 0.7:
                # 解冻特征提取器
                for param in model.gru.parameters():
                    param.requires_grad = True
            else:
                for param in model.gru.parameters():
                    param.requires_grad = False

class StepUnfreezingStrategy(UnfreezingStrategy):
    """策略5: 阶段性解冻"""
    def __init__(self, schedule=None):
        if schedule is None:
            schedule = {0: 'classifier', 5: 'sequence_processor', 10: 'feature_extractor'}
        super().__init__("Step Unfreezing", f"Step-wise unfreezing at epochs {list(schedule.keys())}")
        self.schedule = schedule
        self.unfrozen_groups = set(['classifier'])  # 默认解冻分类器
    
    def apply_strategy(self, model, epoch, total_epochs, **kwargs):
        # 检查是否需要解冻新的层组
        if epoch in self.schedule:
            group_to_unfreeze = self.schedule[epoch]
            self.unfrozen_groups.add(group_to_unfreeze)
            print(f"    🔓 Unfreezing {group_to_unfreeze} at epoch {epoch}")
        
        # 冻结所有层
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻已计划的层组
        if 'classifier' in self.unfrozen_groups:
            for param in model.dense.parameters():
                param.requires_grad = True
            for param in model.output.parameters():
                param.requires_grad = True
        
        if 'sequence_processor' in self.unfrozen_groups:
            for param in model.lstm.parameters():
                param.requires_grad = True
        
        if 'feature_extractor' in self.unfrozen_groups:
            for param in model.gru.parameters():
                param.requires_grad = True

class AdaptiveUnfreezingStrategy(UnfreezingStrategy):
    """策略6: 自适应解冻"""
    def __init__(self, patience=3, improvement_threshold=0.01):
        super().__init__("Adaptive Unfreezing", "Adaptive unfreezing based on validation loss")
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_stage = 0
        self.stages = ['classifier', 'sequence_processor', 'feature_extractor']
        self.unfrozen_groups = set(['classifier'])  # 分类器默认解冻
    
    def apply_strategy(self, model, epoch, total_epochs, val_loss=None, **kwargs):
        # 检查是否需要解冻更多层
        if val_loss is not None:
            improvement = self.best_val_loss - val_loss
            
            if improvement > self.improvement_threshold:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 如果验证损失停止改善，解冻更多层
            if (self.patience_counter >= self.patience and 
                self.current_stage < len(self.stages) - 1):
                self.current_stage += 1
                new_group = self.stages[self.current_stage]
                self.unfrozen_groups.add(new_group)
                self.patience_counter = 0
                print(f"    📈 Adaptive unfreezing: {new_group} at epoch {epoch}")
        
        # 冻结所有层
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻当前阶段的层组
        if 'classifier' in self.unfrozen_groups:
            for param in model.dense.parameters():
                param.requires_grad = True
            for param in model.output.parameters():
                param.requires_grad = True
        
        if 'sequence_processor' in self.unfrozen_groups:
            for param in model.lstm.parameters():
                param.requires_grad = True
        
        if 'feature_extractor' in self.unfrozen_groups:
            for param in model.gru.parameters():
                param.requires_grad = True

class MilanoUnfreezingExperiment:
    """Milano解冻策略对比实验类"""
    
    def __init__(self, sequence_length=8, max_grids=300):
        self.sequence_length = sequence_length
        self.max_grids = max_grids  # 匹配baseline: 300个网格
        self.optimal_features = [
            'internet_traffic', 'grid_feature', 'is_morning', 'is_afternoon', 
            'is_evening', 'is_night', 'is_peak_hour', 'is_workday',
            'traffic_rolling_mean_3', 'traffic_rolling_mean_6', 'traffic_rolling_mean_12',
            'traffic_pct_change', 'traffic_log_return'
        ]
        self.results = {}
        self.data = None
        
    def load_and_prepare_data(self):
        """加载并准备目标域数据 - 匹配baseline方法"""
        print(f"📥 Loading target domain data (max_grids: {self.max_grids})...")
        
        # 加载数据 - 与baseline相同
        df = pd.read_csv(DATA_PATH)
        df['time_interval'] = pd.to_datetime(df['time_interval'])
        df = df.sort_values(['square_id', 'time_interval'])
        
        print(f"原始数据: {len(df):,} 行, {df['square_id'].nunique()} 个网格")
        
        # 选择高质量网格 - 与baseline相同的策略
        grid_stats = df.groupby('square_id')['internet_traffic'].agg([
            'mean', 'count', 'std', 'min', 'max'
        ]).reset_index()
        
        quality_grids = grid_stats[
            (grid_stats['count'] == 48) &
            (grid_stats['std'] > 3) &
            (grid_stats['mean'] > 5) &
            (grid_stats['max'] > 20)
        ].sort_values('mean', ascending=False)
        
        selected_grids = quality_grids['square_id'].head(self.max_grids).tolist()
        self.data = df[df['square_id'].isin(selected_grids)].copy()
        
        print(f"选择的网格: {len(selected_grids)} / {df['square_id'].nunique()}")
        print(f"过滤后数据: {len(self.data):,} 行")
        
        # 创建所有特征 - 与baseline相同
        self.create_all_features()
        
        # 创建序列数据
        X, y = self.create_sequences(self.optimal_features)
        
        print(f"   Sequence data shape: X={X.shape}, y={y.shape}")
        return X, y
    
    def create_all_features(self):
        """
        与baseline完全相同的特征工程
        """
        print("🔧 Creating comprehensive feature set...")
        
        data = self.data.copy()  # 使用副本避免修改原始数据
        
        # 基础时间特征
        data['hour'] = data['time_interval'].dt.hour
        data['day_of_week'] = data['time_interval'].dt.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_workday'] = (data['day_of_week'] < 5).astype(int)
        
        # 周期性编码
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # 时段分类 - 修正的定义（与baseline完全一致）
        data['is_morning'] = ((data['hour'] >= 6) & (data['hour'] < 12)).astype(int)
        data['is_afternoon'] = ((data['hour'] >= 12) & (data['hour'] < 18)).astype(int)
        data['is_evening'] = ((data['hour'] >= 18) & (data['hour'] < 24)).astype(int)  # 修正：到24点
        data['is_night'] = ((data['hour'] >= 0) & (data['hour'] < 6)).astype(int)      # 修正：0-6点
        data['is_peak_hour'] = ((data['hour'] >= 8) & (data['hour'] <= 10) | 
                               (data['hour'] >= 17) & (data['hour'] <= 19)).astype(int)
        
        # 滞后特征
        for lag in range(1, 7):  # lag1 到 lag6
            data[f'traffic_lag{lag}'] = data.groupby('square_id')['internet_traffic'].shift(lag)
        
        # 滚动统计特征
        for window in [3, 6, 12]:
            data[f'traffic_rolling_mean_{window}'] = data.groupby('square_id')['internet_traffic'].rolling(
                window=window, min_periods=1).mean().reset_index(0, drop=True)
            data[f'traffic_rolling_std_{window}'] = data.groupby('square_id')['internet_traffic'].rolling(
                window=window, min_periods=1).std().reset_index(0, drop=True)
            data[f'traffic_rolling_max_{window}'] = data.groupby('square_id')['internet_traffic'].rolling(
                window=window, min_periods=1).max().reset_index(0, drop=True)
            data[f'traffic_rolling_min_{window}'] = data.groupby('square_id')['internet_traffic'].rolling(
                window=window, min_periods=1).min().reset_index(0, drop=True)
        
        # 差分特征
        for diff in [1, 2]:
            data[f'traffic_diff{diff}'] = data.groupby('square_id')['internet_traffic'].diff(diff)
        
        # 变化率特征 - 修正的定义（与baseline完全一致）
        data['traffic_pct_change'] = data.groupby('square_id')['internet_traffic'].pct_change()
        data['traffic_log_return'] = data.groupby('square_id')['internet_traffic'].transform(
            lambda x: np.log(x / x.shift(1)))  # 修正：使用原始公式
        
        # 网格特征 - 与baseline一致
        grid_mapping = {grid_id: idx for idx, grid_id in enumerate(self.data['square_id'].unique())}
        data['grid_feature'] = data['square_id'].map(grid_mapping)
        data['grid_normalized'] = data['grid_feature'] / len(grid_mapping)
        
        # 填充缺失值 - 与baseline一致
        try:
            self.data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        except Exception as e:
            print(f"⚠️  Warning: Using alternative fillna method due to: {e}")
            # 使用新的pandas语法
            self.data = data.ffill().bfill().fillna(0)
        
        print(f"✅ Created {len(self.data.columns) - 3} features")  # 减去时间、square_id、internet_traffic
    
    def create_sequences(self, features):
        """
        与baseline完全相同的序列创建
        """
        X_all, y_all = [], []
        
        # 确保所有特征都存在
        available_features = [f for f in features if f in self.data.columns]
        if len(available_features) != len(features):
            missing = set(features) - set(available_features)
            print(f"⚠️  缺少特征: {missing}")
            return None, None
        
        for grid_id in self.data['square_id'].unique():
            grid_data = self.data[self.data['square_id'] == grid_id].sort_values('time_interval')
            
            if len(grid_data) < self.sequence_length + 1:
                continue
                
            feature_data = grid_data[available_features].values
            target_data = grid_data['internet_traffic'].values
            
            for i in range(len(feature_data) - self.sequence_length):
                X_all.append(feature_data[i:(i + self.sequence_length)])
                y_all.append(target_data[i + self.sequence_length])
        
        if len(X_all) == 0:
            return None, None
            
        X = np.array(X_all)
        y = np.array(y_all)
        
        # 特征标准化（与baseline相同）
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # 目标变换（与baseline相同）
        y_transformed = np.log1p(y + 1.0)
        
        # 保存scaler用于逆变换
        self.feature_scaler = scaler
        
        return X_scaled, y_transformed
    
    def load_pretrained_model(self):
        """加载预训练模型"""
        print("📥 Loading pretrained model...")
        
        if not os.path.exists(PRETRAINED_MODEL_PATH):
            raise FileNotFoundError(f"Pretrained model not found: {PRETRAINED_MODEL_PATH}")
        
        model = FastTrainingModel(len(self.optimal_features), self.sequence_length)
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        
        print("✅ Pretrained model loaded successfully")
        return model
    
    def create_baseline_model(self):
        """创建基线模型（从头训练）"""
        print("🆕 Creating baseline model (from scratch)...")
        model = FastTrainingModel(len(self.optimal_features), self.sequence_length)
        model = model.to(device)
        print("✅ Baseline model created")
        return model
    
    def train_with_strategy(self, strategy, X, y, epochs=30, batch_size=32, learning_rate=None):
        """使用指定策略训练模型"""
        print(f"\n🚀 Training with {strategy.name}...")
        print(f"   Strategy: {strategy.description}")
        
        # 数据分割（与baseline相同：70%-15%-15%）
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        print(f"   Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
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
        if isinstance(strategy, BaselineStrategy):
            model = self.create_baseline_model()
        else:
            model = self.load_pretrained_model()
        
        # 与baseline相同的学习率设置
        if learning_rate is None:
            learning_rate = 0.001  # 与baseline一致
        
        print(f"   📚 Using learning rate: {learning_rate} (baseline compatible)")
        
        # 优化器和损失函数 - 与baseline一致
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 训练记录
        train_losses = []
        val_losses = []
        unfrozen_params_history = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # 应用解冻策略
            val_loss_for_adaptive = val_losses[-1] if val_losses else None
            strategy.apply_strategy(model, epoch, epochs, val_loss=val_loss_for_adaptive)
            
            # 记录可训练参数比例
            trainable_ratio = strategy.get_trainable_ratio(model)
            unfrozen_params_history.append(trainable_ratio)
            
            # 训练
            model.train()
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
            
            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}, "
                      f"Val Loss={val_loss:.6f}, Trainable={trainable_ratio:.1%}")
        
        training_time = time.time() - start_time
        
        # 最终评估
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor).cpu().numpy().squeeze()
            test_pred = model(X_test_tensor).cpu().numpy().squeeze()
        
        # 逆变换（与baseline相同）
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
        
        results = {
            'strategy_name': strategy.name,
            'strategy_description': strategy.description,
            'is_baseline': isinstance(strategy, BaselineStrategy),
            'training_time': training_time,
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'unfrozen_params_history': unfrozen_params_history,
            'final_trainable_ratio': unfrozen_params_history[-1],
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
        
        print(f"   ✅ Training completed in {training_time:.1f}s")
        print(f"   📊 Val: RMSE={val_rmse:.2f}, R²={val_r2:.4f}, MAE={val_mae:.2f}")
        print(f"   📊 Test: RMSE={test_rmse:.2f}, R²={test_r2:.4f}, MAE={test_mae:.2f}")
        
        return results
    
    def run_unfreezing_comparison(self, epochs=30):
        """运行解冻策略对比实验"""
        print("🎯 Milano PyTorch Unfreezing Strategies Comparison")
        print("=" * 80)
        
        # 加载目标域数据
        X, y = self.load_and_prepare_data()
        
        # 定义实验策略
        strategies = [
            BaselineStrategy(),  # 基线模型（从头训练）
            NoUnfreezingStrategy(),  # 不解冻
            FullUnfreezingStrategy(),  # 全量解冻
            LinearProgressiveStrategy(start_epoch=5, end_epoch=15),  # 线性渐进
            ExponentialProgressiveStrategy(start_epoch=3, decay_rate=0.3),  # 指数渐进
            StepUnfreezingStrategy(),  # 阶段性解冻
            AdaptiveUnfreezingStrategy(patience=3, improvement_threshold=0.01)  # 自适应解冻
        ]
        
        print(f"\n🧪 Running experiments with {len(strategies)} strategies...")
        print(f"📊 Epochs per strategy: {epochs}")
        print(f"📊 Target domain data: {len(X):,} samples")
        
        # 运行每个策略
        for i, strategy in enumerate(strategies, 1):
            print(f"\n📋 Experiment {i}/{len(strategies)}: {strategy.name}")
            print("-" * 60)
            
            try:
                results = self.train_with_strategy(strategy, X, y, epochs=epochs)
                self.results[strategy.name] = results
                
            except Exception as e:
                print(f"❌ Strategy {strategy.name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n🎉 All experiments completed! Results for {len(self.results)} strategies")
        return self.results
    
    def create_comprehensive_visualization(self, save_dir='pytorch_unfreezing_results'):
        """创建全面的对比可视化"""
        if not self.results:
            print("❌ No results to visualize")
            return
        
        print("📊 Creating comprehensive visualizations...")
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置样式
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        
        # 1. 主要性能对比图
        self.create_main_comparison_plot(save_dir)
        
        # 2. 训练过程对比图
        self.create_training_process_plot(save_dir)
        
        # 3. 解冻策略时间线图
        self.create_unfreezing_timeline_plot(save_dir)
        
        # 4. 基线模型vs迁移学习对比图
        self.create_baseline_comparison_plot(save_dir)
        
        # 5. 效率分析图
        self.create_efficiency_analysis_plot(save_dir)
    
    def create_main_comparison_plot(self, save_dir):
        """创建主要性能对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('PyTorch Unfreezing Strategies: Performance Comparison', fontsize=16, fontweight='bold')
        
        strategies = list(self.results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
        
        # 为基线模型使用不同颜色
        baseline_color = '#FF6B6B'
        
        # 计算迁移学习策略数量
        transfer_count = sum(1 for strategy in strategies if not self.results[strategy]['is_baseline'])
        
        if transfer_count > 0:
            transfer_colors = plt.cm.Set2(np.linspace(0, 1, transfer_count))
        else:
            transfer_colors = []
        
        final_colors = []
        transfer_idx = 0
        for strategy in strategies:
            if self.results[strategy]['is_baseline']:
                final_colors.append(baseline_color)
            else:
                if transfer_idx < len(transfer_colors):
                    final_colors.append(transfer_colors[transfer_idx])
                    transfer_idx += 1
                else:
                    final_colors.append('#CCCCCC')  # 默认颜色
        
        # 性能指标对比
        metrics = ['val_r2', 'test_r2', 'val_rmse', 'test_rmse', 'val_mae', 'test_mae']
        metric_names = ['Validation R²', 'Test R²', 'Validation RMSE', 'Test RMSE', 'Validation MAE', 'Test MAE']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 3, idx % 3]
            
            values = [self.results[strategy][metric] for strategy in strategies]
            bars = ax.bar(range(len(strategies)), values, color=final_colors, alpha=0.8)
            
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Strategy')
            ax.set_ylabel(name.split()[-1])
            ax.set_xticks(range(len(strategies)))
            ax.set_xticklabels([s.replace(' ', '\n') for s in strategies], rotation=0, fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 标记基线模型
            for i, (bar, strategy) in enumerate(zip(bars, strategies)):
                if self.results[strategy]['is_baseline']:
                    bar.set_edgecolor('red')
                    bar.set_linewidth(2)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}' if 'r2' in metric else f'{value:.1f}',
                       ha='center', va='bottom', fontsize=9)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=baseline_color, label='Baseline (From Scratch)')
        ]
        
        if len(transfer_colors) > 0:
            legend_elements.append(
                Patch(facecolor=transfer_colors[0], label='Transfer Learning')
            )
        
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        main_plot_path = os.path.join(save_dir, 'pytorch_unfreezing_comparison.png')
        plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Main comparison plot: {main_plot_path}")
    
    def create_baseline_comparison_plot(self, save_dir):
        """创建基线vs迁移学习对比图"""
        baseline_results = None
        transfer_results = {}
        
        for strategy, results in self.results.items():
            if results['is_baseline']:
                baseline_results = results
            else:
                transfer_results[strategy] = results
        
        if not baseline_results:
            print("⚠️  No baseline results found for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Baseline vs Transfer Learning Comparison', fontsize=16, fontweight='bold')
        
        # R² 改进对比
        strategies = list(transfer_results.keys())
        val_r2_improvements = [(transfer_results[s]['val_r2'] - baseline_results['val_r2']) 
                              for s in strategies]
        test_r2_improvements = [(transfer_results[s]['test_r2'] - baseline_results['test_r2']) 
                               for s in strategies]
        
        axes[0, 0].barh(strategies, val_r2_improvements, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('R² Improvement over Baseline (Validation)')
        axes[0, 0].set_xlabel('R² Improvement')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].barh(strategies, test_r2_improvements, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('R² Improvement over Baseline (Test)')
        axes[0, 1].set_xlabel('R² Improvement')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)
        
        # RMSE 改进对比（负值表示改进）
        val_rmse_improvements = [(baseline_results['val_rmse'] - transfer_results[s]['val_rmse']) 
                                 for s in strategies]
        test_rmse_improvements = [(baseline_results['test_rmse'] - transfer_results[s]['test_rmse']) 
                                 for s in strategies]
        
        axes[1, 0].barh(strategies, val_rmse_improvements, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('RMSE Improvement over Baseline (Validation)')
        axes[1, 0].set_xlabel('RMSE Improvement')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].barh(strategies, test_rmse_improvements, alpha=0.7, color='gold')
        axes[1, 1].set_title('RMSE Improvement over Baseline (Test)')
        axes[1, 1].set_xlabel('RMSE Improvement')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        baseline_plot_path = os.path.join(save_dir, 'baseline_vs_transfer_comparison.png')
        plt.savefig(baseline_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Baseline comparison plot: {baseline_plot_path}")
    
    def create_training_process_plot(self, save_dir):
        """创建训练过程对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Training Process Comparison', fontsize=14, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for (strategy, results), color in zip(self.results.items(), colors):
            epochs = range(1, len(results['train_losses']) + 1)
            
            # 训练损失
            axes[0, 0].plot(epochs, results['train_losses'], 
                           color=color, label=strategy, linewidth=2, alpha=0.8)
            
            # 验证损失
            axes[0, 1].plot(epochs, results['val_losses'], 
                           color=color, label=strategy, linewidth=2, alpha=0.8)
            
            # 可训练参数比例
            axes[1, 0].plot(epochs, results['unfrozen_params_history'], 
                           color=color, label=strategy, linewidth=2, alpha=0.8)
        
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Trainable Parameters Ratio')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1.1)
        
        # 效率对比（R²/time）
        strategies = list(self.results.keys())
        efficiency = [self.results[s]['val_r2'] / self.results[s]['training_time'] 
                     for s in strategies]
        
        bars = axes[1, 1].bar(range(len(strategies)), efficiency, 
                             color=colors, alpha=0.7)
        axes[1, 1].set_title('Training Efficiency (R² / Training Time)')
        axes[1, 1].set_xlabel('Strategy')
        axes[1, 1].set_ylabel('Efficiency')
        axes[1, 1].set_xticks(range(len(strategies)))
        axes[1, 1].set_xticklabels([s.replace(' ', '\n') for s in strategies], 
                                  rotation=0, fontsize=8)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        process_plot_path = os.path.join(save_dir, 'training_process_comparison.png')
        plt.savefig(process_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Training process plot: {process_plot_path}")
    
    def create_unfreezing_timeline_plot(self, save_dir):
        """创建解冻时间线图"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        fig.suptitle('Unfreezing Timeline Comparison', fontsize=14, fontweight='bold')
        
        y_pos = 0
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for (strategy, results), color in zip(self.results.items(), colors):
            epochs = range(1, len(results['unfrozen_params_history']) + 1)
            ratios = results['unfrozen_params_history']
            
            # 绘制解冻比例变化
            ax.plot(epochs, [y_pos + r * 0.8 for r in ratios], 
                   color=color, linewidth=3, alpha=0.8, label=strategy)
            
            # 添加策略名称
            ax.text(0.5, y_pos + 0.4, strategy, fontsize=10, fontweight='bold', 
                   verticalalignment='center')
            
            y_pos += 1
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Strategy')
        ax.set_title('Parameter Unfreezing Progress Over Time')
        ax.set_yticks(range(len(self.results)))
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        timeline_plot_path = os.path.join(save_dir, 'unfreezing_timeline.png')
        plt.savefig(timeline_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Unfreezing timeline plot: {timeline_plot_path}")
    
    def create_efficiency_analysis_plot(self, save_dir):
        """创建效率分析图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Efficiency Analysis', fontsize=14, fontweight='bold')
        
        strategies = list(self.results.keys())
        training_times = [self.results[s]['training_time'] for s in strategies]
        val_r2_scores = [self.results[s]['val_r2'] for s in strategies]
        test_r2_scores = [self.results[s]['test_r2'] for s in strategies]
        
        # 训练时间vs性能散点图
        colors = ['red' if self.results[s]['is_baseline'] else 'blue' for s in strategies]
        
        scatter = axes[0].scatter(training_times, val_r2_scores, 
                                 c=colors, s=100, alpha=0.7)
        axes[0].set_xlabel('Training Time (seconds)')
        axes[0].set_ylabel('Validation R²')
        axes[0].set_title('Training Time vs Performance')
        axes[0].grid(True, alpha=0.3)
        
        # 添加标签
        for i, strategy in enumerate(strategies):
            axes[0].annotate(strategy.replace(' ', '\n'), 
                           (training_times[i], val_r2_scores[i]),
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
        
        # 性能/时间比率
        efficiency_ratios = [val_r2_scores[i] / training_times[i] for i in range(len(strategies))]
        bars = axes[1].bar(range(len(strategies)), efficiency_ratios, 
                          color=colors, alpha=0.7)
        axes[1].set_title('Performance Efficiency (R² / Training Time)')
        axes[1].set_xlabel('Strategy')
        axes[1].set_ylabel('Efficiency Ratio')
        axes[1].set_xticks(range(len(strategies)))
        axes[1].set_xticklabels([s.replace(' ', '\n') for s in strategies], 
                               rotation=0, fontsize=8)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Baseline'),
            Patch(facecolor='blue', label='Transfer Learning')
        ]
        axes[0].legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        efficiency_plot_path = os.path.join(save_dir, 'efficiency_analysis.png')
        plt.savefig(efficiency_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Efficiency analysis plot: {efficiency_plot_path}")
    
    def save_results(self, save_dir='pytorch_unfreezing_results'):
        """保存实验结果"""
        if not self.results:
            print("❌ No results to save")
            return
        
        print("💾 Saving experiment results...")
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 保存汇总对比表
        comparison_data = []
        for strategy, results in self.results.items():
            comparison_data.append({
                'Strategy': strategy,
                'Type': 'Baseline' if results['is_baseline'] else 'Transfer Learning',
                'Description': results['strategy_description'],
                'Training_Time(s)': results['training_time'],
                'Final_Trainable_Ratio': results['final_trainable_ratio'],
                'Val_RMSE': results['val_rmse'],
                'Val_R2': results['val_r2'],
                'Val_MAE': results['val_mae'],
                'Val_MAPE': results['val_mape'],
                'Test_RMSE': results['test_rmse'],
                'Test_R2': results['test_r2'],
                'Test_MAE': results['test_mae'],
                'Test_MAPE': results['test_mape']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = os.path.join(save_dir, 'pytorch_unfreezing_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"📊 Comparison table: {comparison_path}")
        
        # 2. 保存详细结果JSON
        summary_data = {
            'experiment_info': {
                'name': 'milano_pytorch_unfreezing_strategies',
                'description': 'PyTorch implementation of unfreezing strategies with baseline comparison',
                'datetime': datetime.now().isoformat(),
                'target_domain_grids': self.max_grids,
                'sequence_length': self.sequence_length,
                'device': str(device)
            },
            'strategies_tested': list(self.results.keys()),
            'detailed_results': {}
        }
        
        for strategy, results in self.results.items():
            strategy_key = strategy.replace(' ', '_').lower().replace('(', '').replace(')', '')
            
            # 保存预测结果
            predictions_df = pd.DataFrame({
                'val_true': results['y_val_true'],
                'val_pred': results['y_val_pred'],
                'val_error': results['y_val_pred'] - results['y_val_true'],
                'test_true': results['y_test_true'][:len(results['y_val_true'])],
                'test_pred': results['y_test_pred'][:len(results['y_val_true'])],
                'test_error': (results['y_test_pred'] - results['y_test_true'])[:len(results['y_val_true'])]
            })
            pred_path = os.path.join(save_dir, f'{strategy_key}_predictions.csv')
            predictions_df.to_csv(pred_path, index=False)
            
            # 添加到汇总
            summary_data['detailed_results'][strategy_key] = {
                'strategy_name': strategy,
                'is_baseline': bool(results['is_baseline']),
                'training_time': float(results['training_time']),
                'epochs': int(results['epochs']),
                'final_trainable_ratio': float(results['final_trainable_ratio']),
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
        
        summary_path = os.path.join(save_dir, 'pytorch_unfreezing_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"📈 Summary JSON: {summary_path}")
        
        print(f"💾 All results saved to: {save_dir}/")
        return save_dir
    
    def print_final_summary(self):
        """打印最终汇总"""
        if not self.results:
            print("❌ No results to summarize")
            return
        
        print("\n" + "=" * 90)
        print("🎉 PYTORCH UNFREEZING STRATEGIES COMPARISON WITH BASELINE")
        print("=" * 90)
        
        # 分离基线和迁移学习结果
        baseline_results = None
        transfer_results = {}
        
        for strategy, results in self.results.items():
            if results['is_baseline']:
                baseline_results = results
            else:
                transfer_results[strategy] = results
        
        # 按验证R²排序迁移学习结果
        sorted_transfer = sorted(transfer_results.items(), 
                               key=lambda x: x[1]['val_r2'], reverse=True)
        
        print("📊 PERFORMANCE RANKING:")
        print("-" * 90)
        
        if baseline_results:
            print("🔥 BASELINE (From Scratch):")
            print(f"   📈 Val R²: {baseline_results['val_r2']:.4f}, Test R²: {baseline_results['test_r2']:.4f}")
            print(f"   📉 Val RMSE: {baseline_results['val_rmse']:.2f}, Test RMSE: {baseline_results['test_rmse']:.2f}")
            print(f"   ⏱️  Training Time: {baseline_results['training_time']:.1f}s")
            print()
        
        print("🚀 TRANSFER LEARNING STRATEGIES:")
        for rank, (strategy, results) in enumerate(sorted_transfer, 1):
            print(f"{rank}. {strategy}")
            print(f"   📈 Val R²: {results['val_r2']:.4f}, Test R²: {results['test_r2']:.4f}")
            print(f"   📉 Val RMSE: {results['val_rmse']:.2f}, Test RMSE: {results['test_rmse']:.2f}")
            print(f"   ⏱️  Training Time: {results['training_time']:.1f}s")
            
            # 与基线对比
            if baseline_results:
                r2_improvement = results['val_r2'] - baseline_results['val_r2']
                rmse_improvement = baseline_results['val_rmse'] - results['val_rmse']
                print(f"   🎯 vs Baseline: R² {r2_improvement:+.4f}, RMSE {rmse_improvement:+.2f}")
            print()
        
        # 最佳策略分析
        if sorted_transfer:
            best_strategy, best_results = sorted_transfer[0]
            print(f"🏆 BEST TRANSFER LEARNING STRATEGY: {best_strategy}")
            print(f"   📊 Validation R²: {best_results['val_r2']:.4f}")
            print(f"   📊 Test R²: {best_results['test_r2']:.4f}")
            print(f"   📊 Description: {best_results['strategy_description']}")
        
        # 迁移学习vs基线对比
        if baseline_results and sorted_transfer:
            print(f"\n💡 TRANSFER LEARNING IMPACT:")
            best_transfer = sorted_transfer[0][1]
            r2_gain = best_transfer['val_r2'] - baseline_results['val_r2']
            rmse_gain = baseline_results['val_rmse'] - best_transfer['val_rmse']
            time_saving = baseline_results['training_time'] - best_transfer['training_time']
            
            print(f"   📈 Best R² gain: {r2_gain:+.4f}")
            print(f"   📉 Best RMSE improvement: {rmse_gain:+.2f}")
            print(f"   ⚡ Time saving: {time_saving:+.1f}s ({time_saving/baseline_results['training_time']*100:+.1f}%)")
        
        print("=" * 90)

def main():
    """主函数"""
    print("🚀 Milano PyTorch Unfreezing Strategies Comparison")
    print("=" * 80)
    
    # 检查文件是否存在
    if not os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"❌ Pretrained model not found: {PRETRAINED_MODEL_PATH}")
        return
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found: {DATA_PATH}")
        return
    
    print("✅ All required files found")
    
    # 创建实验实例 - 使用与baseline相同的配置
    experiment = MilanoUnfreezingExperiment(sequence_length=8, max_grids=300)
    
    print(f"🎯 Experiment Configuration:")
    print(f"   Target domain grids: {experiment.max_grids}")
    print(f"   Sequence length: {experiment.sequence_length}")
    print(f"   Features: {len(experiment.optimal_features)}")
    print(f"   Including baseline (from scratch) comparison")
    
    # 询问用户是否继续
    response = input("\nStart PyTorch unfreezing experiments? (y/n): ").lower().strip()
    if response not in ['y', 'yes', '是', '']:
        print("❌ Experiment cancelled")
        return
    
    try:
        # 运行实验
        print(f"\n🧪 Starting PyTorch unfreezing strategies comparison...")
        start_time = time.time()
        
        results = experiment.run_unfreezing_comparison(epochs=30)
        
        if not results:
            print("❌ No successful experiments!")
            return
        
        total_time = time.time() - start_time
        
        # 生成可视化和保存结果
        experiment.create_comprehensive_visualization()
        save_dir = experiment.save_results()
        
        # 打印最终汇总
        experiment.print_final_summary()
        
        print(f"\n⏱️  Total experiment time: {total_time:.1f} seconds")
        print(f"📁 Results saved to: {save_dir}/")
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
