"""
Intelligent Feature Selection for Milano Traffic Prediction
智能特征选择工具 - 通过多次训练找到最佳特征组合
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FastTrainingModel(nn.Module):
    """快速训练的简化模型"""
    def __init__(self, input_size, sequence_length):
        super(FastTrainingModel, self).__init__()
        
        self.gru = nn.GRU(input_size, 64, batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm = nn.LSTM(128, 32, batch_first=True, dropout=0.2)
        self.dense = nn.Linear(32, 64)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        x, _ = self.gru(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后时间步
        x = torch.relu(self.dense(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

class IntelligentFeatureSelector:
    """智能特征选择器"""
    
    def __init__(self, data_path, max_grids=500, sequence_length=8):
        self.data_path = data_path
        self.max_grids = max_grids
        self.sequence_length = sequence_length
        self.results_history = []
        self.feature_importance_scores = defaultdict(list)
        
        # 加载和预处理数据
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """加载和预处理数据"""
        print("📊 Loading and preparing data...")
        
        df = pd.read_csv(self.data_path)
        df['time_interval'] = pd.to_datetime(df['time_interval'])
        df = df.sort_values(['square_id', 'time_interval'])
        
        # 选择高质量网格（减少数量以加速实验）
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
        
        # 创建所有可能的特征
        try:
            self.create_all_features()
        except Exception as e:
            print(f"⚠️  Complex feature creation failed: {e}")
            print("🔄 Falling back to basic features...")
            self.create_basic_features_only()
        
        print(f"✅ Data prepared: {len(self.data):,} records, {len(selected_grids)} grids")
        
    def create_all_features(self):
        """创建所有可能的特征"""
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
        
        # 时段分类
        data['is_morning'] = ((data['hour'] >= 6) & (data['hour'] < 12)).astype(int)
        data['is_afternoon'] = ((data['hour'] >= 12) & (data['hour'] < 18)).astype(int)
        data['is_evening'] = ((data['hour'] >= 18) & (data['hour'] < 24)).astype(int)
        data['is_night'] = ((data['hour'] >= 0) & (data['hour'] < 6)).astype(int)
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
        
        # 变化率特征
        data['traffic_pct_change'] = data.groupby('square_id')['internet_traffic'].pct_change()
        data['traffic_log_return'] = data.groupby('square_id')['internet_traffic'].transform(
            lambda x: np.log(x / x.shift(1)))
        
        # 网格特征
        grid_mapping = {grid_id: idx for idx, grid_id in enumerate(self.data['square_id'].unique())}
        data['grid_feature'] = data['square_id'].map(grid_mapping)
        data['grid_normalized'] = data['grid_feature'] / len(grid_mapping)
        
        # 填充缺失值
        try:
            self.data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        except Exception as e:
            print(f"⚠️  Warning: Using alternative fillna method due to: {e}")
            # 使用新的pandas语法
            self.data = data.ffill().bfill().fillna(0)
        
        print(f"✅ Created {len(self.data.columns) - 3} features")  # 减去时间、square_id、internet_traffic
    
    def create_basic_features_only(self):
        """创建基础特征集合（备用方案）"""
        print("🔧 Creating basic feature set (fallback)...")
        
        data = self.data.copy()
        
        # 基础时间特征
        data['hour'] = data['time_interval'].dt.hour
        data['day_of_week'] = data['time_interval'].dt.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        # 周期性编码
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        
        # 滞后特征
        for lag in range(1, 4):
            data[f'traffic_lag{lag}'] = data.groupby('square_id')['internet_traffic'].shift(lag)
        
        # 简单滚动均值
        data['rolling_mean_3'] = data.groupby('square_id')['internet_traffic'].rolling(
            window=3, min_periods=1).mean().reset_index(0, drop=True)
        
        # 网格特征
        grid_mapping = {grid_id: idx for idx, grid_id in enumerate(self.data['square_id'].unique())}
        data['grid_feature'] = data['square_id'].map(grid_mapping)
        
        # 填充缺失值
        try:
            self.data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        except:
            self.data = data.ffill().bfill().fillna(0)
        
        print(f"✅ Created {len(self.data.columns) - 3} basic features")
        
    def define_feature_groups(self):
        """定义特征分组"""
        return {
            'core': ['internet_traffic', 'grid_feature'],
            'basic_time': ['hour', 'day_of_week', 'is_weekend'],
            'time_cyclical': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'],
            'time_categories': ['is_morning', 'is_afternoon', 'is_evening', 'is_night', 'is_peak_hour', 'is_workday'],
            'lag_basic': ['traffic_lag1', 'traffic_lag2', 'traffic_lag3'],
            'lag_extended': ['traffic_lag4', 'traffic_lag5', 'traffic_lag6'],
            'rolling_mean': ['traffic_rolling_mean_3', 'traffic_rolling_mean_6', 'traffic_rolling_mean_12'],
            'rolling_stats': ['traffic_rolling_std_3', 'traffic_rolling_std_6', 'traffic_rolling_std_12',
                             'traffic_rolling_max_3', 'traffic_rolling_min_3'],
            'differences': ['traffic_diff1', 'traffic_diff2'],
            'changes': ['traffic_pct_change', 'traffic_log_return'],
            'grid_features': ['grid_normalized']
        }
    
    def create_sequences(self, features):
        """创建序列数据"""
        X_all, y_all = [], []
        
        # 确保所有特征都存在
        available_features = [f for f in features if f in self.data.columns]
        if len(available_features) != len(features):
            missing = set(features) - set(available_features)
            print(f"⚠️  Missing features: {missing}")
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
        
        # 特征标准化
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # 目标变换
        y_transformed = np.log1p(y + 1.0)
        
        return X_scaled, y_transformed
    
    def train_and_evaluate(self, features, epochs=30, batch_size=32):
        """训练和评估模型"""
        X, y = self.create_sequences(features)
        
        if X is None or len(X) < 100:
            return None
        
        # 数据分割
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
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
        model = FastTrainingModel(X.shape[2], self.sequence_length).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 训练模型
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        # 评估
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor).cpu().numpy().squeeze()
            test_pred = model(X_test_tensor).cpu().numpy().squeeze()
        
        # 逆变换
        y_val_orig = np.expm1(y_val) - 1.0
        y_test_orig = np.expm1(y_test) - 1.0
        val_pred_orig = np.expm1(val_pred) - 1.0
        test_pred_orig = np.expm1(test_pred) - 1.0
        
        # 计算指标
        val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred_orig))
        test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred_orig))
        val_r2 = r2_score(y_val_orig, val_pred_orig)
        test_r2 = r2_score(y_test_orig, test_pred_orig)
        
        return {
            'val_rmse': val_rmse,
            'test_rmse': test_rmse,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'feature_count': len(features)
        }
    
    def forward_selection(self, max_features=15):
        """前向特征选择"""
        print("🔍 Starting Forward Feature Selection...")
        
        feature_groups = self.define_feature_groups()
        
        # 从核心特征开始
        selected_features = feature_groups['core'].copy()
        remaining_groups = {k: v for k, v in feature_groups.items() if k != 'core'}
        
        best_score = float('inf')
        selection_history = []
        
        print(f"Starting with core features: {selected_features}")
        
        # 评估初始特征集
        initial_result = self.train_and_evaluate(selected_features)
        if initial_result:
            best_score = initial_result['val_rmse']
            selection_history.append({
                'features': selected_features.copy(),
                'group_added': 'core',
                'result': initial_result
            })
            print(f"Initial RMSE: {best_score:.4f}")
        
        # 逐步添加特征组
        for step in range(max_features):
            if not remaining_groups:
                break
                
            best_group = None
            best_improvement = 0
            best_result = None
            
            print(f"\n--- Step {step + 1}: Testing {len(remaining_groups)} feature groups ---")
            
            for group_name, group_features in remaining_groups.items():
                test_features = selected_features + group_features
                
                print(f"Testing group '{group_name}': {group_features}")
                
                result = self.train_and_evaluate(test_features)
                
                if result and result['val_rmse'] < best_score:
                    improvement = best_score - result['val_rmse']
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_group = group_name
                        best_result = result
                
                print(f"  RMSE: {result['val_rmse']:.4f} (improvement: {best_score - result['val_rmse']:.4f})")
            
            # 添加最佳特征组
            if best_group:
                selected_features.extend(remaining_groups[best_group])
                best_score = best_result['val_rmse']
                
                selection_history.append({
                    'features': selected_features.copy(),
                    'group_added': best_group,
                    'result': best_result
                })
                
                print(f"✅ Added group '{best_group}': RMSE improved by {best_improvement:.4f}")
                print(f"Current features ({len(selected_features)}): {selected_features}")
                
                del remaining_groups[best_group]
            else:
                print("❌ No improvement found, stopping selection")
                break
        
        return selection_history
    
    def random_search(self, num_trials=50):
        """随机搜索特征组合"""
        print("🎲 Starting Random Search...")
        
        feature_groups = self.define_feature_groups()
        all_features = []
        for group_features in feature_groups.values():
            all_features.extend(group_features)
        
        # 移除重复特征
        all_features = list(set(all_features))
        
        results = []
        
        for trial in range(num_trials):
            # 随机选择特征数量（6-20）
            num_features = np.random.randint(6, min(21, len(all_features)))
            
            # 随机选择特征
            selected_features = np.random.choice(all_features, num_features, replace=False).tolist()
            
            # 确保包含核心特征
            core_features = feature_groups['core']
            for feature in core_features:
                if feature not in selected_features:
                    selected_features.append(feature)
            
            print(f"Trial {trial + 1}/{num_trials}: Testing {len(selected_features)} features")
            
            result = self.train_and_evaluate(selected_features)
            
            if result:
                results.append({
                    'trial': trial + 1,
                    'features': selected_features,
                    'result': result
                })
                print(f"  RMSE: {result['val_rmse']:.4f}, R²: {result['val_r2']:.4f}")
        
        # 按RMSE排序
        results.sort(key=lambda x: x['result']['val_rmse'])
        
        return results
    
    def exhaustive_group_search(self):
        """穷举特征组合搜索"""
        print("🔍 Starting Exhaustive Group Search...")
        
        feature_groups = self.define_feature_groups()
        core_features = feature_groups['core']
        optional_groups = {k: v for k, v in feature_groups.items() if k != 'core'}
        
        results = []
        total_combinations = 2 ** len(optional_groups)
        
        print(f"Testing {total_combinations} group combinations...")
        
        # 生成所有可能的组合
        group_names = list(optional_groups.keys())
        
        for i in range(total_combinations):
            selected_groups = []
            features = core_features.copy()
            
            # 根据二进制表示选择特征组
            for j, group_name in enumerate(group_names):
                if i & (1 << j):
                    selected_groups.append(group_name)
                    features.extend(optional_groups[group_name])
            
            if len(features) < 4:  # 至少4个特征
                continue
            
            print(f"Combination {i + 1}/{total_combinations}: Groups {selected_groups}")
            
            result = self.train_and_evaluate(features)
            
            if result:
                results.append({
                    'combination_id': i,
                    'selected_groups': selected_groups,
                    'features': features,
                    'result': result
                })
                print(f"  RMSE: {result['val_rmse']:.4f}, Features: {len(features)}")
        
        # 按RMSE排序
        results.sort(key=lambda x: x['result']['val_rmse'])
        
        return results
    
    def analyze_results(self, forward_results, random_results, exhaustive_results):
        """分析和可视化结果"""
        print("\n" + "="*80)
        print("📊 FEATURE SELECTION ANALYSIS RESULTS")
        print("="*80)
        
        # 找出每种方法的最佳结果
        best_forward = min(forward_results, key=lambda x: x['result']['val_rmse'])
        best_random = min(random_results, key=lambda x: x['result']['val_rmse'])
        best_exhaustive = min(exhaustive_results, key=lambda x: x['result']['val_rmse'])
        
        print("\n🏆 BEST RESULTS FROM EACH METHOD:")
        print("-" * 50)
        print(f"Forward Selection:")
        print(f"  RMSE: {best_forward['result']['val_rmse']:.4f}")
        print(f"  R²: {best_forward['result']['val_r2']:.4f}")
        print(f"  Features ({len(best_forward['features'])}): {best_forward['features']}")
        
        print(f"\nRandom Search:")
        print(f"  RMSE: {best_random['result']['val_rmse']:.4f}")
        print(f"  R²: {best_random['result']['val_r2']:.4f}")
        print(f"  Features ({len(best_random['features'])}): {best_random['features']}")
        
        print(f"\nExhaustive Search:")
        print(f"  RMSE: {best_exhaustive['result']['val_rmse']:.4f}")
        print(f"  R²: {best_exhaustive['result']['val_r2']:.4f}")
        print(f"  Groups: {best_exhaustive['selected_groups']}")
        print(f"  Features ({len(best_exhaustive['features'])}): {best_exhaustive['features']}")
        
        # 找出全局最佳
        all_best = [
            ('Forward Selection', best_forward),
            ('Random Search', best_random),
            ('Exhaustive Search', best_exhaustive)
        ]
        
        global_best = min(all_best, key=lambda x: x[1]['result']['val_rmse'])
        
        print(f"\n🥇 GLOBAL BEST RESULT:")
        print(f"Method: {global_best[0]}")
        print(f"RMSE: {global_best[1]['result']['val_rmse']:.4f}")
        print(f"R²: {global_best[1]['result']['val_r2']:.4f}")
        
        return global_best
    
    def visualize_results(self, forward_results, random_results, exhaustive_results):
        """可视化结果"""
        plt.figure(figsize=(15, 10))
        
        # 1. 前向选择进度
        plt.subplot(2, 3, 1)
        forward_rmse = [r['result']['val_rmse'] for r in forward_results]
        forward_features = [len(r['features']) for r in forward_results]
        plt.plot(forward_features, forward_rmse, 'o-', color='blue')
        plt.xlabel('Number of Features')
        plt.ylabel('Validation RMSE')
        plt.title('Forward Selection Progress')
        plt.grid(True, alpha=0.3)
        
        # 2. 随机搜索分布
        plt.subplot(2, 3, 2)
        random_rmse = [r['result']['val_rmse'] for r in random_results]
        random_features = [len(r['features']) for r in random_results]
        plt.scatter(random_features, random_rmse, alpha=0.6, color='orange')
        plt.xlabel('Number of Features')
        plt.ylabel('Validation RMSE')
        plt.title('Random Search Results')
        plt.grid(True, alpha=0.3)
        
        # 3. 穷举搜索结果
        plt.subplot(2, 3, 3)
        exhaustive_rmse = [r['result']['val_rmse'] for r in exhaustive_results[:20]]  # 只显示前20
        exhaustive_features = [len(r['features']) for r in exhaustive_results[:20]]
        plt.scatter(exhaustive_features, exhaustive_rmse, alpha=0.8, color='green')
        plt.xlabel('Number of Features')
        plt.ylabel('Validation RMSE')
        plt.title('Exhaustive Search (Top 20)')
        plt.grid(True, alpha=0.3)
        
        # 4. 方法对比
        plt.subplot(2, 3, 4)
        methods = ['Forward\nSelection', 'Random\nSearch', 'Exhaustive\nSearch']
        best_rmse = [
            min(forward_results, key=lambda x: x['result']['val_rmse'])['result']['val_rmse'],
            min(random_results, key=lambda x: x['result']['val_rmse'])['result']['val_rmse'],
            min(exhaustive_results, key=lambda x: x['result']['val_rmse'])['result']['val_rmse']
        ]
        bars = plt.bar(methods, best_rmse, color=['blue', 'orange', 'green'], alpha=0.7)
        plt.ylabel('Best RMSE')
        plt.title('Method Comparison')
        
        # 添加数值标签
        for bar, value in zip(bars, best_rmse):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 5. 特征数量 vs 性能
        plt.subplot(2, 3, 5)
        all_rmse = random_rmse + [r['result']['val_rmse'] for r in exhaustive_results]
        all_features = random_features + [len(r['features']) for r in exhaustive_results]
        plt.scatter(all_features, all_rmse, alpha=0.5, s=20)
        plt.xlabel('Number of Features')
        plt.ylabel('Validation RMSE')
        plt.title('Feature Count vs Performance')
        plt.grid(True, alpha=0.3)
        
        # 6. R² vs RMSE
        plt.subplot(2, 3, 6)
        all_r2 = [r['result']['val_r2'] for r in random_results + exhaustive_results]
        all_rmse_full = [r['result']['val_rmse'] for r in random_results + exhaustive_results]
        plt.scatter(all_r2, all_rmse_full, alpha=0.5, s=20)
        plt.xlabel('R² Score')
        plt.ylabel('RMSE')
        plt.title('R² vs RMSE')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_selection_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualization saved as 'feature_selection_analysis.png'")
    
    def save_results(self, forward_results, random_results, exhaustive_results, global_best):
        """保存结果"""
        results = {
            'forward_selection': forward_results,
            'random_search': random_results,
            'exhaustive_search': exhaustive_results,
            'global_best': {
                'method': global_best[0],
                'features': global_best[1].get('features', []),
                'result': global_best[1]['result']
            },
            'summary': {
                'total_experiments': len(forward_results) + len(random_results) + len(exhaustive_results),
                'best_rmse': global_best[1]['result']['val_rmse'],
                'best_r2': global_best[1]['result']['val_r2'],
                'optimal_feature_count': len(global_best[1].get('features', []))
            }
        }
        
        with open('feature_selection_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("💾 Results saved as 'feature_selection_results.json'")
        
    def run_complete_feature_selection(self):
        """运行完整的特征选择流程"""
        print("🚀 Starting Comprehensive Feature Selection...")
        print(f"Dataset: {len(self.data):,} records")
        print(f"Grids: {self.data['square_id'].nunique()}")
        print(f"Available features: {len(self.data.columns) - 3}")
        
        start_time = time.time()
        
        # 1. 前向选择
        forward_results = self.forward_selection(max_features=10)
        
        # 2. 随机搜索
        random_results = self.random_search(num_trials=30)
        
        # 3. 穷举组合搜索
        exhaustive_results = self.exhaustive_group_search()
        
        # 4. 分析结果
        global_best = self.analyze_results(forward_results, random_results, exhaustive_results)
        
        # 5. 可视化
        self.visualize_results(forward_results, random_results, exhaustive_results)
        
        # 6. 保存结果
        self.save_results(forward_results, random_results, exhaustive_results, global_best)
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total execution time: {elapsed_time:.1f} seconds")
        
        return global_best

def main():
    """主函数"""
    print("🎯 Intelligent Feature Selection for Milano Traffic Prediction")
    print("="*80)
    
    # 数据路径（请根据实际情况修改）
    data_path = "/root/autodl-tmp/xiaorong0802/data/cleaned_dataset_30/cleaned-sms-call-internet-mi-2013-11-01.csv"
    
    # 检查数据文件是否存在
    import os
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("💡 Please check the file path and try again")
        return
    
    print(f"📁 Using data file: {data_path}")
    
    # 创建特征选择器
    selector = IntelligentFeatureSelector(
        data_path=data_path,
        max_grids=300,  # 减少网格数量以加速实验
        sequence_length=8
    )
    
    # 运行完整的特征选择
    best_result = selector.run_complete_feature_selection()
    
    print("\n🎉 Feature selection completed!")
    print(f"🏆 Best feature combination found:")
    print(f"   Method: {best_result[0]}")
    print(f"   RMSE: {best_result[1]['result']['val_rmse']:.4f}")
    print(f"   R²: {best_result[1]['result']['val_r2']:.4f}")
    
    if 'features' in best_result[1]:
        print(f"   Features ({len(best_result[1]['features'])}): {best_result[1]['features']}")

if __name__ == "__main__":
    main()