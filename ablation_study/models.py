#!/usr/bin/env python3
"""
Milano网络流量预测 - 消融实验模型架构
=====================================

基于基线模型实现四种不同的模型架构用于消融实验：
1. 单向GRU模型
2. 单向GRU+LSTM组合模型
3. 纯LSTM模型
4. 基线模型(双向GRU+LSTM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnidirectionalGRUModel(nn.Module):
    """消融实验1: 只使用单向GRU"""
    def __init__(self, input_size, sequence_length):
        super(UnidirectionalGRUModel, self).__init__()
        self.model_name = "单向GRU"
        
        # 单向GRU层 (输出64维)
        self.gru = nn.GRU(input_size, 64, batch_first=True, bidirectional=False, dropout=0.2)
        
        # 全连接层
        self.dense = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        # GRU处理
        x, _ = self.gru(x)
        x = x[:, -1, :]  # 取最后时间步
        
        # 全连接层
        x = torch.relu(self.dense(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


class UnidirectionalGRULSTMModel(nn.Module):
    """消融实验3: 单向GRU + LSTM组合"""
    def __init__(self, input_size, sequence_length):
        super(UnidirectionalGRULSTMModel, self).__init__()
        self.model_name = "单向GRU+LSTM"
        
        # 单向GRU层 (输出64维)
        self.gru = nn.GRU(input_size, 64, batch_first=True, bidirectional=False, dropout=0.2)
        
        # LSTM层 (64输入，32输出)
        self.lstm = nn.LSTM(64, 32, batch_first=True, dropout=0.2)
        
        # 全连接层
        self.dense = nn.Linear(32, 64)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        # GRU处理
        x, _ = self.gru(x)
        
        # LSTM处理
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后时间步
        
        # 全连接层
        x = torch.relu(self.dense(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


class PureLSTMModel(nn.Module):
    """消融实验4: 只使用LSTM"""
    def __init__(self, input_size, sequence_length):
        super(PureLSTMModel, self).__init__()
        self.model_name = "纯LSTM"
        
        # LSTM层 (输出64维)
        self.lstm = nn.LSTM(input_size, 64, batch_first=True, dropout=0.2)
        
        # 全连接层
        self.dense = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        # LSTM处理
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后时间步
        
        # 全连接层
        x = torch.relu(self.dense(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


class BaselineModel(nn.Module):
    """基线模型: 双向GRU + LSTM (参考模型)"""
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


def get_model_by_name(model_name, input_size, sequence_length):
    """根据模型名称获取对应的模型实例"""
    models = {
        'unidirectional_gru': UnidirectionalGRUModel,
        'unidirectional_gru_lstm': UnidirectionalGRULSTMModel,
        'pure_lstm': PureLSTMModel,
        'baseline': BaselineModel
    }
    
    if model_name not in models:
        raise ValueError(f"未知的模型名称: {model_name}. 可选: {list(models.keys())}")
    
    return models[model_name](input_size, sequence_length)


def get_all_model_info():
    """获取所有模型的信息"""
    return {
        'unidirectional_gru': {
            'name': '单向GRU',
            'description': '使用单向GRU进行序列建模',
            'architecture': 'GRU(64) → Dense(64) → Output(1)'
        },
        'unidirectional_gru_lstm': {
            'name': '单向GRU+LSTM',
            'description': '单向GRU与LSTM的级联组合',
            'architecture': 'GRU(64) → LSTM(32) → Dense(64) → Output(1)'
        },
        'pure_lstm': {
            'name': '纯LSTM',
            'description': '只使用LSTM进行序列建模',
            'architecture': 'LSTM(64) → Dense(64) → Output(1)'
        },
        'baseline': {
            'name': '基线模型',
            'description': '双向GRU+LSTM组合(原始架构)',
            'architecture': 'BiGRU(64*2) → LSTM(32) → Dense(64) → Output(1)'
        }
    }


if __name__ == "__main__":
    # 测试所有模型
    input_size = 13  # 基于最佳特征数量
    sequence_length = 8
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🧪 消融实验模型架构测试")
    print("=" * 50)
    
    model_info = get_all_model_info()
    
    for model_key, info in model_info.items():
        print(f"\n📋 {info['name']}")
        print(f"   描述: {info['description']}")
        print(f"   架构: {info['architecture']}")
        
        # 创建模型实例
        model = get_model_by_name(model_key, input_size, sequence_length).to(device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   参数量: {total_params:,} (可训练: {trainable_params:,})")
        
        # 测试前向传播
        test_input = torch.randn(batch_size, sequence_length, input_size).to(device)
        try:
            with torch.no_grad():
                output = model(test_input)
            print(f"   输出形状: {output.shape}")
            print(f"   ✅ 模型测试通过")
        except Exception as e:
            print(f"   ❌ 模型测试失败: {e}")
    
    print(f"\n🎯 所有模型准备就绪，可以开始消融实验!")