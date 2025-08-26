#!/usr/bin/env python3
"""
Milano网络流量预测消融实验 - 模型快速测试
======================================

快速测试所有消融实验模型的架构和前向传播是否正常工作
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from models import get_model_by_name, get_all_model_info

# GPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model_architecture(model_key, input_size=13, sequence_length=8, batch_size=16):
    """
    测试单个模型架构
    
    Args:
        model_key: 模型键名
        input_size: 输入特征维度
        sequence_length: 序列长度
        batch_size: 批次大小
        
    Returns:
        dict: 测试结果
    """
    try:
        print(f"\n🧪 测试模型: {model_key}")
        print("-" * 40)
        
        # 获取模型信息
        model_info = get_all_model_info()[model_key]
        print(f"名称: {model_info['name']}")
        print(f"描述: {model_info['description']}")
        print(f"架构: {model_info['architecture']}")
        
        # 创建模型实例
        model = get_model_by_name(model_key, input_size, sequence_length).to(device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 创建测试输入
        test_input = torch.randn(batch_size, sequence_length, input_size).to(device)
        print(f"测试输入形状: {test_input.shape}")
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 测试梯度计算
        model.train()
        output = model(test_input)
        target = torch.randn(batch_size, 1).to(device)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        print(f"损失值: {loss.item():.6f}")
        print("✅ 模型测试通过")
        
        return {
            'model_key': model_key,
            'model_name': model_info['name'],
            'success': True,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'output_shape': tuple(output.shape),
            'loss_value': loss.item(),
            'error': None
        }
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return {
            'model_key': model_key,
            'model_name': model_info.get('name', 'Unknown'),
            'success': False,
            'total_params': 0,
            'trainable_params': 0,
            'output_shape': None,
            'loss_value': None,
            'error': str(e)
        }


def test_all_models():
    """测试所有模型"""
    print("🧪 Milano消融实验 - 模型架构测试")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)
    
    # 测试参数
    input_size = 13  # 基于最佳特征数量
    sequence_length = 8
    batch_size = 16
    
    print(f"测试配置:")
    print(f"  输入维度: {input_size}")
    print(f"  序列长度: {sequence_length}")
    print(f"  批次大小: {batch_size}")
    
    # 获取所有模型
    model_info = get_all_model_info()
    model_keys = list(model_info.keys())
    
    print(f"  测试模型数: {len(model_keys)}")
    
    # 测试每个模型
    test_results = []
    success_count = 0
    
    for i, model_key in enumerate(model_keys, 1):
        print(f"\n📋 [{i}/{len(model_keys)}]", end="")
        result = test_model_architecture(
            model_key, input_size, sequence_length, batch_size
        )
        test_results.append(result)
        
        if result['success']:
            success_count += 1
    
    # 打印汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    print(f"✅ 成功: {success_count}/{len(model_keys)} 个模型")
    print(f"❌ 失败: {len(model_keys) - success_count}/{len(model_keys)} 个模型")
    
    if success_count > 0:
        print("\n📈 成功的模型:")
        print("-" * 60)
        print(f"{'模型名称':<20} {'参数量':<12} {'输出形状':<15} {'损失值':<10}")
        print("-" * 60)
        
        for result in test_results:
            if result['success']:
                print(f"{result['model_name']:<20} "
                      f"{result['total_params']:,<12} "
                      f"{str(result['output_shape']):<15} "
                      f"{result['loss_value']:<10.6f}")
    
    if success_count < len(model_keys):
        print("\n❌ 失败的模型:")
        print("-" * 60)
        for result in test_results:
            if not result['success']:
                print(f"模型: {result['model_name']}")
                print(f"错误: {result['error']}")
                print("")
    
    # 性能对比
    if success_count > 1:
        print("\n📊 模型复杂度对比:")
        print("-" * 40)
        successful_results = [r for r in test_results if r['success']]
        
        # 按参数量排序
        successful_results.sort(key=lambda x: x['total_params'])
        
        min_params = successful_results[0]['total_params']
        max_params = successful_results[-1]['total_params']
        
        print(f"最少参数: {min_params:,} ({successful_results[0]['model_name']})")
        print(f"最多参数: {max_params:,} ({successful_results[-1]['model_name']})")
        print(f"参数差异: {max_params - min_params:,} ({(max_params/min_params-1)*100:.1f}%)")
    
    print(f"\n🎯 测试完成!")
    
    if success_count == len(model_keys):
        print("✅ 所有模型测试通过，可以开始消融实验!")
        return True
    else:
        print("⚠️  部分模型测试失败，请检查模型定义")
        return False


def test_data_compatibility():
    """测试数据兼容性"""
    print("\n🔗 测试数据兼容性...")
    
    try:
        # 尝试导入基线模型数据处理器
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)
        
        from optimal_features_training.milano_optimal_features_corrected import CorrectedOptimalFeaturesMilanoPredictor
        
        # 创建数据处理器实例
        processor = CorrectedOptimalFeaturesMilanoPredictor(sequence_length=8, max_grids=50)  # 使用小的max_grids进行快速测试
        
        print("✅ 基线模型数据处理器导入成功")
        print(f"   最佳特征数量: {len(processor.optimal_features)}")
        print(f"   序列长度: {processor.sequence_length}")
        print(f"   最大网格数: {processor.max_grids}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 基线模型导入失败: {e}")
        print("   请确保基线模型文件存在：optimal_features_training/milano_optimal_features_corrected.py")
        return False
    except Exception as e:
        print(f"❌ 数据兼容性测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🧪 Milano消融实验 - 完整测试套件")
    print("=" * 80)
    
    # 测试模型架构
    models_ok = test_all_models()
    
    # 测试数据兼容性
    data_ok = test_data_compatibility()
    
    # 最终结果
    print("\n" + "=" * 80)
    print("🎯 最终测试结果")
    print("=" * 80)
    print(f"模型架构测试: {'✅ 通过' if models_ok else '❌ 失败'}")
    print(f"数据兼容性测试: {'✅ 通过' if data_ok else '❌ 失败'}")
    
    if models_ok and data_ok:
        print("\n🚀 所有测试通过! 可以运行完整的消融实验:")
        print("   python run_all_experiments.py")
        print("   或")
        print("   python ablation_experiment.py")
    else:
        print("\n⚠️  测试未完全通过，请解决上述问题后再运行实验")
    
    print("=" * 80)


if __name__ == "__main__":
    main()