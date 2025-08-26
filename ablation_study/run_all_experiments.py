#!/usr/bin/env python3
"""
Milano网络流量预测 - 一键运行所有消融实验
=====================================

便捷脚本：一次性运行所有消融实验并生成完整报告
"""

import os
import sys
import time
import json
from datetime import datetime

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ablation_experiment import AblationExperiment


def print_header():
    """打印程序头部信息"""
    print("=" * 100)
    print("🧪 Milano网络流量预测 - 消融实验套件")
    print("=" * 100)
    print("实验内容:")
    print("  1️⃣  单向GRU模型")
    print("  2️⃣  单向GRU+LSTM组合模型")
    print("  3️⃣  纯LSTM模型")
    print("  4️⃣  基线模型(双向GRU+LSTM)")
    print("")
    print("实验目标:")
    print("  🎯 比较不同序列建模架构的性能")
    print("  📊 分析模型复杂度与性能的权衡")
    print("  📈 生成详细的对比报告和可视化")
    print("=" * 100)


def print_model_info():
    """打印模型架构信息"""
    from models import get_all_model_info
    
    print("\n📋 模型架构详情:")
    print("-" * 80)
    
    model_info = get_all_model_info()
    for i, (key, info) in enumerate(model_info.items(), 1):
        print(f"{i}. {info['name']}")
        print(f"   描述: {info['description']}")
        print(f"   架构: {info['architecture']}")
        print("")


def run_experiment_with_config(config):
    """使用指定配置运行实验"""
    print(f"\n🚀 开始运行实验...")
    print(f"配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 50)
    
    # 创建实验实例
    experiment = AblationExperiment(
        sequence_length=config['sequence_length'],
        max_grids=config['max_grids']
    )
    
    # 运行实验
    start_time = time.time()
    
    results = experiment.run_ablation_study(
        models_to_test=config.get('models_to_test', None),
        epochs=config['epochs'],
        batch_size=config['batch_size']
    )
    
    total_time = time.time() - start_time
    
    if not results:
        print("❌ 实验失败!")
        return None, None
    
    print(f"\n⏱️  总实验时间: {total_time:.1f} 秒")
    
    # 生成报告和可视化
    comparison_df = experiment.create_comparison_report()
    experiment.create_visualizations()
    save_dir = experiment.save_results()
    
    return results, save_dir, comparison_df


def print_final_summary(results, comparison_df, save_dir, total_time):
    """打印最终总结"""
    print("\n" + "=" * 100)
    print("🎉 消融实验完成!")
    print("=" * 100)
    
    if comparison_df is not None:
        print("📊 实验结果汇总:")
        print("-" * 80)
        
        # 显示简化的对比表
        summary_cols = ['模型', '参数量', '验证_R²', '测试_R²', '训练时间(秒)']
        print(comparison_df[summary_cols].round(4).to_string(index=False))
        
        # 最佳模型信息
        best_idx = comparison_df['验证_R²'].idxmax()
        best_model = comparison_df.iloc[best_idx]
        
        print(f"\n🏆 最佳模型: {best_model['模型']}")
        print(f"   📈 验证集R²: {best_model['验证_R²']:.4f}")
        print(f"   📈 测试集R²: {best_model['测试_R²']:.4f}")
        print(f"   🔧 参数量: {best_model['参数量']:,}")
        print(f"   ⏱️  训练时间: {best_model['训练时间(秒)']:.1f}秒")
        
        # 性能分析
        print(f"\n📈 性能分析:")
        print(f"   最高R²: {comparison_df['验证_R²'].max():.4f}")
        print(f"   最低R²: {comparison_df['验证_R²'].min():.4f}")
        print(f"   R²差异: {comparison_df['验证_R²'].max() - comparison_df['验证_R²'].min():.4f}")
        print(f"   最少参数: {comparison_df['参数量'].min():,}")
        print(f"   最多参数: {comparison_df['参数量'].max():,}")
        
        # 效率分析
        best_efficiency_idx = (comparison_df['验证_R²'] / comparison_df['训练时间(秒)']).idxmax()
        efficient_model = comparison_df.iloc[best_efficiency_idx]
        print(f"   最高效模型: {efficient_model['模型']} (R²/时间 = {efficient_model['验证_R²']/efficient_model['训练时间(秒)']:.4f})")
    
    print(f"\n📁 结果文件:")
    print(f"   📊 对比图表: {save_dir}/ablation_study_comparison.png")
    print(f"   📈 汇总数据: {save_dir}/ablation_study_summary.json")
    print(f"   📋 对比表格: {save_dir}/ablation_study_comparison.csv")
    print(f"   💾 模型文件: {save_dir}/*_model.pth")
    
    print(f"\n⏱️  总实验耗时: {total_time:.1f} 秒")
    print(f"📁 所有结果保存在: {save_dir}")
    print("=" * 100)


def create_experiment_config():
    """创建实验配置"""
    return {
        'sequence_length': 8,      # 序列长度 
        'max_grids': 300,          # 最大网格数（与基线模型一致）
        'epochs': 30,              # 训练轮数
        'batch_size': 32,          # 批次大小
        'models_to_test': [        # 要测试的模型
            'unidirectional_gru',     # 单向GRU
            'unidirectional_gru_lstm', # 单向GRU+LSTM
            'pure_lstm',              # 纯LSTM
            'baseline'                # 基线模型
        ]
    }


def save_experiment_config(config, save_dir):
    """保存实验配置"""
    config_path = os.path.join(save_dir, 'experiment_config.json')
    config_with_timestamp = {
        'timestamp': datetime.now().isoformat(),
        'config': config
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_with_timestamp, f, indent=2, ensure_ascii=False)
    
    print(f"⚙️  实验配置保存: {config_path}")


def main():
    """主函数"""
    # 打印头部信息
    print_header()
    
    # 显示模型信息
    print_model_info()
    
    # 询问用户是否继续
    print("🚀 准备开始消融实验...")
    response = input("是否继续? (y/n): ").lower().strip()
    
    if response not in ['y', 'yes', '是', '']:
        print("❌ 实验已取消")
        sys.exit(0)
    
    # 创建实验配置
    config = create_experiment_config()
    
    print(f"\n⚙️  使用配置:")
    for key, value in config.items():
        if key == 'models_to_test':
            print(f"  {key}: {len(value)} 个模型")
        else:
            print(f"  {key}: {value}")
    
    # 运行实验
    total_start_time = time.time()
    
    try:
        results, save_dir, comparison_df = run_experiment_with_config(config)
        
        if results is None:
            print("❌ 实验失败!")
            sys.exit(1)
        
        total_time = time.time() - total_start_time
        
        # 保存实验配置
        save_experiment_config(config, save_dir)
        
        # 打印最终总结
        print_final_summary(results, comparison_df, save_dir, total_time)
        
    except KeyboardInterrupt:
        print("\n⚠️  实验被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 实验过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()