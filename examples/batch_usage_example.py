#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量推理使用示例
展示如何使用阿里云百炼的批量推理功能来生成数据集，成本仅为实时推理的50%
"""

import os
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from batch_inference import BatchInferenceManager, QianWenBatchInference
from data import QianWenDataGenerator


async def example_basic_batch_inference():
    """基本批量推理示例"""
    print("=== 基本批量推理示例 ===")
    
    # 获取API密钥
    api_key = os.getenv('QIANWEN_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("错误: 请设置环境变量 QIANWEN_API_KEY 或 DASHSCOPE_API_KEY")
        return
    
    # 创建测试提示
    prompts = [
        "请简单解释一下量子叠加原理",
        "什么是量子纠缠？请用通俗易懂的语言解释",
        "Shor算法是如何工作的？",
        "量子计算与经典计算的主要区别是什么？"
    ]
    
    # 创建批量推理管理器
    manager = BatchInferenceManager(api_key, model='qwen-plus')
    
    try:
        # 运行批量推理
        result = await manager.run_batch_inference(
            prompts=prompts,
            job_name="basic_example",
            completion_window="24h",
            wait_for_completion=True,  # 等待完成
            temperature=0.7,
            max_tokens=1000
        )
        
        print(f"批量任务完成: {result['job_id']}")
        print(f"状态: {result['status']}")
        print(f"结果数量: {len(result.get('results', []))}")
        
        return result
        
    except Exception as e:
        print(f"批量推理失败: {e}")
        return None


async def example_dataset_generation_with_batch():
    """使用批量推理生成数据集的示例"""
    print("\n=== 数据集批量生成示例 ===")
    
    # 获取API密钥
    api_key = os.getenv('QIANWEN_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("错误: 请设置环境变量")
        return
    
    # 使用数据生成器的批量模式
    async with QianWenDataGenerator(api_key, model='qwen-plus') as generator:
        await generator.generate_dataset_from_papers(
            output_file="batch_example_dataset.jsonl",
            max_samples_per_paper=2,  # 每篇论文2个样本
            use_batch=True,  # 启用批量推理
            batch_completion_window="24h"
        )
    
    print("数据集批量生成完成!")


async def example_batch_job_management():
    """批量任务管理示例"""
    print("\n=== 批量任务管理示例 ===")
    
    api_key = os.getenv('QIANWEN_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("错误: 请设置环境变量")
        return
    
    client = QianWenBatchInference(api_key, model='qwen-plus')
    
    try:
        # 列出所有批量任务
        jobs = client.list_batch_jobs(limit=5)
        print(f"找到 {len(jobs)} 个批量任务:")
        
        for job in jobs:
            print(f"  - ID: {job.id}")
            print(f"    状态: {job.status.value}")
            print(f"    创建时间: {job.created_at}")
            if job.completed_at:
                print(f"    完成时间: {job.completed_at}")
            print()
            
    except Exception as e:
        print(f"获取批量任务列表失败: {e}")


def example_cli_usage():
    """CLI使用示例"""
    print("\n=== CLI使用示例 ===")
    
    print("1. 创建批量任务:")
    print("   python batch_cli.py create --prompt '解释量子计算原理' --job-name test --wait")
    print()
    
    print("2. 从文件创建批量任务:")
    print("   python batch_cli.py create --input-file prompts.txt --job-name my_batch")
    print()
    
    print("3. 检查任务状态:")
    print("   python batch_cli.py status batch_12345")
    print()
    
    print("4. 列出所有任务:")
    print("   python batch_cli.py list")
    print()
    
    print("5. 取消任务:")
    print("   python batch_cli.py cancel batch_12345")


def show_cost_comparison():
    """显示成本对比"""
    print("\n=== 成本对比 ===")
    print("假设处理1000个请求：")
    print("• 实时推理成本: 100% (基准)")
    print("• 批量推理成本: 50% (节省50%)")
    print("• 节省金额: 如果实时成本为100元，批量推理仅需50元")
    print("\n批量推理特点:")
    print("• ✅ 成本降低50%")
    print("• ✅ 适合大规模数据处理")
    print("• ✅ 支持离线处理")
    print("• ⏰ 需要等待时间(24h-336h)")
    print("• 📊 最适合非紧急的大批量任务")


async def main():
    """主函数"""
    print("阿里云百炼批量推理使用示例")
    print("="*50)
    
    # 显示成本对比
    show_cost_comparison()
    
    # 基本批量推理示例
    result = await example_basic_batch_inference()
    
    # 如果用户确认，运行数据集生成示例
    if result:
        user_input = input("\n是否运行数据集批量生成示例? (y/N): ")
        if user_input.lower() in ['y', 'yes']:
            await example_dataset_generation_with_batch()
    
    # 批量任务管理示例
    await example_batch_job_management()
    
    # CLI使用示例
    example_cli_usage()
    
    print("\n示例演示完成!")
    print("要开始使用批量推理，请:")
    print("1. 设置API密钥: export DASHSCOPE_API_KEY='your-key'")
    print("2. 运行: python data.py --batch --completion-window 24h")
    print("3. 或使用CLI: python batch_cli.py create --input-file prompts.txt")


if __name__ == "__main__":
    # 创建必要的目录
    Path("batch_jobs").mkdir(exist_ok=True)
    
    # 运行示例
    asyncio.run(main())