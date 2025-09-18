#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量推理命令行工具
提供批量任务的创建、监控和管理功能
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from batch_inference import BatchInferenceManager, QianWenBatchInference
# from utils import setup_logging  # 暂时注释掉，使用简单的日志设置


def setup_cli_logging():
    """设置CLI日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def get_api_key():
    """获取API密钥"""
    api_key = os.getenv('QIANWEN_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("错误: 请设置环境变量 QIANWEN_API_KEY 或 DASHSCOPE_API_KEY")
        print("使用方法: export DASHSCOPE_API_KEY='your-api-key-here'")
        sys.exit(1)
    return api_key


def create_batch_job(args):
    """创建批量推理任务"""
    api_key = get_api_key()
    manager = BatchInferenceManager(api_key, args.model)
    
    # 从文件读取提示
    if args.input_file:
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                if args.input_file.endswith('.jsonl'):
                    prompts = []
                    for line in f:
                        data = json.loads(line)
                        prompts.append(data.get('prompt', ''))
                else:
                    prompts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"读取输入文件失败: {e}")
            return
    else:
        prompts = [args.prompt] if args.prompt else []
    
    if not prompts:
        print("错误: 没有找到有效的提示")
        return
    
    print(f"准备创建批量任务，包含 {len(prompts)} 个请求")
    
    # 模型参数
    model_params = {
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'top_p': args.top_p
    }
    
    try:
        import asyncio
        result = asyncio.run(manager.run_batch_inference(
            prompts=prompts,
            job_name=args.job_name,
            completion_window=args.completion_window,
            wait_for_completion=args.wait,
            **model_params
        ))
        
        print(f"批量任务创建成功!")
        print(f"任务ID: {result['job_id']}")
        print(f"任务名称: {result['job_name']}")
        print(f"状态: {result['status']}")
        print(f"请求数量: {result['request_count']}")
        print(f"完成窗口: {result['completion_window']}")
        
        if args.wait and result.get('results'):
            print(f"任务已完成，获得 {len(result['results'])} 个结果")
            if args.output:
                save_results(result['results'], args.output)
        
        # 保存任务信息
        job_info_file = Path("batch_jobs") / f"{result['job_name']}_info.json"
        job_info_file.parent.mkdir(exist_ok=True)
        with open(job_info_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"任务信息已保存: {job_info_file}")
        
    except Exception as e:
        print(f"创建批量任务失败: {e}")


def check_batch_status(args):
    """检查批量任务状态"""
    api_key = get_api_key()
    client = QianWenBatchInference(api_key, args.model)
    
    try:
        batch_job = client.get_batch_status(args.job_id)
        
        print(f"任务ID: {batch_job.id}")
        print(f"状态: {batch_job.status.value}")
        print(f"创建时间: {batch_job.created_at}")
        if batch_job.completed_at:
            print(f"完成时间: {batch_job.completed_at}")
        print(f"完成窗口: {batch_job.completion_window}")
        if batch_job.metadata:
            print(f"元数据: {json.dumps(batch_job.metadata, ensure_ascii=False, indent=2)}")
        
        # 如果任务完成且有输出文件，下载结果
        if batch_job.status.value == "completed" and batch_job.output_file_id:
            if args.download:
                output_file = Path("batch_jobs") / f"{args.job_id}_output.jsonl"
                client.download_batch_results(batch_job.output_file_id, str(output_file))
                
                results = client.parse_batch_results(str(output_file))
                print(f"结果已下载: {output_file} ({len(results)} 个结果)")
                
                if args.output:
                    save_results(results, args.output)
    
    except Exception as e:
        print(f"检查任务状态失败: {e}")


def list_batch_jobs(args):
    """列出批量任务"""
    api_key = get_api_key()
    client = QianWenBatchInference(api_key, args.model)
    
    try:
        jobs = client.list_batch_jobs(limit=args.limit)
        
        if not jobs:
            print("没有找到批量任务")
            return
        
        print(f"找到 {len(jobs)} 个批量任务:")
        print("-" * 80)
        
        for job in jobs:
            print(f"ID: {job.id}")
            print(f"状态: {job.status.value}")
            print(f"创建时间: {job.created_at}")
            if job.completed_at:
                print(f"完成时间: {job.completed_at}")
            if job.metadata:
                name = job.metadata.get('ds_name', 'Unknown')
                print(f"任务名称: {name}")
            print("-" * 40)
    
    except Exception as e:
        print(f"列出批量任务失败: {e}")


def cancel_batch_job(args):
    """取消批量任务"""
    api_key = get_api_key()
    client = QianWenBatchInference(api_key, args.model)
    
    try:
        success = client.cancel_batch_job(args.job_id)
        if success:
            print(f"批量任务 {args.job_id} 已取消")
        else:
            print(f"取消批量任务 {args.job_id} 失败")
    
    except Exception as e:
        print(f"取消批量任务失败: {e}")


def save_results(results: List[Dict[str, Any]], output_file: str):
    """保存结果到文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                # 提取实际的响应内容
                if 'response' in result and 'body' in result['response']:
                    response_body = result['response']['body']
                    if 'choices' in response_body and response_body['choices']:
                        content = response_body['choices'][0]['message']['content']
                        custom_id = result.get('custom_id', '')
                        
                        # 保存为指令微调格式
                        try:
                            # 尝试解析JSON内容
                            if content.startswith('{'):
                                data = json.loads(content)
                                sample = {
                                    "instruction": data.get("instruction", ""),
                                    "input": data.get("input", ""),
                                    "output": data.get("output", ""),
                                    "metadata": {
                                        "custom_id": custom_id,
                                        "generated_at": datetime.now().isoformat(),
                                        "batch_inference": True
                                    }
                                }
                                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                        except json.JSONDecodeError:
                            # 如果不是JSON，直接保存原始内容
                            sample = {
                                "instruction": "batch_generated",
                                "input": "",
                                "output": content,
                                "metadata": {
                                    "custom_id": custom_id,
                                    "generated_at": datetime.now().isoformat(),
                                    "batch_inference": True
                                }
                            }
                            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"保存结果失败: {e}")


def main():
    setup_cli_logging()
    
    parser = argparse.ArgumentParser(description='阿里云百炼批量推理工具')
    parser.add_argument('--model', '-m', default='qwen-plus', help='模型名称')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 创建批量任务
    create_parser = subparsers.add_parser('create', help='创建批量推理任务')
    create_parser.add_argument('--input-file', '-i', help='输入文件路径（每行一个提示或JSONL格式）')
    create_parser.add_argument('--prompt', '-p', help='单个提示（仅用于测试）')
    create_parser.add_argument('--job-name', '-n', help='任务名称')
    create_parser.add_argument('--completion-window', '-w', default='24h', 
                              choices=['24h', '48h', '72h', '168h', '336h'], help='完成时间窗口')
    create_parser.add_argument('--wait', action='store_true', help='等待任务完成')
    create_parser.add_argument('--output', '-o', help='输出文件路径')
    create_parser.add_argument('--temperature', type=float, default=0.7, help='温度参数')
    create_parser.add_argument('--max-tokens', type=int, default=2000, help='最大token数')
    create_parser.add_argument('--top-p', type=float, default=0.9, help='top_p参数')
    
    # 检查任务状态
    status_parser = subparsers.add_parser('status', help='检查批量任务状态')
    status_parser.add_argument('job_id', help='批量任务ID')
    status_parser.add_argument('--download', action='store_true', help='下载结果文件')
    status_parser.add_argument('--output', '-o', help='输出文件路径')
    
    # 列出批量任务
    list_parser = subparsers.add_parser('list', help='列出批量任务')
    list_parser.add_argument('--limit', type=int, default=20, help='返回数量限制')
    
    # 取消批量任务
    cancel_parser = subparsers.add_parser('cancel', help='取消批量任务')
    cancel_parser.add_argument('job_id', help='批量任务ID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'create':
        create_batch_job(args)
    elif args.command == 'status':
        check_batch_status(args)
    elif args.command == 'list':
        list_batch_jobs(args)
    elif args.command == 'cancel':
        cancel_batch_job(args)


if __name__ == "__main__":
    main()