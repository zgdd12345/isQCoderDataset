#!/usr/bin/env python3
"""
数据处理工具函数
提供数据集处理、验证和分析功能
"""

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import asdict
from datetime import datetime


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        logger: 配置好的logger对象
    """
    logger = logging.getLogger(__name__)
    
    if not logger.handlers:
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        log_file = f'dataset_processing_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.setLevel(getattr(logging, log_level.upper()))
    return logger


def extract_paper_title(markdown_content: str) -> str:
    """
    从markdown内容中提取论文标题
    
    Args:
        markdown_content: markdown文件内容
    
    Returns:
        str: 论文标题，如果找不到则返回"未知标题"
    """
    lines = markdown_content.split('\n')
    
    # 查找第一个一级标题
    for line in lines:
        line = line.strip()
        if line.startswith('# ') and len(line) > 2:
            return line[2:].strip()
    
    return "未知标题"


def extract_paper_abstract(markdown_content: str) -> str:
    """
    从markdown内容中提取论文摘要
    
    Args:
        markdown_content: markdown文件内容
    
    Returns:
        str: 论文摘要，如果找不到则返回空字符串
    """
    lines = markdown_content.split('\n')
    abstract_started = False
    abstract_lines = []
    
    for line in lines:
        line = line.strip()
        
        # 查找Abstract部分
        if re.match(r'^#+\s*(abstract|摘要)', line, re.IGNORECASE):
            abstract_started = True
            continue
        
        # 如果已经开始Abstract部分
        if abstract_started:
            # 遇到下一个标题则停止
            if line.startswith('#'):
                break
            # 收集非空行
            if line:
                abstract_lines.append(line)
    
    return ' '.join(abstract_lines)


def clean_text(text: str) -> str:
    """
    清理文本内容，移除多余的空白字符和格式
    
    Args:
        text: 待清理的文本
    
    Returns:
        str: 清理后的文本
    """
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除markdown图片标记
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # 移除markdown链接，保留文本
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # 移除latex数学公式的部分格式
    text = re.sub(r'\$+', '', text)
    
    return text.strip()


def validate_dataset_sample(sample: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    验证数据集样本的格式和内容
    
    Args:
        sample: 数据集样本字典
    
    Returns:
        Tuple[bool, Optional[str]]: (是否有效, 错误信息)
    """
    required_fields = ['instruction', 'input', 'output']
    
    # 检查必需字段
    for field in required_fields:
        if field not in sample:
            return False, f"缺少必需字段: {field}"
    
    # 检查instruction和output不能为空
    if not sample['instruction'].strip():
        return False, "instruction字段不能为空"
    
    if not sample['output'].strip():
        return False, "output字段不能为空"
    
    # 检查字段长度限制
    if len(sample['instruction']) > 1000:
        return False, "instruction字段过长（超过1000字符）"
    
    if len(sample['output']) > 5000:
        return False, "output字段过长（超过5000字符）"
    
    return True, None


def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    加载JSONL格式的数据集文件
    
    Args:
        file_path: JSONL文件路径
    
    Returns:
        List[Dict[str, Any]]: 数据集样本列表
    """
    samples = []
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                logging.warning(f"第{line_num}行JSON解析失败: {e}")
                continue
    
    return samples


def save_jsonl_dataset(samples: List[Dict[str, Any]], file_path: str) -> None:
    """
    保存数据集为JSONL格式
    
    Args:
        samples: 数据集样本列表
        file_path: 输出文件路径
    """
    # 确保输出目录存在
    output_dir = Path(file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            json_line = json.dumps(sample, ensure_ascii=False)
            f.write(json_line + '\n')


def analyze_dataset(file_path: str) -> Dict[str, Any]:
    """
    分析数据集的统计信息
    
    Args:
        file_path: 数据集文件路径
    
    Returns:
        Dict[str, Any]: 数据集统计信息
    """
    samples = load_jsonl_dataset(file_path)
    
    if not samples:
        return {"total_samples": 0, "error": "数据集为空"}
    
    # 基本统计
    total_samples = len(samples)
    valid_samples = 0
    invalid_samples = 0
    
    # 文本长度统计
    instruction_lengths = []
    output_lengths = []
    
    # 分类统计（如果有metadata中的信息）
    categories = {}
    
    for sample in samples:
        is_valid, _ = validate_dataset_sample(sample)
        if is_valid:
            valid_samples += 1
            instruction_lengths.append(len(sample['instruction']))
            output_lengths.append(len(sample['output']))
        else:
            invalid_samples += 1
        
        # 统计分类信息
        if 'metadata' in sample and isinstance(sample['metadata'], dict):
            model = sample['metadata'].get('model', 'unknown')
            categories[model] = categories.get(model, 0) + 1
    
    stats = {
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "validity_rate": valid_samples / total_samples if total_samples > 0 else 0,
    }
    
    if instruction_lengths:
        stats["instruction_stats"] = {
            "avg_length": sum(instruction_lengths) / len(instruction_lengths),
            "min_length": min(instruction_lengths),
            "max_length": max(instruction_lengths)
        }
    
    if output_lengths:
        stats["output_stats"] = {
            "avg_length": sum(output_lengths) / len(output_lengths),
            "min_length": min(output_lengths),
            "max_length": max(output_lengths)
        }
    
    if categories:
        stats["model_distribution"] = categories
    
    return stats


def filter_dataset_by_quality(file_path: str, output_path: str, min_output_length: int = 50) -> None:
    """
    按质量过滤数据集
    
    Args:
        file_path: 输入数据集文件路径
        output_path: 输出数据集文件路径
        min_output_length: 最小输出长度要求
    """
    samples = load_jsonl_dataset(file_path)
    filtered_samples = []
    
    for sample in samples:
        is_valid, _ = validate_dataset_sample(sample)
        if not is_valid:
            continue
        
        # 检查输出长度
        if len(sample['output']) < min_output_length:
            continue
        
        # 检查是否包含有用信息（避免太多重复内容）
        if sample['instruction'].lower() in sample['output'].lower():
            # 如果输出只是简单重复指令，跳过
            continue
        
        filtered_samples.append(sample)
    
    save_jsonl_dataset(filtered_samples, output_path)
    
    logger = logging.getLogger(__name__)
    logger.info(f"原始样本数: {len(samples)}, 过滤后样本数: {len(filtered_samples)}")


def merge_datasets(input_files: List[str], output_file: str) -> None:
    """
    合并多个数据集文件
    
    Args:
        input_files: 输入文件列表
        output_file: 输出文件路径
    """
    all_samples = []
    
    for file_path in input_files:
        if os.path.exists(file_path):
            samples = load_jsonl_dataset(file_path)
            all_samples.extend(samples)
            logging.info(f"从 {file_path} 加载了 {len(samples)} 个样本")
        else:
            logging.warning(f"文件不存在，跳过: {file_path}")
    
    save_jsonl_dataset(all_samples, output_file)
    logging.info(f"合并完成，总计 {len(all_samples)} 个样本保存至 {output_file}")


def preview_dataset(file_path: str, num_samples: int = 3) -> None:
    """
    预览数据集内容
    
    Args:
        file_path: 数据集文件路径
        num_samples: 预览样本数量
    """
    samples = load_jsonl_dataset(file_path)
    
    print(f"\n数据集预览 ({file_path}):")
    print(f"总样本数: {len(samples)}")
    print("=" * 50)
    
    for i, sample in enumerate(samples[:num_samples]):
        print(f"\n样本 {i+1}:")
        print(f"Instruction: {sample.get('instruction', 'N/A')[:200]}...")
        print(f"Input: {sample.get('input', 'N/A')[:100]}...")
        print(f"Output: {sample.get('output', 'N/A')[:300]}...")
        if sample.get('metadata'):
            print(f"Metadata: {sample['metadata']}")
        print("-" * 30)