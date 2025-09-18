#!/usr/bin/env python3
"""
配置管理模块
安全地处理API密钥和配置参数
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class Config:
    """配置类"""
    # API配置
    qianwen_api_key: str
    qianwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qianwen_model: str = "qwen-plus"
    
    # 数据集配置
    max_samples_per_paper: int = 4
    data_dir: str = "data"
    output_file: str = "quantum_instruction_dataset.jsonl"
    
    # 生成参数
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    
    # 批量推理配置
    use_batch: bool = False
    batch_completion_window: str = "24h"
    
    # 为了兼容性保留的旧配置（标记为deprecated）
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    deepseek_model: str = "deepseek-chat"


def load_config() -> Config:
    """
    从环境变量和.env文件加载配置
    
    Returns:
        Config: 配置对象
        
    Raises:
        ValueError: 当必需的API密钥缺失时
    """
    # 加载.env文件（如果存在）
    load_dotenv()
    
    # 获取API密钥 (支持新旧两种方式)
    api_key = (os.getenv('QIANWEN_API_KEY') or 
               os.getenv('DASHSCOPE_API_KEY') or 
               os.getenv('DEEPSEEK_API_KEY'))
    
    if not api_key:
        raise ValueError(
            "未找到API密钥环境变量。\n"
            "请设置环境变量: export QIANWEN_API_KEY='your-api-key-here'\n"
            "或者: export DASHSCOPE_API_KEY='your-api-key-here'\n"
            "或者创建.env文件并添加相应的API密钥"
        )
    
    return Config(
        qianwen_api_key=api_key,
        qianwen_base_url=os.getenv('QIANWEN_BASE_URL', Config.qianwen_base_url),
        qianwen_model=os.getenv('QIANWEN_MODEL', Config.qianwen_model),
        max_samples_per_paper=int(os.getenv('MAX_SAMPLES_PER_PAPER', Config.max_samples_per_paper)),
        data_dir=os.getenv('DATA_DIR', Config.data_dir),
        output_file=os.getenv('OUTPUT_FILE', Config.output_file),
        temperature=float(os.getenv('TEMPERATURE', Config.temperature)),
        max_tokens=int(os.getenv('MAX_TOKENS', Config.max_tokens)),
        top_p=float(os.getenv('TOP_P', Config.top_p)),
        use_batch=os.getenv('USE_BATCH', '').lower() in ('true', '1', 'yes'),
        batch_completion_window=os.getenv('BATCH_COMPLETION_WINDOW', Config.batch_completion_window),
        # 兼容性支持
        deepseek_api_key=os.getenv('DEEPSEEK_API_KEY'),
        deepseek_base_url=os.getenv('DEEPSEEK_BASE_URL', Config.deepseek_base_url),
        deepseek_model=os.getenv('DEEPSEEK_MODEL', Config.deepseek_model)
    )


def validate_config(config: Config) -> None:
    """
    验证配置参数
    
    Args:
        config: 配置对象
        
    Raises:
        ValueError: 当配置参数无效时
    """
    if not config.qianwen_api_key.strip():
        raise ValueError("API密钥不能为空")
    
    if config.max_samples_per_paper <= 0:
        raise ValueError("每篇论文最大样本数必须大于0")
        
    if not (0.0 <= config.temperature <= 2.0):
        raise ValueError("temperature必须在0.0到2.0之间")
        
    if config.max_tokens <= 0:
        raise ValueError("max_tokens必须大于0")
        
    if not (0.0 <= config.top_p <= 1.0):
        raise ValueError("top_p必须在0.0到1.0之间")
    
    # 验证批量推理配置
    valid_windows = ['24h', '48h', '72h', '168h', '336h']
    if config.batch_completion_window not in valid_windows:
        raise ValueError(f"批量完成窗口必须是以下值之一: {valid_windows}")


def get_safe_config() -> Config:
    """
    安全地获取验证过的配置
    
    Returns:
        Config: 验证过的配置对象
    """
    config = load_config()
    validate_config(config)
    return config