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
    # NVIDIA API配置（优先使用）
    nvidia_api_key: Optional[str] = "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"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "deepseek-ai/deepseek-r1"
    use_nvidia: bool = True  # 是否优先使用NVIDIA API

    # API配置（作为备用）
    qianwen_api_key: str = ""
    qianwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qianwen_model: str = "qwen-plus"

    # 数据集配置
    max_samples_per_paper: int = 4
    data_dir: str = "data"
    output_file: str = "quantum_instruction_dataset.jsonl"

    # 生成参数
    temperature: float = 0.6
    max_tokens: int = 4096
    top_p: float = 0.7

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

    # 获取NVIDIA API密钥（优先）
    nvidia_api_key = os.getenv('NVIDIA_API_KEY')

    # 获取QianWen API密钥 (支持多种方式)
    qianwen_api_key = (os.getenv('QIANWEN_API_KEY') or
                       os.getenv('DASHSCOPE_API_KEY') or
                       os.getenv('DEEPSEEK_API_KEY'))

    # 至少需要一个API密钥
    if not nvidia_api_key and not qianwen_api_key:
        raise ValueError(
            "未找到API密钥环境变量。\n"
            "请设置环境变量: export NVIDIA_API_KEY='your-nvidia-api-key'\n"
            "或者: export QIANWEN_API_KEY='your-api-key-here'\n"
            "或者: export DASHSCOPE_API_KEY='your-api-key-here'\n"
            "或者创建.env文件并添加相应的API密钥"
        )

    # 确定是否使用NVIDIA API
    use_nvidia = os.getenv('USE_NVIDIA', 'true').lower() in ('true', '1', 'yes')
    if use_nvidia and not nvidia_api_key:
        use_nvidia = False  # 如果没有NVIDIA密钥，回退到QianWen

    return Config(
        # NVIDIA配置
        nvidia_api_key=nvidia_api_key,
        nvidia_base_url=os.getenv('NVIDIA_BASE_URL', Config.nvidia_base_url),
        nvidia_model=os.getenv('NVIDIA_MODEL', Config.nvidia_model),
        use_nvidia=use_nvidia,
        # QianWen配置
        qianwen_api_key=qianwen_api_key or "",
        qianwen_base_url=os.getenv('QIANWEN_BASE_URL', Config.qianwen_base_url),
        qianwen_model=os.getenv('QIANWEN_MODEL', Config.qianwen_model),
        # 通用配置
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
    # 至少需要一个有效的API密钥
    has_nvidia_key = config.nvidia_api_key and config.nvidia_api_key.strip()
    has_qianwen_key = config.qianwen_api_key and config.qianwen_api_key.strip()

    if not has_nvidia_key and not has_qianwen_key:
        raise ValueError("至少需要配置一个API密钥（NVIDIA或QianWen）")

    if config.use_nvidia and not has_nvidia_key:
        raise ValueError("启用了NVIDIA API但未提供NVIDIA_API_KEY")

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