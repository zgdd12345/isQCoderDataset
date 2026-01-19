#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享数据结构与基础工具函数
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """通过tqdm.write输出日志，避免打断进度条。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


def _configure_logging(log_file: str, log_level: int = logging.INFO) -> logging.Logger:
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 移除普通控制台输出，避免破坏tqdm进度条
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)

    if not any(isinstance(h, TqdmLoggingHandler) for h in root_logger.handlers):
        console_handler = TqdmLoggingHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    abs_log_file = os.path.abspath(log_file)
    log_dir = os.path.dirname(abs_log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == abs_log_file
        for h in root_logger.handlers
    ):
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


def _configure_paper_stats_logging(log_file: str, log_level: int = logging.INFO) -> logging.Logger:
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger_name = f"paper_stats.{os.path.basename(log_file)}"
    stats_logger = logging.getLogger(logger_name)
    stats_logger.setLevel(log_level)
    stats_logger.propagate = False

    abs_log_file = os.path.abspath(log_file)
    log_dir = os.path.dirname(abs_log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == abs_log_file
        for h in stats_logger.handlers
    ):
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        stats_logger.addHandler(file_handler)

    return stats_logger


def normalize_latex(text: str) -> str:
    """统一LaTeX分隔符为$...$或$$...$$。"""
    if not text:
        return text
    normalized = text.replace("\\[", "$$").replace("\\]", "$$")
    normalized = normalized.replace("\\(", "$").replace("\\)", "$")
    return normalized


@dataclass
class DatasetSample:
    """数据集样本结构"""
    instruction: str
    input: str
    output: str
    metadata: Dict[str, Any] = None


@dataclass
class SegmentedPaper:
    """论文段落结构，用于代理工作流"""
    paper_title: str
    paper_abstract: str  # 论文摘要，用于提供额外上下文
    segment_title: str
    segment_content: str
    full_paper_content: str  # 用于验证步骤


@dataclass
class GeneratedInstruction:
    """指令生成结果"""
    instruction: str
    input: str
    instruction_type: str  # concept, implementation, analysis, comparison, application
    key_concepts: List[str]
    segment: SegmentedPaper


@dataclass
class GeneratedAnswer:
    """回答生成结果"""
    instruction: GeneratedInstruction
    output: str


@dataclass
class VerificationResult:
    """验证结果"""
    passed: bool
    confidence_score: float
    issues: List[str]
    context_leakage: List[str] = None  # 上下文泄漏实例
    self_contained: bool = True  # 是否自包含
    suggestion: str = ""
