#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
isQCoder数据集生成工具
使用大模型API生成指令微调数据集
"""

import os
import sys
import json
import glob
import asyncio
import time
from typing import List, Dict, Any, Optional, Coroutine
from dataclasses import dataclass
from pathlib import Path
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from llm_client import LLMClient


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
    if not any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == abs_log_file
        for h in root_logger.handlers
    ):
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


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
    """论文段落结构，用于4步工作流"""
    paper_title: str
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
    suggestion: str = ""


class TokenBucketRateLimiter:
    """
    令牌桶速率限制器

    特点：
    - 允许突发流量（burst）
    - 平滑的请求分布
    - 精确控制每分钟请求数
    """

    def __init__(
        self,
        rate_per_minute: int = 30,
        burst_size: int = 5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            rate_per_minute: 每分钟允许的请求数
            burst_size: 允许的突发请求数（令牌桶最大容量）
            logger: 日志记录器
        """
        self.rate_per_minute = rate_per_minute
        self.burst_size = min(burst_size, rate_per_minute)
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()
        self.token_interval = 60.0 / rate_per_minute
        self._lock = asyncio.Lock()
        self.logger = logger or logging.getLogger(__name__)

        # 统计信息
        self._total_requests = 0
        self._total_wait_time = 0.0

    async def acquire(self) -> float:
        """
        获取一个令牌，如果没有可用令牌则等待

        关键：整个过程持有锁，确保请求严格串行

        Returns:
            等待时间（秒）
        """
        async with self._lock:
            now = time.monotonic()

            # 补充令牌
            elapsed = now - self.last_update
            new_tokens = elapsed / self.token_interval
            self.tokens = min(self.burst_size, self.tokens + new_tokens)
            self.last_update = now

            # 检查是否有可用令牌
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                self._total_requests += 1
                self.logger.debug(f"速率限制：立即获取令牌（剩余={self.tokens:.2f}）")
                return 0.0

            # 计算需要等待的时间
            wait_time = (1.0 - self.tokens) * self.token_interval
            self.logger.debug(f"速率限制：需等待 {wait_time:.2f}s")

            # 关键：在持有锁的情况下等待，确保串行
            await asyncio.sleep(wait_time)

            # 等待完成后消费令牌
            self.tokens = 0.0
            self.last_update = time.monotonic()
            self._total_requests += 1
            self._total_wait_time += wait_time

            return wait_time

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_requests": self._total_requests,
            "total_wait_time": self._total_wait_time,
            "avg_wait_time": self._total_wait_time / max(1, self._total_requests),
            "current_tokens": self.tokens
        }


class ISQCoderDataGenerator:
    """生成指令微调数据的类，与模型与API解耦"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

        # 设置日志（使用tqdm友好的控制台输出）
        self.logger = _configure_logging(
            f'./log/dataset_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def load_markdown_content(self, file_path: str) -> str:
        """加载markdown文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"读取文件 {file_path} 失败: {e}")
            return ""
    
    def get_paper_files(self, data_dir: str = "data") -> List[str]:
        """获取所有论文文件路径"""
        pattern = os.path.join(data_dir, "*.md")
        return glob.glob(pattern)
    
    def segment_paper_content(self, paper_content: str) -> List[Dict[str, str]]:
        """将论文内容按大标题分割，只处理以单个#开头的主要章节"""
        lines = paper_content.split('\n')
        segments = []
        current_segment = []
        current_title = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检测大标题行（只处理以单个#开头的标题，忽略##、###等子标题）
            if line.startswith('# ') and not line.startswith('## '):
                # 保存上一个段落
                if current_segment and current_title:
                    segment_content = '\n'.join(current_segment).strip()
                    if len(segment_content) > 1000:  # 过滤过短的段落
                        segments.append({
                            'title': current_title,
                            'content': segment_content
                        })
                
                # 开始新段落
                current_title = line[2:].strip()  # 移除"# "前缀
                current_segment = []
            else:
                current_segment.append(line)
        
        # 处理最后一个段落
        if current_segment and current_title:
            segment_content = '\n'.join(current_segment).strip()
            if len(segment_content) > 1000:
                segments.append({
                    'title': current_title,
                    'content': segment_content
                })
        
        return segments
    
    def create_instruction_prompts(self, paper_content: str, paper_title: str) -> List[str]:
        """基于论文段落内容创建指令微调提示"""
        segments = self.segment_paper_content(paper_content)
        prompts = []
        
        for segment in segments:
            segment_title = segment['title']
            segment_content = segment['content']
            print(f"Processing segment: {segment_title}, length: {len(segment_content)}")
            # 为每个段落生成指令微调数据对的提示
            prompt = f"""你是一个专业的量子计算指令数据生成专家。请基于以下论文段落内容，分析其核心知识点，并生成一个高质量的指令微调数据对。

论文标题：《{paper_title}》
段落标题：{segment_title}
段落内容：
{segment_content}

要求：
1. 深入理解段落的核心内容和关键概念
2. 根据内容的特点，自然地生成一个合适的instruction（指令）
3. 根据instruction的具体需要生成input字段（补充背景、约束或数据），若不需要补充则置空字符串
4. output应该是对instruction的完整、准确、专业的回答，并使用input中的信息（如有）
5. instruction、input、output语言保持一致（英文提问英文回答，中文提问中文回答）
6. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$，不要使用\\(\\)或\\[\\]

指令类型可以包括但不限于：
- 概念解释和定义
- 方法步骤说明  
- 算法原理阐述
- 公式推导过程
- 实现方案描述
- 问题分析讨论
- 应用场景介绍

请确保生成的指令微调对具有教育价值，能够帮助学习者理解量子计算相关知识。

返回JSON格式的结果：
{{
    "instruction": "这里是指令",
    "input": "这里是输入（可为空）",
    "output": "这里是详细的回答"
}}"""
            
            prompts.append(prompt)
        
        return prompts
    
    async def generate_dataset_sample(self, prompt: str) -> DatasetSample:
        """生成单个数据集样本"""
        response = await self.llm_client.generate(prompt)
        
        if not response.text:
            self.logger.error("API返回空响应")
            return None
        
        self.logger.info(f"API响应长度: {len(response.text)}")
        try:
            # 尝试解析JSON响应
            if "```json" in response.text:
                json_start = response.text.find("```json") + 7
                json_end = response.text.find("```", json_start)
                json_content = response.text[json_start:json_end].strip()
            else:
                json_content = response.text
            
            data = json.loads(json_content)
            raw_input = data.get("input", "")
            if raw_input is None:
                raw_input = ""
            if not isinstance(raw_input, str):
                raw_input = json.dumps(raw_input, ensure_ascii=False)

            raw_output = data.get("output", "")
            if raw_output is None:
                raw_output = ""
            if not isinstance(raw_output, str):
                raw_output = json.dumps(raw_output, ensure_ascii=False)
            normalized_output = normalize_latex(raw_output)

            return DatasetSample(
                instruction=data.get("instruction", ""),
                input=raw_input,
                output=normalized_output,
                metadata={
                    "generated_at": datetime.now().isoformat(),
                    "model": response.model,
                    "provider": response.provider
                }
            )
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}")
            self.logger.error(f"响应内容: {response.text}")
            return None
    
    async def generate_dataset_from_papers(self, output_file: str = "quantum_instruction_dataset.jsonl", 
                                          max_samples_per_paper: int = 4):
        """从论文生成完整的指令微调数据集"""
        paper_files = self.get_paper_files()
        
        self.logger.info(f"找到 {len(paper_files)} 个论文文件")
        
        await self._generate_dataset_realtime_mode(paper_files, output_file, max_samples_per_paper)
    
    async def _generate_dataset_realtime_mode(self, paper_files: List[str], output_file: str, 
                                            max_samples_per_paper: int):
        """实时推理模式生成数据集"""
        all_samples = []
        
        for paper_file in paper_files:
            paper_name = Path(paper_file).stem
            self.logger.info(f"处理论文: {paper_name}")
            
            paper_content = self.load_markdown_content(paper_file)
            if not paper_content:
                continue
            
            # 提取论文标题
            lines = paper_content.split('\n')
            paper_title = next((line.strip('# ') for line in lines if line.startswith('# ')), paper_name)
            
            # 生成多种类型的提示
            prompts = self.create_instruction_prompts(paper_content, paper_title)
            
            # 限制每篇论文的样本数量
            prompts = prompts[:max_samples_per_paper]
            
            # 并发生成样本
            tasks = [self.generate_dataset_sample(prompt) for prompt in prompts]
            samples = await asyncio.gather(*tasks)
            
            # 过滤有效样本
            valid_samples = [s for s in samples if s is not None and s.instruction and s.output]
            all_samples.extend(valid_samples)
            
            self.logger.info(f"从 {paper_name} 生成了 {len(valid_samples)} 个有效样本")
            
            # 避免API频率限制
            await asyncio.sleep(1)
        
        # 保存数据集
        self.save_dataset(all_samples, output_file)
        self.logger.info(f"数据集生成完成，共 {len(all_samples)} 个样本，保存至 {output_file}")
    
    def save_dataset(self, samples: List[DatasetSample], output_file: str):
        """保存数据集到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                json_line = {
                    "instruction": sample.instruction,
                    "input": sample.input,
                    "output": sample.output,
                    "metadata": sample.metadata
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')


class ISQCoderDGAgent:
    """
    数据生成代理，实现4步工作流：
    1. 读取论文，分割段落
    2. 基于段落生成指令
    3. 根据指令和段落生成回答
    4. 重新阅读论文验证生成内容符合原意
    """

    @staticmethod
    def _get_instruction_prompt(language_mode: str) -> str:
        """
        根据语言模式返回指令生成提示词（只生成instruction，不生成input）

        Args:
            language_mode: "en" (英文), "mixed" (中英混合), "zh" (纯中文)
        """
        if language_mode == "en":
            return """You are a professional quantum computing education expert. Please generate a valuable learning instruction based on the following paper segment.

Paper Title: {paper_title}
Segment Title: {segment_title}
Segment Content:
{segment_content}

Requirements:
1. Carefully analyze the core knowledge points and key concepts in the segment
2. Generate a clear and specific instruction that can guide learners to deeply understand this content
3. The instruction should be a question or task, not a simple paraphrase of the content
4. Keep instruction in English; do not mix languages
5. If the segment is not suitable for generating an instruction, return an empty string for instruction
6. Use standard LaTeX delimiters $...$ or $$...$$ for any formulas (do not use \\(\\) or \\[\\])
7. Do NOT generate questions that analyze graphs or tables
8. Focus ONLY on generating the instruction - do NOT generate input field (it will be handled separately)

Instruction Type Options:
- concept: Concept explanation and definition
- implementation: Method steps or algorithm implementation
- analysis: Problem analysis and discussion
- comparison: Comparative analysis
- application: Application scenario introduction
- others: Other related types

Please return in JSON format:
{{
    "instruction": "The generated instruction content IN ENGLISH",
    "instruction_type": "instruction type",
    "key_concepts": ["key concept 1", "key concept 2"]
}}"""
        elif language_mode == "mixed":
            return """你是一个专业的量子计算教育专家。请基于以下论文段落，生成一个有价值的学习指令。

论文标题：《{paper_title}》
段落标题：{segment_title}
段落内容：
{segment_content}

要求：
1. 仔细分析段落中的核心知识点和关键概念
2. 生成一个清晰、具体的指令，这个指令应该能够引导学习者深入理解这段内容
3. 指令应该是一个问题或任务，而不是对内容的简单复述
4. **IMPORTANT: 指令必须使用中文**
5. 选择一个合适的指令类型
6. 如果该段落不适合生成指令，请返回空字符串
7. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$，不要使用\\(\\)或\\[\\]
8. 禁止生成分析图或者表格的问题
9. 只生成instruction，不要生成input字段（input将由单独步骤处理）

指令类型选项：
- concept: 概念解释和定义
- implementation: 方法步骤或算法实现
- analysis: 问题分析和讨论
- comparison: 比较分析
- application: 应用场景介绍
- others: 其他相关类型

请以JSON格式返回：
{{
    "instruction": "生成的指令内容（中文）",
    "instruction_type": "指令类型",
    "key_concepts": ["关键概念1", "关键概念2"]
}}"""
        else:  # zh
            return """你是一个专业的量子计算教育专家。请基于以下论文段落，生成一个有价值的学习指令。

论文标题：《{paper_title}》
段落标题：{segment_title}
段落内容：
{segment_content}

要求：
1. 仔细分析段落中的核心知识点和关键概念
2. 生成一个清晰、具体的指令，这个指令应该能够引导学习者深入理解这段内容
3. 指令应该是一个问题或任务，而不是对内容的简单复述
4. 指令必须使用中文
5. 选择一个合适的指令类型
6. 如果该段落不适合生成指令，请返回空字符串
7. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$，不要使用\\(\\)或\\[\\]
8. 只生成instruction，不要生成input字段（input将由单独步骤处理）

指令类型选项：
- concept: 概念解释和定义
- implementation: 方法步骤或算法实现
- analysis: 问题分析和讨论
- comparison: 比较分析
- application: 应用场景介绍
- others: 其他相关类型

请以JSON格式返回：
{{
    "instruction": "生成的指令内容",
    "instruction_type": "指令类型",
    "key_concepts": ["关键概念1", "关键概念2"]
}}"""

    @staticmethod
    def _get_answer_prompt(language_mode: str) -> str:
        """
        根据语言模式返回回答生成提示词

        Args:
            language_mode: "en" (英文), "mixed" (中英混合), "zh" (纯中文)
        """
        if language_mode == "en":
            return """You are a professional quantum computing expert. Please generate an accurate, professional, and detailed answer based on the following instruction and original paper segment.

Original Paper Title: {paper_title}
Original Segment Title: {segment_title}
Original Segment Content:
{segment_content}

Instruction: {instruction}
Input: {input}
Instruction Type: {instruction_type}

Requirements:
1. The answer must be based on the original segment content to ensure accuracy
2. The answer should be comprehensive and professional, helping learners truly understand the relevant knowledge
3. You can expand and explain appropriately, but the core content must come from the original text
4. If it involves formulas or algorithms, please explain their meanings and derivation processes in detail
5. Use a clear structure to organize the answer (you can use numbering, bullet points, etc.)
6. **IMPORTANT: The entire answer must be in English and match the instruction language**
7. Use standard LaTeX delimiters $...$ or $$...$$ for all formulas (do not use \\(\\) or \\[\\])

Please return the answer content directly, no JSON format needed."""
        elif language_mode == "mixed":
            return """你是一个专业的量子计算专家。请根据以下指令和原始论文段落，生成一个准确、专业、详尽的回答。

原始论文标题：《{paper_title}》
原始段落标题：{segment_title}
原始段落内容：
{segment_content}

指令：{instruction}
输入：{input}
指令类型：{instruction_type}

要求：
1. 回答必须基于原始段落的内容，确保准确性
2. 回答应该全面、专业，帮助学习者真正理解相关知识
3. 可以适当扩展和解释，但核心内容必须来自原文
4. 如果涉及公式或算法，请详细解释其含义和推导过程
5. 使用清晰的结构组织回答（可使用编号、分点等）
6. **IMPORTANT: 回答语言必须与指令一致（中文）**
7. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$，不要使用\\(\\)或\\[\\]

请直接返回回答内容，不需要JSON格式。"""
        else:  # zh
            return """你是一个专业的量子计算专家。请根据以下指令和原始论文段落，生成一个准确、专业、详尽的回答。

原始论文标题：《{paper_title}》
原始段落标题：{segment_title}
原始段落内容：
{segment_content}

指令：{instruction}
输入：{input}
指令类型：{instruction_type}

要求：
1. 回答必须基于原始段落的内容，确保准确性
2. 回答应该全面、专业，帮助学习者真正理解相关知识
3. 可以适当扩展和解释，但核心内容必须来自原文
4. 如果涉及公式或算法，请详细解释其含义和推导过程
5. 使用清晰的结构组织回答（可使用编号、分点等）
6. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$，不要使用\\(\\)或\\[\\]

请直接返回回答内容，不需要JSON格式。"""

    @staticmethod
    def _get_input_prompt(language_mode: str) -> str:
        """
        根据语言模式返回input生成提示词

        Args:
            language_mode: "en" (英文), "mixed" (中英混合), "zh" (纯中文)
        """
        if language_mode == "en":
            return """You are a professional instruction data quality expert. Please analyze the following instruction and determine whether it needs an input field, and if so, generate appropriate input content.

Original Paper Title: {paper_title}
Original Segment Title: {segment_title}
Original Segment Content:
{segment_content}

Generated Instruction: {instruction}
Instruction Type: {instruction_type}

Analysis Guidelines:
1. Determine if this instruction requires additional context, constraints, parameters, or data to be answered properly
2. An input field is NEEDED when:
   - The instruction references specific parameters, thresholds, or configurations that should be specified
   - The instruction asks about a specific scenario that needs context
   - The instruction requires specific data, code snippets, or examples to work with
   - The instruction mentions "given", "for", "with" implying external input
3. An input field is NOT NEEDED when:
   - The instruction is self-contained and can be answered directly
   - The instruction asks for general explanations or definitions
   - All necessary context is already in the instruction itself
4. If input is needed, generate concise but complete input content that:
   - Provides necessary context without being redundant with the instruction
   - Uses concrete examples or values when appropriate
   - Is relevant to the paper segment content
5. Keep input in English to match the instruction language
6. Use standard LaTeX delimiters $...$ or $$...$$ for any formulas

Please return in JSON format:
{{
    "needs_input": true or false,
    "reasoning": "Brief explanation of why input is or isn't needed",
    "input": "The generated input content, or empty string if not needed"
}}"""
        elif language_mode == "mixed":
            return """你是一个专业的指令数据质量专家。请分析以下指令，判断是否需要input字段，如果需要则生成合适的input内容。

原始论文标题：《{paper_title}》
原始段落标题：{segment_title}
原始段落内容：
{segment_content}

生成的指令：{instruction}
指令类型：{instruction_type}

分析指南：
1. 判断这个指令是否需要额外的上下文、约束、参数或数据才能被正确回答
2. 以下情况需要input字段：
   - 指令涉及特定的参数、阈值或配置需要指定
   - 指令询问需要上下文的特定场景
   - 指令需要特定的数据、代码片段或示例
   - 指令中提到"给定"、"对于"、"在...条件下"等暗示外部输入
   - 指令中提到"根据图"或"根据表"等类似内容，需要补充图表的数据或说明
3. 以下情况不需要input字段：
   - 指令是自包含的，可以直接回答
   - 指令询问一般性的解释或定义
   - 所有必要的上下文已经在指令本身中
4. 如果需要input，生成简洁但完整的input内容：
   - 提供必要的上下文，但不与指令重复
   - 在适当时使用具体的示例或数值
   - 与论文段落内容相关
5. **input必须使用中文**（与指令语言一致）
6. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$

请以JSON格式返回：
{{
    "needs_input": true或false,
    "reasoning": "简要说明为什么需要或不需要input",
    "input": "生成的input内容（中文），如果不需要则为空字符串"
}}"""
        else:  # zh
            return """你是一个专业的指令数据质量专家。请分析以下指令，判断是否需要input字段，如果需要则生成合适的input内容。

原始论文标题：《{paper_title}》
原始段落标题：{segment_title}
原始段落内容：
{segment_content}

生成的指令：{instruction}
指令类型：{instruction_type}

分析指南：
1. 判断这个指令是否需要额外的上下文、约束、参数或数据才能被正确回答
2. 以下情况需要input字段：
   - 指令涉及特定的参数、阈值或配置需要指定
   - 指令询问需要上下文的特定场景
   - 指令需要特定的数据、代码片段或示例
   - 指令中提到"给定"、"对于"、"在...条件下"等暗示外部输入
3. 以下情况不需要input字段：
   - 指令是自包含的，可以直接回答
   - 指令询问一般性的解释或定义
   - 所有必要的上下文已经在指令本身中
4. 如果需要input，生成简洁但完整的input内容：
   - 提供必要的上下文，但不与指令重复
   - 在适当时使用具体的示例或数值
   - 与论文段落内容相关
5. input必须使用中文
6. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$

请以JSON格式返回：
{{
    "needs_input": true或false,
    "reasoning": "简要说明为什么需要或不需要input",
    "input": "生成的input内容，如果不需要则为空字符串"
}}"""

    # 验证提示模板
    VERIFICATION_PROMPT = """你是一个严格的学术审核专家。请验证以下生成的指令-回答对是否准确反映了原始论文的内容和本意。

原始论文标题：《{paper_title}》
原始论文相关段落：
{segment_content}

生成的指令：{instruction}
生成的输入：{input}
生成的回答：{output}

请严格检查以下几点：
1. 准确性：回答中的事实、公式、概念是否与原文一致
2. 完整性：回答是否涵盖了指令所问的关键点
3. 一致性：回答的观点和论述是否与原文的立场一致
4. 无幻觉：是否存在原文未提及的信息被当作事实陈述
5. 语言表达：回答是否清晰、专业，符合学术规范
6. 指令和输入是否一致，是否应该生成输入而未生成。

请以JSON格式返回验证结果：
{{
    "passed": true或false,
    "confidence_score": 0.0到1.0的数值,
    "issues": ["发现的问题1", "发现的问题2"],
    "suggestion": "如果未通过，建议如何修改"
}}"""

    def __init__(
        self,
        llm_client: LLMClient,
        verification_strategy: str = "retry",
        max_retries: int = 2,
        verification_threshold: float = 0.7,
        skip_verification: bool = False,
        concurrency: int = 5,
        rate_limit_per_minute: int = 30,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化数据生成代理

        Args:
            llm_client: LLM客户端
            verification_strategy: 验证失败策略 ("retry", "discard", "flag")
            max_retries: 最大重试次数
            verification_threshold: 验证通过阈值 (0.0-1.0)
            skip_verification: 是否跳过验证步骤
            concurrency: 并发处理数量（保留用于向后兼容）
            rate_limit_per_minute: API调用速率限制（次/分钟）
            logger: 日志记录器
        """
        self.llm_client = llm_client
        self.verification_strategy = verification_strategy
        self.max_retries = max_retries
        self.verification_threshold = verification_threshold
        self.skip_verification = skip_verification
        self.concurrency = concurrency
        self.rate_limit_per_minute = rate_limit_per_minute

        if logger is None:
            self.logger = _configure_logging(
                f'./log/agent_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        else:
            self.logger = logger

        # 初始化全局速率限制器
        self.rate_limiter = TokenBucketRateLimiter(
            rate_per_minute=rate_limit_per_minute,
            burst_size=1,
            logger=self.logger
        )

        # 语言分布计数器（用于控制语言比例）
        self._language_counter = {"en": 0, "mixed": 0, "zh": 0}
        self._language_lock = asyncio.Lock()

    def segment_paper(self, paper_path: str) -> List[SegmentedPaper]:
        """
        步骤1：读取论文并分割为段落

        Args:
            paper_path: 论文文件路径

        Returns:
            分割后的段落列表
        """
        try:
            with open(paper_path, 'r', encoding='utf-8') as f:
                full_content = f.read()
        except Exception as e:
            self.logger.error(f"读取文件 {paper_path} 失败: {e}")
            return []

        # 提取论文标题
        lines = full_content.split('\n')
        paper_title = next(
            (line.strip('# ').strip() for line in lines if line.startswith('# ')),
            Path(paper_path).stem
        )

        # 按H1标题分割段落
        segments = []
        current_segment = []
        current_title = ""

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 检测H1标题（只处理单个#开头的标题）
            if stripped.startswith('# ') and not stripped.startswith('## '):
                if current_segment and current_title:
                    segment_content = '\n'.join(current_segment).strip()
                    if len(segment_content) > 1000:
                        segments.append(SegmentedPaper(
                            paper_title=paper_title,
                            segment_title=current_title,
                            segment_content=segment_content,
                            full_paper_content=full_content
                        ))

                current_title = stripped[2:].strip()
                current_segment = []
            else:
                current_segment.append(line)

        # 处理最后一个段落
        if current_segment and current_title:
            segment_content = '\n'.join(current_segment).strip()
            if len(segment_content) > 1000:
                segments.append(SegmentedPaper(
                    paper_title=paper_title,
                    segment_title=current_title,
                    segment_content=segment_content,
                    full_paper_content=full_content
                ))

        self.logger.info(f"从 {paper_path} 分割出 {len(segments)} 个段落")
        return segments

    async def _rate_limited_call(self, coro: Coroutine) -> Any:
        """
        带速率限制的API调用包装器

        Args:
            coro: 要执行的协程

        Returns:
            协程执行结果
        """
        await self.rate_limiter.acquire()
        return await coro

    async def _determine_language_mode(self) -> str:
        """
        根据当前分布决定下一个样本使用的语言模式

        目标分布: 50% 英文, 50% 纯中文

        Returns:
            "en" (英文), "mixed" (中英混合), "zh" (纯中文)
        """
        async with self._language_lock:
            total = sum(self._language_counter.values())

            if total == 0:
                # 第一个样本，随机选择
                import random
                return "en" if random.random() < 0.5 else "zh"

            # 计算当前比例
            en_ratio = self._language_counter["en"] / total
            zh_ratio = self._language_counter["zh"] / total

            # 根据与目标的差距决定下一个样本的语言
            # 计算每种语言与目标的差距
            en_gap = 0.5 - en_ratio
            zh_gap = 0.5 - zh_ratio

            # 选择差距最大的（需要补充最多的）
            if en_gap >= zh_gap:
                return "en"
            return "zh"

    async def _increment_language_counter(self, language_mode: str):
        """增加语言计数器"""
        async with self._language_lock:
            self._language_counter[language_mode] += 1

    async def generate_instruction(
        self,
        segment: SegmentedPaper,
        language_mode: Optional[str] = None
    ) -> Optional[GeneratedInstruction]:
        """
        步骤2：基于段落生成指令

        Args:
            segment: 论文段落
            language_mode: 语言模式 ("en", "mixed", "zh")，如果为None则自动决定

        Returns:
            生成的指令，失败返回None
        """
        # 如果未指定语言模式，自动决定
        if language_mode is None:
            language_mode = await self._determine_language_mode()

        # 获取对应语言的提示词
        prompt_template = self._get_instruction_prompt(language_mode)
        prompt = prompt_template.format(
            paper_title=segment.paper_title,
            segment_title=segment.segment_title,
            segment_content=segment.segment_content
        )

        response = await self.llm_client.generate(prompt)

        if not response.text:
            self.logger.error("指令生成API返回空响应")
            return None

        try:
            # 解析JSON响应
            json_content = self._extract_json(response.text)
            data = json.loads(json_content)

            instruction = GeneratedInstruction(
                instruction=data.get("instruction", ""),
                input="",  # input 将在单独的 generate_input 步骤中生成
                instruction_type=data.get("instruction_type", "concept"),
                key_concepts=data.get("key_concepts", []),
                segment=segment
            )
            # 在 instruction 对象上附加语言模式信息（用于后续步骤）
            instruction.language_mode = language_mode
            return instruction
        except json.JSONDecodeError as e:
            self.logger.error(f"指令生成JSON解析失败: {e}")
            self.logger.debug(f"响应内容: {response.text}")
            return None

    async def generate_input(self, instruction: GeneratedInstruction) -> GeneratedInstruction:
        """
        步骤2.5：分析指令并生成合适的input

        Args:
            instruction: 生成的指令（不含input或input质量不高）

        Returns:
            更新了input字段的指令
        """
        language_mode = getattr(instruction, 'language_mode', 'en')

        prompt_template = self._get_input_prompt(language_mode)
        prompt = prompt_template.format(
            paper_title=instruction.segment.paper_title,
            segment_title=instruction.segment.segment_title,
            segment_content=instruction.segment.segment_content,
            instruction=instruction.instruction,
            instruction_type=instruction.instruction_type
        )

        response = await self.llm_client.generate(prompt)

        if not response.text:
            self.logger.warning("Input生成API返回空响应，保留原input")
            return instruction

        try:
            json_content = self._extract_json(response.text)
            data = json.loads(json_content)

            needs_input = data.get("needs_input", False)
            reasoning = data.get("reasoning", "")
            generated_input = data.get("input", "")

            if generated_input is None:
                generated_input = ""
            if not isinstance(generated_input, str):
                generated_input = json.dumps(generated_input, ensure_ascii=False)

            if needs_input and generated_input:
                instruction.input = generated_input
                self.logger.info(f"生成input: {generated_input[:50]}... (原因: {reasoning[:30]}...)")
            else:
                instruction.input = ""
                self.logger.info(f"不需要input (原因: {reasoning[:50]}...)")

            return instruction
        except json.JSONDecodeError as e:
            self.logger.warning(f"Input生成JSON解析失败: {e}，保留原input")
            return instruction

    async def generate_answer(self, instruction: GeneratedInstruction) -> Optional[GeneratedAnswer]:
        """
        步骤3：根据指令和段落生成回答

        Args:
            instruction: 生成的指令

        Returns:
            生成的回答，失败返回None
        """
        # 使用 instruction 中附加的语言模式
        language_mode = getattr(instruction, 'language_mode', 'en')

        # 获取对应语言的提示词
        prompt_template = self._get_answer_prompt(language_mode)
        prompt = prompt_template.format(
            paper_title=instruction.segment.paper_title,
            segment_title=instruction.segment.segment_title,
            segment_content=instruction.segment.segment_content,
            instruction=instruction.instruction,
            input=instruction.input,
            instruction_type=instruction.instruction_type
        )

        response = await self.llm_client.generate(prompt)

        if not response.text:
            self.logger.error("回答生成API返回空响应")
            return None

        normalized_output = normalize_latex(response.text.strip())
        return GeneratedAnswer(
            instruction=instruction,
            output=normalized_output
        )

    async def verify_sample(self, answer: GeneratedAnswer) -> VerificationResult:
        """
        步骤4：验证生成内容符合论文原意

        Args:
            answer: 生成的回答

        Returns:
            验证结果
        """
        prompt = self.VERIFICATION_PROMPT.format(
            paper_title=answer.instruction.segment.paper_title,
            segment_content=answer.instruction.segment.segment_content,
            instruction=answer.instruction.instruction,
            input=answer.instruction.input,
            output=answer.output
        )

        response = await self.llm_client.generate(prompt)

        if not response.text:
            self.logger.error("验证API返回空响应")
            return VerificationResult(
                passed=False,
                confidence_score=0.0,
                issues=["API返回空响应"],
                suggestion="重新生成"
            )

        try:
            json_content = self._extract_json(response.text)
            data = json.loads(json_content)

            return VerificationResult(
                passed=data.get("passed", False),
                confidence_score=float(data.get("confidence_score", 0.0)),
                issues=data.get("issues", []),
                suggestion=data.get("suggestion", "")
            )
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"验证结果JSON解析失败: {e}")
            return VerificationResult(
                passed=False,
                confidence_score=0.0,
                issues=["验证响应解析失败"],
                suggestion="重新验证"
            )

    async def process_segment(self, segment: SegmentedPaper) -> Optional[DatasetSample]:
        """
        处理单个段落，执行完整的5步工作流（带速率限制）

        工作流：
        1. 分割段落（已在上层完成）
        2. 生成指令（instruction）
        3. 分析指令并生成input
        4. 根据指令和input生成回答
        5. 验证生成内容

        Args:
            segment: 论文段落

        Returns:
            生成的数据集样本，失败返回None
        """
        for attempt in range(self.max_retries + 1):
            try:
                # 步骤2：生成指令（速率限制）
                instruction = await self._rate_limited_call(
                    self.generate_instruction(segment)
                )
                if not instruction or not instruction.instruction:
                    self.logger.warning(f"指令生成失败: {segment.segment_title}")
                    continue

                self.logger.info(f"生成指令: {instruction.instruction[:50]}...")

                # 步骤3：分析指令并生成input（速率限制）
                instruction = await self._rate_limited_call(
                    self.generate_input(instruction)
                )
                self.logger.info(f"Input生成完成: {'有input' if instruction.input else '无input'}")

                # 步骤4：生成回答（速率限制）
                answer = await self._rate_limited_call(
                    self.generate_answer(instruction)
                )
                if not answer or not answer.output:
                    self.logger.warning(f"回答生成失败: {instruction.instruction[:50]}...")
                    continue

                self.logger.info(f"生成回答长度: {len(answer.output)}")

                # 步骤4：验证（可选，速率限制）
                if self.skip_verification:
                    verification = VerificationResult(
                        passed=True,
                        confidence_score=1.0,
                        issues=[],
                        suggestion=""
                    )
                else:
                    verification = await self._rate_limited_call(
                        self.verify_sample(answer)
                    )
                    self.logger.info(f"验证结果: passed={verification.passed}, score={verification.confidence_score}")

                # 检查验证是否通过
                if verification.passed or verification.confidence_score >= self.verification_threshold:
                    # 获取语言模式并增加计数
                    language_mode = getattr(instruction, 'language_mode', 'en')
                    await self._increment_language_counter(language_mode)

                    return DatasetSample(
                        instruction=instruction.instruction,
                        input=instruction.input,
                        output=answer.output,
                        metadata={
                            "generated_at": datetime.now().isoformat(),
                            "model": self.llm_client.provider.model,
                            "provider": self.llm_client.provider.name,
                            "instruction_type": instruction.instruction_type,
                            "key_concepts": instruction.key_concepts,
                            "verification_score": verification.confidence_score,
                            "language_mode": language_mode,
                            "attempt": attempt + 1
                        }
                    )

                # 处理验证失败
                if self.verification_strategy == "flag":
                    # 获取语言模式并增加计数（即使验证失败也计入）
                    language_mode = getattr(instruction, 'language_mode', 'en')
                    await self._increment_language_counter(language_mode)

                    return DatasetSample(
                        instruction=instruction.instruction,
                        input=instruction.input,
                        output=answer.output,
                        metadata={
                            "generated_at": datetime.now().isoformat(),
                            "model": self.llm_client.provider.model,
                            "provider": self.llm_client.provider.name,
                            "instruction_type": instruction.instruction_type,
                            "language_mode": language_mode,
                            "verification_failed": True,
                            "verification_issues": verification.issues,
                            "verification_score": verification.confidence_score
                        }
                    )
                elif self.verification_strategy == "retry":
                    self.logger.info(f"验证未通过，重试 ({attempt + 1}/{self.max_retries + 1})")
                    continue
                else:  # discard
                    self.logger.info("验证未通过，丢弃样本")
                    return None

            except Exception as e:
                self.logger.error(f"处理段落时发生错误: {e}")
                if attempt == self.max_retries:
                    return None

        return None

    async def generate_dataset(
        self,
        data_dir: str = "data",
        output_file: str = "./results/results.jsonl",
        max_samples_per_paper: int = 10
    ) -> List[DatasetSample]:
        """
        从论文目录生成完整数据集（并发处理，全局速率限制，实时保存）

        Args:
            data_dir: 论文目录
            output_file: 输出文件路径
            max_samples_per_paper: 每篇论文最大样本数

        Returns:
            生成的数据集样本列表
        """
        import glob as glob_module

        paper_files = glob_module.glob(os.path.join(data_dir, "*.md"))
        self.logger.info(f"找到 {len(paper_files)} 个论文文件")

        # 收集所有段落
        all_segments = []
        for paper_file in paper_files:
            paper_name = Path(paper_file).stem
            self.logger.info(f"分割论文: {paper_name}")

            segments = self.segment_paper(paper_file)
            if not segments:
                self.logger.warning(f"论文 {paper_name} 没有有效段落")
                continue

            # 限制每篇论文的段落数
            segments = segments[:max_samples_per_paper]
            all_segments.extend(segments)

        self.logger.info(
            f"共收集 {len(all_segments)} 个段落，开始并发处理（速率限制={self.rate_limit_per_minute}次/分钟）"
        )

        # 清空输出文件（如果存在）
        if os.path.exists(output_file):
            os.remove(output_file)

        # 创建进度条和统计计数器
        success_count = 0
        error_count = 0
        all_samples = []

        # 包装 process_segment 以支持实时保存和进度更新
        async def process_and_save(segment: SegmentedPaper, pbar) -> Optional[DatasetSample]:
            nonlocal success_count, error_count
            try:
                result = await self.process_segment(segment)
                if isinstance(result, DatasetSample):
                    # 实时保存到文件
                    self._save_sample(result, output_file)
                    success_count += 1
                    pbar.set_postfix({
                        '成功': success_count,
                        '失败': error_count,
                        '速率': f"{self.rate_limit_per_minute}/min"
                    })
                    return result
                else:
                    error_count += 1
                    pbar.set_postfix({
                        '成功': success_count,
                        '失败': error_count,
                        '速率': f"{self.rate_limit_per_minute}/min"
                    })
                    return None
            except Exception as e:
                error_count += 1
                self.logger.error(f"段落处理异常: {segment.segment_title} - {e}")
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': error_count,
                    '速率': f"{self.rate_limit_per_minute}/min"
                })
                return None

        # 使用 tqdm 进度条并发处理所有段落
        tasks = []
        with atqdm(total=len(all_segments), desc="生成数据集", unit="段落") as pbar:
            for seg in all_segments:
                task = asyncio.create_task(process_and_save(seg, pbar))
                # 添加完成回调来更新进度条
                task.add_done_callback(lambda _: pbar.update(1))
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # 收集成功的样本
        for result in results:
            if isinstance(result, DatasetSample):
                all_samples.append(result)

        # 输出速率限制统计
        rate_stats = self.rate_limiter.get_stats()

        # 输出语言分布统计
        total_lang = sum(self._language_counter.values())
        lang_stats = ""
        if total_lang > 0:
            en_pct = (self._language_counter["en"] / total_lang) * 100
            mixed_pct = (self._language_counter["mixed"] / total_lang) * 100
            zh_pct = (self._language_counter["zh"] / total_lang) * 100
            lang_stats = (
                f"语言分布：英文={self._language_counter['en']}({en_pct:.1f}%)，"
                f"中英混合={self._language_counter['mixed']}({mixed_pct:.1f}%)，"
                f"纯中文={self._language_counter['zh']}({zh_pct:.1f}%)"
            )

        self.logger.info(
            f"数据集生成完成，共 {len(all_samples)} 个样本，保存至 {output_file}。"
            f"成功: {success_count}，失败: {error_count}。"
            f"{lang_stats}。"
            f"速率限制统计：总请求={rate_stats['total_requests']}，"
            f"总等待时间={rate_stats['total_wait_time']:.2f}s，"
            f"平均等待={rate_stats['avg_wait_time']:.2f}s"
        )

        return all_samples

    def _extract_json(self, text: str) -> str:
        """从响应文本中提取JSON内容"""
        if "```json" in text:
            json_start = text.find("```json") + 7
            json_end = text.find("```", json_start)
            return text[json_start:json_end].strip()
        elif "```" in text:
            json_start = text.find("```") + 3
            json_end = text.find("```", json_start)
            return text[json_start:json_end].strip()
        return text.strip()

    def _save_sample(self, sample: DatasetSample, output_file: str):
        """实时追加保存单条样本到文件"""
        json_line = {
            "instruction": sample.instruction,
            "input": sample.input,
            "output": sample.output,
            "metadata": sample.metadata
        }
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

    def _save_dataset(self, samples: List[DatasetSample], output_file: str):
        """批量保存数据集到文件（向后兼容）"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                json_line = {
                    "instruction": sample.instruction,
                    "input": sample.input,
                    "output": sample.output,
                    "metadata": sample.metadata
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')


async def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='生成量子计算指令微调数据集')
    parser.add_argument('--output', '-o', default='./results/results.jsonl', help='输出文件路径')
    parser.add_argument('--max-samples', '-m', type=int, default=10, help='每篇论文最大样本数')
    parser.add_argument('--data-dir', '-d', default='data', help='论文文件目录')
    parser.add_argument('--provider', choices=['qianwen', 'nvidia'], default='nvidia', help='选择调用的大模型提供方')
    parser.add_argument('--model', default='deepseek-ai/deepseek-r1', help='模型名称（根据provider解释）')

    # Agent模式参数
    parser.add_argument('--agent', action='store_true', default=True, help='使用Agent模式（4步工作流：分段、生成指令、生成回答、验证）')
    parser.add_argument('--verification-strategy', choices=['retry', 'discard', 'flag'], default='retry',
                        help='验证失败策略: retry(重试), discard(丢弃), flag(标记)')
    parser.add_argument('--verification-threshold', type=float, default=0.7,
                        help='验证通过阈值 (0.0-1.0)')
    parser.add_argument('--max-retries', type=int, default=2, help='最大重试次数')
    parser.add_argument('--skip-verification', action='store_true', help='跳过验证步骤（加快生成速度）')
    parser.add_argument('--concurrency', type=int, default=5, help='并发处理数量（保留用于向后兼容）')
    parser.add_argument('--rate-limit', type=int, default=35, help='API调用速率限制（次/分钟）')

    args = parser.parse_args()

    # 从环境变量获取API密钥
    nvidia_api_key = os.getenv('NVIDIA_API_KEY')
    qianwen_api_key = os.getenv('QIANWEN_API_KEY') or os.getenv('DASHSCOPE_API_KEY')

    if args.provider == 'nvidia':
        if not nvidia_api_key:
            print("错误: 请选择NVIDIA提供方时设置NVIDIA_API_KEY")
            return
        from llm_client import NvidiaProvider
        # 禁用Provider内置限速，由Agent级别的速率限制器统一控制
        provider = NvidiaProvider(api_key=nvidia_api_key, model=args.model, rate_limit_per_minute=0)
        print(f"使用NVIDIA API，模型: {args.model}")
    else:
        if not qianwen_api_key:
            print("错误: 请选择QianWen提供方时设置QIANWEN_API_KEY或DASHSCOPE_API_KEY")
            return
        from llm_client import QianWenProvider
        provider = QianWenProvider(api_key=qianwen_api_key, model=args.model)
        print(f"使用QianWen API，模型: {args.model}")

    llm_client = LLMClient(provider)

    if args.agent:
        # 使用Agent模式（4步工作流）
        print("使用Agent模式（4步工作流）")
        print(f"速率限制: {args.rate_limit} 次/分钟")
        if args.skip_verification:
            print("跳过验证步骤")
        else:
            print(f"验证策略: {args.verification_strategy}, 阈值: {args.verification_threshold}")

        agent = ISQCoderDGAgent(
            llm_client=llm_client,
            verification_strategy=args.verification_strategy,
            max_retries=args.max_retries,
            verification_threshold=args.verification_threshold,
            skip_verification=args.skip_verification,
            concurrency=args.concurrency,
            rate_limit_per_minute=args.rate_limit
        )
        await agent.generate_dataset(
            data_dir=args.data_dir,
            output_file=args.output,
            max_samples_per_paper=args.max_samples
        )
    else:
        # 使用原有的单步模式
        print("使用传统模式（单步生成）")
        async with ISQCoderDataGenerator(llm_client) as generator:
            await generator.generate_dataset_from_papers(
                output_file=args.output,
                max_samples_per_paper=args.max_samples
            )


if __name__ == "__main__":
    asyncio.run(main())
