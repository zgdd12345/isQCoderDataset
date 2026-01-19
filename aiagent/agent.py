#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent模式的数据生成逻辑
"""

import os
import json
import asyncio
import time
import re
import logging
from typing import List, Any, Optional, Coroutine
from pathlib import Path
from datetime import datetime
from tqdm.asyncio import tqdm as atqdm
from llm_client import LLMClient
from dataset_core import (
    DatasetSample,
    SegmentedPaper,
    GeneratedInstruction,
    GeneratedAnswer,
    VerificationResult,
    normalize_latex,
    _configure_logging,
    _configure_paper_stats_logging,
)
from aiagent.agent_prompts import (
    get_instruction_prompt,
    get_answer_prompt,
    get_input_prompt,
    VERIFICATION_PROMPT,
    SEGMENT_RELEVANCE_PROMPT,
)


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
        max_concurrent: int = 8,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            rate_per_minute: 每分钟允许的请求数
            burst_size: 允许的突发请求数（令牌桶最大容量）
            max_concurrent: 最大并发请求数（信号量大小）
            logger: 日志记录器
        """
        self.rate_per_minute = rate_per_minute
        self.burst_size = min(burst_size, rate_per_minute)
        self.max_concurrent = max_concurrent
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()
        self.token_interval = 60.0 / rate_per_minute

        # 关键改变：使用信号量控制并发，而非序列化锁
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()  # 仅用于保护令牌状态读写

        self.logger = logger or logging.getLogger(__name__)

        # 统计信息
        self._total_requests = 0
        self._total_wait_time = 0.0

    async def acquire(self) -> float:
        """
        获取一个令牌，如果没有可用令牌则等待

        关键改进：使用信号量允许 max_concurrent 个任务并发

        Returns:
            等待时间（秒）
        """
        async with self._semaphore:  # 允许 max_concurrent 个任务并发进入
            wait_time = await self._calculate_wait_time()
            if wait_time > 0:
                self.logger.debug(f"速率限制：需等待 {wait_time:.2f}s")
                await asyncio.sleep(wait_time)  # 不在锁内睡眠
            else:
                self.logger.debug(f"速率限制：立即获取令牌")
            return wait_time

    async def _calculate_wait_time(self) -> float:
        """
        快速计算等待时间，不阻塞其他任务

        Returns:
            等待时间（秒）
        """
        async with self._lock:  # 仅锁保护令牌计算
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
                return 0.0

            # 计算等待时间并预扣令牌
            wait_time = (1.0 - self.tokens) * self.token_interval
            self.tokens = 0.0
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


class ISQCoderDGAgent:
    """
    数据生成代理，实现6步工作流：
    1. 读取论文，分割段落
    2. 分析段落是否适合生成instruction
    3. 基于段落生成指令
    4. 分析指令并生成input
    5. 根据指令和段落生成回答
    6. 重新阅读论文验证生成内容符合原意
    """

    def __init__(
        self,
        llm_client: LLMClient,
        verification_strategy: str = "retry",
        max_retries: int = 2,
        verification_threshold: float = 0.7,
        skip_verification: bool = False,
        concurrency: int = 5,
        rate_limit_per_minute: int = 30,
        min_segment_words: int = 120,
        max_concurrent_papers: int = 3,
        max_concurrent_segments: int = 15,
        language_mode: str = "auto",
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
            min_segment_words: 段落最少字数，低于则丢弃
            max_concurrent_papers: 最大并发论文数
            max_concurrent_segments: 最大并发段落数
            language_mode: 语言模式 ("auto", "en", "zh")
            logger: 日志记录器
        """
        self.llm_client = llm_client
        self.verification_strategy = verification_strategy
        self.max_retries = max_retries
        self.verification_threshold = verification_threshold
        self.skip_verification = skip_verification
        self.concurrency = concurrency
        self.rate_limit_per_minute = rate_limit_per_minute
        self.min_segment_words = min_segment_words
        self.max_concurrent_papers = max_concurrent_papers
        self.max_concurrent_segments = max_concurrent_segments
        self.language_mode = language_mode

        if logger is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.logger = _configure_logging(
                f'./log/agent_{run_id}.log'
            )
            self.paper_stats_logger = _configure_paper_stats_logging(
                f'./log/paper_stats_{run_id}.log'
            )
        else:
            self.logger = logger
            self.paper_stats_logger = _configure_paper_stats_logging(
                f'./log/paper_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )

        # 初始化全局速率限制器（优化配置）
        self.rate_limiter = TokenBucketRateLimiter(
            rate_per_minute=min(rate_limit_per_minute, 38),  # 接近40但留缓冲
            burst_size=8,  # 允许初始8次快速请求
            max_concurrent=8,  # 8个任务并发
            logger=self.logger
        )

        # 多级并发控制
        self._paper_semaphore = asyncio.Semaphore(max_concurrent_papers)
        self._segment_semaphore = asyncio.Semaphore(max_concurrent_segments)

        # 语言分布计数器（用于控制语言比例）
        self._language_counter = {"en": 0, "zh": 0}
        self._language_lock = asyncio.Lock()

    def _extract_abstract(self, content: str) -> str:
        """
        从论文内容中提取摘要

        Args:
            content: 完整的论文Markdown内容

        Returns:
            摘要文本，如果未找到则返回空字符串
        """
        # 匹配多种语言的摘要标题
        abstract_patterns = [
            r'#+\s*Abstract\s*\n',
            r'#+\s*ABSTRACT\s*\n',
            r'#+\s*摘要\s*\n',
            r'#+\s*摘\s*要\s*\n',
        ]

        for pattern in abstract_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start = match.end()

                # 查找下一个主要section (H1或H2标题)
                next_section = re.search(r'\n#{1,2}\s+[^\n]+\n', content[start:])
                if next_section:
                    end = start + next_section.start()
                else:
                    # 如果没有找到下一个section,最多取2000字符
                    end = start + 2000

                abstract = content[start:end].strip()

                # 限制最大长度为1000字符以控制token成本
                if len(abstract) > 1000:
                    abstract = abstract[:1000] + "..."

                return abstract

        return ""  # 未找到摘要

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

        # 提取论文标题和摘要
        lines = full_content.split('\n')
        paper_title = next(
            (line.strip('# ').strip() for line in lines if line.startswith('# ')),
            Path(paper_path).stem
        )
        paper_abstract = self._extract_abstract(full_content)

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
                            paper_abstract=paper_abstract,
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
                    paper_abstract=paper_abstract,
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
            "en" (英文), "zh" (纯中文)
        """
        if self.language_mode in ("en", "zh"):
            return self.language_mode

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
            language_mode: 语言模式 ("en", "zh")，如果为None则自动决定

        Returns:
            生成的指令，失败返回None
        """
        # 如果未指定语言模式，自动决定
        if language_mode is None:
            language_mode = await self._determine_language_mode()

        # 获取对应语言的提示词
        prompt_template = get_instruction_prompt(language_mode)
        prompt = prompt_template.format(
            paper_title=segment.paper_title,
            paper_abstract=segment.paper_abstract,
            segment_title=segment.segment_title,
            segment_content=segment.segment_content
        )

        response = await self.llm_client.generate(prompt)

        if not response.text:
            self.logger.error("指令生成API返回空响应")
            return None

        try:
            data = self._parse_json_from_text(response.text)

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
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"指令生成JSON解析失败: {e}")
            self.logger.debug(f"响应内容: {response.text}")
            return None

    @staticmethod
    def _estimate_text_units(text: str) -> int:
        """粗略估计字数：中文按字符计，英文按单词计。"""
        if not text:
            return 0
        cjk_chars = re.findall(r'[\u4e00-\u9fff]', text)
        word_tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)
        return len(cjk_chars) + len(word_tokens)

    def _is_segment_long_enough(self, segment: SegmentedPaper) -> bool:
        """根据字数过滤过短段落。"""
        word_count = self._estimate_text_units(segment.segment_content)
        if word_count < self.min_segment_words:
            self.logger.info(
                f"段落过短，跳过: {segment.segment_title} (字数={word_count}, 阈值={self.min_segment_words})"
            )
            return False
        return True

    async def check_segment_relevance(self, segment: SegmentedPaper) -> bool:
        """用大模型判断段落是否与量子计算相关。"""
        prompt = SEGMENT_RELEVANCE_PROMPT.format(
            paper_title=segment.paper_title,
            segment_title=segment.segment_title,
            segment_content=segment.segment_content
        )
        response = await self.llm_client.generate(prompt)

        if not response.text:
            self.logger.warning(f"段落相关性判定API返回空响应: {segment.segment_title}")
            return False

        try:
            data = self._parse_json_from_text(response.text)
            is_quantum = bool(data.get("is_quantum", False))
            reason = data.get("reason", "")
            if not isinstance(reason, str):
                reason = str(reason)
            self.logger.info(
                f"段落相关性判定: {segment.segment_title} -> {is_quantum} ({reason[:60]}...)"
            )
            return is_quantum
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"段落相关性判定JSON解析失败: {segment.segment_title} - {e}")
            return False

    async def generate_input(self, instruction: GeneratedInstruction) -> GeneratedInstruction:
        """
        步骤2.5：分析指令并生成合适的input

        Args:
            instruction: 生成的指令（不含input或input质量不高）

        Returns:
            更新了input字段的指令
        """
        language_mode = getattr(instruction, 'language_mode', 'en')

        prompt_template = get_input_prompt(language_mode)
        prompt = prompt_template.format(
            paper_title=instruction.segment.paper_title,
            paper_abstract=instruction.segment.paper_abstract,
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
            data = self._parse_json_from_text(response.text)

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
        except (json.JSONDecodeError, ValueError) as e:
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
        prompt_template = get_answer_prompt(language_mode)
        prompt = prompt_template.format(
            paper_title=instruction.segment.paper_title,
            paper_abstract=instruction.segment.paper_abstract,
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
        prompt = VERIFICATION_PROMPT.format(
            paper_title=answer.instruction.segment.paper_title,
            paper_abstract=answer.instruction.segment.paper_abstract,
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
            data = self._parse_json_from_text(response.text)

            return VerificationResult(
                passed=data.get("passed", False),
                confidence_score=float(data.get("confidence_score", 0.0)),
                issues=data.get("issues", []),
                context_leakage=data.get("context_leakage", []),
                self_contained=data.get("self_contained", True),
                suggestion=data.get("suggestion", "")
            )
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"验证结果JSON解析失败: {e}")
            self.logger.debug(f"验证响应内容: {response.text}")
            return VerificationResult(
                passed=False,
                confidence_score=0.0,
                issues=["验证响应解析失败"],
                suggestion="重新验证"
            )

    async def process_segment(self, segment: SegmentedPaper) -> Optional[DatasetSample]:
        """
        处理单个段落，执行完整的6步工作流（带速率限制）

        工作流：
        1. 分割段落（已在上层完成）
        2. 分析段落是否适合生成instruction（字数 + 相关性）
        3. 生成指令（instruction）
        4. 分析指令并生成input
        5. 根据指令和input生成回答
        6. 验证生成内容

        Args:
            segment: 论文段落

        Returns:
            生成的数据集样本，失败返回None
        """
        if not self._is_segment_long_enough(segment):
            return None

        is_relevant = await self._rate_limited_call(
            self.check_segment_relevance(segment)
        )
        if not is_relevant:
            self.logger.info(f"段落与量子计算无关，跳过: {segment.segment_title}")
            return None

        for attempt in range(self.max_retries + 1):
            try:
                # 步骤3：生成指令（速率限制）
                instruction = await self._rate_limited_call(
                    self.generate_instruction(segment)
                )
                if not instruction or not instruction.instruction:
                    self.logger.warning(f"指令生成失败: {segment.segment_title}")
                    continue

                self.logger.info(f"生成指令: {instruction.instruction[:50]}...")

                # 步骤4：分析指令并生成input（速率限制）
                instruction = await self._rate_limited_call(
                    self.generate_input(instruction)
                )
                self.logger.info(f"Input生成完成: {'有input' if instruction.input else '无input'}")

                # 步骤5：生成回答（速率限制）
                answer = await self._rate_limited_call(
                    self.generate_answer(instruction)
                )
                if not answer or not answer.output:
                    self.logger.warning(f"回答生成失败: {instruction.instruction[:50]}...")
                    continue

                self.logger.info(f"生成回答长度: {len(answer.output)}")

                # 步骤6：验证（可选，速率限制）
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
                if self.verification_strategy == "retry":
                    self.logger.info(f"验证未通过，重试 ({attempt + 1}/{self.max_retries + 1})")
                    continue

                self.logger.info("验证未通过，丢弃样本")
                return None

            except Exception as e:
                self.logger.error(f"处理段落时发生错误: {e}")
                if attempt == self.max_retries:
                    return None

        return None

    async def _process_single_paper(
        self,
        paper_file: str,
        max_samples_per_paper: int,
        output_file: str
    ) -> tuple[int, int, List[DatasetSample]]:
        """
        处理单篇论文（受论文级信号量控制）

        Args:
            paper_file: 论文文件路径
            max_samples_per_paper: 每篇论文最大样本数
            output_file: 输出文件路径前缀

        Returns:
            (成功数, 失败数, 样本列表)
        """
        async with self._paper_semaphore:
            paper_name = Path(paper_file).stem
            self.logger.info(f"分割论文: {paper_name}")

            segments = self.segment_paper(paper_file)
            if not segments:
                self.logger.warning(f"论文 {paper_name} 没有有效段落")
                self.paper_stats_logger.info(
                    f"{paper_name}\t成功=0\t失败=0\t总数=0"
                )
                return (0, 0, [])

            # 限制每篇论文的段落数（0表示不限制）
            if max_samples_per_paper > 0:
                segments = segments[:max_samples_per_paper]

            paper_output_file = self._build_paper_output_path(output_file, paper_file)
            if os.path.exists(paper_output_file):
                os.remove(paper_output_file)

            success_count = 0
            error_count = 0
            paper_samples: List[DatasetSample] = []

            # 包装 process_segment 以支持实时保存和进度更新
            async def process_and_save(segment: SegmentedPaper, pbar) -> Optional[DatasetSample]:
                nonlocal success_count, error_count
                # 添加段落级并发控制
                async with self._segment_semaphore:
                    try:
                        result = await self.process_segment(segment)
                        if isinstance(result, DatasetSample):
                            # 实时保存到文件
                            self._save_sample(result, paper_output_file)
                            success_count += 1
                            pbar.set_postfix({
                                '成功': success_count,
                                '失败': error_count,
                                '速率': f"{self.rate_limit_per_minute}/min"
                            })
                            return result
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

            # 使用 tqdm 进度条并发处理该论文的段落
            tasks = []
            with atqdm(total=len(segments), desc=f"生成数据集: {paper_name}", unit="段落") as pbar:
                for seg in segments:
                    task = asyncio.create_task(process_and_save(seg, pbar))
                    # 添加完成回调来更新进度条
                    task.add_done_callback(lambda _: pbar.update(1))
                    tasks.append(task)

                results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, DatasetSample):
                    paper_samples.append(result)

            self.logger.info(
                f"论文 {paper_name} 完成，生成 {len(paper_samples)} 个样本，保存至 {paper_output_file}"
            )
            self.paper_stats_logger.info(
                f"{paper_name}\t成功={success_count}\t失败={error_count}\t总数={len(segments)}"
            )

            return (success_count, error_count, paper_samples)

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
            output_file: 输出文件路径（作为前缀生成每篇论文的jsonl文件）
            max_samples_per_paper: 每篇论文最大样本数

        Returns:
            生成的数据集样本列表
        """
        import glob as glob_module

        paper_files = glob_module.glob(os.path.join(data_dir, "*.md"))
        self.logger.info(f"找到 {len(paper_files)} 个论文文件")

        total_success = 0
        total_error = 0
        all_samples: List[DatasetSample] = []

        # 并行处理所有论文
        tasks = [
            self._process_single_paper(
                paper_file,
                max_samples_per_paper,
                output_file
            )
            for paper_file in paper_files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 聚合结果
        for result in results:
            if isinstance(result, tuple):
                success, error, samples = result
                total_success += success
                total_error += error
                all_samples.extend(samples)
            elif isinstance(result, Exception):
                self.logger.error(f"论文处理失败: {result}")

        # 输出速率限制统计
        rate_stats = self.rate_limiter.get_stats()

        # 输出语言分布统计
        total_lang = sum(self._language_counter.values())
        lang_stats = ""
        if total_lang > 0:
            en_pct = (self._language_counter["en"] / total_lang) * 100
            zh_pct = (self._language_counter["zh"] / total_lang) * 100
            lang_stats = (
                f"语言分布：英文={self._language_counter['en']}({en_pct:.1f}%)，"
                f"纯中文={self._language_counter['zh']}({zh_pct:.1f}%)"
            )

        self.logger.info(
            f"数据集生成完成，共 {len(all_samples)} 个样本，按论文分别保存。"
            f"成功: {total_success}, 失败: {total_error}。{lang_stats}"
        )
        self.logger.info(
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
        if "```" in text:
            json_start = text.find("```") + 3
            json_end = text.find("```", json_start)
            return text[json_start:json_end].strip()
        return text.strip()

    @staticmethod
    def _strip_to_json_start(text: str) -> str:
        for idx, char in enumerate(text):
            if char in "{[":
                return text[idx:]
        return text

    @staticmethod
    def _escape_invalid_backslashes(text: str) -> str:
        result = []
        i = 0
        length = len(text)
        while i < length:
            char = text[i]
            if char != "\\":
                result.append(char)
                i += 1
                continue
            if i + 1 >= length:
                result.append("\\\\")
                i += 1
                continue
            nxt = text[i + 1]
            if nxt in ["\"", "\\", "/", "b", "f", "n", "r", "t"]:
                result.append("\\" + nxt)
                i += 2
                continue
            if nxt == "u" and i + 5 < length:
                hex_part = text[i + 2:i + 6]
                if all(c in "0123456789abcdefABCDEF" for c in hex_part):
                    result.append("\\u" + hex_part)
                    i += 6
                    continue
            result.append("\\\\")
            i += 1
        return "".join(result)

    def _parse_json_from_text(self, text: str) -> dict:
        json_text = self._extract_json(text).strip()
        json_text = self._strip_to_json_start(json_text)
        decoder = json.JSONDecoder()
        try:
            data, _ = decoder.raw_decode(json_text)
        except json.JSONDecodeError:
            sanitized = self._escape_invalid_backslashes(json_text)
            data, _ = decoder.raw_decode(sanitized)
        if not isinstance(data, dict):
            raise ValueError("JSON root is not an object")
        return data

    @staticmethod
    def _sanitize_filename(value: str) -> str:
        sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', value)
        sanitized = sanitized.strip("._-")
        return sanitized or "paper"

    def _build_paper_output_path(self, output_file: str, paper_path: str) -> str:
        output_path = Path(output_file)
        if output_path.suffix == ".jsonl":
            output_dir = output_path.parent
            prefix = output_path.stem
        else:
            output_dir = output_path
            prefix = "results"

        output_dir.mkdir(parents=True, exist_ok=True)
        paper_stem = self._sanitize_filename(Path(paper_path).stem)
        return str(output_dir / f"{prefix}_{paper_stem}.jsonl")

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
