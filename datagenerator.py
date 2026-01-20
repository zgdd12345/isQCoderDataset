#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
isQCoder数据集生成工具
使用大模型API生成指令微调数据集
"""

import os
import json
import glob
import asyncio
from typing import List, Dict, Any
from pathlib import Path
import argparse
from datetime import datetime
from dotenv import load_dotenv
from llm_client import LLMClient
from dataset_core import (
    DatasetSample,
    normalize_latex,
    _configure_logging,
    _configure_paper_stats_logging,
)
from aiagent.agent import ISQCoderDGAgent


class ISQCoderDataGenerator:
    """生成指令微调数据的类，与模型与API解耦"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

        # 设置日志（使用tqdm友好的控制台输出）
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = _configure_logging(
            f'./log/dataset_generation_{run_id}.log'
        )
        self.paper_stats_logger = _configure_paper_stats_logging(
            f'./log/paper_stats_{run_id}.log'
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
    
    def _resolve_language_mode(self, language_mode: str) -> str:
        if language_mode in ("en", "zh"):
            return language_mode
        return "zh"

    def create_instruction_prompts(
        self,
        paper_content: str,
        paper_title: str,
        language_mode: str = "auto"
    ) -> List[str]:
        """基于论文段落内容创建指令微调提示"""
        language_mode = self._resolve_language_mode(language_mode)
        segments = self.segment_paper_content(paper_content)
        prompts = []
        
        for segment in segments:
            segment_title = segment['title']
            segment_content = segment['content']
            print(f"Processing segment: {segment_title}, length: {len(segment_content)}")
            # 为每个段落生成指令微调数据对的提示
            if language_mode == "en":
                prompt = f"""You are a professional quantum computing instruction data generator. Based on the paper segment below, identify the core knowledge points and generate a high-quality instruction-tuning data pair.

Paper Title: {paper_title}
Segment Title: {segment_title}
Segment Content:
{segment_content}

Requirements:
1. Deeply understand the core content and key concepts in the segment
2. Generate a natural, suitable instruction (task/question) based on the content
3. Generate the input field as needed (background/constraints/data); if not needed, use an empty string
4. The output must be a complete, accurate, and professional answer to the instruction, using the input if provided
5. Language consistency: instruction, input, and output must be entirely in English; do not mix Chinese
6. All formulas must use standard LaTeX delimiters $...$ or $$...$$ (do not use \\(\\) or \\[\\])
7. No citation tags or source labels such as `` or 【来源: x】 may appear in the output
8. Input purity: input must be pure knowledge or data, with no meta prefixes like "Key context from paper:", "Based on segment:", or "Context:"
9. Do not reference figures or tables not fully converted to text in input (avoid "see Fig. 1", "as shown in Table 2")

Instruction types can include but are not limited to:
- Concept explanation and definition
- Method steps explanation
- Algorithm principle description
- Formula derivation process
- Implementation plan description
- Problem analysis and discussion
- Application scenario introduction

Ensure the generated instruction pair has educational value and helps learners understand quantum computing knowledge.

Return a JSON object:
{{
    "instruction": "Instruction text in English",
    "input": "Input text (may be empty)",
    "output": "Detailed answer in English"
}}"""
            else:
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
5. 语言一致性：instruction、input、output必须为纯中文，严禁中英混杂
6. 所有数学公式统一使用标准LaTeX格式$...$或$$...$$，不要使用\\(\\)或\\[\\]
7. 禁止在output中出现 ``、【来源: x】 或类似引用标签，必须直接陈述事实
8. 纯净输入：input必须是纯粹的知识片段或数据，不要包含"Key context from paper:"、"Based on segment:"、"Context:"等元数据前缀
9. output中禁止引用未转化为文本的图/表（如"见图1"、"如表2所示"），除非表格内容已完整写入input

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
    
    async def generate_dataset_from_papers(
        self,
        output_file: str = "quantum_instruction_dataset.jsonl",
        max_samples_per_paper: int = 4,
        language_mode: str = "auto"
    ):
        """从论文生成完整的指令微调数据集"""
        paper_files = self.get_paper_files()
        
        self.logger.info(f"找到 {len(paper_files)} 个论文文件")
        
        await self._generate_dataset_realtime_mode(
            paper_files,
            output_file,
            max_samples_per_paper,
            language_mode
        )
    
    async def _generate_dataset_realtime_mode(
        self,
        paper_files: List[str],
        output_file: str,
        max_samples_per_paper: int,
        language_mode: str = "auto"
    ):
        """实时推理模式生成数据集"""
        all_samples = []

        for paper_file in paper_files:
            paper_name = Path(paper_file).stem
            self.logger.info(f"处理论文: {paper_name}")
            
            paper_content = self.load_markdown_content(paper_file)
            if not paper_content:
                self.paper_stats_logger.info(
                    f"{paper_name}\t成功=0\t失败=0\t总数=0"
                )
                continue
            
            # 提取论文标题
            lines = paper_content.split('\n')
            paper_title = next((line.strip('# ') for line in lines if line.startswith('# ')), paper_name)
            
            # 生成多种类型的提示
            prompts = self.create_instruction_prompts(
                paper_content,
                paper_title,
                language_mode
            )
            
            # 限制每篇论文的样本数量（0表示不限制）
            if max_samples_per_paper > 0:
                prompts = prompts[:max_samples_per_paper]

            total_prompts = len(prompts)

            # 并发生成样本
            tasks = [self.generate_dataset_sample(prompt) for prompt in prompts]
            samples = await asyncio.gather(*tasks)

            # 过滤有效样本
            valid_samples = [s for s in samples if s is not None and s.instruction and s.output]
            all_samples.extend(valid_samples)

            self.logger.info(f"从 {paper_name} 生成了 {len(valid_samples)} 个有效样本")
            failure_count = total_prompts - len(valid_samples)
            self.paper_stats_logger.info(
                f"{paper_name}\t成功={len(valid_samples)}\t失败={failure_count}\t总数={total_prompts}"
            )
            
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


async def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='生成量子计算指令微调数据集')
    parser.add_argument('--output', '-o', default='./results_paper/results.jsonl', help='输出文件路径（作为前缀生成每篇论文的jsonl文件）')
    parser.add_argument('--max-samples', '-m', type=int, default=0, help='每篇论文最大样本数（0表示不限制）')
    parser.add_argument('--data-dir', '-d', default='data/raw', help='论文文件目录')
    parser.add_argument('--provider', choices=['qianwen', 'nvidia'], default='nvidia', help='选择调用的大模型提供方')
    parser.add_argument('--language-mode', choices=['auto', 'en', 'zh'], default='auto', help='语言模式: auto/en/zh')
    # parser.add_argument('--model', default='deepseek-ai/deepseek-r1-0528', help='模型名称（根据provider解释）')
    parser.add_argument('--model', default='z-ai/glm4.7', help='模型名称（根据provider解释）')
    
    # parser.add_argument('--nvidia-proxy', default='', help='NVIDIA API代理地址（例如 http://user:pass@host:port ）')

    # Agent模式参数
    parser.add_argument('--agent', action='store_true', default=True, help='使用Agent模式（6步工作流：分段、适配性、指令、input、回答、验证）')
    parser.add_argument('--verification-strategy', choices=['retry', 'discard', 'flag'], default='retry', help='验证失败策略: retry(重试), discard(丢弃), flag(标记)')
    parser.add_argument('--verification-threshold', type=float, default=0.7, help='验证通过阈值 (0.0-1.0)')
    parser.add_argument('--max-retries', type=int, default=2, help='最大重试次数')
    parser.add_argument('--skip-verification', action='store_true', help='跳过验证步骤（加快生成速度）')
    parser.add_argument('--concurrency', type=int, default=5, help='并发处理数量（保留用于向后兼容）')
    parser.add_argument('--rate-limit', type=int, default=38, help='API调用速率限制（次/分钟）')
    parser.add_argument('--min-segment-words', type=int, default=120, help='段落最少字数，低于则跳过')
    parser.add_argument('--max-concurrent-papers', type=int, default=10, help='最大并发论文数')
    parser.add_argument('--max-concurrent-segments', type=int, default=50, help='最大并发段落数')

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
        # nvidia_proxy = args.nvidia_proxy or os.getenv("NVIDIA_PROXY", "")
        provider = NvidiaProvider(
            api_key=nvidia_api_key,
            model=args.model,
            rate_limit_per_minute=0,
            # proxy=nvidia_proxy or None,
        )
        print(f"使用NVIDIA API，模型: {args.model}")
        # if nvidia_proxy:
        #     print("NVIDIA API代理: 已设置")
        if hasattr(provider, "test_connection"):
            ok, detail = await provider.test_connection()
            status = "成功" if ok else "失败"
            detail_text = f" ({detail})" if detail else ""
            print(f"NVIDIA API链路测试: {status}{detail_text}")
    else:
        if not qianwen_api_key:
            print("错误: 请选择QianWen提供方时设置QIANWEN_API_KEY或DASHSCOPE_API_KEY")
            return
        from llm_client import QianWenProvider
        provider = QianWenProvider(api_key=qianwen_api_key, model=args.model)
        print(f"使用QianWen API，模型: {args.model}")

    llm_client = LLMClient(provider)

    if args.agent:
        # 使用Agent模式（6步工作流）
        print("使用Agent模式（6步工作流）")
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
            rate_limit_per_minute=args.rate_limit,
            min_segment_words=args.min_segment_words,
            max_concurrent_papers=args.max_concurrent_papers,
            max_concurrent_segments=args.max_concurrent_segments,
            language_mode=args.language_mode
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
                max_samples_per_paper=args.max_samples,
                language_mode=args.language_mode
            )


if __name__ == "__main__":
    asyncio.run(main())
