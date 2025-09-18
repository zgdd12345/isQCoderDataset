#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
isQCoder数据集生成工具
使用阿里云通义千问API生成指令微调数据集
"""

import os
import json
import glob
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import argparse
import logging
from datetime import datetime
import dashscope
from dashscope import Generation
from batch_inference import BatchInferenceManager


@dataclass
class DatasetSample:
    """数据集样本结构"""
    instruction: str
    input: str
    output: str
    metadata: Dict[str, Any] = None


class QianWenDataGenerator:
    """使用阿里云通义千问模型生成指令微调数据的类"""
    
    def __init__(self, api_key: str, model: str = "qwen-plus"):
        # 设置API密钥
        dashscope.api_key = api_key
        self.model = model
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'./log/dataset_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
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
    
    async def call_qianwen_api(self, prompt: str) -> str:
        """调用通义千问API"""
        try:
            self.logger.info(f"开始调用API，模型: {self.model}")
            # 在异步环境中运行同步的API调用
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: Generation.call(
                    model=self.model,
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=2000,
                    top_p=0.9
                )
            )
            
            self.logger.info(f"API响应状态: {response.status_code}")
            if response.status_code == 200:
                return response.output.text
            else:
                self.logger.error(f"API调用失败: {response.status_code} - {response.message}")
                return ""
        except Exception as e:
            self.logger.error(f"API调用异常: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return ""
    
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
3. 如果需要额外的上下文信息，可以在input字段中提供，否则保持为空
4. output应该是对instruction的完整、准确、专业的回答

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
        response = await self.call_qianwen_api(prompt)
        
        if not response:
            self.logger.error("API返回空响应")
            return None
        
        self.logger.info(f"API响应长度: {len(response)}")
        try:
            # 尝试解析JSON响应
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_content = response[json_start:json_end].strip()
            else:
                json_content = response
            
            data = json.loads(json_content)
            
            return DatasetSample(
                instruction=data.get("instruction", ""),
                input=data.get("input", ""),
                output=data.get("output", ""),
                metadata={
                    "generated_at": datetime.now().isoformat(),
                    "model": self.model
                }
            )
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}")
            self.logger.error(f"响应内容: {response}")
            return None
    
    async def generate_dataset_from_papers(self, output_file: str = "quantum_instruction_dataset.jsonl", 
                                          max_samples_per_paper: int = 4, use_batch: bool = False,
                                          batch_completion_window: str = "24h"):
        """从论文生成完整的指令微调数据集"""
        paper_files = self.get_paper_files()
        
        self.logger.info(f"找到 {len(paper_files)} 个论文文件")
        
        if use_batch:
            # 使用批量推理模式
            await self._generate_dataset_batch_mode(paper_files, output_file, 
                                                   max_samples_per_paper, batch_completion_window)
        else:
            # 使用实时推理模式
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
    
    async def _generate_dataset_batch_mode(self, paper_files: List[str], output_file: str,
                                         max_samples_per_paper: int, completion_window: str):
        """批量推理模式生成数据集"""
        # 收集所有提示
        all_prompts = []
        prompt_metadata = []
        
        for paper_file in paper_files:
            paper_name = Path(paper_file).stem
            self.logger.info(f"准备论文: {paper_name}")
            
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
            
            all_prompts.extend(prompts)
            
            # 保存元数据，用于后续结果处理
            for i, prompt in enumerate(prompts):
                prompt_metadata.append({
                    'paper_name': paper_name,
                    'paper_title': paper_title,
                    'prompt_index': i
                })
        
        if not all_prompts:
            self.logger.warning("没有找到有效的提示")
            return
        
        self.logger.info(f"总共准备了 {len(all_prompts)} 个提示，开始批量推理")
        
        # 使用批量推理
        batch_manager = BatchInferenceManager(dashscope.api_key, self.model)
        
        job_name = f"dataset_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = await batch_manager.run_batch_inference(
            prompts=all_prompts,
            job_name=job_name,
            completion_window=completion_window,
            wait_for_completion=True,
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9
        )
        
        if 'results' not in result:
            self.logger.error("批量推理未返回结果")
            return
        
        # 处理批量推理结果
        all_samples = []
        for i, batch_result in enumerate(result['results']):
            try:
                if 'response' in batch_result and 'body' in batch_result['response']:
                    response_body = batch_result['response']['body']
                    if 'choices' in response_body and response_body['choices']:
                        content = response_body['choices'][0]['message']['content']
                        
                        # 解析JSON响应
                        if "```json" in content:
                            json_start = content.find("```json") + 7
                            json_end = content.find("```", json_start)
                            json_content = content[json_start:json_end].strip()
                        else:
                            json_content = content
                        
                        data = json.loads(json_content)
                        
                        # 创建数据集样本
                        sample = DatasetSample(
                            instruction=data.get("instruction", ""),
                            input=data.get("input", ""),
                            output=data.get("output", ""),
                            metadata={
                                "generated_at": datetime.now().isoformat(),
                                "model": self.model,
                                "batch_inference": True,
                                "paper_name": prompt_metadata[i]['paper_name'] if i < len(prompt_metadata) else 'unknown',
                                "paper_title": prompt_metadata[i]['paper_title'] if i < len(prompt_metadata) else 'unknown'
                            }
                        )
                        
                        if sample.instruction and sample.output:
                            all_samples.append(sample)
                        
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                self.logger.warning(f"解析第 {i+1} 个批量结果失败: {e}")
                continue
        
        # 保存数据集
        self.save_dataset(all_samples, output_file)
        self.logger.info(f"批量推理数据集生成完成，共 {len(all_samples)} 个有效样本，保存至 {output_file}")
    
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
    parser = argparse.ArgumentParser(description='生成量子计算指令微调数据集')
    parser.add_argument('--output', '-o', default='./results/results.jsonl', help='输出文件路径')
    parser.add_argument('--max-samples', '-m', type=int, default=10, help='每篇论文最大样本数')
    parser.add_argument('--data-dir', '-d', default='data', help='论文文件目录')
    parser.add_argument('--batch', '-b', action='store_true', help='使用批量推理模式（成本更低）')
    parser.add_argument('--completion-window', '-w', default='24h', 
                       choices=['24h', '48h', '72h', '168h', '336h'], 
                       help='批量推理完成时间窗口')
    parser.add_argument('--model', default='qwen-plus', help='使用的模型名称: qwen-plus, qwen-7b, qwen-14b等')
    
    args = parser.parse_args()
    
    # 从环境变量获取API密钥
    api_key = os.getenv('QIANWEN_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("错误: 请设置环境变量 QIANWEN_API_KEY 或 DASHSCOPE_API_KEY")
        print("使用方法: export QIANWEN_API_KEY='your-api-key-here'")
        print("或者: export DASHSCOPE_API_KEY='your-api-key-here'")
        return
    
    if args.batch:
        print(f"使用批量推理模式，完成窗口: {args.completion_window}")
        print("注意：批量推理成本仅为实时推理的50%，但需要等待任务完成")
    else:
        print("使用实时推理模式")
    
    async with QianWenDataGenerator(api_key, model=args.model) as generator:
        await generator.generate_dataset_from_papers(
            output_file=args.output,
            max_samples_per_paper=args.max_samples,
            use_batch=args.batch,
            batch_completion_window=args.completion_window
        )


if __name__ == "__main__":
    asyncio.run(main())