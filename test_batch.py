#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量推理功能测试脚本
"""

import os
import sys
import asyncio
import json
from batch_inference import BatchInferenceManager
from pathlib import Path


def test_batch_inference():
    """测试批量推理功能"""
    # 获取API密钥
    api_key = os.getenv('QIANWEN_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("错误: 请设置环境变量 QIANWEN_API_KEY 或 DASHSCOPE_API_KEY")
        return False
    
    # 创建测试提示
    test_prompts = [
        """你是一个专业的量子计算指令数据生成专家。请基于以下内容，生成一个高质量的指令微调数据对。

        内容：量子计算中的量子比特是量子信息的基本单位，类似于经典计算中的比特，但具有叠加态和纠缠的特性。

        要求：
        1. 生成一个合适的instruction（指令）
        2. 如果需要额外的上下文信息，可以在input字段中提供，否则保持为空
        3. output应该是对instruction的完整、准确、专业的回答

        返回JSON格式的结果：
        {
            "instruction": "这里是指令",
            "input": "这里是输入（可为空）",
            "output": "这里是详细的回答"
        }""",
                
                """你是一个专业的量子计算指令数据生成专家。请基于以下内容，生成一个高质量的指令微调数据对。

        内容：量子门是量子计算中的基本操作，用于操作量子比特的状态。常见的量子门包括Pauli-X门、Pauli-Y门、Pauli-Z门和Hadamard门等。

        要求：
        1. 生成一个合适的instruction（指令）
        2. 如果需要额外的上下文信息，可以在input字段中提供，否则保持为空
        3. output应该是对instruction的完整、准确、专业的回答

        返回JSON格式的结果：
        {
            "instruction": "这里是指令",
            "input": "这里是输入（可为空）",
            "output": "这里是详细的回答"
        }"""
    ]
    
    print("开始测试批量推理功能...")
    print(f"测试提示数量: {len(test_prompts)}")
    
    # 创建批量推理管理器
    manager = BatchInferenceManager(api_key, model='qwen-plus')
    
    try:
        # 运行批量推理
        result = asyncio.run(manager.run_batch_inference(
            prompts=test_prompts,
            job_name="test_batch_inference",
            completion_window="24h",
            wait_for_completion=True,
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9
        ))
        
        print(f"批量推理任务完成!")
        print(f"任务ID: {result['job_id']}")
        print(f"状态: {result['status']}")
        
        if 'results' in result:
            print(f"获得结果数量: {len(result['results'])}")
            
            # 显示前两个结果的预览
            for i, batch_result in enumerate(result['results'][:2]):
                print(f"\n--- 结果 {i+1} 预览 ---")
                try:
                    if 'response' in batch_result and 'body' in batch_result['response']:
                        response_body = batch_result['response']['body']
                        if 'choices' in response_body and response_body['choices']:
                            content = response_body['choices'][0]['message']['content']
                            print(f"原始响应: {content[:200]}...")
                            
                            # 尝试解析JSON
                            if "```json" in content:
                                json_start = content.find("```json") + 7
                                json_end = content.find("```", json_start)
                                json_content = content[json_start:json_end].strip()
                            else:
                                json_content = content
                            
                            data = json.loads(json_content)
                            print(f"指令: {data.get('instruction', '')[:100]}...")
                            print(f"输出: {data.get('output', '')[:100]}...")
                            
                except Exception as e:
                    print(f"解析结果 {i+1} 失败: {e}")
            
            print("\n批量推理测试成功!")
            return True
        else:
            print("批量推理未返回结果")
            return False
            
    except Exception as e:
        print(f"批量推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if test_batch_inference():
        print("\n✅ 批量推理功能测试通过")
        sys.exit(0)
    else:
        print("\n❌ 批量推理功能测试失败")
        sys.exit(1)