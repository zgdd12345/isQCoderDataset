#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阿里云百炼批量推理模块
使用OpenAI兼容接口进行批量推理，成本仅为实时推理的50%
"""

import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging
from openai import OpenAI


class BatchStatus(Enum):
    """批量任务状态"""
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress" 
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


@dataclass
class BatchRequest:
    """单个批量请求"""
    custom_id: str
    method: str
    url: str
    body: Dict[str, Any]


@dataclass
class BatchJob:
    """批量任务信息"""
    id: str
    status: BatchStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    input_file_id: str = ""
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    completion_window: str = "24h"
    metadata: Optional[Dict[str, Any]] = None


class QianWenBatchInference:
    """阿里云百炼批量推理类"""
    
    def __init__(self, api_key: str, model: str = "qwen-plus", base_url: str = None):
        """
        初始化批量推理客户端
        
        Args:
            api_key: 阿里云百炼API密钥 
            model: 模型名称，默认为qwen-plus
            base_url: API基础URL，默认使用OpenAI兼容接口
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url
        )
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    def create_batch_requests(self, prompts: List[str], 
                            temperature: float = 0.7,
                            max_tokens: int = 2000,
                            top_p: float = 0.9) -> List[BatchRequest]:
        """
        创建批量请求列表
        
        Args:
            prompts: 提示列表
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top_p参数
            
        Returns:
            BatchRequest列表
        """
        requests = []
        
        for i, prompt in enumerate(prompts):
            custom_id = f"request_{i}_{int(time.time())}"
            
            request = BatchRequest(
                custom_id=custom_id,
                method="POST",
                url="/v1/chat/completions",
                body={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p
                }
            )
            requests.append(request)
            
        return requests
    
    def save_batch_file(self, requests: List[BatchRequest], 
                       file_path: str) -> str:
        """
        保存批量请求到JSONL文件
        
        Args:
            requests: 批量请求列表
            file_path: 文件路径
            
        Returns:
            文件路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for request in requests:
                json_line = {
                    "custom_id": request.custom_id,
                    "method": request.method,
                    "url": request.url,
                    "body": request.body
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
        
        self.logger.info(f"批量请求文件已保存: {file_path}")
        return file_path
    
    def upload_batch_file(self, file_path: str) -> str:
        """
        上传批量请求文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件ID
        """
        try:
            with open(file_path, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose='batch'
                )
            
            file_id = response.id
            self.logger.info(f"文件上传成功，文件ID: {file_id}")
            return file_id
            
        except Exception as e:
            self.logger.error(f"文件上传失败: {e}")
            raise
    
    def create_batch_job(self, input_file_id: str,
                        completion_window: str = "24h",
                        metadata: Optional[Dict[str, Any]] = None) -> BatchJob:
        """
        创建批量推理任务
        
        Args:
            input_file_id: 输入文件ID
            completion_window: 完成窗口时间（24h到336h）
            metadata: 元数据
            
        Returns:
            BatchJob对象
        """
        try:
            response = self.client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window=completion_window,
                metadata=metadata or {}
            )
            
            batch_job = BatchJob(
                id=response.id,
                status=BatchStatus(response.status),
                created_at=datetime.fromtimestamp(response.created_at),
                input_file_id=input_file_id,
                completion_window=completion_window,
                metadata=metadata
            )
            
            self.logger.info(f"批量任务创建成功，任务ID: {batch_job.id}")
            return batch_job
            
        except Exception as e:
            self.logger.error(f"批量任务创建失败: {e}")
            raise
    
    def get_batch_status(self, batch_id: str) -> BatchJob:
        """
        获取批量任务状态
        
        Args:
            batch_id: 批量任务ID
            
        Returns:
            BatchJob对象
        """
        try:
            response = self.client.batches.retrieve(batch_id)
            
            batch_job = BatchJob(
                id=response.id,
                status=BatchStatus(response.status),
                created_at=datetime.fromtimestamp(response.created_at),
                completed_at=datetime.fromtimestamp(response.completed_at) if response.completed_at else None,
                input_file_id=response.input_file_id,
                output_file_id=response.output_file_id,
                error_file_id=response.error_file_id,
                completion_window=response.completion_window,
                metadata=response.metadata
            )
            
            return batch_job
            
        except Exception as e:
            self.logger.error(f"获取批量任务状态失败: {e}")
            raise
    
    def download_batch_results(self, output_file_id: str, 
                              local_path: str) -> str:
        """
        下载批量推理结果
        
        Args:
            output_file_id: 输出文件ID
            local_path: 本地保存路径
            
        Returns:
            本地文件路径
        """
        try:
            response = self.client.files.content(output_file_id)
            
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"批量结果已下载: {local_path}")
            return local_path
            
        except Exception as e:
            self.logger.error(f"下载批量结果失败: {e}")
            raise
    
    def wait_for_completion(self, batch_id: str, 
                           check_interval: int = 60,
                           max_wait_time: int = 86400) -> BatchJob:
        """
        等待批量任务完成
        
        Args:
            batch_id: 批量任务ID
            check_interval: 检查间隔（秒）
            max_wait_time: 最大等待时间（秒）
            
        Returns:
            完成的BatchJob对象
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            batch_job = self.get_batch_status(batch_id)
            
            self.logger.info(f"批量任务状态: {batch_job.status.value}")
            
            if batch_job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, 
                                  BatchStatus.EXPIRED, BatchStatus.CANCELLED]:
                return batch_job
            
            time.sleep(check_interval)
        
        raise TimeoutError(f"批量任务在{max_wait_time}秒内未完成")
    
    def parse_batch_results(self, results_file_path: str) -> List[Dict[str, Any]]:
        """
        解析批量推理结果
        
        Args:
            results_file_path: 结果文件路径
            
        Returns:
            解析后的结果列表
        """
        results = []
        
        try:
            with open(results_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        results.append(result)
            
            self.logger.info(f"解析了 {len(results)} 个批量推理结果")
            return results
            
        except Exception as e:
            self.logger.error(f"解析批量结果失败: {e}")
            raise
    
    def cancel_batch_job(self, batch_id: str) -> bool:
        """
        取消批量任务
        
        Args:
            batch_id: 批量任务ID
            
        Returns:
            是否成功取消
        """
        try:
            response = self.client.batches.cancel(batch_id)
            self.logger.info(f"批量任务 {batch_id} 已取消")
            return response.status == "cancelling"
            
        except Exception as e:
            self.logger.error(f"取消批量任务失败: {e}")
            return False
    
    def list_batch_jobs(self, limit: int = 20) -> List[BatchJob]:
        """
        列出批量任务
        
        Args:
            limit: 返回数量限制
            
        Returns:
            BatchJob列表
        """
        try:
            response = self.client.batches.list(limit=limit)
            
            jobs = []
            for batch_data in response.data:
                job = BatchJob(
                    id=batch_data.id,
                    status=BatchStatus(batch_data.status),
                    created_at=datetime.fromtimestamp(batch_data.created_at),
                    completed_at=datetime.fromtimestamp(batch_data.completed_at) if batch_data.completed_at else None,
                    input_file_id=batch_data.input_file_id,
                    output_file_id=batch_data.output_file_id,
                    error_file_id=batch_data.error_file_id,
                    completion_window=batch_data.completion_window,
                    metadata=batch_data.metadata
                )
                jobs.append(job)
            
            return jobs
            
        except Exception as e:
            self.logger.error(f"获取批量任务列表失败: {e}")
            raise


class BatchInferenceManager:
    """批量推理管理器"""
    
    def __init__(self, api_key: str, model: str = "qwen-plus"):
        self.batch_client = QianWenBatchInference(api_key, model)
        self.logger = logging.getLogger(__name__)
        
        # 创建批量任务工作目录
        self.work_dir = Path("batch_jobs")
        self.work_dir.mkdir(exist_ok=True)
    
    async def run_batch_inference(self, prompts: List[str],
                                 job_name: str = None,
                                 completion_window: str = "24h",
                                 wait_for_completion: bool = True,
                                 **model_params) -> Dict[str, Any]:
        """
        运行批量推理任务
        
        Args:
            prompts: 提示列表
            job_name: 任务名称
            completion_window: 完成时间窗口
            wait_for_completion: 是否等待完成
            **model_params: 模型参数
            
        Returns:
            包含任务信息和结果的字典
        """
        if not prompts:
            raise ValueError("提示列表不能为空")
        
        job_name = job_name or f"batch_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"开始批量推理任务: {job_name}, 共 {len(prompts)} 个请求")
        
        # 创建批量请求
        requests = self.batch_client.create_batch_requests(prompts, **model_params)
        
        # 保存请求文件
        input_file = self.work_dir / f"{job_name}_input.jsonl"
        self.batch_client.save_batch_file(requests, str(input_file))
        
        # 上传文件
        input_file_id = self.batch_client.upload_batch_file(str(input_file))
        
        # 创建批量任务
        metadata = {
            "ds_name": job_name,
            "ds_description": f"量子计算数据集批量推理任务，包含{len(prompts)}个请求"
        }
        
        batch_job = self.batch_client.create_batch_job(
            input_file_id=input_file_id,
            completion_window=completion_window,
            metadata=metadata
        )
        
        result = {
            "job_id": batch_job.id,
            "job_name": job_name,
            "status": batch_job.status.value,
            "created_at": batch_job.created_at,
            "input_file_id": input_file_id,
            "request_count": len(prompts),
            "completion_window": completion_window
        }
        
        # 如果需要等待完成
        if wait_for_completion:
            self.logger.info(f"等待批量任务完成: {batch_job.id}")
            completed_job = self.batch_client.wait_for_completion(batch_job.id)
            
            result.update({
                "status": completed_job.status.value,
                "completed_at": completed_job.completed_at,
                "output_file_id": completed_job.output_file_id,
                "error_file_id": completed_job.error_file_id
            })
            
            # 下载结果
            if completed_job.output_file_id:
                output_file = self.work_dir / f"{job_name}_output.jsonl"
                self.batch_client.download_batch_results(
                    completed_job.output_file_id, 
                    str(output_file)
                )
                
                # 解析结果
                results = self.batch_client.parse_batch_results(str(output_file))
                result["results"] = results
                result["output_file"] = str(output_file)
                
                self.logger.info(f"批量推理任务完成: {job_name}, 获得 {len(results)} 个结果")
            
            # 下载错误文件（如果有）
            if completed_job.error_file_id:
                error_file = self.work_dir / f"{job_name}_errors.jsonl"
                self.batch_client.download_batch_results(
                    completed_job.error_file_id,
                    str(error_file)
                )
                result["error_file"] = str(error_file)
        
        return result