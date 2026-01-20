#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM provider client.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import dashscope
from dashscope import Generation
import httpx
from openai import OpenAI


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str


class LLMProvider(ABC):
    def __init__(self, name: str, model: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.model = model
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError


class QianWenProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "qwen-plus",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 0.9,
        timeout: int = 300,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__("qianwen", model, logger)
        if not api_key:
            raise ValueError("QianWen API key is required")
        dashscope.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout

    async def generate(self, prompt: str) -> str:
        try:
            self.logger.info(f"开始调用QianWen API，模型: {self.model}")
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: Generation.call(
                    model=self.model,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    timeout=self.timeout,
                ),
            )

            self.logger.info(f"QianWen API响应状态: {response.status_code}")
            if response.status_code == 200:
                return response.output.text
            self.logger.error(
                f"QianWen API调用失败: {response.status_code} - {response.message}"
            )
            return ""
        except Exception as exc:
            self.logger.error(f"QianWen API调用异常: {exc}")
            import traceback

            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return ""


class NvidiaProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-ai/deepseek-r1-0528",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        rate_limit_per_minute: int = 30,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        top_p: float = 0.7,
        timeout: float = 300.0,  # 增加到5分钟，DeepSeek R1推理模型需要更长时间
        proxy: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__("nvidia", model, logger)
        if not api_key:
            raise ValueError("NVIDIA API key is required")
        self.proxy = proxy
        http_client = None
        if proxy:
            # 使用异步httpx client，并增加连接和读取超时
            http_client = httpx.Client(
                proxy=proxy,
                timeout=httpx.Timeout(
                    connect=30.0,  # 连接超时30秒
                    read=timeout,   # 读取超时使用设定值
                    write=30.0,     # 写入超时30秒
                    pool=10.0       # 连接池超时10秒
                ),
                trust_env=False
            )
        else:
            # 即使不用代理，也设置合理的超时
            http_client = httpx.Client(
                timeout=httpx.Timeout(
                    connect=30.0,
                    read=timeout,
                    write=30.0,
                    pool=10.0
                )
            )
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            http_client=http_client,
        )
        self.timeout = timeout
        self.rate_limit_per_minute = rate_limit_per_minute
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._rate_lock = asyncio.Lock()
        self._last_request_time = 0.0
        self._min_interval = 0.0
        if self.rate_limit_per_minute > 0:
            self._min_interval = 60.0 / self.rate_limit_per_minute

    async def test_connection(self) -> tuple[bool, str]:
        """测试NVIDIA API链路是否畅通。"""
        try:
            self.logger.info(f"开始测试NVIDIA API连接，模型: {self.model}")
            await self._throttle()
            loop = asyncio.get_running_loop()

            def _call() -> str:
                self.logger.info("发送测试请求到NVIDIA API（使用流式模式）...")
                # 使用流式API，可能更稳定
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Say OK"}],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=10,
                    stream=True,  # 使用流式响应
                )
                response_text = ""
                for chunk in stream:
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = getattr(chunk.choices[0], "delta", None)
                    content = getattr(delta, "content", None)
                    if content:
                        response_text += content
                self.logger.info("收到NVIDIA API响应")
                return response_text

            response = await loop.run_in_executor(None, _call)
            self.logger.info(f"NVIDIA API测试成功: {response[:100]}")
            return True, response.strip()
        except Exception as exc:
            error_msg = str(exc)
            self.logger.error(f"NVIDIA API链路测试异常: {exc}")

            # 提供更具体的错误提示
            if "Connection error" in error_msg or "disconnected" in error_msg:
                hint = "网络连接问题。可能需要配置代理（NVIDIA_PROXY）或检查防火墙设置"
            elif "timeout" in error_msg.lower():
                hint = "请求超时。DeepSeek R1 是推理模型，可能需要更长时间"
            elif "401" in error_msg or "authentication" in error_msg.lower():
                hint = "API密钥无效或过期"
            elif "429" in error_msg or "rate" in error_msg.lower():
                hint = "API速率限制"
            else:
                hint = error_msg

            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            return False, hint

    async def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._rate_lock:
            now = time.monotonic()
            wait_time = self._min_interval - (now - self._last_request_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_request_time = time.monotonic()

    async def generate(self, prompt: str) -> str:
        max_retries = 3
        retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                self.logger.info(f"开始调用NVIDIA API，模型: {self.model} (尝试 {attempt + 1}/{max_retries})")
                await self._throttle()
                loop = asyncio.get_running_loop()

                def _call_nvidia() -> str:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        stream=False,
                    )
                    reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
                    content = completion.choices[0].message.content
                    if reasoning:
                        self.logger.debug(f"Reasoning: {reasoning[:200]}...")
                    return content

                response = await loop.run_in_executor(None, _call_nvidia)
                self.logger.info(
                    f"NVIDIA API响应成功，长度: {len(response) if response else 0}"
                )
                return response or ""
            except Exception as exc:
                error_msg = str(exc)
                is_connection_error = any(keyword in error_msg.lower() for keyword in
                    ['ssl', 'eof', 'connection', 'timeout', 'network'])

                if is_connection_error and attempt < max_retries - 1:
                    self.logger.warning(
                        f"NVIDIA API连接错误 (尝试 {attempt + 1}/{max_retries}): {exc}"
                    )
                    self.logger.info(f"等待 {retry_delay}s 后重试...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # 指数退避
                    continue
                else:
                    self.logger.error(f"NVIDIA API调用失败: {exc}")
                    import traceback
                    self.logger.error(f"详细错误: {traceback.format_exc()}")
                    return ""

        return ""


class LLMClient:
    def __init__(self, provider: LLMProvider, logger: Optional[logging.Logger] = None):
        if provider is None:
            raise ValueError("LLM provider is required")
        self.provider = provider
        self.logger = logger or logging.getLogger(__name__)

    async def generate(self, prompt: str) -> LLMResult:
        try:
            text = await self.provider.generate(prompt)
        except Exception as exc:
            self.logger.error(
                f"Provider {self.provider.name} failed with exception: {exc}"
            )
            text = ""
        return LLMResult(text=text or "", provider=self.provider.name, model=self.provider.model)
