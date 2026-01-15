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
        model: str = "deepseek-ai/deepseek-r1",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        rate_limit_per_minute: int = 30,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        top_p: float = 0.7,
        timeout: float = 300.0,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__("nvidia", model, logger)
        if not api_key:
            raise ValueError("NVIDIA API key is required")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
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
        try:
            self.logger.info(f"开始调用NVIDIA API，模型: {self.model}")
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
            self.logger.error(f"NVIDIA API调用异常: {exc}")
            import traceback

            self.logger.error(f"详细错误: {traceback.format_exc()}")
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
