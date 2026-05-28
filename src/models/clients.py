from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import httpx
from dotenv import load_dotenv

from src.utils.settings import ModelEndpointConfig

load_dotenv()


class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int = 1024) -> str:
        raise NotImplementedError


class BaseEmbeddingClient(ABC):
    @abstractmethod
    async def embed(self, inputs: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAICompatibleLLMClient(BaseLLMClient):
    def __init__(self, cfg: ModelEndpointConfig) -> None:
        self.cfg = cfg

    async def chat(self, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int = 1024) -> str:
        api_key = os.getenv(self.cfg.api_key_env, "EMPTY")
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(timeout=self.cfg.timeout_s) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]


class OpenAICompatibleEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, cfg: ModelEndpointConfig) -> None:
        self.cfg = cfg

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        api_key = os.getenv(self.cfg.api_key_env, "EMPTY")
        url = f"{self.cfg.base_url.rstrip('/')}/embeddings"
        payload = {"model": self.cfg.model, "input": inputs}
        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(timeout=self.cfg.timeout_s) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return [item["embedding"] for item in data["data"]]


def build_llm_client(cfg: ModelEndpointConfig) -> BaseLLMClient:
    if cfg.provider == "openai_compatible":
        return OpenAICompatibleLLMClient(cfg)
    raise ValueError(f"Unsupported LLM provider: {cfg.provider}")


def build_embedding_client(cfg: ModelEndpointConfig) -> BaseEmbeddingClient:
    if cfg.provider == "openai_compatible":
        return OpenAICompatibleEmbeddingClient(cfg)
    raise ValueError(f"Unsupported embedding provider: {cfg.provider}")
