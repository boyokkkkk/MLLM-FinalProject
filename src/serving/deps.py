from __future__ import annotations

from functools import lru_cache

from src.models.clients import BaseEmbeddingClient, BaseLLMClient, build_embedding_client, build_llm_client
from src.utils.settings import settings


@lru_cache(maxsize=1)
def get_vlm_client() -> BaseLLMClient:
    return build_llm_client(settings.vlm)


@lru_cache(maxsize=1)
def get_text_embedding_client() -> BaseEmbeddingClient:
    return build_embedding_client(settings.text_embedding)


@lru_cache(maxsize=1)
def get_vision_embedding_client() -> BaseEmbeddingClient:
    return build_embedding_client(settings.vision_embedding)
