from __future__ import annotations

from functools import cached_property
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_CONFIG_FILE = PROJECT_ROOT / "configs" / "retrieval.yaml"


class ModelEndpointConfig(BaseModel):
    provider: str = Field(default="openai_compatible")
    model: str
    base_url: str
    api_key_env: str = Field(default="OPENAI_API_KEY")
    timeout_s: int = 60


class RetrievalConfig(BaseModel):
    enable_text_retrieval: bool = True
    top_k_text: int = 5
    top_k_vision: int = 5
    score_threshold: float = 0.0
    rerank: bool = False
    index_path: Path = Field(default=Path("data/processed/retrieval/text_vectors.npy"))
    metadata_path: Path = Field(default=Path("data/processed/retrieval/text_chunks.jsonl"))
    context_max_chars: int = 4000
    fallback_to_request_context: bool = True
    default_temperature: float = 0.2
    default_max_tokens: int = 512


def _resolve_project_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return data


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_title: str = "Multimodal Doc RAG API"
    api_version: str = "0.1.0"
    api_prefix: str = "/api/v1"

    host: str = "0.0.0.0"
    port: int = 8000

    vlm_provider: str = "openai_compatible"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_base_url: str = "http://localhost:8001/v1"
    vlm_api_key_env: str = "OPENAI_API_KEY"
    vlm_timeout_s: int = 120

    text_emb_provider: str = "openai_compatible"
    text_emb_model: str = "Qwen/Qwen3-Embedding-4B"
    text_emb_base_url: str = "http://localhost:8001/v1"
    text_emb_api_key_env: str = "OPENAI_API_KEY"
    text_emb_timeout_s: int = 60

    vision_emb_provider: str = "openai_compatible"
    vision_emb_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vision_emb_base_url: str = "http://localhost:8001/v1"
    vision_emb_api_key_env: str = "OPENAI_API_KEY"
    vision_emb_timeout_s: int = 60

    data_root: Path = Path("data")
    output_root: Path = Path("outputs")

    @property
    def vlm(self) -> ModelEndpointConfig:
        return ModelEndpointConfig(
            provider=self.vlm_provider,
            model=self.vlm_model,
            base_url=self.vlm_base_url,
            api_key_env=self.vlm_api_key_env,
            timeout_s=self.vlm_timeout_s,
        )

    @property
    def text_embedding(self) -> ModelEndpointConfig:
        return ModelEndpointConfig(
            provider=self.text_emb_provider,
            model=self.text_emb_model,
            base_url=self.text_emb_base_url,
            api_key_env=self.text_emb_api_key_env,
            timeout_s=self.text_emb_timeout_s,
        )

    @property
    def vision_embedding(self) -> ModelEndpointConfig:
        return ModelEndpointConfig(
            provider=self.vision_emb_provider,
            model=self.vision_emb_model,
            base_url=self.vision_emb_base_url,
            api_key_env=self.vision_emb_api_key_env,
            timeout_s=self.vision_emb_timeout_s,
        )

    @cached_property
    def retrieval(self) -> RetrievalConfig:
        raw_config = _load_yaml(RETRIEVAL_CONFIG_FILE).get("retrieval", {})
        config = RetrievalConfig(**raw_config)
        return config.model_copy(
            update={
                "index_path": _resolve_project_path(config.index_path),
                "metadata_path": _resolve_project_path(config.metadata_path),
            }
        )


settings = AppSettings()
