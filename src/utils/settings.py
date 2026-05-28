from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelEndpointConfig(BaseModel):
    provider: str = Field(default="openai_compatible")
    model: str
    base_url: str
    api_key_env: str = Field(default="OPENAI_API_KEY")
    timeout_s: int = 60


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


settings = AppSettings()
