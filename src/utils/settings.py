from __future__ import annotations

from functools import cached_property
import os
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
    rerank: bool = True
    query_type_aware_rerank: bool = True
    rerank_profile: str = "stronger"
    rerank_pool_size: int = 20
    diversify_results: bool = False
    fingerprint_duplicate_penalty: float = 0.10
    docpage_duplicate_penalty: float = 0.08
    same_sample_penalty: float = 0.04
    dense_rerank: bool = False
    dense_rerank_pool_size: int = 12
    dense_score_weight: float = 0.35
    index_path: Path = Field(default=Path("data/processed/retrieval/text_vectors.npy"))
    metadata_path: Path = Field(default=Path("data/processed/retrieval/text_chunks.jsonl"))
    sparse_index_path: Path = Field(default=Path("data/processed/indexes/text/doc_store.json"))
    visual_index_path: Path = Field(default=Path("data/processed/indexes/vision/visual_store.json"))
    visual_fusion: bool = False
    visual_pool_size: int = 16
    visual_fusion_weight: float = 0.45
    visual_dense_metadata_path: Path = Field(default=Path("data/processed/indexes/vision/visual_descriptor_store.jsonl"))
    visual_dense_vectors_path: Path = Field(default=Path("data/processed/indexes/vision/visual_descriptor_vectors.json"))
    visual_dense_fusion: bool = False
    visual_dense_pool_size: int = 24
    visual_dense_weight: float = 0.45
    text_fusion_weight: float = 1.0
    fusion_k: int = 60
    chart_table_specialist: bool = False
    chart_table_visual_boost: float = 0.18
    query_image_aware_rerank: bool = False
    query_image_pool_size: int = 20
    query_image_weight: float = 0.35
    generation_visual_assist: bool = False
    generation_visual_top_n: int = 2
    generation_visual_include_descriptors: bool = True
    generation_visual_include_images: bool = True
    generation_visual_prefer_crops: bool = True
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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return Path(value)


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
        config = config.model_copy(
            update={
                "rerank": _env_bool("RETRIEVAL_RERANK", config.rerank),
                "query_type_aware_rerank": _env_bool("RETRIEVAL_QUERY_TYPE_AWARE_RERANK", config.query_type_aware_rerank),
                "rerank_profile": _env_str("RETRIEVAL_RERANK_PROFILE", config.rerank_profile),
                "rerank_pool_size": _env_int("RETRIEVAL_RERANK_POOL_SIZE", config.rerank_pool_size),
                "diversify_results": _env_bool("RETRIEVAL_DIVERSIFY_RESULTS", config.diversify_results),
                "fingerprint_duplicate_penalty": _env_float("RETRIEVAL_FINGERPRINT_DUPLICATE_PENALTY", config.fingerprint_duplicate_penalty),
                "docpage_duplicate_penalty": _env_float("RETRIEVAL_DOCPAGE_DUPLICATE_PENALTY", config.docpage_duplicate_penalty),
                "same_sample_penalty": _env_float("RETRIEVAL_SAME_SAMPLE_PENALTY", config.same_sample_penalty),
                "dense_rerank": _env_bool("RETRIEVAL_DENSE_RERANK", config.dense_rerank),
                "dense_rerank_pool_size": _env_int("RETRIEVAL_DENSE_RERANK_POOL_SIZE", config.dense_rerank_pool_size),
                "dense_score_weight": _env_float("RETRIEVAL_DENSE_SCORE_WEIGHT", config.dense_score_weight),
                "index_path": _env_path("RETRIEVAL_INDEX_PATH", config.index_path),
                "metadata_path": _env_path("RETRIEVAL_METADATA_PATH", config.metadata_path),
                "sparse_index_path": _env_path("RETRIEVAL_SPARSE_INDEX_PATH", config.sparse_index_path),
                "visual_index_path": _env_path("RETRIEVAL_VISUAL_INDEX_PATH", config.visual_index_path),
                "visual_fusion": _env_bool("RETRIEVAL_VISUAL_FUSION", config.visual_fusion),
                "visual_pool_size": _env_int("RETRIEVAL_VISUAL_POOL_SIZE", config.visual_pool_size),
                "visual_fusion_weight": _env_float("RETRIEVAL_VISUAL_FUSION_WEIGHT", config.visual_fusion_weight),
                "visual_dense_metadata_path": _env_path("RETRIEVAL_VISUAL_DENSE_METADATA_PATH", config.visual_dense_metadata_path),
                "visual_dense_vectors_path": _env_path("RETRIEVAL_VISUAL_DENSE_VECTORS_PATH", config.visual_dense_vectors_path),
                "visual_dense_fusion": _env_bool("RETRIEVAL_VISUAL_DENSE_FUSION", config.visual_dense_fusion),
                "visual_dense_pool_size": _env_int("RETRIEVAL_VISUAL_DENSE_POOL_SIZE", config.visual_dense_pool_size),
                "visual_dense_weight": _env_float("RETRIEVAL_VISUAL_DENSE_WEIGHT", config.visual_dense_weight),
                "text_fusion_weight": _env_float("RETRIEVAL_TEXT_FUSION_WEIGHT", config.text_fusion_weight),
                "fusion_k": _env_int("RETRIEVAL_FUSION_K", config.fusion_k),
                "chart_table_specialist": _env_bool("RETRIEVAL_CHART_TABLE_SPECIALIST", config.chart_table_specialist),
                "chart_table_visual_boost": _env_float("RETRIEVAL_CHART_TABLE_VISUAL_BOOST", config.chart_table_visual_boost),
                "query_image_aware_rerank": _env_bool("RETRIEVAL_QUERY_IMAGE_AWARE_RERANK", config.query_image_aware_rerank),
                "query_image_pool_size": _env_int("RETRIEVAL_QUERY_IMAGE_POOL_SIZE", config.query_image_pool_size),
                "query_image_weight": _env_float("RETRIEVAL_QUERY_IMAGE_WEIGHT", config.query_image_weight),
                "generation_visual_assist": _env_bool("RETRIEVAL_GENERATION_VISUAL_ASSIST", config.generation_visual_assist),
                "generation_visual_top_n": _env_int("RETRIEVAL_GENERATION_VISUAL_TOP_N", config.generation_visual_top_n),
                "generation_visual_include_descriptors": _env_bool("RETRIEVAL_GENERATION_VISUAL_INCLUDE_DESCRIPTORS", config.generation_visual_include_descriptors),
                "generation_visual_include_images": _env_bool("RETRIEVAL_GENERATION_VISUAL_INCLUDE_IMAGES", config.generation_visual_include_images),
                "generation_visual_prefer_crops": _env_bool("RETRIEVAL_GENERATION_VISUAL_PREFER_CROPS", config.generation_visual_prefer_crops),
            }
        )
        return config.model_copy(
            update={
                "index_path": _resolve_project_path(config.index_path),
                "metadata_path": _resolve_project_path(config.metadata_path),
                "sparse_index_path": _resolve_project_path(config.sparse_index_path),
                "visual_index_path": _resolve_project_path(config.visual_index_path),
                "visual_dense_metadata_path": _resolve_project_path(config.visual_dense_metadata_path),
                "visual_dense_vectors_path": _resolve_project_path(config.visual_dense_vectors_path),
            }
        )


settings = AppSettings()
