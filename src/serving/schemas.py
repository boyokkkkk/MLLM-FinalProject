from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, description="User natural-language question.")
    context: list[str] = Field(default_factory=list, description="Optional retrieved evidences.")
    image_data_urls: list[str] = Field(
        default_factory=list,
        description="Optional image data URLs for multimodal chat, e.g. data:image/png;base64,...",
    )
    temperature: float = 0.2
    max_tokens: int = 512


class Citation(BaseModel):
    source_ref: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    model: str


class EmbeddingRequest(BaseModel):
    inputs: list[str] = Field(min_length=1)


class EmbeddingResponse(BaseModel):
    model: str
    vectors: list[list[float]]


class ErrorResponse(BaseModel):
    detail: str
