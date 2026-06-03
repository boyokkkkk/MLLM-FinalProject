from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, description="User natural-language question.")
    context: list[str] = Field(
        default_factory=list,
        description="Optional fallback or debug context supplied by the caller.",
    )
    image_data_urls: list[str] = Field(
        default_factory=list,
        description="Optional image data URLs for multimodal chat, e.g. data:image/png;base64,...",
    )
    temperature: float | None = Field(
        default=None,
        description="Optional override for generation temperature; uses configured default when omitted.",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Optional override for generation max tokens; uses configured default when omitted.",
    )


class Citation(BaseModel):
    chunk_id: str
    source: str
    page: int | None = None
    snippet: str
    source_ref: str | None = Field(
        default=None,
        description="Legacy compatibility field; prefer `source` for new clients.",
    )


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
