from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.models.retrieval import BaseTextRetriever, Evidence
from src.serving.deps import (
    get_text_embedding_client,
    get_text_retriever,
    get_vision_embedding_client,
    get_vlm_client,
)
from src.serving.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
)
from src.utils.settings import settings

app = FastAPI(title=settings.api_title, version=settings.api_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_fallback_evidence(context_items: list[str]) -> list[Evidence]:
    evidences: list[Evidence] = []
    for index, item in enumerate(context_items, start=1):
        text = item.strip()
        evidences.append(
            Evidence(
                chunk_id=f"request_context_{index:04d}",
                source="request_context",
                page=None,
                text=text,
                snippet=text.replace("\n", " ")[:240],
                score=1.0,
            )
        )
    return evidences


def _render_evidence_block(evidences: list[Evidence], context_max_chars: int) -> str:
    if not evidences:
        return "No retrieved evidence available."

    blocks: list[str] = []
    total_chars = 0
    for evidence in evidences:
        page_text = evidence.page if evidence.page is not None else "unknown"
        block = (
            f"[chunk_id={evidence.chunk_id} | source={evidence.source} | page={page_text}]\n"
            f"{evidence.text.strip()}"
        )
        projected_length = total_chars + len(block) + (2 if blocks else 0)
        if blocks and projected_length > context_max_chars:
            break
        if not blocks and len(block) > context_max_chars:
            block = block[:context_max_chars]
        blocks.append(block)
        total_chars += len(block) + (2 if len(blocks) > 1 else 0)
    return "\n\n".join(blocks)


def _build_citations(evidences: list[Evidence]) -> list[Citation]:
    return [
        Citation(
            chunk_id=evidence.chunk_id,
            source=evidence.source,
            page=evidence.page,
            snippet=evidence.snippet,
            source_ref=evidence.source,
        )
        for evidence in evidences
    ]


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=settings.api_version)


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    client=Depends(get_vlm_client),
    retriever: BaseTextRetriever = Depends(get_text_retriever),
) -> ChatResponse:
    retrieval_cfg = settings.retrieval
    try:
        retrieved_evidences = await retriever.retrieve(req.query, top_k=retrieval_cfg.top_k_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"retrieval_failed: {exc}") from exc

    evidences = retrieved_evidences
    if not evidences and retrieval_cfg.fallback_to_request_context and req.context:
        evidences = _build_fallback_evidence(req.context)

    system_prompt = (
        "You are a multimodal document QA assistant. Prioritize the retrieved evidence when answering. "
        "If the evidence is insufficient, say that clearly. Do not fabricate citations, page numbers, or sources."
    )
    evidence_block = _render_evidence_block(evidences, retrieval_cfg.context_max_chars)
    user_text = (
        f"Question:\n{req.query}\n\n"
        f"Retrieved Evidence:\n{evidence_block}\n\n"
        "Answer concisely and ground the answer in the evidence above."
    )

    if req.image_data_urls:
        user_content: list[dict[str, object]] = [{"type": "text", "text": user_text}]
        for data_url in req.image_data_urls:
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})
        user_message: dict[str, Any] = {"role": "user", "content": user_content}
    else:
        user_message = {"role": "user", "content": user_text}

    messages = [
        {"role": "system", "content": system_prompt},
        user_message,
    ]

    temperature = req.temperature if req.temperature is not None else retrieval_cfg.default_temperature
    max_tokens = req.max_tokens if req.max_tokens is not None else retrieval_cfg.default_max_tokens

    try:
        answer = await client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        citations = _build_citations(evidences)
        return ChatResponse(answer=answer, citations=citations, model=settings.vlm.model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"chat_failed: {exc}") from exc


@app.post(f"{settings.api_prefix}/embed/text", response_model=EmbeddingResponse)
async def embed_text(
    req: EmbeddingRequest,
    client=Depends(get_text_embedding_client),
) -> EmbeddingResponse:
    try:
        vectors = await client.embed(req.inputs)
        return EmbeddingResponse(model=settings.text_embedding.model, vectors=vectors)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"text_embedding_failed: {exc}") from exc


@app.post(f"{settings.api_prefix}/embed/vision", response_model=EmbeddingResponse)
async def embed_vision(
    req: EmbeddingRequest,
    client=Depends(get_vision_embedding_client),
) -> EmbeddingResponse:
    try:
        vectors = await client.embed(req.inputs)
        return EmbeddingResponse(model=settings.vision_embedding.model, vectors=vectors)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"vision_embedding_failed: {exc}") from exc
