from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.serving.deps import get_text_embedding_client, get_vision_embedding_client, get_vlm_client
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


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=settings.api_version)


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    client=Depends(get_vlm_client),
) -> ChatResponse:
    try:
        system_prompt = (
            "You are a multimodal document QA assistant. Answer strictly based on provided context when possible. "
            "If context is insufficient, say so clearly."
        )
        context_block = "\n\n".join(req.context) if req.context else "No external context provided."
        user_text = f"Question:\n{req.query}\n\nContext:\n{context_block}\n\nReturn concise answer."
        if req.image_data_urls:
            user_content: list[dict[str, object]] = [{"type": "text", "text": user_text}]
            for data_url in req.image_data_urls:
                user_content.append({"type": "image_url", "image_url": {"url": data_url}})
            user_message = {"role": "user", "content": user_content}
        else:
            user_message = {"role": "user", "content": user_text}

        messages = [
            {"role": "system", "content": system_prompt},
            user_message,
        ]
        answer = await client.chat(messages, temperature=req.temperature, max_tokens=req.max_tokens)
        citations = [Citation(source_ref=f"ctx-{i+1}", snippet=item[:120]) for i, item in enumerate(req.context)]
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
