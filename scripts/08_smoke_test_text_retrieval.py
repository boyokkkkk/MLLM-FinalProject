from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from src.serving.api import app
from src.serving.deps import get_text_retriever
from src.utils.settings import settings


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _post_with_retry(client: TestClient, path: str, payload: dict, attempts: int = 3) -> object:
    last_response = None
    for _ in range(attempts):
        response = client.post(path, json=payload)
        if response.status_code == 200:
            return response
        last_response = response
    return last_response


async def _smoke_retriever() -> None:
    retriever = get_text_retriever()
    evidences = await retriever.retrieve("6月4日前 B 负责什么任务？")
    _assert(bool(evidences), "retriever should return at least one evidence for B task query")
    top_evidence = evidences[0]
    _assert(bool(top_evidence.chunk_id), "top evidence should include chunk_id")
    _assert(bool(top_evidence.source), "top evidence should include source")
    _assert(top_evidence.page == 1, "mock markdown chunks should use page=1")
    print(f"[ok] retriever top hit: {top_evidence.chunk_id} from {top_evidence.source}")


def _smoke_chat_positive(client: TestClient) -> None:
    retrieval_cfg = settings.retrieval
    original_context_max_chars = retrieval_cfg.context_max_chars
    try:
        retrieval_cfg.context_max_chars = min(retrieval_cfg.context_max_chars, 1200)
        response = _post_with_retry(
            client,
            f"{settings.api_prefix}/chat",
            {"query": "6月4日前 B 负责什么任务？"},
        )
        _assert(
            response.status_code == 200,
            f"chat positive path should return 200, got {response.status_code}: {response.text}",
        )

        payload = response.json()
        _assert(bool(payload.get("answer", "").strip()), "chat positive path should return a non-empty answer")
        citations = payload.get("citations", [])
        _assert(bool(citations), "chat positive path should return citations")
        _assert(
            len(citations) == settings.retrieval.top_k_text,
            f"chat positive path should return top_k_text citations, got {len(citations)}",
        )
        first = citations[0]
        for field in ("chunk_id", "source", "snippet"):
            _assert(field in first and first[field], f"citation should include non-empty field: {field}")
        _assert("page" in first, "citation should include page field")
        print(f"[ok] chat positive path citations={len(citations)}")
    finally:
        retrieval_cfg.context_max_chars = original_context_max_chars


def _smoke_chat_fallback(client: TestClient) -> None:
    retrieval_cfg = settings.retrieval
    original_threshold = retrieval_cfg.score_threshold
    try:
        retrieval_cfg.score_threshold = 1.1
        response = _post_with_retry(
            client,
            f"{settings.api_prefix}/chat",
            {
                "query": "当检索没有命中时应该怎么办？",
                "context": ["调用方提供的备用上下文：如果证据不足，应明确说明并保留稳定响应。"],
            },
        )
        _assert(
            response.status_code == 200,
            f"chat fallback path should return 200, got {response.status_code}: {response.text}",
        )
        payload = response.json()
        citations = payload.get("citations", [])
        _assert(bool(citations), "chat fallback path should still return citations")
        _assert(citations[0]["source"] == "request_context", "fallback citation should come from request context")
        print("[ok] chat fallback path")
    finally:
        retrieval_cfg.score_threshold = original_threshold


def main() -> None:
    metadata_path = settings.retrieval.metadata_path
    index_path = settings.retrieval.index_path
    _assert(metadata_path.exists(), f"mock metadata file missing: {metadata_path}")
    _assert(index_path.exists(), f"mock vector file missing: {index_path}")

    asyncio.run(_smoke_retriever())

    with TestClient(app) as client:
        _smoke_chat_positive(client)
        _smoke_chat_fallback(client)

    print("[smoke] OK")


if __name__ == "__main__":
    main()
