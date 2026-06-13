from __future__ import annotations

import base64
import csv
import json
import mimetypes
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any
import re

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

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
from src.serving.workspaces import workspace_manager
from src.utils.settings import settings

app = FastAPI(title=settings.api_title, version=settings.api_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SHORT_ANSWER_RE = re.compile(r"^\s*(?:short answer|answer)\s*:\s*(.+)$", re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_STATIC_DIR = Path(__file__).resolve().parents[1] / "ui" / "web_static"
EVAL_ASSET_DIR = (PROJECT_ROOT / settings.output_root / "eval" / "final_assets").resolve()


def _should_enable_visual_assist(query: str, evidences: list[Evidence]) -> bool:
    value = query.lower().strip()
    if not value:
        return False

    enabled_markers = (
        "logo",
        "pack",
        "written on the pack",
        "written within the logo",
        "heading",
        "title",
        "page no",
        "page number",
        "handwritten",
    )
    disabled_markers = (
        "x axis",
        "y axis",
        "axis",
        "plot",
        "graph",
        "chart",
        "table",
        "how many",
        "how much",
        "amount",
        "total",
        "value",
        "during the year",
        "what does",
        "indicate",
        "why",
        "summary",
        "abstract",
    )

    if any(marker in value for marker in disabled_markers):
        return False
    if any(marker in value for marker in enabled_markers):
        return True

    # Short page-reading questions with top evidence from page/image regions are good candidates.
    if len(value.split()) <= 8 and any((e.chunk_type or "") in {"page_image", "figure"} for e in evidences[:2]):
        return True
    return False


def _build_fallback_evidence(context_items: list[str]) -> list[Evidence]:
    evidences: list[Evidence] = []
    for index, item in enumerate(context_items, start=1):
        source, text = _parse_request_context_item(item)
        if source.startswith("workspace_file:"):
            evidences.extend(_build_workspace_file_evidences(source, text, index))
            continue
        evidences.append(
            Evidence(
                chunk_id=f"request_context_{index:04d}",
                source=source,
                page=None,
                text=text,
                snippet=_summarize_snippet(text),
                score=1.0,
                section_title="Pasted context" if source.startswith("workspace_note:") else "Inline context",
                citation_kind="workspace",
            )
        )
    return evidences


def _parse_request_context_item(item: str) -> tuple[str, str]:
    text = item.strip()
    if text.startswith("[workspace_file:"):
        first_line, _, remainder = text.partition("\n")
        label = first_line.removeprefix("[workspace_file:").rstrip("]").strip() or "workspace_file"
        return f"workspace_file:{label}", remainder.strip() or text
    if text.startswith("[workspace_note]"):
        _, _, remainder = text.partition("\n")
        return "workspace_note:Pasted context", remainder.strip() or text
    return "workspace_context:Inline context", text


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_title = "Document overview"
    current_lines: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_title, body))
                current_lines = []
            heading = stripped.lstrip("#").strip()
            current_title = heading or "Untitled section"
            continue
        current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_title, body))

    if sections:
        return sections

    plain = text.strip()
    if not plain:
        return []
    return [("Document overview", plain)]


def _summarize_snippet(text: str, limit: int = 220) -> str:
    normalized = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _build_workspace_file_evidences(source: str, text: str, index: int) -> list[Evidence]:
    sections = _split_markdown_sections(text)
    evidences: list[Evidence] = []

    for section_index, (section_title, section_body) in enumerate(sections, start=1):
        evidences.append(
            Evidence(
                chunk_id=f"request_context_{index:04d}_section_{section_index:03d}",
                source=source,
                page=None,
                text=section_body,
                snippet=_summarize_snippet(section_body),
                score=1.0,
                section_title=section_title,
                citation_kind="workspace",
            )
        )
    return evidences


def _split_request_context(context_items: list[str]) -> tuple[str, list[str]]:
    scope = "global"
    filtered: list[str] = []

    for item in context_items:
        text = item.strip()
        if not text:
            continue
        if text.startswith("[ui_scope]"):
            if "workspace context first" in text.lower():
                scope = "workspace-first"
            elif "prioritize the pasted context" in text.lower():
                scope = "context-only"
            else:
                scope = "global"
            continue
        filtered.append(text)

    return scope, filtered


def _merge_evidences(
    workspace_evidences: list[Evidence],
    retrieved_evidences: list[Evidence],
    request_context_evidences: list[Evidence],
    scope: str,
) -> list[Evidence]:
    if scope == "context-only":
        if workspace_evidences:
            return workspace_evidences + request_context_evidences
        if request_context_evidences:
            return request_context_evidences
        return []
    if scope == "workspace-first":
        return workspace_evidences + request_context_evidences + retrieved_evidences
    if scope == "context-only" and request_context_evidences:
        return request_context_evidences
    if scope == "workspace-first" and request_context_evidences:
        return request_context_evidences + retrieved_evidences
    if workspace_evidences:
        return retrieved_evidences + workspace_evidences + request_context_evidences
    if request_context_evidences:
        return retrieved_evidences + request_context_evidences
    return retrieved_evidences


def _tokenize_local(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def _score_request_context_evidence(query: str, evidence: Evidence) -> float:
    query_terms = set(_tokenize_local(query))
    title_terms = set(_tokenize_local(evidence.section_title or ""))
    body_terms = set(_tokenize_local(evidence.text))
    overlap = len(query_terms & body_terms)
    title_overlap = len(query_terms & title_terms)
    return (title_overlap * 2.0) + overlap + min(len(body_terms), 40) * 0.001


def _select_citation_evidences(evidences: list[Evidence], scope: str, query: str) -> list[Evidence]:
    request_context_evidences = [item for item in evidences if item.citation_kind in {"workspace", "workspace_indexed"}]
    corpus_evidences = [item for item in evidences if item.citation_kind not in {"workspace", "workspace_indexed"}]
    request_context_evidences.sort(key=lambda item: _score_request_context_evidence(query, item), reverse=True)

    if scope == "context-only":
        return request_context_evidences[:5]
    if scope == "workspace-first" and request_context_evidences:
        return request_context_evidences[:5]
    return (request_context_evidences + corpus_evidences)[:5]


@lru_cache(maxsize=1)
def _load_visual_descriptor_map() -> dict[str, dict[str, Any]]:
    path = settings.retrieval.visual_dense_metadata_path
    if not path.exists():
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("chunk_id") or "").strip()
        if chunk_id:
            mapping[chunk_id] = row
    return mapping


def _resolve_local_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    candidate = (Path.cwd() / path).resolve()
    if candidate.exists():
        return candidate
    return None


def _image_file_to_data_url(path: Path) -> str | None:
    if not path.exists():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _crop_image_to_data_url(path: Path, bbox: list[int] | None) -> str | None:
    if not path.exists():
        return None
    try:
        with Image.open(path) as image:
            crop = image.convert("RGB")
            if bbox and len(bbox) == 4:
                left, top, right, bottom = [max(0, int(v)) for v in bbox]
                right = min(right, crop.width)
                bottom = min(bottom, crop.height)
                if right > left and bottom > top:
                    crop = crop.crop((left, top, right, bottom))
            buffer = BytesIO()
            crop.save(buffer, format="PNG")
    except Exception:
        return None
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _build_generation_visual_assist(query: str, evidences: list[Evidence]) -> tuple[list[str], list[str]]:
    cfg = settings.retrieval
    if not cfg.generation_visual_assist:
        return [], []
    if not _should_enable_visual_assist(query, evidences):
        return [], []

    descriptor_map = _load_visual_descriptor_map() if cfg.generation_visual_include_descriptors else {}
    image_urls: list[str] = []
    hints: list[str] = []
    seen_images: set[str] = set()

    for evidence in evidences[: max(1, cfg.generation_visual_top_n)]:
        if cfg.generation_visual_include_descriptors:
            descriptor = descriptor_map.get(evidence.chunk_id, {})
            text = " ".join(str(descriptor.get("text", "")).split()).strip()
            if text:
                hints.append(f"[{evidence.chunk_id}] {text}")

        if not cfg.generation_visual_include_images:
            continue
        image_path = _resolve_local_path(evidence.source_path) or _resolve_local_path(evidence.image_path)
        if not image_path:
            continue
        image_key = f"{image_path}::{evidence.bbox}"
        if image_key in seen_images:
            continue
        data_url = None
        if cfg.generation_visual_prefer_crops and evidence.bbox:
            data_url = _crop_image_to_data_url(image_path, evidence.bbox)
        if not data_url:
            data_url = _image_file_to_data_url(image_path)
        if not data_url:
            continue
        image_urls.append(data_url)
        seen_images.add(image_key)
    return image_urls, hints


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
            section_title=evidence.section_title,
            citation_kind=evidence.citation_kind,
            source_ref=evidence.source,
        )
        for evidence in evidences
    ]


def _clean_answer_text(text: str) -> str:
    value = text.strip()
    value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
    value = re.sub(r"__([^_]+)__", r"\1", value)
    value = value.strip(" \n\t`\"'")
    return value


def _postprocess_answer(answer: str) -> str:
    value = answer.strip()
    if not value:
        return ""

    lines = [line.rstrip() for line in value.splitlines()]
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        match = SHORT_ANSWER_RE.match(stripped)
        if match:
            cleaned_lines.append(_clean_answer_text(match.group(1)))
            continue
        if re.match(r"^\s*evidence\s*:\s*", stripped, re.IGNORECASE):
            stripped = re.sub(r"^\s*evidence\s*:\s*", "", stripped, flags=re.IGNORECASE)
        cleaned_lines.append(_clean_answer_text(stripped))

    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()
    return "\n".join(cleaned_lines).strip()


def _build_history_messages(history: list[Any], max_turns: int = 8) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in history[-max_turns:]:
        if isinstance(item, dict):
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
        else:
            role = str(getattr(item, "role", "")).strip()
            content = str(getattr(item, "content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _build_ui_bootstrap() -> dict[str, Any]:
    docs_path = (PROJECT_ROOT / settings.data_root / "processed" / "documents" / "documents.jsonl").resolve()
    chunks_path = settings.retrieval.metadata_path
    descriptor_path = settings.retrieval.visual_dense_metadata_path
    manifest = _load_json_file(EVAL_ASSET_DIR / "asset_manifest.json")
    main_results = _load_csv_rows(EVAL_ASSET_DIR / "table_closeout_main_results.csv")
    error_rows = _load_csv_rows(EVAL_ASSET_DIR / "table_appendix_error_by_type.csv")

    figure_urls = []
    for item in manifest.get("figures_and_tables", []):
        path = Path(item)
        if path.suffix.lower() == ".svg":
            figure_urls.append(f"/artifacts/{path.name}")

    return {
        "app": {
            "title": settings.api_title,
            "version": settings.api_version,
            "api_prefix": settings.api_prefix,
        },
        "services": {
            "health": {"method": "GET", "path": "/health"},
            "chat": {"method": "POST", "path": f"{settings.api_prefix}/chat"},
            "embed_text": {"method": "POST", "path": f"{settings.api_prefix}/embed/text"},
            "embed_vision": {"method": "POST", "path": f"{settings.api_prefix}/embed/vision"},
        },
        "models": {
            "vlm": {
                "provider": settings.vlm.provider,
                "model": settings.vlm.model,
                "base_url": settings.vlm.base_url,
            },
            "text_embedding": {
                "provider": settings.text_embedding.provider,
                "model": settings.text_embedding.model,
                "base_url": settings.text_embedding.base_url,
            },
            "vision_embedding": {
                "provider": settings.vision_embedding.provider,
                "model": settings.vision_embedding.model,
                "base_url": settings.vision_embedding.base_url,
            },
        },
        "retrieval": {
            "top_k_text": settings.retrieval.top_k_text,
            "rerank": settings.retrieval.rerank,
            "query_type_aware_rerank": settings.retrieval.query_type_aware_rerank,
            "visual_fusion": settings.retrieval.visual_fusion,
            "visual_dense_fusion": settings.retrieval.visual_dense_fusion,
            "query_image_aware_rerank": settings.retrieval.query_image_aware_rerank,
            "generation_visual_assist": settings.retrieval.generation_visual_assist,
            "context_max_chars": settings.retrieval.context_max_chars,
            "default_temperature": settings.retrieval.default_temperature,
            "default_max_tokens": settings.retrieval.default_max_tokens,
        },
        "corpus": {
            "documents": _count_jsonl_rows(docs_path),
            "chunks": _count_jsonl_rows(chunks_path),
            "visual_descriptors": _count_jsonl_rows(descriptor_path),
        },
        "evaluation": {
            "available": bool(manifest),
            "manifest_summary": manifest.get("manifest_summary", {}),
            "repair_stats": manifest.get("repair_stats", {}),
            "error_summary": manifest.get("current_error_summary", {}),
            "main_results": main_results,
            "error_breakdown": error_rows,
            "figure_urls": figure_urls,
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=settings.api_version)


@app.get(f"{settings.api_prefix}/ui/bootstrap")
async def ui_bootstrap() -> dict[str, Any]:
    return _build_ui_bootstrap()


@app.post(f"{settings.api_prefix}/workspaces")
async def create_workspace() -> dict[str, Any]:
    return workspace_manager.create_workspace()


@app.get(f"{settings.api_prefix}/workspaces/{{workspace_id}}")
async def get_workspace(workspace_id: str) -> dict[str, Any]:
    try:
        return workspace_manager.get_workspace(workspace_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post(f"{settings.api_prefix}/workspaces/{{workspace_id}}/assets")
async def upload_workspace_assets(workspace_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    try:
        payloads: list[tuple[str, bytes]] = []
        for file in files:
            payloads.append((file.filename or "asset", await file.read()))
        return workspace_manager.add_assets(workspace_id, payloads)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"workspace_upload_failed: {exc}") from exc


@app.delete(f"{settings.api_prefix}/workspaces/{{workspace_id}}/assets/{{asset_id}}")
async def delete_workspace_asset(workspace_id: str, asset_id: str) -> dict[str, Any]:
    try:
        return workspace_manager.delete_asset(workspace_id, asset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"workspace_delete_failed: {exc}") from exc


@app.post(f"{settings.api_prefix}/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    client=Depends(get_vlm_client),
    retriever: BaseTextRetriever = Depends(get_text_retriever),
) -> ChatResponse:
    retrieval_cfg = settings.retrieval
    request_scope, request_context_items = _split_request_context(req.context)
    workspace_evidences: list[Evidence] = []

    if req.workspace_id:
        try:
            workspace_evidences = workspace_manager.workspace_retrieve(
                req.workspace_id,
                req.query,
                top_k=retrieval_cfg.top_k_text,
            )
        except FileNotFoundError:
            workspace_evidences = []

    try:
        retrieved_evidences = []
        if request_scope != "context-only":
            retrieved_evidences = await retriever.retrieve(req.query, top_k=retrieval_cfg.top_k_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"retrieval_failed: {exc}") from exc

    request_context_evidences = _build_fallback_evidence(request_context_items)
    evidences = _merge_evidences(
        workspace_evidences=workspace_evidences,
        retrieved_evidences=retrieved_evidences,
        request_context_evidences=request_context_evidences,
        scope=request_scope,
    )

    if not evidences and retrieval_cfg.fallback_to_request_context and request_context_items:
        evidences = _build_fallback_evidence(request_context_items)

    system_prompt = (
        "You are a multimodal document QA assistant for a RAG application. "
        "Answer naturally and completely in the user's language, using the retrieved evidence as the grounding source. "
        "Treat workspace evidence as user-uploaded project material and follow the requested workspace priority mode. "
        "Use the prior conversation only to resolve references such as 'this result' or 'the previous figure'; if history conflicts with the current evidence, follow the current evidence. "
        "Synthesize the evidence into a coherent answer instead of copying raw retrieval fragments. "
        "If the evidence is insufficient, say so clearly and explain what is missing. "
        "Do not fabricate citations, page numbers, experimental details, or file contents."
    )
    evidence_block = _render_evidence_block(evidences, retrieval_cfg.context_max_chars)
    visual_assist_images, visual_assist_hints = _build_generation_visual_assist(req.query, evidences)
    visual_hint_block = ""
    if visual_assist_hints:
        visual_hint_block = "Retrieved Visual Hints:\n" + "\n".join(visual_assist_hints) + "\n\n"
    workspace_mode_note = (
        "Workspace mode: prioritize request_context evidence only.\n\n"
        if request_scope == "context-only"
        else "Workspace mode: prioritize request_context evidence first, then retrieved corpus evidence.\n\n"
        if request_scope == "workspace-first"
        else "Workspace mode: prioritize retrieved corpus evidence, but you may also use request_context when relevant.\n\n"
    )
    user_text = (
        f"Question:\n{req.query}\n\n"
        f"{workspace_mode_note}"
        f"Retrieved Evidence:\n{evidence_block}\n\n"
        f"{visual_hint_block}"
        "Write a polished answer grounded in the evidence above. "
        "For summary or analysis requests, provide a short conclusion first and then the key supporting points in complete sentences."
    )

    if req.image_data_urls or visual_assist_images:
        user_content: list[dict[str, object]] = [{"type": "text", "text": user_text}]
        for data_url in req.image_data_urls:
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})
        for data_url in visual_assist_images:
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})
        user_message: dict[str, Any] = {"role": "user", "content": user_content}
    else:
        user_message = {"role": "user", "content": user_text}

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_build_history_messages(req.history))
    messages.append(user_message)

    temperature = req.temperature if req.temperature is not None else retrieval_cfg.default_temperature
    max_tokens = req.max_tokens if req.max_tokens is not None else retrieval_cfg.default_max_tokens

    try:
        answer = await client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        answer = _postprocess_answer(answer)
        citations = _build_citations(_select_citation_evidences(evidences, request_scope, req.query))
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


if EVAL_ASSET_DIR.exists():
    app.mount("/artifacts", StaticFiles(directory=str(EVAL_ASSET_DIR)), name="artifacts")

if WEB_STATIC_DIR.exists():
    # Mount the JS frontend last so API routes remain authoritative.
    app.mount("/", StaticFiles(directory=str(WEB_STATIC_DIR), html=True), name="frontend")
