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
SOURCE_FIGURE_RE = re.compile(r"(?:^|#)figure=(?P<figure_id>[^#]+)")
SOURCE_BLOCK_RE = re.compile(r"(?:^|#)block=(?P<block_id>[^#]+)")
TRAILING_NUMBER_RE = re.compile(r"(\d+)$")
EXHAUSTIVE_QUERY_MARKERS = (
    "all questions",
    "all the questions",
    "every question",
    "solve all",
    "answer all",
    "图中所有题目",
    "图片中的所有题目",
    "回答图中的所有题目",
    "回答图片中的所有题目",
    "这张图里的所有题目",
    "所有题目",
)
FOLLOWUP_QUERY_MARKERS = (
    "继续",
    "第二题",
    "第三题",
    "第四题",
    "上一题",
    "下一题",
    "这个",
    "这个呢",
    "那这个",
    "这个怎么",
    "这个如何",
    "上面的",
    "后面的",
    "前面的",
    "再说",
    "展开",
    "补充",
    "then what",
    "what about",
    "the second",
    "the third",
    "next one",
    "continue",
    "go on",
    "this one",
    "that one",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_STATIC_DIR = Path(__file__).resolve().parents[1] / "ui" / "web_static"
EVAL_ASSET_DIR = (PROJECT_ROOT / settings.output_root / "eval" / "final_assets").resolve()
TRANSPARENT_PIXEL_DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAJElEQVR4nGP8//8/AyWAiSLdowaAARMDhYBp1ACGUQMYKDcAAGOaAx2f8LHwAAAAAElFTkSuQmCC"


def _normalize_visual_assist_policy(policy: str | None) -> str:
    value = (policy or "").strip().lower()
    if value in {"", "gated", "heuristic_gated", "gated_logo_pack_heading_title_page_handwritten_only"}:
        return "gated_logo_pack_heading_title_page_handwritten_only"
    if value in {"logo_only", "logo-only"}:
        return "logo_only"
    if value in {"title_page_only", "title-page-only", "title_or_page_only"}:
        return "title_page_only"
    if value in {"handwritten_only", "handwritten-only"}:
        return "handwritten_only"
    if value in {"strict_visual_gated", "strict-visual-gated"}:
        return "strict_visual_gated"
    if value in {"always", "all", "force_on"}:
        return "always"
    if value in {"off", "disabled", "none"}:
        return "off"
    return value


def _should_enable_visual_assist(query: str, evidences: list[Evidence], policy: str | None = None) -> bool:
    normalized_policy = _normalize_visual_assist_policy(policy)
    if normalized_policy == "off":
        return False
    if normalized_policy == "always":
        return True

    value = query.lower().strip()
    if not value:
        return False

    logo_markers = (
        "logo",
        "pack",
        "written on the pack",
        "written within the logo",
        "brand",
    )
    title_page_markers = (
        "heading",
        "title",
        "page no",
        "page number",
        "tagline",
        "subject",
    )
    handwritten_markers = ("handwritten",)

    enabled_markers = (
        *logo_markers,
        *title_page_markers,
        *handwritten_markers,
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

    if normalized_policy == "logo_only":
        return any(marker in value for marker in logo_markers)
    if normalized_policy == "title_page_only":
        return any(marker in value for marker in title_page_markers)
    if normalized_policy == "handwritten_only":
        return any(marker in value for marker in handwritten_markers)
    if normalized_policy == "strict_visual_gated":
        strict_markers = (*logo_markers, *title_page_markers, *handwritten_markers)
        if any(marker in value for marker in disabled_markers):
            return False
        return any(marker in value for marker in strict_markers)

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


def _build_uploaded_image_evidences(image_data_urls: list[str]) -> list[Evidence]:
    evidences: list[Evidence] = []
    for index, data_url in enumerate(image_data_urls, start=1):
        evidences.append(
            Evidence(
                chunk_id=f"request_image_{index:04d}",
                source=f"temporary_image:{index}",
                page=None,
                text="Temporary uploaded image provided by the user for this turn.",
                snippet="Temporary uploaded image provided by the user for this turn.",
                score=1.0,
                section_title=f"Temporary image {index}",
                citation_kind="workspace",
                chunk_type="temporary_image",
                inline_image_data_url=data_url,
            )
        )
    return evidences


def _needs_vl_placeholder_image(model_name: str, image_data_urls: list[str], visual_assist_images: list[str]) -> bool:
    value = (model_name or "").lower()
    if "vl" not in value:
        return False
    return not image_data_urls and not visual_assist_images


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


def _is_exhaustive_question_request(query: str) -> bool:
    value = (query or "").strip().lower()
    if not value:
        return False
    return any(marker in value for marker in EXHAUSTIVE_QUERY_MARKERS)


def _is_workspace_visual_question(query: str, image_data_urls: list[str], workspace_evidences: list[Evidence]) -> bool:
    value = (query or "").strip().lower()
    if image_data_urls:
        return True
    if any(marker in value for marker in ("图", "图片", "图中", "这张图", "题目", "question in the image", "in the image")):
        return True
    return any((item.source or "").startswith("workspace_file:") and (item.source or "").lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".pdf")) for item in workspace_evidences)


def _should_prefer_uploaded_images_only(
    query: str,
    image_data_urls: list[str],
    workspace_evidences: list[Evidence],
) -> bool:
    if not image_data_urls:
        return False
    if workspace_evidences:
        return False
    return _is_workspace_visual_question(query, image_data_urls, workspace_evidences)


def _is_followup_query(query: str) -> bool:
    value = (query or "").strip().lower()
    if not value:
        return False
    if len(value) <= 12:
        return True
    return any(marker in value for marker in FOLLOWUP_QUERY_MARKERS)


def _build_retrieval_query(query: str, history: list[Any], max_turns: int = 3) -> str:
    current = (query or "").strip()
    if not current:
        return ""
    if not _is_followup_query(current):
        return current

    recent_user_messages: list[str] = []
    for item in history[-max_turns:]:
        if isinstance(item, dict):
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
        else:
            role = str(getattr(item, "role", "")).strip()
            content = str(getattr(item, "content", "")).strip()
        if role == "user" and content:
            recent_user_messages.append(content)

    if not recent_user_messages:
        return current
    anchor = recent_user_messages[-1]
    if anchor == current:
        return current
    return f"{anchor}\nFollow-up request: {current}"


def _should_prefer_workspace_only(
    scope: str,
    query: str,
    image_data_urls: list[str],
    workspace_evidences: list[Evidence],
) -> bool:
    if scope != "workspace-first":
        return False
    if not workspace_evidences:
        return False
    return (
        _is_workspace_visual_question(query, image_data_urls, workspace_evidences)
        or _is_exhaustive_question_request(query)
        or _is_followup_query(query)
    )


def _merge_evidences(
    workspace_evidences: list[Evidence],
    retrieved_evidences: list[Evidence],
    request_context_evidences: list[Evidence],
    scope: str,
    prefer_workspace_only: bool = False,
) -> list[Evidence]:
    if scope == "context-only":
        if workspace_evidences:
            return workspace_evidences + request_context_evidences
        if request_context_evidences:
            return request_context_evidences
        return []
    if scope == "workspace-first":
        if prefer_workspace_only and workspace_evidences:
            return workspace_evidences + request_context_evidences
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
    retrieval_score = float(evidence.score or 0.0)
    body_bonus = min(len(body_terms), 40) * 0.001
    return (title_overlap * 2.0) + overlap + body_bonus + (retrieval_score * 0.75)


def _select_citation_evidences(evidences: list[Evidence], scope: str, query: str) -> list[Evidence]:
    request_context_evidences = [item for item in evidences if item.citation_kind in {"workspace", "workspace_indexed"}]
    corpus_evidences = [item for item in evidences if item.citation_kind not in {"workspace", "workspace_indexed"}]
    request_context_evidences.sort(key=lambda item: _score_request_context_evidence(query, item), reverse=True)
    citation_limit = 8 if _is_exhaustive_question_request(query) else 5

    if scope == "context-only":
        return request_context_evidences[:citation_limit]
    if scope == "workspace-first" and request_context_evidences:
        return request_context_evidences[:citation_limit]
    return (request_context_evidences + corpus_evidences)[:citation_limit]


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
    if not _should_enable_visual_assist(query, evidences, cfg.generation_visual_assist_policy):
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
    citations: list[Citation] = []
    for evidence in evidences:
        figure_id, figure_no, block_id = _extract_source_locator_fields(evidence.source)
        image_data_url = evidence.inline_image_data_url
        if not image_data_url:
            image_path = _resolve_local_path(evidence.source_path) or _resolve_local_path(evidence.image_path)
            if image_path:
                if evidence.bbox:
                    image_data_url = _crop_image_to_data_url(image_path, evidence.bbox)
                if not image_data_url:
                    image_data_url = _image_file_to_data_url(image_path)
        citations.append(
            Citation(
                chunk_id=evidence.chunk_id,
                source=evidence.source,
                page=evidence.page,
                figure_id=figure_id,
                figure_no=figure_no,
                block_id=block_id,
                snippet=evidence.snippet,
                section_title=evidence.section_title,
                citation_kind=evidence.citation_kind,
                chunk_type=evidence.chunk_type,
                score=evidence.score,
                image_data_url=image_data_url,
                source_ref=evidence.source,
            )
        )
    return citations


def _extract_source_locator_fields(source: str | None) -> tuple[str | None, str | None, str | None]:
    value = (source or "").strip()
    if not value:
        return None, None, None

    figure_match = SOURCE_FIGURE_RE.search(value)
    figure_id = figure_match.group("figure_id").strip() if figure_match else None
    block_match = SOURCE_BLOCK_RE.search(value)
    block_id = block_match.group("block_id").strip() if block_match else None
    figure_no = _extract_figure_no(figure_id)
    return figure_id, figure_no, block_id


def _extract_figure_no(figure_id: str | None) -> str | None:
    if not figure_id:
        return None
    match = TRAILING_NUMBER_RE.search(figure_id)
    if match:
        return match.group(1)
    return figure_id


def _format_citation_locator(citation: Citation) -> str:
    parts: list[str] = []
    if citation.page is not None:
        parts.append(f"p.{citation.page}")
    if citation.figure_id:
        parts.append(citation.figure_id)
    elif citation.figure_no:
        parts.append(f"fig-{citation.figure_no}")
    return ", ".join(parts)


def _append_inline_citation_summary(answer: str, citations: list[Citation]) -> str:
    body = answer.strip()
    if not body or not citations:
        return body

    labels: list[str] = []
    for index, citation in enumerate(citations, start=1):
        locator = _format_citation_locator(citation)
        if not locator:
            continue
        labels.append(f"[{index}] [{locator}]")
        if len(labels) >= 3:
            break

    if not labels:
        return body

    summary = "Citations: " + "; ".join(labels)
    if summary in body:
        return body
    return f"{body}\n\n{summary}"


def _clean_answer_text(text: str) -> str:
    return text.strip(" \n\t")


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
        cleaned_lines.append(_clean_answer_text(line))

    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()
    return "\n".join(cleaned_lines).strip()


def _has_unbalanced_delimiters(text: str) -> bool:
    pairs = {
        "(": ")",
        "[": "]",
        "{": "}",
    }
    closing = {value: key for key, value in pairs.items()}
    stack: list[str] = []
    for char in text:
        if char in pairs:
            stack.append(char)
        elif char in closing:
            if stack and stack[-1] == closing[char]:
                stack.pop()
    return bool(stack)


def _has_unbalanced_inline_math(text: str) -> bool:
    dollar_count = 0
    index = 0
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == "$":
            dollar_count += 1
        index += 1
    return dollar_count % 2 == 1


def _looks_truncated_answer(answer: str) -> bool:
    value = answer.strip()
    if len(value) < 80:
        return False
    if _has_unbalanced_inline_math(value):
        return True
    if _has_unbalanced_delimiters(value):
        return True
    if value.endswith(("设", "令", "则", "所以", "因此", "where", "let", "and", "or", "with", "for")):
        return True
    if re.search(r"[\w\u4e00-\u9fff]\s*$", value) and not re.search(r"[。！？.!?:：）\]】$]", value):
        last_line = value.splitlines()[-1].strip()
        if len(last_line) >= 12:
            return True
    return False


def _build_history_messages(history: list[Any], max_turns: int = 8) -> list[dict[str, str]]:
    normalized: list[dict[str, Any]] = []
    for item in history[-max_turns:]:
        if isinstance(item, dict):
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
        else:
            role = str(getattr(item, "role", "")).strip()
            content = str(getattr(item, "content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": [{"type": "text", "text": content}]})
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
            "generation_visual_assist_policy": settings.retrieval.generation_visual_assist_policy,
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
    retrieval_query = _build_retrieval_query(req.query, req.history)
    workspace_evidences: list[Evidence] = []
    uploaded_image_evidences = _build_uploaded_image_evidences(req.image_data_urls)

    if req.workspace_id:
        try:
            workspace_evidences = workspace_manager.workspace_retrieve(
                req.workspace_id,
                retrieval_query,
                top_k=max(retrieval_cfg.top_k_text, 8 if _is_exhaustive_question_request(req.query) else retrieval_cfg.top_k_text),
            )
        except FileNotFoundError:
            workspace_evidences = []

    prefer_workspace_only = _should_prefer_workspace_only(
        scope=request_scope,
        query=req.query,
        image_data_urls=req.image_data_urls,
        workspace_evidences=workspace_evidences,
    )
    prefer_uploaded_images_only = _should_prefer_uploaded_images_only(
        query=req.query,
        image_data_urls=req.image_data_urls,
        workspace_evidences=workspace_evidences,
    )

    try:
        retrieved_evidences = []
        if request_scope != "context-only" and not prefer_workspace_only and not prefer_uploaded_images_only:
            retrieved_evidences = await retriever.retrieve(retrieval_query, top_k=retrieval_cfg.top_k_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"retrieval_failed: {exc}") from exc

    request_context_evidences = _build_fallback_evidence(request_context_items)
    request_context_evidences = uploaded_image_evidences + request_context_evidences
    evidences = _merge_evidences(
        workspace_evidences=workspace_evidences,
        retrieved_evidences=retrieved_evidences,
        request_context_evidences=request_context_evidences,
        scope=request_scope,
        prefer_workspace_only=(prefer_workspace_only or prefer_uploaded_images_only),
    )

    if not evidences and retrieval_cfg.fallback_to_request_context and request_context_items:
        evidences = _build_fallback_evidence(request_context_items)

    system_prompt = (
        "You are a multimodal document QA assistant for a RAG application. "
        "Answer naturally and completely in the user's language, using the retrieved evidence as the grounding source. "
        "Treat workspace evidence as user-uploaded project material and follow the requested workspace priority mode. "
        "Use the prior conversation only to resolve references such as 'this result' or 'the previous figure'; if history conflicts with the current evidence, follow the current evidence. "
        "Synthesize the evidence into a coherent answer instead of copying raw retrieval fragments. "
        "Format the final answer in Markdown when it improves readability, and preserve tables, bullet lists, headings, and code spans when they are useful. "
        "For mathematical expressions, keep formulas compact and readable using standard LaTeX such as `$L(w)=\\\\frac{1}{n}(Xw-y)^T(Xw-y)$`; never split symbols across many lines. "
        "When the user asks to answer all questions in an uploaded image or worksheet, identify each question that is supported by workspace evidence and answer them one by one instead of only answering the first matching item. "
        "If workspace evidence is available for an uploaded image task, ignore unrelated corpus evidence instead of discussing it. "
        "For short follow-up requests such as 'continue', 'the second one', or 'what about this one', use the conversation history only to resolve which previously identified item the user means, then answer with the current evidence. "
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
    task_note = (
        "Task handling note: this request asks for all identifiable questions from the uploaded workspace material. "
        "First list the identifiable questions briefly, then answer each one in order with clear numbering. "
        "If some questions remain unreadable, separate them into a final 'Unresolved items' section.\n\n"
        if _is_exhaustive_question_request(req.query)
        else ""
    )
    workspace_focus_note = (
        "Evidence filtering note: this is a workspace-centered visual question. Use the workspace evidence as primary grounding and do not bring in unrelated corpus questions.\n\n"
        if prefer_workspace_only
        else ""
    )
    direct_image_focus_note = (
        "Evidence filtering note: the user supplied temporary images for this turn. Treat those uploaded images as the primary grounding source and do not introduce unrelated global corpus evidence.\n\n"
        if prefer_uploaded_images_only
        else ""
    )
    retrieval_note = (
        f"Retrieval helper note: the search query was expanded using recent conversation context as:\n{retrieval_query}\n\n"
        if retrieval_query != req.query
        else ""
    )
    user_text = (
        f"Question:\n{req.query}\n\n"
        f"{workspace_mode_note}"
        f"{task_note}"
        f"{workspace_focus_note}"
        f"{direct_image_focus_note}"
        f"{retrieval_note}"
        f"Retrieved Evidence:\n{evidence_block}\n\n"
        f"{visual_hint_block}"
        "Write a polished answer grounded in the evidence above. "
        "For summary or analysis requests, provide a short conclusion first and then the key supporting points in complete sentences."
    )

    user_content: list[dict[str, object]] = [{"type": "text", "text": user_text}]
    for data_url in req.image_data_urls:
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})
    for data_url in visual_assist_images:
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})
    if _needs_vl_placeholder_image(settings.vlm.model, req.image_data_urls, visual_assist_images):
        # Some VL endpoints reject text-only requests. A transparent 1x1 image keeps the
        # request format valid without exposing extra document evidence to the model.
        user_content.append({"type": "image_url", "image_url": {"url": TRANSPARENT_PIXEL_DATA_URL}})
    user_message: dict[str, Any] = {"role": "user", "content": user_content}

    messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    messages.extend(_build_history_messages(req.history))
    messages.append(user_message)

    temperature = req.temperature if req.temperature is not None else retrieval_cfg.default_temperature
    max_tokens = req.max_tokens if req.max_tokens is not None else retrieval_cfg.default_max_tokens

    try:
        answer = await client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        answer = _postprocess_answer(answer)
        if _looks_truncated_answer(answer):
            retry_max_tokens = min(max(max_tokens + 512, int(max_tokens * 1.5)), 4096)
            if retry_max_tokens > max_tokens:
                retry_messages = list(messages)
                retry_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "The previous answer appears incomplete or cut off. "
                                    "Please provide the complete answer again from the beginning, preserving Markdown structure and complete LaTeX expressions."
                                ),
                            }
                        ],
                    }
                )
                retry_answer = await client.chat(retry_messages, temperature=temperature, max_tokens=retry_max_tokens)
                retry_answer = _postprocess_answer(retry_answer)
                if retry_answer and len(retry_answer) >= len(answer):
                    answer = retry_answer
        citations = _build_citations(_select_citation_evidences(evidences, request_scope, req.query))
        answer = _append_inline_citation_summary(answer, citations)
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
