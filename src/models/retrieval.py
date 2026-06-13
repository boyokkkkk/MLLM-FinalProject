from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.clients import BaseEmbeddingClient
from src.utils.settings import RetrievalConfig

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
TIME_RE = re.compile(r"\b\d{1,2}[:.]\d{2}\b|\b(?:a\.m\.|p\.m\.|am|pm)\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*%?\b")
PAGE_MARKER_RE = re.compile(r"^\s*(?:-?\d+-?|page\s*\d+)\s*$", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "by",
    "during",
    "for",
    "from",
    "how",
    "in",
    "is",
    "name",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
}
GENERIC_REGION_TEXTS = {"figure region", "text region", "table region", "formula region"}


@dataclass(slots=True)
class Evidence:
    chunk_id: str
    source: str
    page: int | None
    text: str
    snippet: str
    score: float
    section_title: str | None = None
    citation_kind: str | None = None
    chunk_type: str | None = None
    image_path: str | None = None
    source_path: str | None = None
    bbox: list[int] | None = None


class BaseTextRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        raise NotImplementedError


def rank_sparse_chunks(
    query: str,
    doc_store: dict[str, dict[str, Any]],
    idf: dict[str, float],
    limit: int,
    score_threshold: float = 0.0,
    rerank: bool = False,
    query_type_aware_rerank: bool = True,
    rerank_profile: str = "basic",
    rerank_pool_size: int = 20,
    diversify_results: bool = False,
    fingerprint_duplicate_penalty: float = 0.10,
    docpage_duplicate_penalty: float = 0.08,
    same_sample_penalty: float = 0.04,
) -> list[tuple[float, dict[str, Any]]]:
    # In the stronger profile, drop low-information stopwords during sparse pre-ranking.
    # This reduces domination by synthetic fallback chunks that simply restate the question.
    query_tokens = _tokenize(query)
    if rerank_profile == "stronger":
        filtered_tokens = [token for token in query_tokens if token not in STOPWORDS]
        if filtered_tokens:
            query_tokens = filtered_tokens
    query_tf = _term_frequency(query_tokens)
    ranked_sparse: list[tuple[float, dict[str, Any]]] = []
    for chunk in doc_store.values():
        score = _cosine_sparse(query_tf, _coerce_tf(chunk.get("tf")), idf)
        if score < score_threshold:
            continue
        ranked_sparse.append((score, chunk))

    ranked_sparse.sort(key=lambda pair: pair[0], reverse=True)
    if not rerank or not ranked_sparse:
        return ranked_sparse[:limit]

    pool_size = max(limit, rerank_pool_size)
    candidates = ranked_sparse[:pool_size]
    reranked: list[tuple[float, float, dict[str, Any]]] = []
    for base_score, chunk in candidates:
        rerank_score = base_score + _rerank_adjustment(
            query,
            chunk,
            rerank_profile=rerank_profile,
            query_type_aware=query_type_aware_rerank,
        )
        reranked.append((rerank_score, base_score, chunk))
    reranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    ranked = [(rerank_score, chunk) for rerank_score, _, chunk in reranked]
    if diversify_results:
        return _diversify_ranked_results(
            ranked,
            limit,
            fingerprint_duplicate_penalty=fingerprint_duplicate_penalty,
            docpage_duplicate_penalty=docpage_duplicate_penalty,
            same_sample_penalty=same_sample_penalty,
        )
    return ranked[:limit]


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def _term_frequency(tokens: list[str]) -> dict[str, float]:
    return {term: float(count) for term, count in Counter(tokens).items()}


def _coerce_tf(raw_tf: Any) -> dict[str, float]:
    if not isinstance(raw_tf, dict):
        return {}
    tf: dict[str, float] = {}
    for term, value in raw_tf.items():
        try:
            tf[str(term)] = float(value)
        except (TypeError, ValueError):
            continue
    return tf


def _cosine_sparse(query_tf: dict[str, float], doc_tf: dict[str, float], idf: dict[str, float]) -> float:
    if not query_tf or not doc_tf:
        return 0.0
    dot = 0.0
    q_norm = 0.0
    d_norm = 0.0
    terms = set(query_tf) | set(doc_tf)
    for term in terms:
        weight = idf.get(term, 1.0)
        qv = query_tf.get(term, 0.0) * weight
        dv = doc_tf.get(term, 0.0) * weight
        dot += qv * dv
        q_norm += qv * qv
        d_norm += dv * dv
    if q_norm <= 0.0 or d_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(q_norm) * math.sqrt(d_norm))


def _query_terms_for_rerank(query: str) -> list[str]:
    return [token for token in _tokenize(query) if token not in STOPWORDS and len(token) > 1]


def _normalized_text(text: str) -> str:
    return " ".join(_tokenize(text))


def _query_phrases(query_terms: list[str]) -> list[str]:
    phrases: list[str] = []
    max_size = min(3, len(query_terms))
    for size in range(max_size, 1, -1):
        for index in range(0, len(query_terms) - size + 1):
            phrases.append(" ".join(query_terms[index : index + size]))
    return phrases


def _ordered_overlap_ratio(query_terms: list[str], doc_tokens: list[str]) -> float:
    if not query_terms or not doc_tokens:
        return 0.0
    cursor = 0
    hits = 0
    for token in doc_tokens:
        if cursor >= len(query_terms):
            break
        if token == query_terms[cursor]:
            hits += 1
            cursor += 1
    return hits / len(query_terms)


def _count_list_markers(text: str) -> int:
    return text.count("|") + text.count(":")


def _count_year_markers(text: str) -> int:
    return len(re.findall(r"\b(?:19|20)\d{2}\b", text or ""))


def _contains_address_pattern(text: str) -> bool:
    value = text.lower()
    return any(marker in value for marker in ("street", "st.", "road", "rd.", "avenue", "ave", "washington", "california"))


def _is_location_question(query: str) -> bool:
    return "where is" in query.lower() or "address" in query.lower() or "located" in query.lower()


def _is_photo_question(query: str) -> bool:
    value = query.lower()
    return any(marker in value for marker in ("picture", "advert", "advertise", "photo", "brand", "shown in the picture"))


def _is_title_question(query: str) -> bool:
    value = query.lower()
    return any(marker in value for marker in ("title", "subject", "heading", "name of the company", "name of university"))


def _is_layout_question(query: str) -> bool:
    value = query.lower()
    layout_markers = (
        "page number",
        "heading",
        "subheading",
        "title",
        "logo",
        "word in double quote",
        "day and date",
        "year mentioned at the top",
        "group no.",
    )
    return any(marker in value for marker in layout_markers)


def _is_page_number_question(query: str) -> bool:
    value = query.lower()
    return "page number" in value or value.strip() == "what is the page number?"


def _is_heading_question(query: str) -> bool:
    value = query.lower()
    heading_markers = ("heading", "subheading", "title", "subject", "logo", "heading of the page")
    return any(marker in value for marker in heading_markers)


def _is_time_question(query: str) -> bool:
    value = query.lower()
    return "what time" in value or "time is" in value or "session" in value


def _is_numeric_question(query: str) -> bool:
    value = query.lower()
    numeric_markers = (
        "how many",
        "amount",
        "value",
        "number",
        "total",
        "committee strength",
        "persons present",
        "during the year",
        "what year",
    )
    return any(marker in value for marker in numeric_markers)


def _is_entity_question(query: str) -> bool:
    value = query.lower()
    entity_markers = (
        "what is the name",
        "what is name",
        "who is",
        "where is",
        "what is the company",
        "what is the university",
    )
    return any(marker in value for marker in entity_markers)


def _has_digits(text: str) -> bool:
    return any(char.isdigit() for char in text or "")


def _extract_numbers(text: str) -> set[str]:
    return {match.group(0).lower() for match in NUMBER_RE.finditer(text or "")}


def _is_chart_question(query: str) -> bool:
    value = query.lower()
    return any(marker in value for marker in ("chart", "graph", "figure", "diagram", "per 1000"))


def _is_table_question(query: str) -> bool:
    value = query.lower()
    return any(marker in value for marker in ("table", "committee", "meeting", "attendance", "present", "strength"))


def _is_chart_or_table_question(query: str) -> bool:
    return _is_chart_question(query) or _is_table_question(query)


def _looks_like_structured_numeric_text(text: str) -> bool:
    return _count_list_markers(text) >= 4 or len(_extract_numbers(text)) >= 3 or _count_year_markers(text) >= 2


def _text_fingerprint(text: str) -> str:
    tokens = [token for token in _tokenize(text) if token not in STOPWORDS]
    return " ".join(tokens[:12])


def _chunk_docpage_key(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    raw_doc_id = metadata.get("ucsf_document_id")
    raw_page_no = metadata.get("ucsf_document_page_no")
    if raw_doc_id not in (None, "") and raw_page_no not in (None, ""):
        return f"{raw_doc_id}|{raw_page_no}"
    sample_id = str(chunk.get("sample_id", ""))
    page_no = str(chunk.get("page_no", ""))
    dataset = str(chunk.get("dataset", ""))
    split = str(chunk.get("split", ""))
    return f"{dataset}|{split}|{sample_id}|{page_no}"


def _looks_like_page_marker(text: str) -> bool:
    value = text.strip()
    return bool(PAGE_MARKER_RE.match(value))


def _looks_like_brief_heading(text: str) -> bool:
    tokens = _tokenize(text)
    return 1 <= len(tokens) <= 10 and len(text.strip()) <= 80 and not URL_RE.search(text)


def _diversify_ranked_results(
    ranked: list[tuple[float, dict[str, Any]]],
    limit: int,
    *,
    fingerprint_duplicate_penalty: float = 0.10,
    docpage_duplicate_penalty: float = 0.08,
    same_sample_penalty: float = 0.04,
) -> list[tuple[float, dict[str, Any]]]:
    selected: list[tuple[float, dict[str, Any]]] = []
    seen_fingerprints: set[str] = set()
    seen_docpages: set[str] = set()
    per_sample_count: Counter[str] = Counter()
    for score, chunk in ranked:
        text = str(chunk.get("text", ""))
        fingerprint = _text_fingerprint(text)
        sample_id = str(chunk.get("sample_id", ""))
        docpage_key = _chunk_docpage_key(chunk)
        adjusted = score
        if fingerprint and fingerprint in seen_fingerprints:
            adjusted -= fingerprint_duplicate_penalty
        if docpage_key and docpage_key in seen_docpages:
            adjusted -= docpage_duplicate_penalty
        adjusted -= same_sample_penalty * per_sample_count[sample_id]
        if adjusted < 0:
            adjusted = 0.0
        selected.append((adjusted, chunk))
        seen_fingerprints.add(fingerprint)
        seen_docpages.add(docpage_key)
        per_sample_count[sample_id] += 1
    selected.sort(key=lambda item: item[0], reverse=True)
    return selected[:limit]


def _basic_rerank_adjustment(query: str, chunk: dict[str, Any]) -> float:
    text = str(chunk.get("text", "")).strip()
    normalized_text = text.lower()
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    raw_type = str(metadata.get("raw_type", "")).lower()
    chunk_type = str(chunk.get("chunk_type", "")).lower()
    token_count = int(chunk.get("token_count") or 0)
    doc_terms = set(_tokenize(text))
    query_terms = _query_terms_for_rerank(query)
    overlap_ratio = 0.0
    if query_terms:
        overlap_ratio = sum(1 for term in query_terms if term in doc_terms) / len(query_terms)

    score = 0.14 * overlap_ratio
    numeric_overlap = len(_extract_numbers(query) & _extract_numbers(text))
    if numeric_overlap:
        score += 0.05 * min(numeric_overlap, 2)

    if normalized_text in GENERIC_REGION_TEXTS:
        score -= 0.34
    if raw_type == "footer" or normalized_text.startswith("source:") or URL_RE.search(text):
        score -= 0.20
    if token_count <= 2:
        score -= 0.06
    if chunk_type == "figure" and normalized_text == "figure region":
        score -= 0.10

    if _is_time_question(query) and TIME_RE.search(text):
        score += 0.14
    if _is_numeric_question(query) and _has_digits(text):
        score += 0.08
    if _is_numeric_question(query) and chunk_type == "table":
        score += 0.08
    if _is_table_question(query) and chunk_type == "table":
        score += 0.10
    if _is_chart_question(query) and chunk_type in {"figure", "table"}:
        score += 0.08
    if _is_entity_question(query) and raw_type in {"header", "title"}:
        score += 0.10
    if _is_entity_question(query) and 2 <= token_count <= 12:
        score += 0.03
    if raw_type == "header" and overlap_ratio > 0:
        score += 0.04
    if raw_type in {"title", "header"} and chunk_type == "text" and token_count <= 16:
        score += 0.03

    return score


def _strong_rerank_adjustment(query: str, chunk: dict[str, Any], query_type_aware: bool = True) -> float:
    text = str(chunk.get("text", "")).strip()
    normalized_text = text.lower()
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    raw_type = str(metadata.get("raw_type", "")).lower()
    chunk_type = str(chunk.get("chunk_type", "")).lower()
    token_count = int(chunk.get("token_count") or 0)
    query_terms = _query_terms_for_rerank(query)
    doc_tokens = _tokenize(text)
    doc_terms = set(doc_tokens)
    score = _basic_rerank_adjustment(query, chunk)

    if query_terms:
        overlap_count = sum(1 for term in query_terms if term in doc_terms)
        overlap_ratio = overlap_count / len(query_terms)
        score += 0.08 * overlap_ratio
        score += 0.10 * _ordered_overlap_ratio(query_terms, doc_tokens)
        normalized_doc = _normalized_text(text)
        for phrase in _query_phrases(query_terms):
            if phrase in normalized_doc:
                score += 0.035 if len(phrase.split()) == 2 else 0.055

    if _is_numeric_question(query):
        numbers = _extract_numbers(text)
        score += 0.02 * min(len(numbers), 4)
        if chunk_type in {"table", "figure"}:
            score += 0.05
        if _count_list_markers(text) >= 4:
            score += 0.04
    if _is_table_question(query) and _count_list_markers(text) >= 4:
        score += 0.06
    if _is_chart_question(query) and any(marker in normalized_text for marker in ("year", "line", "dotted", "solid")):
        score += 0.08
    if _is_entity_question(query) or _is_title_question(query):
        if raw_type in {"title", "header"}:
            score += 0.08
        if 2 <= token_count <= 14:
            score += 0.05
        if token_count > 40:
            score -= 0.05
    if _is_location_question(query):
        if _contains_address_pattern(text):
            score += 0.08
        if "," in text:
            score += 0.03
    if _is_photo_question(query):
        if chunk_type in {"figure", "page_image"}:
            score += 0.10
        if raw_type in {"title", "header"}:
            score += 0.04
        if token_count > 40 and chunk_type == "text":
            score -= 0.08
    if query_type_aware and _is_layout_question(query):
        if raw_type in {"title", "header"}:
            score += 0.12
        if _looks_like_brief_heading(text):
            score += 0.08
        if token_count > 60 and chunk_type == "text":
            score -= 0.06
    if query_type_aware and _is_heading_question(query):
        if raw_type in {"title", "header"}:
            score += 0.10
        if _looks_like_brief_heading(text):
            score += 0.10
        if token_count > 30:
            score -= 0.06
    if query_type_aware and _is_page_number_question(query):
        if _looks_like_page_marker(text):
            score += 0.18
        if raw_type in {"footer", "header"} and _has_digits(text):
            score += 0.08
        if token_count > 12:
            score -= 0.08
    if normalized_text in GENERIC_REGION_TEXTS:
        score -= 0.08
    if raw_type == "footer":
        score -= 0.05
    return score


def _rerank_adjustment(
    query: str,
    chunk: dict[str, Any],
    rerank_profile: str = "basic",
    query_type_aware: bool = True,
) -> float:
    if rerank_profile == "stronger":
        return _strong_rerank_adjustment(query, chunk, query_type_aware=query_type_aware)
    return _basic_rerank_adjustment(query, chunk)


def _chunk_visual_key(chunk: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(chunk.get("sample_id") or ""),
        str(chunk.get("page_no") or ""),
        str(chunk.get("image_path") or chunk.get("source_path") or ""),
    )


def _select_visual_support_chunk(
    query: str,
    visual_item: dict[str, Any],
    doc_store: dict[str, dict[str, Any]],
    idf: dict[str, float],
    chart_table_specialist: bool = False,
) -> dict[str, Any]:
    direct = doc_store.get(str(visual_item.get("chunk_id") or ""))
    key = _chunk_visual_key(visual_item)
    candidates = [chunk for chunk in doc_store.values() if _chunk_visual_key(chunk) == key]
    if chart_table_specialist and _is_chart_or_table_question(query):
        block_id = str(visual_item.get("block_id") or "")
        page_no = visual_item.get("page_no")
        focused: list[dict[str, Any]] = []
        for chunk in candidates:
            chunk_block_id = str(chunk.get("block_id") or "")
            same_block = block_id and chunk_block_id == block_id
            same_page = page_no is not None and chunk.get("page_no") == page_no
            chunk_text = str(chunk.get("text", ""))
            if same_block or (same_page and _looks_like_structured_numeric_text(chunk_text)):
                focused.append(chunk)
        if focused:
            candidates = focused
    if direct and direct not in candidates:
        candidates.append(direct)
    if not candidates and direct:
        return direct
    if not candidates:
        pseudo_chunk = dict(visual_item)
        pseudo_chunk.setdefault("text", str(visual_item.get("caption", "")))
        pseudo_chunk.setdefault("metadata", {})
        pseudo_chunk.setdefault("token_count", len(_tokenize(str(pseudo_chunk.get("text", "")))))
        return pseudo_chunk

    query_tf = _term_frequency(_tokenize(query))
    best_chunk = candidates[0]
    best_score = -1.0
    for chunk in candidates:
        text = str(chunk.get("text", "")).strip()
        base_score = _cosine_sparse(query_tf, _coerce_tf(chunk.get("tf")), idf)
        rerank_score = _strong_rerank_adjustment(query, chunk)
        if chart_table_specialist and _is_chart_or_table_question(query):
            if _looks_like_structured_numeric_text(text):
                rerank_score += 0.12
            if str(chunk.get("chunk_type", "")).lower() in {"table", "figure"}:
                rerank_score += 0.08
        penalty = 0.15 if text.lower() in GENERIC_REGION_TEXTS else 0.0
        candidate_score = base_score + rerank_score - penalty
        if candidate_score > best_score:
            best_score = candidate_score
            best_chunk = chunk
    return best_chunk


def rank_visual_chunks(
    query: str,
    doc_store: dict[str, dict[str, Any]],
    visual_store: list[dict[str, Any]],
    idf: dict[str, float],
    limit: int,
    chart_table_specialist: bool = False,
    chart_table_visual_boost: float = 0.18,
) -> list[tuple[float, dict[str, Any]]]:
    query_tf = _term_frequency(_tokenize(query))
    ranked: list[tuple[float, dict[str, Any]]] = []
    seen_chunk_ids: set[str] = set()
    specialist_mode = chart_table_specialist and _is_chart_or_table_question(query)
    for item in visual_store:
        item_chunk_type = str(item.get("chunk_type", "")).lower()
        if specialist_mode and item_chunk_type not in {"figure", "table", "text"}:
            continue
        support_chunk = _select_visual_support_chunk(
            query,
            item,
            doc_store,
            idf,
            chart_table_specialist=chart_table_specialist,
        )
        chunk_id = str(support_chunk.get("chunk_id") or "")
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        caption = str(item.get("caption", ""))
        support_text = str(support_chunk.get("text", ""))
        merged_text = " ".join(part for part in (caption, support_text) if part).strip()
        merged_tf = _term_frequency(_tokenize(merged_text))
        score = _cosine_sparse(query_tf, merged_tf, idf)
        score += _strong_rerank_adjustment(query, support_chunk)

        chunk_type = str(item.get("chunk_type", support_chunk.get("chunk_type", ""))).lower()
        if _is_chart_question(query) and chunk_type in {"figure", "table"}:
            score += 0.10
        if _is_table_question(query) and chunk_type == "table":
            score += 0.12
        if _is_photo_question(query) and chunk_type in {"figure", "page_image", "text"}:
            score += 0.08
        if _is_title_question(query) and chunk_type == "text":
            score += 0.05
        if merged_text.lower() in GENERIC_REGION_TEXTS:
            score -= 0.12
        if specialist_mode:
            if _looks_like_structured_numeric_text(merged_text):
                score += chart_table_visual_boost
            if chunk_type == "table":
                score += chart_table_visual_boost
            if _count_year_markers(merged_text) >= 2:
                score += 0.08
            if chunk_type == "figure" and _is_chart_question(query):
                score += 0.10
            if chunk_type == "text" and not _looks_like_structured_numeric_text(merged_text):
                score -= 0.06
        ranked.append((score, support_chunk))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return ranked[:limit]


def late_fuse_ranked_chunks(
    text_ranked: list[tuple[float, dict[str, Any]]],
    visual_ranked: list[tuple[float, dict[str, Any]]],
    limit: int,
    text_weight: float,
    visual_weight: float,
    fusion_k: int,
) -> list[tuple[float, dict[str, Any]]]:
    fused_scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict[str, Any]] = {}

    for rank, (_, chunk) in enumerate(text_ranked, start=1):
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id:
            continue
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (text_weight / (fusion_k + rank))
        chunk_lookup[chunk_id] = chunk

    for rank, (_, chunk) in enumerate(visual_ranked, start=1):
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id:
            continue
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (visual_weight / (fusion_k + rank))
        chunk_lookup[chunk_id] = chunk

    ranked = [(score, chunk_lookup[chunk_id]) for chunk_id, score in fused_scores.items()]
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return ranked[:limit]


def rerank_chart_table_candidates(
    query: str,
    ranked: list[tuple[float, dict[str, Any]]],
    limit: int,
) -> list[tuple[float, dict[str, Any]]]:
    if not ranked or not _is_chart_or_table_question(query):
        return ranked[:limit]

    reranked: list[tuple[float, float, dict[str, Any]]] = []
    query_numbers = _extract_numbers(query)
    for base_score, chunk in ranked:
        text = str(chunk.get("text", ""))
        chunk_type = str(chunk.get("chunk_type", "")).lower()
        struct_bonus = 0.0
        numbers = _extract_numbers(text)
        if _looks_like_structured_numeric_text(text):
            struct_bonus += 0.16
        if chunk_type == "table":
            struct_bonus += 0.14
        if chunk_type == "figure" and _is_chart_question(query):
            struct_bonus += 0.12
        if _count_year_markers(text) >= 2:
            struct_bonus += 0.10
        if len(numbers) >= 4:
            struct_bonus += 0.08
        if query_numbers and (numbers & query_numbers):
            struct_bonus += 0.08
        if any(marker in text.lower() for marker in ("solid line", "dotted line", "per 1000")):
            struct_bonus += 0.10
        reranked.append((base_score + struct_bonus, base_score, chunk))

    reranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [(base_score, chunk) for _, base_score, chunk in reranked[:limit]]


class LocalTextRetriever(BaseTextRetriever):
    def __init__(self, embedding_client: BaseEmbeddingClient, config: RetrievalConfig) -> None:
        self.embedding_client = embedding_client
        self.config = config
        self._dense_metadata: list[dict[str, Any]] | None = None
        self._dense_vectors: list[list[float]] | None = None
        self._sparse_doc_store: dict[str, dict[str, Any]] | None = None
        self._sparse_idf: dict[str, float] | None = None
        self._visual_store: list[dict[str, Any]] | None = None
        self._visual_dense_metadata: list[dict[str, Any]] | None = None
        self._visual_dense_vectors: list[list[float]] | None = None

    async def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        if not self.config.enable_text_retrieval:
            return []

        normalized_query = query.strip()
        if not normalized_query:
            return []

        limit = top_k or self.config.top_k_text

        if self._sparse_available():
            doc_store, idf = self._load_sparse_resources()
            ranked_sparse = rank_sparse_chunks(
                query=normalized_query,
                doc_store=doc_store,
                idf=idf,
                limit=max(limit, self.config.dense_rerank_pool_size if self.config.dense_rerank else limit),
                score_threshold=self.config.score_threshold,
                rerank=self.config.rerank,
                query_type_aware_rerank=self.config.query_type_aware_rerank,
                rerank_profile=self.config.rerank_profile,
                rerank_pool_size=self.config.rerank_pool_size,
                diversify_results=self.config.diversify_results,
                fingerprint_duplicate_penalty=self.config.fingerprint_duplicate_penalty,
                docpage_duplicate_penalty=self.config.docpage_duplicate_penalty,
                same_sample_penalty=self.config.same_sample_penalty,
            )
            if self.config.dense_rerank and ranked_sparse:
                ranked_sparse = await self._dense_rerank_candidates(normalized_query, ranked_sparse, limit)
            if self.config.visual_fusion and self._visual_available():
                ranked_sparse = self._late_fuse_visual_candidates(normalized_query, ranked_sparse, doc_store, idf, limit)
            if self.config.visual_dense_fusion and self._visual_dense_available():
                ranked_sparse = await self._late_fuse_visual_dense_candidates(normalized_query, ranked_sparse, doc_store, limit)
            if self.config.chart_table_specialist:
                ranked_sparse = rerank_chart_table_candidates(normalized_query, ranked_sparse, limit)
            return [self._build_evidence(item, score) for score, item in ranked_sparse[:limit]]

        if self._dense_available():
            metadata, vectors = self._load_dense_resources()
            query_vectors = await self.embedding_client.embed([normalized_query])
            if not query_vectors:
                return []

            query_vector = query_vectors[0]
            ranked: list[tuple[float, dict[str, Any]]] = []
            for item, vector in zip(metadata, vectors, strict=True):
                score = self._cosine_similarity(query_vector, vector)
                if score < self.config.score_threshold:
                    continue
                ranked.append((score, item))

            ranked.sort(key=lambda pair: pair[0], reverse=True)
            return [self._build_evidence(item, score) for score, item in ranked[:limit]]

        raise FileNotFoundError(
            "No retrieval sources were found. Expected sparse doc store or dense retrieval artifacts."
        )

    def _dense_available(self) -> bool:
        return self.config.metadata_path.exists() and self.config.index_path.exists()

    def _sparse_available(self) -> bool:
        return self.config.sparse_index_path.exists()

    def _visual_available(self) -> bool:
        return self.config.visual_index_path.exists()

    def _visual_dense_available(self) -> bool:
        return self.config.visual_dense_metadata_path.exists() and self.config.visual_dense_vectors_path.exists()

    def _load_dense_resources(self) -> tuple[list[dict[str, Any]], list[list[float]]]:
        if self._dense_metadata is None:
            self._dense_metadata = self._load_metadata(self.config.metadata_path)
        if self._dense_vectors is None:
            self._dense_vectors = self._load_vectors(self.config.index_path)
        if len(self._dense_metadata) != len(self._dense_vectors):
            raise ValueError(
                "Metadata and vector counts do not match: "
                f"{len(self._dense_metadata)} metadata rows vs {len(self._dense_vectors)} vectors."
            )
        return self._dense_metadata, self._dense_vectors

    def _load_sparse_resources(self) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
        if self._sparse_doc_store is None:
            self._sparse_doc_store = self._load_sparse_doc_store(self.config.sparse_index_path)
        if self._sparse_idf is None:
            n_docs = max(1, len(self._sparse_doc_store))
            doc_frequency: Counter[str] = Counter()
            for chunk in self._sparse_doc_store.values():
                doc_frequency.update(_coerce_tf(chunk.get("tf")).keys())
            self._sparse_idf = {
                term: math.log((n_docs + 1) / (freq + 1)) + 1.0
                for term, freq in doc_frequency.items()
            }
        return self._sparse_doc_store, self._sparse_idf

    def _load_visual_store(self) -> list[dict[str, Any]]:
        if self._visual_store is None:
            path = self.config.visual_index_path
            if not path.exists():
                raise FileNotFoundError(f"Visual store not found: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"Visual store must be a JSON list: {path}")
            self._visual_store = [item for item in payload if isinstance(item, dict)]
        return self._visual_store

    def _load_visual_dense_resources(self) -> tuple[list[dict[str, Any]], list[list[float]]]:
        if self._visual_dense_metadata is None:
            self._visual_dense_metadata = self._load_metadata(self.config.visual_dense_metadata_path)
        if self._visual_dense_vectors is None:
            self._visual_dense_vectors = self._load_vectors(self.config.visual_dense_vectors_path)
        if len(self._visual_dense_metadata) != len(self._visual_dense_vectors):
            raise ValueError(
                "Visual descriptor metadata and vector counts do not match: "
                f"{len(self._visual_dense_metadata)} metadata rows vs {len(self._visual_dense_vectors)} vectors."
            )
        return self._visual_dense_metadata, self._visual_dense_vectors

    def _load_metadata(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Retrieval metadata file not found: {path}")

        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("chunks", payload) if isinstance(payload, dict) else payload
        else:
            raise ValueError(f"Unsupported metadata format: {path.suffix}")

        if not isinstance(rows, list):
            raise ValueError(f"Metadata file must contain a list-like payload: {path}")

        validated: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                raise ValueError(f"Metadata row must be an object: {item!r}")
            self._validate_metadata_row(item, path)
            validated.append(item)
        return validated

    def _load_sparse_doc_store(self, path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(
                "No retrieval sources were found. Expected either dense retrieval artifacts "
                f"({self.config.metadata_path}, {self.config.index_path}) or sparse doc store {path}."
            )

        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Sparse doc store must be a JSON object keyed by chunk_id: {path}")

        validated: dict[str, dict[str, Any]] = {}
        for chunk_id, item in payload.items():
            if not isinstance(item, dict):
                raise ValueError(f"Sparse doc store row must be an object: {chunk_id!r}")
            row = dict(item)
            row.setdefault("chunk_id", str(chunk_id))
            self._validate_metadata_row(row, path)
            validated[str(chunk_id)] = row
        return validated

    def _load_vectors(self, path: Path) -> list[list[float]]:
        if not path.exists():
            raise FileNotFoundError(f"Retrieval vector file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("vectors", payload) if isinstance(payload, dict) else payload
        elif suffix == ".jsonl":
            rows = [self._extract_vector(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif suffix == ".npy":
            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError("Loading .npy vector files requires numpy to be installed.") from exc
            rows = np.load(path, allow_pickle=False).tolist()
        else:
            raise ValueError(f"Unsupported vector format: {path.suffix}")

        if not isinstance(rows, list):
            raise ValueError(f"Vector file must contain a list-like payload: {path}")

        vectors: list[list[float]] = []
        for row in rows:
            vector = self._extract_vector(row)
            if not vector:
                raise ValueError("Vector rows must be non-empty.")
            vectors.append(vector)
        return vectors

    def _extract_vector(self, row: Any) -> list[float]:
        if isinstance(row, dict):
            if "vector" in row:
                row = row["vector"]
            elif "embedding" in row:
                row = row["embedding"]
        if not isinstance(row, list):
            raise ValueError(f"Vector row must be a list of floats: {row!r}")
        return [float(value) for value in row]

    def _validate_metadata_row(self, item: dict[str, Any], path: Path) -> None:
        required_fields = ("chunk_id", "text")
        missing = [field for field in required_fields if not item.get(field)]
        if missing:
            raise ValueError(f"Metadata row in {path} is missing required fields: {', '.join(missing)}")
        if not self._pick_source(item):
            raise ValueError(f"Metadata row in {path} must include source/source_ref/source_path: {item!r}")
        page = item.get("page", item.get("page_no"))
        if page is not None:
            try:
                int(page)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Metadata row page must be int-like or null: {item!r}") from exc

    def _pick_source(self, item: dict[str, Any]) -> str | None:
        for key in ("source", "source_ref", "source_path"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _build_evidence(self, item: dict[str, Any], score: float) -> Evidence:
        text = str(item.get("text", "")).strip()
        snippet = text.replace("\n", " ")[:240]
        page_value = item.get("page", item.get("page_no"))
        page = int(page_value) if page_value not in (None, "") else None
        return Evidence(
            chunk_id=str(item["chunk_id"]),
            source=self._pick_source(item) or "unknown",
            page=page,
            text=text,
            snippet=snippet,
            score=score,
            section_title=None,
            citation_kind="corpus",
            chunk_type=str(item.get("chunk_type", "")).strip() or None,
            image_path=str(item.get("image_path", "")).strip() or None,
            source_path=str(item.get("source_path", "")).strip() or None,
            bbox=item.get("bbox") if isinstance(item.get("bbox"), list) else None,
        )

    async def _dense_rerank_candidates(
        self,
        query: str,
        ranked_sparse: list[tuple[float, dict[str, Any]]],
        limit: int,
    ) -> list[tuple[float, dict[str, Any]]]:
        pool_size = max(limit, self.config.dense_rerank_pool_size)
        candidates = ranked_sparse[:pool_size]
        texts = [str(item.get("text", ""))[:1200] for _, item in candidates]
        if not texts:
            return ranked_sparse[:limit]
        try:
            vectors = await self.embedding_client.embed([query, *texts])
        except Exception:
            return ranked_sparse[:limit]
        if len(vectors) != len(texts) + 1:
            return ranked_sparse[:limit]
        query_vector = vectors[0]
        reranked: list[tuple[float, float, dict[str, Any]]] = []
        for (sparse_score, item), vector in zip(candidates, vectors[1:], strict=True):
            dense_score = self._cosine_similarity(query_vector, vector)
            hybrid_score = ((1.0 - self.config.dense_score_weight) * sparse_score) + (self.config.dense_score_weight * dense_score)
            reranked.append((hybrid_score, sparse_score, item))
        reranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        dense_ranked = [(hybrid_score, item) for hybrid_score, _, item in reranked]
        if self.config.diversify_results:
            return _diversify_ranked_results(
                dense_ranked,
                limit,
                fingerprint_duplicate_penalty=self.config.fingerprint_duplicate_penalty,
                docpage_duplicate_penalty=self.config.docpage_duplicate_penalty,
                same_sample_penalty=self.config.same_sample_penalty,
            )
        return dense_ranked[:limit]

    def _late_fuse_visual_candidates(
        self,
        query: str,
        ranked_sparse: list[tuple[float, dict[str, Any]]],
        doc_store: dict[str, dict[str, Any]],
        idf: dict[str, float],
        limit: int,
    ) -> list[tuple[float, dict[str, Any]]]:
        try:
            visual_store = self._load_visual_store()
        except Exception:
            return ranked_sparse[:limit]

        visual_ranked = rank_visual_chunks(
            query=query,
            doc_store=doc_store,
            visual_store=visual_store,
            idf=idf,
            limit=max(limit, self.config.visual_pool_size),
            chart_table_specialist=self.config.chart_table_specialist,
            chart_table_visual_boost=self.config.chart_table_visual_boost,
        )
        if not visual_ranked:
            return ranked_sparse[:limit]
        text_pool = ranked_sparse[: max(limit, self.config.rerank_pool_size)]
        visual_weight = self.config.visual_fusion_weight
        if self.config.chart_table_specialist and _is_chart_or_table_question(query):
            visual_weight += self.config.chart_table_visual_boost
        fused = late_fuse_ranked_chunks(
            text_ranked=text_pool,
            visual_ranked=visual_ranked,
            limit=limit,
            text_weight=self.config.text_fusion_weight,
            visual_weight=visual_weight,
            fusion_k=max(1, self.config.fusion_k),
        )
        if self.config.diversify_results:
            return _diversify_ranked_results(
                fused,
                limit,
                fingerprint_duplicate_penalty=self.config.fingerprint_duplicate_penalty,
                docpage_duplicate_penalty=self.config.docpage_duplicate_penalty,
                same_sample_penalty=self.config.same_sample_penalty,
            )
        return fused

    async def _late_fuse_visual_dense_candidates(
        self,
        query: str,
        ranked_sparse: list[tuple[float, dict[str, Any]]],
        doc_store: dict[str, dict[str, Any]],
        limit: int,
    ) -> list[tuple[float, dict[str, Any]]]:
        try:
            metadata, vectors = self._load_visual_dense_resources()
            query_vector = (await self.embedding_client.embed([query]))[0]
        except Exception:
            return ranked_sparse[:limit]

        ranked_visual: list[tuple[float, dict[str, Any]]] = []
        for item, vector in zip(metadata, vectors, strict=True):
            chunk_id = str(item.get("chunk_id") or "")
            chunk = doc_store.get(chunk_id)
            if not chunk:
                continue
            score = self._cosine_similarity(query_vector, vector)
            ranked_visual.append((score, chunk))
        ranked_visual.sort(key=lambda pair: pair[0], reverse=True)
        if not ranked_visual:
            return ranked_sparse[:limit]
        fused = late_fuse_ranked_chunks(
            text_ranked=ranked_sparse[: max(limit, self.config.rerank_pool_size)],
            visual_ranked=ranked_visual[: max(limit, self.config.visual_dense_pool_size)],
            limit=limit,
            text_weight=self.config.text_fusion_weight,
            visual_weight=self.config.visual_dense_weight,
            fusion_k=max(1, self.config.fusion_k),
        )
        if self.config.diversify_results:
            return _diversify_ranked_results(
                fused,
                limit,
                fingerprint_duplicate_penalty=self.config.fingerprint_duplicate_penalty,
                docpage_duplicate_penalty=self.config.docpage_duplicate_penalty,
                same_sample_penalty=self.config.same_sample_penalty,
            )
        return fused

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError(f"Vector dimension mismatch: {len(left)} vs {len(right)}")

        dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot_product / (left_norm * right_norm)
