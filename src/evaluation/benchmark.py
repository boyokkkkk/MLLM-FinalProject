from __future__ import annotations

import asyncio
import base64
import json
import math
import mimetypes
import re
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_index_utils import read_jsonl, read_yaml
from src.evaluation.metrics import anls, answer_contains, citation_accuracy, exact_match, hit_at_k, precision_at_k, recall_at_k, token_f1
from src.models.retrieval import late_fuse_ranked_chunks, rank_sparse_chunks, rank_visual_chunks, rerank_chart_table_candidates
from src.models.clients import build_embedding_client, build_llm_client
from src.utils.settings import RetrievalConfig, settings


@dataclass(slots=True)
class EvalSample:
    sample_id: str
    dataset: str
    split: str
    question: str
    answers: list[str]
    image_path: str | None
    expected_source_prefix: str
    expected_doc_page_key: str | None = None


_VISION_VECTOR_CACHE: dict[str, list[float]] = {}
_QUERY_IMAGE_DESCRIPTOR_CACHE: dict[str, str] = {}
_TEXT_VECTOR_CACHE: dict[str, list[float]] = {}
_PAGE_TEXT_CACHE: dict[str, str] = {}
_QUERY_IMAGE_TIEBREAK_CACHE: dict[str, int] = {}


def _sample_doc_page_key(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    doc_id = metadata.get("ucsf_document_id")
    page_no = metadata.get("ucsf_document_page_no")
    if doc_id in (None, "") or page_no in (None, ""):
        return None
    return f"{doc_id}|{page_no}"


def _chunk_doc_page_key(chunk: dict[str, Any]) -> str | None:
    explicit_key = chunk.get("doc_page_key")
    if explicit_key not in (None, ""):
        return str(explicit_key)
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    doc_id = metadata.get("ucsf_document_id")
    page_no = metadata.get("ucsf_document_page_no")
    if doc_id in (None, "") or page_no in (None, ""):
        return None
    return f"{doc_id}|{page_no}"


def load_eval_config(project_root: Path, config_path: Path) -> dict[str, Any]:
    cfg = read_yaml(config_path)
    return cfg.get("eval", {})


def collect_eval_samples(
    project_root: Path,
    datasets: list[str],
    splits: list[str],
    limit_per_split: int = 0,
    sample_manifest: Path | None = None,
) -> list[EvalSample]:
    samples: list[EvalSample] = []
    if sample_manifest is not None:
        rows = read_jsonl(sample_manifest)
        for row in rows:
            sample_id = str(row.get("id"))
            dataset = str(row.get("dataset", ""))
            split = str(row.get("split", ""))
            image = row.get("image")
            image_path = str(image) if isinstance(image, str) else None
            samples.append(
                EvalSample(
                    sample_id=sample_id,
                    dataset=dataset,
                    split=split,
                    question=str(row.get("question", "")),
                    answers=[str(item) for item in row.get("answers", [])],
                    image_path=image_path,
                    expected_source_prefix=f"{dataset}/{split}/{sample_id}",
                    expected_doc_page_key=_sample_doc_page_key(row),
                )
            )
        return samples
    for dataset in datasets:
        for split in splits:
            path = project_root / "data" / "processed" / dataset / f"{split}.jsonl"
            rows = read_jsonl(path)
            if limit_per_split > 0:
                rows = rows[:limit_per_split]
            for row in rows:
                sample_id = str(row.get("id"))
                image = row.get("image")
                image_path = str(image) if isinstance(image, str) else None
                samples.append(
                    EvalSample(
                        sample_id=sample_id,
                        dataset=dataset,
                        split=split,
                        question=str(row.get("question", "")),
                        answers=[str(item) for item in row.get("answers", [])],
                        image_path=image_path,
                        expected_source_prefix=f"{dataset}/{split}/{sample_id}",
                        expected_doc_page_key=_sample_doc_page_key(row),
                    )
                )
    return samples


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(l * r for l, r in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _fuse_query_image_ranked_chunks(
    query: str,
    query_image_path: str | None,
    ranked_text: list[tuple[float, dict[str, Any]]],
    doc_store: dict[str, dict[str, Any]],
    top_k: int,
    retrieval_cfg: RetrievalConfig,
) -> list[tuple[float, dict[str, Any]]]:
    if not query_image_path or not ranked_text:
        return ranked_text[:top_k]
    query_lower = query.lower()
    if not any(marker in query_lower for marker in ("title", "heading", "page no", "page number")):
        return ranked_text[:top_k]

    pool = ranked_text[: max(top_k, retrieval_cfg.query_image_pool_size)]
    query_key = f"{query_image_path}::{query}"
    if query_key not in _QUERY_IMAGE_DESCRIPTOR_CACHE:
        data_urls = _image_to_data_url(query_image_path)
        if not data_urls:
            return ranked_text[:top_k]
        llm_client = build_llm_client(settings.vlm)
        prompt = (
            "Write one concise retrieval descriptor for this document page image. "
            "Focus on page title, heading, page number, form fields, key entities, short visible phrases, "
            f"and any layout cues that would help distinguish this page from similar pages for the question: {query}. "
            "Return exactly one line."
        )
        messages = [
            {"role": "system", "content": "You produce concise retrieval descriptors for document pages."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_urls[0]}},
                ],
            },
        ]
        try:
            descriptor = asyncio.run(llm_client.chat(messages=messages, temperature=0.0, max_tokens=120))
        except Exception:
            return ranked_text[:top_k]
        _QUERY_IMAGE_DESCRIPTOR_CACHE[query_key] = " ".join((descriptor or "").split())[:500]

    query_descriptor = _QUERY_IMAGE_DESCRIPTOR_CACHE.get(query_key, "")
    if not query_descriptor:
        return ranked_text[:top_k]

    embed_client = build_embedding_client(settings.text_embedding)
    try:
        query_vector = asyncio.run(embed_client.embed([query_descriptor]))[0]
    except Exception:
        return ranked_text[:top_k]

    def page_key(chunk: dict[str, Any]) -> str:
        sample_id = str(chunk.get("sample_id") or "")
        page_no = str(chunk.get("page_no") or "")
        return f"{sample_id}|{page_no}"

    def aggregate_page_text(chunk: dict[str, Any]) -> str:
        key = page_key(chunk)
        if key in _PAGE_TEXT_CACHE:
            return _PAGE_TEXT_CACHE[key]
        sample_id = str(chunk.get("sample_id") or "")
        page_no = chunk.get("page_no")
        page_chunks = [
            item
            for item in doc_store.values()
            if str(item.get("sample_id") or "") == sample_id and item.get("page_no") == page_no
        ]
        prioritized: list[str] = []
        fallback: list[str] = []
        for item in page_chunks:
            text = " ".join(str(item.get("text", "")).split()).strip()
            if not text:
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            parser = str(metadata.get("parser") or "")
            chunk_type = str(item.get("chunk_type") or "")
            if parser == "fallback" or chunk_type in {"figure", "page_image"}:
                fallback.append(text)
                continue
            prioritized.append(text)
        value = " ".join(prioritized[:8] or fallback[:4])[:1600]
        _PAGE_TEXT_CACHE[key] = value
        return value

    missing_chunks: list[tuple[str, str]] = []
    for _, chunk in pool:
        key = page_key(chunk)
        if key and key not in _TEXT_VECTOR_CACHE:
            text = aggregate_page_text(chunk)
            if text:
                missing_chunks.append((key, text))
    if missing_chunks:
        try:
            vectors = asyncio.run(embed_client.embed([text for _, text in missing_chunks]))
        except Exception:
            return ranked_text[:top_k]
        for (key, _), vector in zip(missing_chunks, vectors, strict=True):
            _TEXT_VECTOR_CACHE[key] = [float(value) for value in vector]

    ranked_visual: list[tuple[float, dict[str, Any]]] = []
    for _, chunk in pool:
        vector = _TEXT_VECTOR_CACHE.get(page_key(chunk))
        if not vector:
            continue
        ranked_visual.append((_cosine_similarity(query_vector, vector), chunk))
    if not ranked_visual:
        return ranked_text[:top_k]

    ranked_visual.sort(key=lambda pair: pair[0], reverse=True)
    return late_fuse_ranked_chunks(
        text_ranked=pool,
        visual_ranked=ranked_visual,
        limit=top_k,
        text_weight=retrieval_cfg.text_fusion_weight,
        visual_weight=retrieval_cfg.query_image_weight,
        fusion_k=max(1, retrieval_cfg.fusion_k),
    )


def _needs_query_image_vlm_tiebreak(query: str, ranked: list[tuple[float, dict[str, Any]]]) -> bool:
    if len(ranked) < 2:
        return False
    value = query.lower()
    if not any(marker in value for marker in ("title", "heading", "page no", "page number")):
        return False
    first_score, first_chunk = ranked[0]
    second_score, second_chunk = ranked[1]
    if abs(float(first_score) - float(second_score)) > 0.0005:
        return False
    first_text = str(first_chunk.get("text", "")).strip().lower()
    second_text = str(second_chunk.get("text", "")).strip().lower()
    return first_text.startswith("question:") and second_text.startswith("question:")


def _maybe_query_image_vlm_tiebreak(
    query: str,
    query_image_path: str | None,
    ranked: list[tuple[float, dict[str, Any]]],
    top_k: int,
) -> list[tuple[float, dict[str, Any]]]:
    if not query_image_path or not _needs_query_image_vlm_tiebreak(query, ranked):
        return ranked[:top_k]
    if len(ranked) < 2:
        return ranked[:top_k]

    query_urls = _image_to_data_url(query_image_path)
    first_path = str(ranked[0][1].get("image_path") or ranked[0][1].get("source_path") or "").strip()
    second_path = str(ranked[1][1].get("image_path") or ranked[1][1].get("source_path") or "").strip()
    first_urls = _image_to_data_url(first_path)
    second_urls = _image_to_data_url(second_path)
    if not query_urls or not first_urls or not second_urls:
        return ranked[:top_k]

    cache_key = f"{query}::{query_image_path}::{first_path}::{second_path}"
    if cache_key not in _QUERY_IMAGE_TIEBREAK_CACHE:
        llm_client = build_llm_client(settings.vlm)
        prompt = (
            "Image A is the query document page. Candidate 1 and Candidate 2 are retrieved document pages. "
            f"Question: {query} "
            "Which candidate page is the same page as Image A, or visually/layout-wise the best match for answering the question? "
            "Reply with only `1` or `2`."
        )
        messages = [
            {"role": "system", "content": "You compare document pages and choose the best matching candidate."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Image A (query page)"},
                    {"type": "image_url", "image_url": {"url": query_urls[0]}},
                    {"type": "text", "text": "Candidate 1"},
                    {"type": "image_url", "image_url": {"url": first_urls[0]}},
                    {"type": "text", "text": "Candidate 2"},
                    {"type": "image_url", "image_url": {"url": second_urls[0]}},
                ],
            },
        ]
        try:
            answer = asyncio.run(llm_client.chat(messages=messages, temperature=0.0, max_tokens=8))
        except Exception:
            return ranked[:top_k]
        match = re.search(r"\b([12])\b", answer or "")
        _QUERY_IMAGE_TIEBREAK_CACHE[cache_key] = int(match.group(1)) if match else 1

    if _QUERY_IMAGE_TIEBREAK_CACHE.get(cache_key) == 2:
        reranked = list(ranked)
        reranked[0], reranked[1] = reranked[1], reranked[0]
        return reranked[:top_k]
    return ranked[:top_k]


def query_local_index(
    project_root: Path,
    config_path: Path,
    query: str,
    top_k: int,
    retrieval_cfg: RetrievalConfig | None = None,
    query_image_path: str | None = None,
) -> list[dict[str, Any]]:
    cfg = read_yaml(config_path)
    index_cfg = cfg.get("indexing", {})
    text_dir = project_root / index_cfg.get("text_index_dir", "data/processed/indexes/text")
    vision_dir = project_root / index_cfg.get("vision_index_dir", "data/processed/indexes/vision")
    doc_store = _load_json(text_dir / "doc_store.json")
    df = _load_json(text_dir / "document_frequency.json")
    n_docs = max(1, len(doc_store))
    idf = {term: math.log((n_docs + 1) / (int(freq) + 1)) + 1.0 for term, freq in df.items()}
    retrieval_cfg = retrieval_cfg or settings.retrieval
    scored = rank_sparse_chunks(
        query=query,
        doc_store=doc_store,
        idf=idf,
        limit=max(top_k, retrieval_cfg.dense_rerank_pool_size if retrieval_cfg.dense_rerank else top_k),
        score_threshold=max(0.0, retrieval_cfg.score_threshold),
        rerank=retrieval_cfg.rerank,
        query_type_aware_rerank=retrieval_cfg.query_type_aware_rerank,
        rerank_profile=retrieval_cfg.rerank_profile,
        rerank_pool_size=retrieval_cfg.rerank_pool_size,
        diversify_results=retrieval_cfg.diversify_results,
        fingerprint_duplicate_penalty=retrieval_cfg.fingerprint_duplicate_penalty,
        docpage_duplicate_penalty=retrieval_cfg.docpage_duplicate_penalty,
        same_sample_penalty=retrieval_cfg.same_sample_penalty,
    )

    if retrieval_cfg.visual_fusion:
        visual_path = vision_dir / "visual_store.json"
        if visual_path.exists():
            visual_store = _load_json(visual_path)
            if isinstance(visual_store, list):
                visual_ranked = rank_visual_chunks(
                    query=query,
                    doc_store=doc_store,
                    visual_store=[item for item in visual_store if isinstance(item, dict)],
                    idf=idf,
                    limit=max(top_k, retrieval_cfg.visual_pool_size),
                    chart_table_specialist=retrieval_cfg.chart_table_specialist,
                    chart_table_visual_boost=retrieval_cfg.chart_table_visual_boost,
                )
                visual_weight = retrieval_cfg.visual_fusion_weight
                if retrieval_cfg.chart_table_specialist and ("chart" in query.lower() or "graph" in query.lower() or "figure" in query.lower() or "diagram" in query.lower() or "per 1000" in query.lower() or "table" in query.lower() or "committee" in query.lower() or "meeting" in query.lower() or "attendance" in query.lower() or "present" in query.lower() or "strength" in query.lower()):
                    visual_weight += retrieval_cfg.chart_table_visual_boost
                scored = late_fuse_ranked_chunks(
                    text_ranked=scored[: max(top_k, retrieval_cfg.rerank_pool_size)],
                    visual_ranked=visual_ranked,
                    limit=top_k,
                    text_weight=retrieval_cfg.text_fusion_weight,
                    visual_weight=visual_weight,
                    fusion_k=max(1, retrieval_cfg.fusion_k),
                )
    if retrieval_cfg.visual_dense_fusion:
        metadata_path = retrieval_cfg.visual_dense_metadata_path
        vectors_path = retrieval_cfg.visual_dense_vectors_path
        if metadata_path.exists() and vectors_path.exists():
            metadata_rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            vectors_payload = _load_json(vectors_path)
            vector_rows = vectors_payload.get("vectors", vectors_payload) if isinstance(vectors_payload, dict) else vectors_payload
            if isinstance(vector_rows, list) and len(vector_rows) == len(metadata_rows):
                try:
                    query_vector = asyncio.run(build_embedding_client(settings.text_embedding).embed([query]))[0]
                except Exception:
                    query_vector = []
                if query_vector:
                    ranked_visual_dense: list[tuple[float, dict[str, Any]]] = []
                    for item, vector in zip(metadata_rows, vector_rows, strict=True):
                        chunk = doc_store.get(str(item.get("chunk_id") or ""))
                        if not chunk or not isinstance(vector, list):
                            continue
                        score = _cosine_similarity(query_vector, [float(v) for v in vector])
                        ranked_visual_dense.append((score, chunk))
                    ranked_visual_dense.sort(key=lambda pair: pair[0], reverse=True)
                    scored = late_fuse_ranked_chunks(
                        text_ranked=scored[: max(top_k, retrieval_cfg.rerank_pool_size)],
                        visual_ranked=ranked_visual_dense[: max(top_k, retrieval_cfg.visual_dense_pool_size)],
                        limit=top_k,
                        text_weight=retrieval_cfg.text_fusion_weight,
                        visual_weight=retrieval_cfg.visual_dense_weight,
                        fusion_k=max(1, retrieval_cfg.fusion_k),
                    )
    if retrieval_cfg.query_image_aware_rerank:
        scored = _fuse_query_image_ranked_chunks(
            query=query,
            query_image_path=query_image_path,
            ranked_text=scored,
            doc_store=doc_store,
            top_k=top_k,
            retrieval_cfg=retrieval_cfg,
        )
        scored = _maybe_query_image_vlm_tiebreak(query, query_image_path, scored, top_k)
    if retrieval_cfg.chart_table_specialist:
        scored = rerank_chart_table_candidates(query, scored, top_k)

    return [
        {
            "score": round(score, 6),
            "chunk_id": chunk.get("chunk_id"),
            "sample_id": chunk.get("sample_id"),
            "dataset": chunk.get("dataset"),
            "split": chunk.get("split"),
            "chunk_type": chunk.get("chunk_type"),
            "source_ref": chunk.get("source_ref"),
            "page_no": chunk.get("page_no"),
            "bbox": chunk.get("bbox"),
            "image_path": chunk.get("image_path"),
            "snippet": str(chunk.get("text", ""))[:300],
        }
        for score, chunk in scored[:top_k]
    ]


def _image_to_data_url(path: str | None) -> list[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    mime = mimetypes.guess_type(file_path.name)[0] or "image/png"
    payload = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return [f"data:{mime};base64,{payload}"]


def _citation_source(citation: dict[str, Any]) -> str:
    value = citation.get("source") or citation.get("source_ref") or ""
    return str(value)


def _sample_relevance_flags(citations: list[dict[str, Any]], expected_source_prefix: str) -> list[bool]:
    return [_citation_source(citation).startswith(expected_source_prefix) for citation in citations]


def _result_doc_page_key(item: dict[str, Any], sample_to_docpage: dict[str, str]) -> str | None:
    sample_id = str(item.get("sample_id") or "")
    if sample_id and sample_id in sample_to_docpage:
        return sample_to_docpage[sample_id]
    source_ref = str(item.get("source_ref") or item.get("source") or "")
    sample_id = source_ref.split("/", 2)[2].split("#", 1)[0] if source_ref.startswith("docvqa/val/") else ""
    if sample_id and sample_id in sample_to_docpage:
        return sample_to_docpage[sample_id]
    return _chunk_doc_page_key(item)


def _doc_page_relevance_flags_from_results(
    results: list[dict[str, Any]],
    expected_doc_page_key: str | None,
    sample_to_docpage: dict[str, str],
) -> list[bool]:
    if not expected_doc_page_key:
        return [False for _ in results]
    return [(_result_doc_page_key(item, sample_to_docpage) == expected_doc_page_key) for item in results]


def _doc_page_relevance_flags_from_citations(
    citations: list[dict[str, Any]],
    expected_doc_page_key: str | None,
    sample_to_docpage: dict[str, str],
) -> list[bool]:
    if not expected_doc_page_key:
        return [False for _ in citations]
    return [(_result_doc_page_key(item, sample_to_docpage) == expected_doc_page_key) for item in citations]


def run_retrieval_eval(
    project_root: Path,
    config_path: Path,
    samples: list[EvalSample],
    top_k: int,
    retrieval_cfg: RetrievalConfig | None = None,
    match_granularity: str = "sample",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for sample in samples:
        results = query_local_index(
            project_root,
            config_path,
            sample.question,
            top_k,
            retrieval_cfg=retrieval_cfg,
            query_image_path=sample.image_path,
        )
        sample_flags = [str(item.get("source_ref", "")).startswith(sample.expected_source_prefix) for item in results]
        sample_to_docpage = {item.sample_id: (item.expected_doc_page_key or "") for item in samples}
        doc_page_flags = _doc_page_relevance_flags_from_results(results, sample.expected_doc_page_key, sample_to_docpage)
        flags = doc_page_flags if match_granularity == "doc_page" else sample_flags
        record = {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "split": sample.split,
            "question": sample.question,
            "answers": sample.answers,
            "mode": "retrieval",
            "top_k": top_k,
            "results": results,
            "metrics": {
                "hit_at_k": hit_at_k(flags),
                "recall_at_k": recall_at_k(flags),
                "precision_at_k": precision_at_k(flags),
                "citation_accuracy": citation_accuracy(flags[0] if flags else False),
                "sample_hit_at_k": hit_at_k(sample_flags),
                "sample_citation_accuracy": citation_accuracy(sample_flags[0] if sample_flags else False),
                "doc_page_hit_at_k": hit_at_k(doc_page_flags),
                "doc_page_citation_accuracy": citation_accuracy(doc_page_flags[0] if doc_page_flags else False),
            },
        }
        records.append(record)

    return records, summarize_records(records)


def run_rag_eval(
    samples: list[EvalSample],
    top_k: int,
    api_base: str,
    temperature: float,
    max_tokens: int,
    include_query_images: bool = False,
    match_granularity: str = "sample",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with httpx.Client(timeout=300.0) as client:
        sample_to_docpage = {item.sample_id: (item.expected_doc_page_key or "") for item in samples}
        for sample in samples:
            payload = {
                "query": sample.question,
                "context": [],
                "image_data_urls": _image_to_data_url(sample.image_path) if include_query_images else [],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            response: httpx.Response | None = None
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    response = client.post(f"{api_base.rstrip('/')}/chat", json=payload)
                    if response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"Server error {response.status_code} during rag eval",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    break
                except (httpx.HTTPError, httpx.TransportError) as exc:
                    last_error = exc
                    if attempt >= 2:
                        raise
                    time.sleep(2.0 * (attempt + 1))
            if response is None:
                raise RuntimeError(f"RAG evaluation request failed without response: {last_error}")
            body = response.json()
            citations = body.get("citations", [])[:top_k]
            sample_flags = _sample_relevance_flags(citations, sample.expected_source_prefix)
            doc_page_flags = _doc_page_relevance_flags_from_citations(citations, sample.expected_doc_page_key, sample_to_docpage)
            flags = doc_page_flags if match_granularity == "doc_page" else sample_flags
            answer = str(body.get("answer", ""))
            record = {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "split": sample.split,
                "question": sample.question,
                "answers": sample.answers,
                "mode": "rag",
                "top_k": top_k,
                "answer": answer,
                "citations": citations,
                "model": body.get("model"),
                "metrics": {
                    "hit_at_k": hit_at_k(flags),
                    "recall_at_k": recall_at_k(flags),
                    "precision_at_k": precision_at_k(flags),
                    "citation_accuracy": citation_accuracy(flags[0] if flags else False),
                    "sample_hit_at_k": hit_at_k(sample_flags),
                    "sample_citation_accuracy": citation_accuracy(sample_flags[0] if sample_flags else False),
                    "doc_page_hit_at_k": hit_at_k(doc_page_flags),
                    "doc_page_citation_accuracy": citation_accuracy(doc_page_flags[0] if doc_page_flags else False),
                    "exact_match": exact_match(answer, sample.answers),
                    "answer_contains": answer_contains(answer, sample.answers),
                    "token_f1": token_f1(answer, sample.answers),
                    "anls": anls(answer, sample.answers),
                },
            }
            records.append(record)

    return records, summarize_records(records)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_samples": 0, "overall": {}, "by_dataset": {}}

    metric_names = sorted(records[0]["metrics"].keys())
    overall = {
        metric: round(mean(float(record["metrics"].get(metric, 0.0)) for record in records), 6)
        for metric in metric_names
    }

    by_dataset: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"{record['dataset']}:{record['split']}"
        grouped.setdefault(key, []).append(record)

    for key, items in grouped.items():
        by_dataset[key] = {
            metric: round(mean(float(item["metrics"].get(metric, 0.0)) for item in items), 6)
            for metric in metric_names
        }

    return {
        "num_samples": len(records),
        "overall": overall,
        "by_dataset": by_dataset,
    }


def write_eval_outputs(output_dir: Path, run_name: str, records: list[dict[str, Any]], summary: dict[str, Any], meta: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{run_name}.jsonl"
    json_path = output_dir / f"{run_name}.summary.json"
    md_path = output_dir / f"{run_name}.summary.md"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    payload = {
        "meta": meta,
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Evaluation Summary: {run_name}",
        "",
        "## Meta",
        f"- mode: {meta['mode']}",
        f"- datasets: {', '.join(meta['datasets'])}",
        f"- splits: {', '.join(meta['splits'])}",
        f"- top_k: {meta['top_k']}",
        f"- samples: {summary['num_samples']}",
        "",
        "## Overall",
    ]
    for metric, value in summary["overall"].items():
        lines.append(f"- {metric}: {value:.6f}")
    lines.append("")
    lines.append("## By Dataset")
    for key, metrics in summary["by_dataset"].items():
        lines.append(f"- {key}")
        for metric, value in metrics.items():
            lines.append(f"  - {metric}: {value:.6f}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
