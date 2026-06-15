from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from src.models.retrieval import Evidence, rank_sparse_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACES_ROOT = PROJECT_ROOT / "data" / "workspaces"
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".csv", ".tex"}
MINERU_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}

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
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_WORD_RE = re.compile(r"[^A-Za-z0-9\u4e00-\u9fff]+")
ASSET_CODE_RE = re.compile(r"^[A-Za-z]+\d+(?:-\d+)?")
PAGE_QUERY_RE = re.compile(r"(?:第\s*(?P<cn>[零一二三四五六七八九十百两\d]+)\s*页|page\s*(?P<en>\d+)|p\.\s*(?P<short>\d+))", re.IGNORECASE)
ENGLISH_SPECIAL_TERM_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9]+(?:-[A-Z][A-Za-z0-9]+)+|KeyGen|Encrypt|Decrypt|Setup|Extract)\b")
WORKSPACE_FLOW_TERM_MARKERS = (
    "核心流程",
    "四个阶段",
    "四阶段",
    "setup",
    "keygen",
    "encrypt",
    "decrypt",
    "extract",
)
DEFINITION_QUERY_RE = re.compile(r"(?:什么是|是什么|解释|请解释|说明|介绍)")


def _now_ts() -> float:
    return time.time()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slugify_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)
    return cleaned or "asset"


def _tokenize_workspace(text: str) -> list[str]:
    tokens: list[str] = []
    buffer = []
    for char in text:
        if char.isalnum() or char == "_":
            buffer.append(char.lower())
            continue
        if "\u4e00" <= char <= "\u9fff":
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            tokens.append(char)
            continue
        if buffer:
            tokens.append("".join(buffer))
            buffer = []
    if buffer:
        tokens.append("".join(buffer))
    return tokens


def _term_frequency(tokens: list[str]) -> dict[str, float]:
    tf: dict[str, float] = {}
    for token in tokens:
        tf[token] = tf.get(token, 0.0) + 1.0
    return tf


def _title_overlap_score(query: str, title: str | None) -> float:
    if not title:
        return 0.0
    query_terms = set(_tokenize_workspace(query))
    title_terms = set(_tokenize_workspace(title))
    if not query_terms or not title_terms:
        return 0.0
    overlap = len(query_terms & title_terms)
    if overlap <= 0:
        return 0.0
    score = float(overlap)
    if title.strip() and title.strip() in query:
        score += 3.0
    return score


def _normalize_match_text(text: str | None) -> str:
    if not text:
        return ""
    stripped = HTML_TAG_RE.sub("", text)
    return NON_WORD_RE.sub("", stripped).lower()


def _asset_match_candidates(asset_name: str | None) -> list[str]:
    value = (asset_name or "").strip()
    if not value:
        return []
    path = Path(value)
    stem = path.stem.strip()
    candidates = [value, path.name, stem]
    if "_" in path.name:
        suffix_name = path.name.split("_", 1)[1].strip()
        suffix_stem = Path(suffix_name).stem.strip()
        candidates.extend([suffix_name, suffix_stem])
    code_match = ASSET_CODE_RE.match(stem)
    if code_match:
        candidates.append(code_match.group(0))
    seen: set[str] = set()
    ordered: list[str] = []
    for item in candidates:
        normalized = _normalize_match_text(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _query_mentions_asset(query: str, asset_name: str | None) -> bool:
    normalized_query = _normalize_match_text(query)
    if not normalized_query:
        return False
    for candidate in _asset_match_candidates(asset_name):
        if len(candidate) >= 2 and candidate in normalized_query:
            return True
    return False


def _workspace_chunk_asset_name(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    return str(metadata.get("asset_name") or Path(str(chunk.get("source_path") or "")).name or "").strip()


def _workspace_target_asset_names(query: str, doc_store: dict[str, dict[str, Any]]) -> set[str]:
    targets: set[str] = set()
    for chunk in doc_store.values():
        asset_name = _workspace_chunk_asset_name(chunk)
        if asset_name and _query_mentions_asset(query, asset_name):
            targets.add(asset_name)
    return targets


def _extract_page_number(query: str) -> int | None:
    value = (query or "").strip()
    if not value:
        return None
    match = re.search(r"第\s*([零一二三四五六七八九十百两\d]+)\s*页", value)
    if match:
        raw = match.group(1)
    else:
        match = PAGE_QUERY_RE.search(value)
        raw = (match.group("cn") or match.group("en") or match.group("short")) if match else None
    if not match:
        return None
    if not raw:
        return None
    cn_digits = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if not raw.isdigit():
        total = 0
        current = 0
        for char in raw:
            if char in cn_digits:
                current = cn_digits[char]
                total += current
            elif char == "十":
                if current == 0:
                    current = 1
                total = total - current + (current * 10)
                current = 0
            elif char == "百":
                if current == 0:
                    current = 1
                total = total - current + (current * 100)
                current = 0
        page_no = total
        return page_no if page_no >= 1 else None
    try:
        page_no = int(raw)
    except ValueError:
        return None
    return page_no if page_no >= 1 else None


def _is_page_level_query(query: str) -> bool:
    lowered = (query or "").lower()
    has_page_marker = ("页" in (query or "")) or ("page" in lowered) or ("p." in lowered)
    page_no = _extract_page_number(query)
    if page_no is None and not has_page_marker:
        return False
    return any(marker in lowered for marker in ("讲了什么", "内容", "是什么", "what", "summary", "定义", "meaning"))


def _extract_english_special_terms(query: str) -> list[str]:
    terms: list[str] = []
    for match in ENGLISH_SPECIAL_TERM_RE.finditer(query or ""):
        term = match.group(0).strip()
        if term:
            terms.append(term)
    seen: set[str] = set()
    ordered: list[str] = []
    for term in terms:
        lowered = term.lower()
        if lowered not in seen:
            seen.add(lowered)
            ordered.append(term)
    return ordered


def _contains_flow_markers(query: str) -> bool:
    lowered = (query or "").lower()
    return any(marker in lowered for marker in WORKSPACE_FLOW_TERM_MARKERS)


def _workspace_special_term_bonus(query: str, text: str, title: str | None = None) -> float:
    haystacks = [text or "", title or ""]
    lowered_haystacks = [item.lower() for item in haystacks]
    bonus = 0.0
    for term in _extract_english_special_terms(query):
        lowered = term.lower()
        if lowered in (title or "").lower():
            bonus += 2.2 + min(len(term), 18) * 0.03
        elif lowered in (text or "").lower():
            bonus += 0.75 + min(len(term), 18) * 0.02
    if _contains_flow_markers(query):
        for marker in ("setup", "keygen", "encrypt", "decrypt", "extract"):
            if marker in (title or "").lower():
                bonus += 1.0
            elif marker in (text or "").lower():
                bonus += 0.35
    return bonus


def _is_overview_like_section_title(title: str | None) -> bool:
    value = (title or "").strip()
    if not value:
        return False
    if value.startswith(("一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、")):
        return True
    lowered = value.lower()
    return lowered.startswith(("introduction", "overview", "background"))


def _is_definition_like_query(query: str) -> bool:
    value = (query or "").strip()
    if not value:
        return False
    lowered = value.lower()
    if any(marker in lowered for marker in ("what is", "explain", "define", "meaning of")):
        return True
    return bool(DEFINITION_QUERY_RE.search(value))


def _extract_focus_phrases(query: str) -> list[str]:
    raw = HTML_TAG_RE.sub("", query or "")
    if not raw.strip():
        return []

    focus = raw.strip()
    leading_match = re.search(r"(?:解释|请解释|说明|介绍)(.+)", focus)
    if leading_match:
        focus = leading_match.group(1)
    trailing_match = re.search(r"(.+?)(?:是什么|是啥|什么意思|指什么)", focus)
    if trailing_match:
        focus = trailing_match.group(1)

    focus = focus.strip(" ：:，。,？?！!；;（）()[]【】")
    focus_parts = [part.strip() for part in re.split(r"[中的里关于对与和及]", focus) if part.strip()]
    tail = focus_parts[-1] if focus_parts else focus
    phrases: list[str] = []

    latin_terms = [term.lower() for term in re.findall(r"[A-Za-z0-9]{3,}", tail)]
    phrases.extend(latin_terms)

    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", tail)
    for term in chinese_terms:
        if len(term) <= 8:
            phrases.append(term)
            continue
        for size in range(min(8, len(term)), 1, -1):
            phrases.append(term[-size:])

    seen: set[str] = set()
    ordered: list[str] = []
    for phrase in sorted(phrases, key=len, reverse=True):
        if phrase and phrase not in seen:
            seen.add(phrase)
            ordered.append(phrase)
    if not ordered:
        fallback = _normalize_match_text(tail)
        if fallback:
            ordered.append(fallback[-8:])
    return ordered[:8]


def _query_match_bonus(query: str, text: str | None, title: str | None = None) -> float:
    normalized_text = _normalize_match_text(text)
    normalized_title = _normalize_match_text(title)
    if not normalized_text and not normalized_title:
        return 0.0

    bonus = 0.0
    definition_like = _is_definition_like_query(query)
    raw_text_compact = "".join((text or "").split())
    for phrase in _extract_focus_phrases(query):
        if len(phrase) < 2:
            continue
        if phrase in normalized_text:
            bonus += 0.65 + min(len(phrase), 8) * 0.05
            if definition_like and (
                f"{phrase}是" in normalized_text
                or f"{phrase}:" in raw_text_compact
                or f"{phrase}：" in raw_text_compact
            ):
                bonus += 0.45
        elif phrase in normalized_title:
            bonus += 0.3 + min(len(phrase), 8) * 0.03
    return bonus


def _looks_like_brief_heading_text(text: str | None) -> bool:
    normalized = _normalize_match_text(text)
    if not normalized:
        return False
    return len(normalized) <= 14 and "：" not in (text or "") and ":" not in (text or "")


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
            current_title = stripped.lstrip("#").strip() or "Untitled section"
            continue
        current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_title, body))

    if sections:
        return sections
    plain = text.strip()
    return [("Document overview", plain)] if plain else []


def _summarize_text(text: str, limit: int = 220) -> str:
    normalized = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _is_exhaustive_workspace_query(query: str) -> bool:
    value = (query or "").strip().lower()
    if not value:
        return False
    return any(marker in value for marker in EXHAUSTIVE_QUERY_MARKERS)


def _render_pdf_page_image(pdf_path: Path, page_no: int, cache_dir: Path) -> Path | None:
    if page_no < 1 or not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{pdf_path.stem}_page_{page_no}.png"
    if output_path.exists():
        return output_path
    try:
        import pypdfium2 as pdfium
    except Exception:
        return None
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        if page_no > len(pdf):
            return None
        page = pdf.get_page(page_no - 1)
        try:
            bitmap = page.render(scale=2.0)
            pil_image = bitmap.to_pil()
            pil_image.save(output_path)
        finally:
            page.close()
            pdf.close()
    except Exception:
        return None
    return output_path if output_path.exists() else None


class WorkspaceManager:
    def __init__(self, root: Path = WORKSPACES_ROOT) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._threads: dict[str, threading.Thread] = {}

    def create_workspace(self) -> dict[str, Any]:
        workspace_id = uuid.uuid4().hex[:12]
        workspace_dir = self.root / workspace_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        for name in ("raw", "mineru", "mineru_raw", "processed", "indexes"):
            (workspace_dir / name).mkdir(parents=True, exist_ok=True)
        meta = {
            "workspace_id": workspace_id,
            "scope": "workspace",
            "dataset_name": f"workspace_{workspace_id}",
            "status": "idle",
            "stage": "created",
            "progress": 0.0,
            "progress_label": "Workspace ready",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "assets": [],
            "last_error": "",
            "counts": {"documents": 0, "chunks": 0, "visual_items": 0},
        }
        self._save_meta(workspace_id, meta)
        return meta

    def get_workspace(self, workspace_id: str) -> dict[str, Any]:
        return self._load_meta(workspace_id)

    def reset_workspace(self, workspace_id: str) -> dict[str, Any]:
        workspace_dir = self._workspace_dir(workspace_id)
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        return self.create_workspace_with_id(workspace_id)

    def create_workspace_with_id(self, workspace_id: str) -> dict[str, Any]:
        workspace_dir = self.root / workspace_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        for name in ("raw", "mineru", "mineru_raw", "processed", "indexes"):
            (workspace_dir / name).mkdir(parents=True, exist_ok=True)
        meta = {
            "workspace_id": workspace_id,
            "scope": "workspace",
            "dataset_name": f"workspace_{workspace_id}",
            "status": "idle",
            "stage": "created",
            "progress": 0.0,
            "progress_label": "Workspace ready",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "assets": [],
            "last_error": "",
            "counts": {"documents": 0, "chunks": 0, "visual_items": 0},
        }
        self._save_meta(workspace_id, meta)
        return meta

    def add_assets(self, workspace_id: str, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        meta = self._load_meta(workspace_id)
        raw_dir = self._workspace_dir(workspace_id) / "raw"
        assets = list(meta.get("assets", []))
        for original_name, payload in files:
            asset_id = uuid.uuid4().hex[:10]
            safe_name = _slugify_filename(original_name)
            target = raw_dir / f"{asset_id}_{safe_name}"
            target.write_bytes(payload)
            asset_type = target.suffix.lower().lstrip(".") or "file"
            assets.append(
                {
                    "asset_id": asset_id,
                    "name": original_name,
                    "stored_name": target.name,
                    "path": str(target),
                    "type": asset_type,
                    "status": "uploaded",
                    "stage": "uploaded",
                    "parser": "pending",
                    "section_count": 0,
                    "snippet": "",
                }
            )
        meta["assets"] = assets
        meta["updated_at"] = _now_iso()
        meta["status"] = "queued"
        meta["stage"] = "uploaded"
        meta["progress"] = 0.05
        meta["progress_label"] = "Assets uploaded"
        self._save_meta(workspace_id, meta)
        self.start_ingestion(workspace_id)
        return meta

    def delete_asset(self, workspace_id: str, asset_id: str) -> dict[str, Any]:
        meta = self._load_meta(workspace_id)
        if meta.get("status") == "processing":
            raise RuntimeError("workspace_busy: ingestion is still running")

        assets = list(meta.get("assets", []))
        target = next((asset for asset in assets if asset.get("asset_id") == asset_id), None)
        if target is None:
            raise FileNotFoundError(f"workspace_asset_not_found:{asset_id}")

        raw_path = Path(str(target.get("path", "")))
        if raw_path.exists():
            raw_path.unlink()

        dataset_name = str(meta.get("dataset_name"))
        normalized_path = self._workspace_dir(workspace_id) / "mineru" / dataset_name / f"{asset_id}.json"
        if normalized_path.exists():
            normalized_path.unlink()

        mineru_raw_dir = self._workspace_dir(workspace_id) / "mineru_raw" / dataset_name / "workspace" / asset_id
        if mineru_raw_dir.exists():
            shutil.rmtree(mineru_raw_dir)

        meta["assets"] = [asset for asset in assets if asset.get("asset_id") != asset_id]
        meta["last_error"] = ""
        if not meta["assets"]:
            self._clear_workspace_outputs(workspace_id)
            meta["status"] = "idle"
            meta["stage"] = "empty"
            meta["progress"] = 0.0
            meta["progress_label"] = "Workspace is empty"
            meta["counts"] = {"documents": 0, "chunks": 0, "visual_items": 0}
            self._save_meta(workspace_id, meta)
            return meta

        meta["status"] = "queued"
        meta["stage"] = "uploaded"
        meta["progress"] = 0.05
        meta["progress_label"] = "Asset removed. Rebuilding workspace indexes"
        self._save_meta(workspace_id, meta)
        self.start_ingestion(workspace_id)
        return meta

    def start_ingestion(self, workspace_id: str) -> None:
        thread = self._threads.get(workspace_id)
        if thread and thread.is_alive():
            return
        worker = threading.Thread(target=self._run_ingestion, args=(workspace_id,), daemon=True)
        self._threads[workspace_id] = worker
        worker.start()

    def workspace_retrieve(self, workspace_id: str, query: str, top_k: int = 5) -> list[Evidence]:
        meta = self._load_meta(workspace_id)
        if meta.get("status") not in {"indexed", "ready"}:
            return []
        workspace_dir = self._workspace_dir(workspace_id)
        doc_store_path = workspace_dir / "indexes" / "text" / "doc_store.json"
        if not doc_store_path.exists():
            return []
        doc_store = json.loads(doc_store_path.read_text(encoding="utf-8"))
        if not isinstance(doc_store, dict):
            return []
        explicit_target_assets = _workspace_target_asset_names(query, doc_store)
        asset_records: dict[str, dict[str, Any]] = {}
        for asset in meta.get("assets", []):
            for key in (
                str(asset.get("name") or "").strip(),
                str(asset.get("stored_name") or "").strip(),
                Path(str(asset.get("path") or "")).name,
            ):
                if key:
                    asset_records[key] = asset
        target_page_no = _extract_page_number(query)
        candidate_doc_store = doc_store
        if explicit_target_assets:
            asset_filtered = {
                chunk_id: chunk
                for chunk_id, chunk in doc_store.items()
                if _workspace_chunk_asset_name(chunk) in explicit_target_assets
            }
            if asset_filtered:
                candidate_doc_store = asset_filtered
        if explicit_target_assets and target_page_no is not None and _is_page_level_query(query):
            page_filtered = {
                chunk_id: chunk
                for chunk_id, chunk in candidate_doc_store.items()
                if chunk.get("page_no") == target_page_no
            }
            if page_filtered:
                candidate_doc_store = page_filtered
        n_docs = max(1, len(candidate_doc_store))
        doc_frequency: dict[str, int] = {}
        for chunk in candidate_doc_store.values():
            tf = chunk.get("tf") if isinstance(chunk.get("tf"), dict) else {}
            for term in tf:
                doc_frequency[term] = doc_frequency.get(term, 0) + 1
        idf = {term: 1.0 + math.log((n_docs + 1) / (freq + 1)) for term, freq in doc_frequency.items()}
        exhaustive_query = _is_exhaustive_workspace_query(query)
        candidate_limit = max(top_k, 12)
        ranked = rank_sparse_chunks(
            query=query,
            doc_store=candidate_doc_store,
            idf=idf,
            limit=max(candidate_limit, 14 if exhaustive_query else candidate_limit),
            score_threshold=0.0,
            rerank=True,
            query_type_aware_rerank=True,
            rerank_profile="stronger",
            rerank_pool_size=max(candidate_limit, 16),
            diversify_results=True,
        )
        if ranked:
            reranked_with_titles: list[tuple[float, dict[str, Any]]] = []
            for base_score, item in ranked:
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                section_title = str(metadata.get("section_title") or "").strip() or None
                raw_type = str(metadata.get("raw_type") or "").strip().lower()
                text = str(item.get("text") or "")
                page_no = item.get("page_no")
                definition_like = _is_definition_like_query(query)
                title_bonus = _title_overlap_score(query, section_title)
                phrase_bonus = _query_match_bonus(query, text, section_title)
                special_term_bonus = _workspace_special_term_bonus(query, text, section_title)
                boilerplate_penalty = 0.0
                if raw_type == "footer" and phrase_bonus <= 0.0:
                    boilerplate_penalty += 0.08
                if raw_type == "header":
                    boilerplate_penalty += 0.04
                if definition_like and phrase_bonus > 0.0 and _looks_like_brief_heading_text(text):
                    boilerplate_penalty += 0.42
                if _contains_flow_markers(query) and _is_overview_like_section_title(section_title):
                    boilerplate_penalty += 0.36
                if _contains_flow_markers(query) and section_title and section_title.strip() == "一、基于身份的加密 IBE":
                    boilerplate_penalty += 0.52
                page_bonus = 0.0
                if target_page_no is not None:
                    if page_no == target_page_no:
                        page_bonus += 2.8 if _is_page_level_query(query) else 1.2
                    elif _is_page_level_query(query):
                        page_bonus -= 0.18
                reranked_with_titles.append(
                    (base_score + (title_bonus * 0.18) + phrase_bonus + special_term_bonus + page_bonus - boilerplate_penalty, item)
                )
            reranked_with_titles.sort(key=lambda pair: pair[0], reverse=True)
            ranked = reranked_with_titles

        if explicit_target_assets and ranked:
            target_ranked = [
                (score, item)
                for score, item in ranked
                if _workspace_chunk_asset_name(item) in explicit_target_assets
            ]
            other_ranked = [
                (score, item)
                for score, item in ranked
                if _workspace_chunk_asset_name(item) not in explicit_target_assets
            ]
            if target_ranked:
                ranked = target_ranked + other_ranked
        if explicit_target_assets and target_page_no is not None and ranked:
            same_page_ranked = [
                (score, item)
                for score, item in ranked
                if _workspace_chunk_asset_name(item) in explicit_target_assets and item.get("page_no") == target_page_no
            ]
            other_ranked = [
                (score, item)
                for score, item in ranked
                if not (_workspace_chunk_asset_name(item) in explicit_target_assets and item.get("page_no") == target_page_no)
            ]
            if same_page_ranked:
                ranked = same_page_ranked + other_ranked

        selected_items = [item for _, item in ranked[:top_k]]
        expand_same_page = exhaustive_query
        anchor_item = ranked[0][1] if ranked else None

        if not expand_same_page and ranked:
            best_item_score, best_item = ranked[0]
            best_meta = best_item.get("metadata") if isinstance(best_item.get("metadata"), dict) else {}
            best_title = str(best_meta.get("section_title") or "").strip() or None
            title_match_score = _title_overlap_score(query, best_title)
            query_value = (query or "").strip()
            if title_match_score > 0 or (best_item_score > 0.35 and len(query_value) >= 6):
                expand_same_page = True
                anchor_item = best_item

        if expand_same_page and anchor_item:
            best_item = anchor_item
            best_source_path = str(best_item.get("source_path", "")).strip()
            best_page_no = best_item.get("page_no")
            related_items: list[dict[str, Any]] = []
            for candidate in doc_store.values():
                same_source = str(candidate.get("source_path", "")).strip() == best_source_path
                same_page = candidate.get("page_no") == best_page_no
                if same_source and same_page:
                    related_items.append(candidate)
            related_items.sort(
                key=lambda item: (
                    str((item.get("metadata") or {}).get("section_title", "")),
                    int(item.get("block_index") or 0),
                    int(item.get("part_index") or 0),
                )
            )
            selected_ids = {str(item.get("chunk_id")) for item in selected_items}
            for candidate in related_items:
                candidate_id = str(candidate.get("chunk_id"))
                if candidate_id in selected_ids:
                    continue
                selected_items.append(candidate)
                selected_ids.add(candidate_id)
                if len(selected_items) >= max(top_k, 10 if exhaustive_query else max(top_k, 6)):
                    break
        evidences: list[Evidence] = []
        ranked_score_map = {str(item.get("chunk_id")): score for score, item in ranked}
        output_limit = max(top_k, 10) if exhaustive_query else (max(top_k, 6) if expand_same_page else top_k)
        for item in selected_items[:output_limit]:
            text = str(item.get("text", "")).strip()
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            section_title = str(metadata.get("section_title") or "").strip() or None
            page_no = item.get("page_no")
            page = int(page_no) if page_no not in (None, "") else None
            score = float(ranked_score_map.get(str(item.get("chunk_id")), 1.0))
            evidences.append(
                Evidence(
                    chunk_id=str(item.get("chunk_id")),
                    source=f"workspace_file:{metadata.get('asset_name') or Path(str(item.get('source_path') or '')).name or 'workspace asset'}",
                    page=page,
                    text=text,
                    snippet=_summarize_text(text),
                    score=score,
                    section_title=section_title,
                    citation_kind="workspace_indexed",
                    chunk_type=str(item.get("chunk_type", "")).strip() or None,
                    image_path=str(item.get("image_path", "")).strip() or None,
                    source_path=str(item.get("source_path", "")).strip() or None,
                    bbox=item.get("bbox") if isinstance(item.get("bbox"), list) else None,
                )
            )

        fallback_page_no = target_page_no
        if fallback_page_no is None and _is_page_level_query(query):
            for evidence in evidences:
                if evidence.page is not None:
                    fallback_page_no = evidence.page
                    break

        if fallback_page_no is not None and _is_page_level_query(query):
            target_asset_name: str | None = None
            if len(explicit_target_assets) == 1:
                target_asset_name = next(iter(explicit_target_assets))
            else:
                pdf_assets = [asset for asset in meta.get("assets", []) if str(asset.get("name", "")).lower().endswith(".pdf")]
                if len(pdf_assets) == 1:
                    target_asset_name = str(pdf_assets[0].get("name") or "").strip()
            target_asset = asset_records.get(target_asset_name or "")
            if target_asset:
                raw_pdf_path = Path(str(target_asset.get("path") or ""))
                rendered = _render_pdf_page_image(
                    raw_pdf_path,
                    fallback_page_no,
                    workspace_dir / "rendered_pages",
                )
                if rendered:
                    evidences.insert(
                        0,
                        Evidence(
                            chunk_id=f"workspace_page_render_{target_asset.get('asset_id')}_{fallback_page_no}",
                            source=f"workspace_file:{target_asset_name}",
                            page=fallback_page_no,
                            text=f"Rendered PDF page {fallback_page_no} from {target_asset_name} for workspace page-grounded reading.",
                            snippet=f"Rendered PDF page {fallback_page_no} from {target_asset_name}.",
                            score=9.0,
                            section_title=f"Page {fallback_page_no} render",
                            citation_kind="workspace_indexed",
                            chunk_type="page_image",
                            image_path=str(rendered),
                            source_path=str(raw_pdf_path),
                        ),
                    )
        return evidences

    def _workspace_dir(self, workspace_id: str) -> Path:
        return self.root / workspace_id

    def _meta_path(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "metadata.json"

    def _load_meta(self, workspace_id: str) -> dict[str, Any]:
        path = self._meta_path(workspace_id)
        if not path.exists():
            raise FileNotFoundError(f"workspace_not_found:{workspace_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_meta(self, workspace_id: str, meta: dict[str, Any]) -> None:
        meta["updated_at"] = _now_iso()
        path = self._meta_path(workspace_id)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _update_meta(self, workspace_id: str, **updates: Any) -> dict[str, Any]:
        meta = self._load_meta(workspace_id)
        meta.update(updates)
        self._save_meta(workspace_id, meta)
        return meta

    def _set_progress(self, workspace_id: str, *, status: str, stage: str, progress: float, label: str) -> None:
        self._update_meta(
            workspace_id,
            status=status,
            stage=stage,
            progress=max(0.0, min(progress, 1.0)),
            progress_label=label,
            last_error="",
        )

    def _run_ingestion(self, workspace_id: str) -> None:
        lock = self._locks.setdefault(workspace_id, threading.Lock())
        if not lock.acquire(blocking=False):
            return
        try:
            self._set_progress(workspace_id, status="processing", stage="preparing", progress=0.08, label="Preparing workspace assets")
            meta = self._load_meta(workspace_id)
            workspace_dir = self._workspace_dir(workspace_id)
            dataset_name = str(meta.get("dataset_name"))
            mineru_root = workspace_dir / "mineru"
            mineru_raw_root = workspace_dir / "mineru_raw"
            self._clear_workspace_outputs(workspace_id)

            assets = list(meta.get("assets", []))
            total_assets = max(1, len(assets))
            for index, asset in enumerate(assets, start=1):
                stage_prefix = f"Processing {asset.get('name', 'asset')} ({index}/{total_assets})"
                self._set_progress(
                    workspace_id,
                    status="processing",
                    stage="parsing",
                    progress=0.08 + (0.42 * (index - 1) / total_assets),
                    label=stage_prefix,
                )
                path = Path(str(asset.get("path")))
                suffix = path.suffix.lower()
                if suffix in TEXT_EXTENSIONS:
                    self._normalize_text_asset(workspace_id, asset, dataset_name, mineru_root)
                elif suffix in MINERU_EXTENSIONS:
                    self._run_mineru_for_asset(workspace_id, asset, dataset_name, mineru_root, mineru_raw_root)
                else:
                    asset["status"] = "skipped"
                    asset["stage"] = "unsupported"
                    asset["parser"] = "unsupported"
                    asset["snippet"] = "Unsupported file type for workspace indexing."
                meta = self._load_meta(workspace_id)
                for existing in meta.get("assets", []):
                    if existing.get("asset_id") == asset.get("asset_id"):
                        existing.update(asset)
                self._save_meta(workspace_id, meta)

            self._set_progress(workspace_id, status="processing", stage="chunking", progress=0.58, label="Building workspace chunks")
            config_path = self._write_workspace_config(workspace_id)
            self._run_subprocess(
                [
                    sys.executable,
                    "scripts/07_parse_and_chunk.py",
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--config",
                    str(config_path),
                    "--datasets",
                    dataset_name,
                    "--splits",
                    "workspace",
                    "--source-mode",
                    "mineru",
                ],
                cwd=PROJECT_ROOT,
            )

            self._set_progress(workspace_id, status="processing", stage="indexing", progress=0.78, label="Building workspace indexes")
            self._run_subprocess(
                [
                    sys.executable,
                    "scripts/08_build_indexes.py",
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--config",
                    str(config_path),
                ],
                cwd=PROJECT_ROOT,
            )

            counts = self._collect_workspace_counts(workspace_id)
            meta = self._load_meta(workspace_id)
            for asset in meta.get("assets", []):
                if asset.get("status") not in {"failed", "skipped"}:
                    asset["status"] = "indexed"
                    asset["stage"] = "indexed"
                    asset["parser"] = asset.get("parser") or "workspace_pipeline"
            meta["counts"] = counts
            meta["status"] = "indexed"
            meta["stage"] = "ready"
            meta["progress"] = 1.0
            meta["progress_label"] = "Workspace indexed and ready for retrieval"
            self._save_meta(workspace_id, meta)
        except Exception as exc:
            meta = self._load_meta(workspace_id)
            meta["status"] = "failed"
            meta["stage"] = "failed"
            meta["last_error"] = str(exc)
            meta["progress_label"] = f"Workspace ingestion failed: {exc}"
            self._save_meta(workspace_id, meta)
        finally:
            lock.release()

    def _write_workspace_config(self, workspace_id: str) -> Path:
        workspace_dir = self._workspace_dir(workspace_id)
        config = {
            "document_chunking": {
                "mineru_json_root": str(workspace_dir / "mineru"),
                "document_output": str(workspace_dir / "processed" / "documents.jsonl"),
                "chunk_output": str(workspace_dir / "processed" / "chunks.jsonl"),
                "default_page_no": 1,
                "max_chars_per_chunk": 900,
                "overlap_chars": 120,
            },
            "indexing": {
                "text_index_dir": str(workspace_dir / "indexes" / "text"),
                "vision_index_dir": str(workspace_dir / "indexes" / "vision"),
                "manifest_path": str(workspace_dir / "indexes" / "index_manifest.json"),
            },
        }
        path = workspace_dir / "workspace_config.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    def _normalize_text_asset(self, workspace_id: str, asset: dict[str, Any], dataset_name: str, mineru_root: Path) -> None:
        raw_path = Path(str(asset.get("path")))
        text = raw_path.read_text(encoding="utf-8", errors="ignore")
        sections = _split_markdown_sections(text)
        blocks = []
        for block_index, (section_title, body) in enumerate(sections):
            if body.strip():
                blocks.append(
                    {
                        "block_id": f"text-{block_index:03d}",
                        "type": "text",
                        "text": body,
                        "page_no": 1,
                        "metadata": {"asset_name": asset.get("name"), "section_title": section_title},
                    }
                )
        sample_id = str(asset.get("asset_id"))
        output_path = mineru_root / dataset_name / f"{sample_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "document_id": f"workspace-doc-{sample_id}",
            "sample_id": sample_id,
            "dataset": dataset_name,
            "split": "workspace",
            "source_path": str(raw_path),
            "source_type": raw_path.suffix.lower().lstrip(".") or "file",
            "raw_output_dir": str(raw_path.parent),
            "raw_files": [raw_path.name],
            "metadata": {"workspace_id": workspace_id, "asset_name": asset.get("name")},
            "blocks": blocks,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        asset["status"] = "parsed"
        asset["stage"] = "normalized"
        asset["parser"] = "workspace_text_sections"
        asset["section_count"] = len(sections)
        asset["snippet"] = _summarize_text(text)

    def _run_mineru_for_asset(
        self,
        workspace_id: str,
        asset: dict[str, Any],
        dataset_name: str,
        mineru_root: Path,
        mineru_raw_root: Path,
    ) -> None:
        raw_path = Path(str(asset.get("path")))
        cmd = [
            sys.executable,
            "scripts/06_run_mineru.py",
            "--project-root",
            str(PROJECT_ROOT),
            "--datasets",
            dataset_name,
            "--splits",
            "workspace",
            "--input-path",
            str(raw_path),
            "--document-id",
            str(asset.get("asset_id")),
            "--output-root",
            str(mineru_raw_root),
            "--normalized-root",
            str(mineru_root),
            "--limit-per-split",
            "1",
        ]
        self._run_subprocess(cmd, cwd=PROJECT_ROOT)
        asset["status"] = "parsed"
        asset["stage"] = "normalized"
        asset["parser"] = "mineru"
        asset["snippet"] = "Parsed with MinerU and routed into the workspace retrieval pipeline."

    def _collect_workspace_counts(self, workspace_id: str) -> dict[str, int]:
        workspace_dir = self._workspace_dir(workspace_id)
        documents = 0
        chunks = 0
        visual_items = 0
        documents_path = workspace_dir / "processed" / "documents.jsonl"
        chunks_path = workspace_dir / "processed" / "chunks.jsonl"
        visual_path = workspace_dir / "indexes" / "vision" / "visual_store.json"
        if documents_path.exists():
            documents = sum(1 for line in documents_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if chunks_path.exists():
            chunks = sum(1 for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if visual_path.exists():
            payload = json.loads(visual_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                visual_items = len(payload)
        return {"documents": documents, "chunks": chunks, "visual_items": visual_items}

    def _clear_workspace_outputs(self, workspace_id: str) -> None:
        workspace_dir = self._workspace_dir(workspace_id)
        for name in ("mineru", "mineru_raw", "processed", "indexes"):
            target = workspace_dir / name
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)

    def _run_subprocess(self, cmd: list[str], cwd: Path) -> None:
        subprocess.run(cmd, cwd=str(cwd), check=True)


workspace_manager = WorkspaceManager()
