from __future__ import annotations

import json
import math
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
        n_docs = max(1, len(doc_store))
        doc_frequency: dict[str, int] = {}
        for chunk in doc_store.values():
            tf = chunk.get("tf") if isinstance(chunk.get("tf"), dict) else {}
            for term in tf:
                doc_frequency[term] = doc_frequency.get(term, 0) + 1
        idf = {term: 1.0 + math.log((n_docs + 1) / (freq + 1)) for term, freq in doc_frequency.items()}
        ranked = rank_sparse_chunks(
            query=query,
            doc_store=doc_store,
            idf=idf,
            limit=top_k,
            score_threshold=0.0,
            rerank=True,
            query_type_aware_rerank=True,
            rerank_profile="stronger",
            rerank_pool_size=max(top_k, 12),
            diversify_results=True,
        )
        evidences: list[Evidence] = []
        for score, item in ranked[:top_k]:
            text = str(item.get("text", "")).strip()
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            section_title = str(metadata.get("section_title") or "").strip() or None
            page_no = item.get("page_no")
            page = int(page_no) if page_no not in (None, "") else None
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
