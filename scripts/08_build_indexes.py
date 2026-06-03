from __future__ import annotations

import argparse
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_index_utils import read_jsonl, read_yaml, stable_id, term_frequency, tokenize, write_json


def _dataset_version(documents: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> str:
    parts = [
        f"documents={len(documents)}",
        f"chunks={len(chunks)}",
        ",".join(sorted({str(x.get("dataset")) for x in documents})),
        ",".join(sorted({str(x.get("split")) for x in documents})),
    ]
    return stable_id(*parts, length=20)


def build_indexes(project_root: Path, config_path: Path) -> dict[str, Any]:
    cfg = read_yaml(config_path)
    chunk_cfg = cfg.get("document_chunking", {})
    index_cfg = cfg.get("indexing", {})

    document_path = project_root / chunk_cfg.get("document_output", "data/processed/documents/documents.jsonl")
    chunk_path = project_root / chunk_cfg.get("chunk_output", "data/processed/chunks/chunks.jsonl")
    text_dir = project_root / index_cfg.get("text_index_dir", "data/processed/indexes/text")
    vision_dir = project_root / index_cfg.get("vision_index_dir", "data/processed/indexes/vision")
    manifest_path = project_root / index_cfg.get("manifest_path", "data/processed/indexes/index_manifest.json")

    documents = read_jsonl(document_path)
    chunks = read_jsonl(chunk_path)
    if not chunks:
        raise ValueError(f"No chunks found at {chunk_path}. Run scripts/07_parse_and_chunk.py first.")

    doc_store: dict[str, dict[str, Any]] = {}
    postings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    document_frequency: Counter[str] = Counter()
    visual_items: list[dict[str, Any]] = []

    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        tokens = tokenize(str(chunk.get("text", "")))
        tf = term_frequency(tokens)
        doc_store[chunk_id] = {
            "chunk_id": chunk_id,
            "document_id": chunk.get("document_id"),
            "sample_id": chunk.get("sample_id"),
            "dataset": chunk.get("dataset"),
            "split": chunk.get("split"),
            "chunk_type": chunk.get("chunk_type"),
            "text": chunk.get("text", ""),
            "page_no": chunk.get("page_no"),
            "bbox": chunk.get("bbox"),
            "source_ref": chunk.get("source_ref"),
            "image_path": chunk.get("image_path"),
            "metadata": chunk.get("metadata", {}),
            "tf": tf,
            "token_count": len(tokens),
        }
        for term, freq in tf.items():
            postings[term].append({"chunk_id": chunk_id, "tf": freq})
        document_frequency.update(tf.keys())
        if chunk.get("chunk_type") in {"figure", "page_image", "table"} or chunk.get("image_path"):
            visual_items.append({
                "chunk_id": chunk_id,
                "document_id": chunk.get("document_id"),
                "sample_id": chunk.get("sample_id"),
                "dataset": chunk.get("dataset"),
                "split": chunk.get("split"),
                "chunk_type": chunk.get("chunk_type"),
                "image_path": chunk.get("image_path"),
                "page_no": chunk.get("page_no"),
                "bbox": chunk.get("bbox"),
                "source_ref": chunk.get("source_ref"),
                "caption": chunk.get("text", ""),
            })

    text_dir.mkdir(parents=True, exist_ok=True)
    vision_dir.mkdir(parents=True, exist_ok=True)
    write_json(text_dir / "doc_store.json", doc_store)
    write_json(text_dir / "postings.json", dict(sorted(postings.items())))
    write_json(text_dir / "document_frequency.json", dict(sorted(document_frequency.items())))
    write_json(vision_dir / "visual_store.json", visual_items)

    manifest = {
        "index_version": "offline-bm25-lite-v1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": _dataset_version(documents, chunks),
        "python": platform.python_version(),
        "inputs": {"documents": str(document_path), "chunks": str(chunk_path)},
        "outputs": {
            "text_doc_store": str(text_dir / "doc_store.json"),
            "text_postings": str(text_dir / "postings.json"),
            "text_document_frequency": str(text_dir / "document_frequency.json"),
            "vision_store": str(vision_dir / "visual_store.json"),
        },
        "counts": {
            "documents": len(documents),
            "chunks": len(chunks),
            "text_terms": len(postings),
            "visual_items": len(visual_items),
        },
        "notes": [
            "Text index uses local tokenization and sparse term frequencies.",
            "Vision index is a metadata store for image/table/figure chunks; embedding backends can replace it later.",
        ],
    }
    write_json(manifest_path, manifest)
    print(f"[build-index] chunks={len(chunks)} terms={len(postings)} visual_items={len(visual_items)}")
    print(f"[build-index] manifest -> {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local text and visual indexes from document chunks.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--config", default="configs/datasets.yaml", help="Dataset config path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    build_indexes(project_root, (project_root / args.config).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
