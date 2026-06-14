from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_index_utils import read_jsonl, stable_id, term_frequency, tokenize, write_json, write_jsonl


TEXTUAL_CHUNK_TYPES = {"text", "title", "table", "formula", "fallback_text", "question_context"}
VISUAL_CHUNK_TYPES = {"figure", "page_image"}


def _base_source_ref(source_ref: str) -> str:
    return str(source_ref or "").split("#block=", 1)[0]


def _is_textual_chunk(chunk: dict[str, Any]) -> bool:
    return str(chunk.get("chunk_type") or "") in TEXTUAL_CHUNK_TYPES


def _build_page_text_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        if not _is_textual_chunk(chunk):
            continue
        key = (
            str(chunk.get("dataset") or ""),
            str(chunk.get("split") or ""),
            str(chunk.get("sample_id") or ""),
            str(chunk.get("page_no") or ""),
        )
        grouped[key].append(chunk)

    merged: list[dict[str, Any]] = []
    for key, items in grouped.items():
        items.sort(key=lambda item: (int(item.get("block_index") or 0), int(item.get("part_index") or 0), str(item.get("chunk_id") or "")))
        texts: list[str] = []
        seen_texts: set[str] = set()
        for item in items:
            text = " ".join(str(item.get("text") or "").split()).strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            texts.append(text)
        if not texts:
            continue
        sample = items[0]
        dataset, split, sample_id, page_no = key
        merged.append(
            {
                "chunk_id": f"chk-{stable_id(dataset, split, sample_id, page_no, 'page_text')}",
                "document_id": sample.get("document_id"),
                "sample_id": sample_id,
                "dataset": dataset,
                "split": split,
                "block_id": "page-text",
                "block_index": None,
                "part_index": 0,
                "chunk_type": "page_text",
                "text": "\n".join(texts),
                "page_no": sample.get("page_no"),
                "bbox": None,
                "source_ref": _base_source_ref(str(sample.get("source_ref") or "")),
                "source_path": sample.get("source_path"),
                "image_path": sample.get("image_path"),
                "metadata": sample.get("metadata", {}),
            }
        )
    return merged


def _variant_chunks(chunks: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    if variant == "page_text":
        return _build_page_text_chunks(chunks)
    if variant == "block_text":
        return [chunk for chunk in chunks if _is_textual_chunk(chunk)]
    if variant == "block_multimodal":
        return list(chunks)
    raise ValueError(f"Unsupported variant: {variant}")


def _build_index_payload(chunks: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    doc_store: dict[str, dict[str, Any]] = {}
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
            "block_id": chunk.get("block_id"),
            "block_index": chunk.get("block_index"),
            "part_index": chunk.get("part_index"),
            "chunk_type": chunk.get("chunk_type"),
            "text": chunk.get("text", ""),
            "page_no": chunk.get("page_no"),
            "bbox": chunk.get("bbox"),
            "source_ref": chunk.get("source_ref"),
            "source_path": chunk.get("source_path"),
            "image_path": chunk.get("image_path"),
            "metadata": chunk.get("metadata", {}),
            "tf": tf,
            "token_count": len(tokens),
        }
        document_frequency.update(tf.keys())
        chunk_type = str(chunk.get("chunk_type") or "")
        if chunk_type in VISUAL_CHUNK_TYPES or chunk.get("image_path"):
            visual_items.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": chunk.get("document_id"),
                    "sample_id": chunk.get("sample_id"),
                    "dataset": chunk.get("dataset"),
                    "split": chunk.get("split"),
                    "block_id": chunk.get("block_id"),
                    "block_index": chunk.get("block_index"),
                    "part_index": chunk.get("part_index"),
                    "chunk_type": chunk_type,
                    "source_path": chunk.get("source_path"),
                    "image_path": chunk.get("image_path"),
                    "page_no": chunk.get("page_no"),
                    "bbox": chunk.get("bbox"),
                    "source_ref": chunk.get("source_ref"),
                    "caption": chunk.get("text", ""),
                }
            )
    return doc_store, dict(sorted(document_frequency.items())), visual_items


def _write_variant(root: Path, variant: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    variant_root = root / variant
    chunks_path = variant_root / "chunks.jsonl"
    doc_store_path = variant_root / "doc_store.json"
    doc_freq_path = variant_root / "document_frequency.json"
    visual_store_path = variant_root / "visual_store.json"

    write_jsonl(chunks_path, chunks)
    doc_store, document_frequency, visual_items = _build_index_payload(chunks)
    write_json(doc_store_path, doc_store)
    write_json(doc_freq_path, document_frequency)
    write_json(visual_store_path, visual_items)

    summary = {
        "variant": variant,
        "counts": {
            "chunks": len(chunks),
            "visual_items": len(visual_items),
            "unique_terms": len(document_frequency),
        },
        "paths": {
            "chunks": str(chunks_path),
            "sparse_index": str(doc_store_path),
            "document_frequency": str(doc_freq_path),
            "visual_index": str(visual_store_path),
        },
    }
    write_json(variant_root / "manifest.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare index variants for ablation experiments.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--chunks-path", default="data/processed/chunks/chunks.jsonl", help="Source chunks JSONL.")
    parser.add_argument("--output-dir", default="outputs/eval/ablation_indexes", help="Output root for ablation index variants.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    chunks_path = (project_root / args.chunks_path).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise RuntimeError(f"No chunks found at {chunks_path}")

    summaries = []
    for variant in ("page_text", "block_text", "block_multimodal"):
        variant_rows = _variant_chunks(chunks, variant)
        summaries.append(_write_variant(output_dir, variant, variant_rows))

    write_json(output_dir / "manifest.json", {"source_chunks": str(chunks_path), "variants": summaries})
    print(f"[ablation-indexes] source={chunks_path}")
    for summary in summaries:
        print(f"[ablation-indexes] {summary['variant']} chunks={summary['counts']['chunks']} visual_items={summary['counts']['visual_items']}")
    print(f"[ablation-indexes] manifest -> {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
