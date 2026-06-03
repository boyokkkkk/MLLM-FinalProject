from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_index_utils import extract_image_path, read_jsonl, read_yaml, source_ref, split_text, stable_id, write_jsonl


def _coerce_page_no(sample: dict[str, Any], default_page_no: int) -> int:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    for key in ("page_no", "page", "ucsf_document_page_no", "page_number"):
        value = sample.get(key, metadata.get(key))
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default_page_no


def _resolve_local_image_path(project_root: Path, dataset: str, split: str, image_path: str | None) -> str | None:
    if not image_path:
        return None
    path = Path(image_path)
    if path.is_absolute() and path.exists():
        return str(path)
    candidates = [
        project_root / image_path,
        project_root / "data" / "images" / dataset / split / image_path,
        project_root / "data" / "images" / dataset / split / Path(image_path).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return image_path


def _answers_text(sample: dict[str, Any]) -> str:
    answers = sample.get("answers") or []
    if not isinstance(answers, list):
        answers = [answers]
    return "; ".join(str(a) for a in answers if a not in (None, ""))


def _load_mineru_blocks(mineru_json_root: Path, sample: dict[str, Any]) -> list[dict[str, Any]]:
    sample_id = str(sample.get("id", ""))
    candidates = [
        mineru_json_root / f"{sample_id}.json",
        mineru_json_root / str(sample.get("dataset", "")) / f"{sample_id}.json",
    ]
    image_path = extract_image_path(sample.get("image"))
    if image_path:
        stem = Path(image_path).stem
        candidates.extend([
            mineru_json_root / f"{stem}.json",
            mineru_json_root / str(sample.get("dataset", "")) / f"{stem}.json",
        ])

    for path in candidates:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("blocks", "content", "items", "spans"):
                if isinstance(payload.get(key), list):
                    return [x for x in payload[key] if isinstance(x, dict)]
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
    return []


def _block_text(block: dict[str, Any]) -> str:
    for key in ("text", "content", "html", "latex", "caption"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _block_type(block: dict[str, Any]) -> str:
    raw = str(block.get("type") or block.get("category") or block.get("block_type") or "text").lower()
    if "table" in raw:
        return "table"
    if "formula" in raw or "equation" in raw:
        return "formula"
    if "image" in raw or "figure" in raw or "chart" in raw:
        return "figure"
    return "text"


def build_documents_and_chunks(project_root: Path, config_path: Path, datasets: list[str], splits: list[str], limit_per_split: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = read_yaml(config_path)
    chunk_cfg = cfg.get("document_chunking", {})
    input_root = project_root / chunk_cfg.get("processed_input_root", "data/processed")
    mineru_json_root = project_root / chunk_cfg.get("mineru_json_root", "data/interim/mineru")
    default_page_no = int(chunk_cfg.get("default_page_no", 1))
    max_chars = int(chunk_cfg.get("max_chars_per_chunk", 900))
    overlap = int(chunk_cfg.get("overlap_chars", 120))

    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []

    for dataset in datasets:
        for split in splits:
            src = input_root / dataset / f"{split}.jsonl"
            rows = read_jsonl(src)
            if limit_per_split is not None:
                rows = rows[:limit_per_split]
            for sample in rows:
                sample_id = str(sample.get("id", stable_id(dataset, split, len(documents))))
                page_no = _coerce_page_no(sample, default_page_no)
                image_path = extract_image_path(sample.get("image"))
                image_path = _resolve_local_image_path(project_root, dataset, split, image_path)
                figure_id = f"fig-{Path(image_path).stem}" if image_path else None
                doc_id = f"doc-{stable_id(dataset, split, sample_id, image_path)}"
                ref = source_ref(dataset, split, sample_id, page_no, figure_id)
                evidence = sample.get("evidence")
                if not isinstance(evidence, str):
                    evidence = "" if evidence is None else json.dumps(evidence, ensure_ascii=False)

                documents.append({
                    "document_id": doc_id,
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "split": split,
                    "source_type": "image" if image_path else "benchmark_record",
                    "image_path": image_path,
                    "figure_id": figure_id,
                    "page_no": page_no,
                    "question": str(sample.get("question", "")),
                    "answers": sample.get("answers", []),
                    "evidence": evidence,
                    "metadata": sample.get("metadata", {}),
                })

                mineru_blocks = _load_mineru_blocks(mineru_json_root, sample)
                if mineru_blocks:
                    question_context = " ".join(
                        x
                        for x in [
                            f"Question: {sample.get('question', '')}",
                            f"Expected answer: {_answers_text(sample)}",
                        ]
                        if x.strip()
                    )
                    if question_context:
                        chunks.append({
                            "chunk_id": f"chk-{stable_id(doc_id, 'question-context')}",
                            "document_id": doc_id,
                            "sample_id": sample_id,
                            "dataset": dataset,
                            "split": split,
                            "chunk_type": "question_context",
                            "text": question_context,
                            "page_no": page_no,
                            "bbox": None,
                            "source_ref": ref,
                            "image_path": image_path,
                            "metadata": {"parser": "mineru", "synthetic": True},
                        })
                    for block_idx, block in enumerate(mineru_blocks):
                        text = _block_text(block)
                        chunk_type = _block_type(block)
                        if not text and chunk_type not in {"figure", "table"}:
                            continue
                        for part_idx, part in enumerate(split_text(text or f"{chunk_type} region", max_chars, overlap)):
                            chunks.append({
                                "chunk_id": f"chk-{stable_id(doc_id, block_idx, part_idx)}",
                                "document_id": doc_id,
                                "sample_id": sample_id,
                                "dataset": dataset,
                                "split": split,
                                "chunk_type": chunk_type,
                                "text": part,
                                "page_no": int(block.get("page_no", page_no) or page_no),
                                "bbox": block.get("bbox"),
                                "source_ref": ref,
                                "image_path": image_path,
                                "metadata": {"parser": "mineru", "block_index": block_idx},
                            })
                    continue

                fallback_text = " ".join(x for x in [
                    f"Question: {sample.get('question', '')}",
                    f"Evidence: {evidence}" if evidence else "",
                ] if x.strip())
                if fallback_text:
                    for part_idx, part in enumerate(split_text(fallback_text, max_chars, overlap)):
                        chunks.append({
                            "chunk_id": f"chk-{stable_id(doc_id, 'text', part_idx)}",
                            "document_id": doc_id,
                            "sample_id": sample_id,
                            "dataset": dataset,
                            "split": split,
                            "chunk_type": "text",
                            "text": part,
                            "page_no": page_no,
                            "bbox": None,
                            "source_ref": ref,
                            "image_path": image_path,
                            "metadata": {"parser": "fallback", "contains_answer": False},
                        })
                if image_path or dataset == "chartqa":
                    visual_text = " ".join(x for x in [
                        f"Visual document for {dataset}.",
                        f"Question: {sample.get('question', '')}",
                        f"Expected answer: {_answers_text(sample)}",
                    ] if x.strip())
                    chunks.append({
                        "chunk_id": f"chk-{stable_id(doc_id, 'visual')}",
                        "document_id": doc_id,
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "split": split,
                        "chunk_type": "figure" if dataset == "chartqa" else "page_image",
                        "text": visual_text,
                        "page_no": page_no,
                        "bbox": None,
                        "source_ref": ref,
                        "image_path": image_path,
                        "metadata": {"parser": "fallback", "contains_answer": True},
                    })

    return documents, chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse normalized benchmark records into document and chunk schemas.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--config", default="configs/datasets.yaml", help="Dataset config path.")
    parser.add_argument("--datasets", default="docvqa,chartqa", help="Comma-separated datasets.")
    parser.add_argument("--splits", default="val,test", help="Comma-separated splits.")
    parser.add_argument("--limit-per-split", type=int, default=0, help="Optional cap per dataset split; 0 means all.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = (project_root / args.config).resolve()
    cfg = read_yaml(config_path)
    chunk_cfg = cfg.get("document_chunking", {})
    document_output = project_root / chunk_cfg.get("document_output", "data/processed/documents/documents.jsonl")
    chunk_output = project_root / chunk_cfg.get("chunk_output", "data/processed/chunks/chunks.jsonl")
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    limit = int(args.limit_per_split) or None

    documents, chunks = build_documents_and_chunks(project_root, config_path, datasets, splits, limit)
    doc_count = write_jsonl(document_output, documents)
    chunk_count = write_jsonl(chunk_output, chunks)
    print(f"[parse-chunk] documents={doc_count} -> {document_output}")
    print(f"[parse-chunk] chunks={chunk_count} -> {chunk_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
