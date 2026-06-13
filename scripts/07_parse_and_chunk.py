from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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


def _load_mineru_payload(mineru_json_root: Path, sample: dict[str, Any]) -> dict[str, Any] | None:
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
            return payload
        if isinstance(payload, list):
            return {"blocks": [x for x in payload if isinstance(x, dict)]}
    return None


def _load_mineru_blocks(mineru_json_root: Path, sample: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _load_mineru_payload(mineru_json_root, sample)
    if not isinstance(payload, dict):
        return []
    for key in ("blocks", "content", "items", "spans"):
        if isinstance(payload.get(key), list):
            return [x for x in payload[key] if isinstance(x, dict)]
    return []


def _iter_mineru_payloads(mineru_json_root: Path, datasets: list[str], splits: list[str], limit_per_split: int | None) -> list[dict[str, Any]]:
    selected_splits = set(splits)
    payloads: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_root = mineru_json_root / dataset
        if not dataset_root.exists():
            continue
        by_split: dict[str, int] = {}
        for path in sorted(dataset_root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            split = str(payload.get("split") or "raw")
            if selected_splits and split not in selected_splits:
                continue
            if limit_per_split is not None and by_split.get(split, 0) >= limit_per_split:
                continue
            payload.setdefault("dataset", dataset)
            payload.setdefault("split", split)
            payloads.append(payload)
            by_split[split] = by_split.get(split, 0) + 1
    return payloads


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
    if "title" in raw:
        return "title"
    return "text"


def _block_id(document_id: str, block: dict[str, Any], block_idx: int) -> str:
    value = block.get("block_id") or block.get("id")
    if value not in (None, ""):
        return str(value)
    return f"blk-{stable_id(document_id, block_idx, length=10)}"


def _chunk_source_ref(dataset: str, split: str, sample_id: str, page_no: int, block_id: str, figure_id: str | None = None) -> str:
    base = source_ref(dataset, split, sample_id, page_no, figure_id)
    return f"{base}#block={block_id}"


def _merged_chunk_metadata(
    sample_metadata: dict[str, Any] | None,
    block_idx: int,
    block_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(sample_metadata, dict):
        merged.update(sample_metadata)
    merged.update({"parser": "mineru", "block_index": block_idx})
    if isinstance(block_metadata, dict):
        merged.update(block_metadata)
    return merged


def _append_block_chunks(
    chunks: list[dict[str, Any]],
    *,
    document_id: str,
    sample_id: str,
    dataset: str,
    split: str,
    source_path: str | None,
    image_path: str | None,
    figure_id: str | None,
    default_page_no: int,
    blocks: list[dict[str, Any]],
    max_chars: int,
    overlap: int,
    sample_metadata: dict[str, Any] | None = None,
) -> None:
    for block_idx, block in enumerate(blocks):
        text = _block_text(block)
        chunk_type = _block_type(block)
        if not text and chunk_type not in {"figure", "table", "formula"}:
            continue
        block_id = _block_id(document_id, block, block_idx)
        try:
            page_no = int(block.get("page_no", default_page_no) or default_page_no)
        except (TypeError, ValueError):
            page_no = default_page_no
        block_image_path = block.get("image_path") if isinstance(block.get("image_path"), str) else None
        final_image_path = block_image_path or image_path
        for part_idx, part in enumerate(split_text(text or f"{chunk_type} region", max_chars, overlap)):
            chunk_id = f"chk-{stable_id(document_id, block_id, part_idx)}"
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "sample_id": sample_id,
                "dataset": dataset,
                "split": split,
                "block_id": block_id,
                "block_index": block_idx,
                "part_index": part_idx,
                "chunk_type": chunk_type,
                "text": part,
                "page_no": page_no,
                "bbox": block.get("bbox"),
                "source_ref": _chunk_source_ref(dataset, split, sample_id, page_no, block_id, figure_id),
                "source_path": source_path,
                "image_path": final_image_path,
                "metadata": _merged_chunk_metadata(
                    sample_metadata,
                    block_idx,
                    block.get("metadata") if isinstance(block.get("metadata"), dict) else None,
                ),
            })


def build_documents_and_chunks(
    project_root: Path,
    config_path: Path,
    datasets: list[str],
    splits: list[str],
    limit_per_split: int | None,
    include_qa_context: bool = False,
    sample_manifest: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = read_yaml(config_path)
    chunk_cfg = cfg.get("document_chunking", {})
    input_root = project_root / chunk_cfg.get("processed_input_root", "data/processed")
    mineru_json_root = project_root / chunk_cfg.get("mineru_json_root", "data/interim/mineru")
    default_page_no = int(chunk_cfg.get("default_page_no", 1))
    max_chars = int(chunk_cfg.get("max_chars_per_chunk", 900))
    overlap = int(chunk_cfg.get("overlap_chars", 120))

    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []

    manifest_rows = read_jsonl(sample_manifest) if sample_manifest is not None else []
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        grouped_rows[(str(row.get("dataset", "")), str(row.get("split", "")))].append(row)

    dataset_split_pairs = list(grouped_rows.keys()) if grouped_rows else [(dataset, split) for dataset in datasets for split in splits]

    for dataset, split in dataset_split_pairs:
        if grouped_rows:
            rows = grouped_rows.get((dataset, split), [])
        else:
            src = input_root / dataset / f"{split}.jsonl"
            rows = read_jsonl(src)
            if limit_per_split is not None:
                rows = rows[:limit_per_split]
        for sample in rows:
                sample_id = str(sample.get("id", stable_id(dataset, split, len(documents))))
                sample_metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
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
                    "source_path": image_path,
                    "image_path": image_path,
                    "figure_id": figure_id,
                    "page_no": page_no,
                    "question": str(sample.get("question", "")),
                    "answers": sample.get("answers", []),
                    "evidence": evidence,
                    "metadata": sample_metadata,
                })

                mineru_blocks = _load_mineru_blocks(mineru_json_root, sample)
                if mineru_blocks:
                    if include_qa_context:
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
                                "block_id": "question-context",
                                "block_index": None,
                                "part_index": 0,
                                "chunk_type": "question_context",
                                "text": question_context,
                                "page_no": page_no,
                                "bbox": None,
                                "source_ref": ref,
                                "source_path": image_path,
                                "image_path": image_path,
                                "metadata": {**sample_metadata, "parser": "benchmark", "synthetic": True, "index_by_default": False},
                            })
                    _append_block_chunks(
                        chunks,
                        document_id=doc_id,
                        sample_id=sample_id,
                        dataset=dataset,
                        split=split,
                        source_path=image_path,
                        image_path=image_path,
                        figure_id=figure_id,
                        default_page_no=page_no,
                        blocks=mineru_blocks,
                        max_chars=max_chars,
                        overlap=overlap,
                        sample_metadata=sample_metadata,
                    )
                    continue

                fallback_text = " ".join(x for x in [
                    f"Question: {sample.get('question', '')}",
                    f"Evidence: {evidence}" if evidence else "",
                ] if x.strip())
                if fallback_text:
                    for part_idx, part in enumerate(split_text(fallback_text, max_chars, overlap)):
                        chunks.append({
                            "chunk_id": f"chk-{stable_id(doc_id, 'fallback-text', part_idx)}",
                            "document_id": doc_id,
                            "sample_id": sample_id,
                            "dataset": dataset,
                            "split": split,
                            "block_id": "fallback-text",
                            "block_index": None,
                            "part_index": part_idx,
                            "chunk_type": "fallback_text",
                            "text": part,
                            "page_no": page_no,
                            "bbox": None,
                            "source_ref": ref,
                            "source_path": image_path,
                            "image_path": image_path,
                            "metadata": {**sample_metadata, "parser": "fallback", "index_by_default": False},
                        })
                if image_path or dataset == "chartqa":
                    chunks.append({
                        "chunk_id": f"chk-{stable_id(doc_id, 'fallback-visual')}",
                        "document_id": doc_id,
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "split": split,
                        "block_id": "fallback-visual",
                        "block_index": None,
                        "part_index": 0,
                        "chunk_type": "page_image" if dataset == "docvqa" else "figure",
                        "text": "Visual page/image fallback chunk. MinerU blocks were not available.",
                        "page_no": page_no,
                        "bbox": None,
                        "source_ref": ref,
                        "source_path": image_path,
                        "image_path": image_path,
                        "metadata": {**sample_metadata, "parser": "fallback", "index_by_default": False},
                    })

    return documents, chunks


def build_documents_and_chunks_from_mineru(
    project_root: Path,
    config_path: Path,
    datasets: list[str],
    splits: list[str],
    limit_per_split: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = read_yaml(config_path)
    chunk_cfg = cfg.get("document_chunking", {})
    mineru_json_root = project_root / chunk_cfg.get("mineru_json_root", "data/interim/mineru")
    default_page_no = int(chunk_cfg.get("default_page_no", 1))
    max_chars = int(chunk_cfg.get("max_chars_per_chunk", 900))
    overlap = int(chunk_cfg.get("overlap_chars", 120))

    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []

    for payload in _iter_mineru_payloads(mineru_json_root, datasets, splits, limit_per_split):
        dataset = str(payload.get("dataset") or "raw_documents")
        split = str(payload.get("split") or "raw")
        sample_id = str(payload.get("sample_id") or Path(str(payload.get("source_path") or "document")).stem)
        source_path = str(payload.get("source_path") or "") or None
        source_suffix = Path(source_path).suffix.lower().lstrip(".") if source_path else "file"
        document_id = str(payload.get("document_id") or f"doc-{stable_id(dataset, split, sample_id, source_path)}")
        blocks = [x for x in payload.get("blocks", []) if isinstance(x, dict)]
        page_numbers = []
        for block in blocks:
            try:
                page_numbers.append(int(block.get("page_no")))
            except (TypeError, ValueError):
                pass
        page_no = min(page_numbers) if page_numbers else default_page_no
        page_count = len(set(page_numbers)) if page_numbers else None
        image_path = source_path if source_suffix in {"png", "jpg", "jpeg", "webp", "bmp"} else None

        documents.append({
            "document_id": document_id,
            "sample_id": sample_id,
            "dataset": dataset,
            "split": split,
            "source_type": source_suffix or "file",
            "source_path": source_path,
            "image_path": image_path,
            "figure_id": None,
            "page_no": page_no,
            "page_count": page_count,
            "question": None,
            "answers": [],
            "evidence": "",
            "metadata": {
                "parser": "mineru",
                "raw_output_dir": payload.get("raw_output_dir"),
                "raw_files": payload.get("raw_files", []),
                **(payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}),
            },
        })
        _append_block_chunks(
            chunks,
            document_id=document_id,
            sample_id=sample_id,
            dataset=dataset,
            split=split,
            source_path=source_path,
            image_path=image_path,
            figure_id=None,
            default_page_no=page_no,
            blocks=blocks,
            max_chars=max_chars,
            overlap=overlap,
        )

    return documents, chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse normalized benchmark records or MinerU JSON into document and block chunk schemas.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--config", default="configs/datasets.yaml", help="Dataset config path.")
    parser.add_argument("--datasets", default="docvqa,chartqa", help="Comma-separated datasets.")
    parser.add_argument("--splits", default="val,test", help="Comma-separated splits.")
    parser.add_argument("--source-mode", choices=["benchmark", "mineru"], default="benchmark", help="benchmark reads processed QA JSONL; mineru reads data/interim/mineru/<dataset>/*.json directly.")
    parser.add_argument("--include-qa-context", action="store_true", help="Benchmark-only debug option: add synthetic question/answer chunks. Disabled by default for production RAG.")
    parser.add_argument("--limit-per-split", type=int, default=0, help="Optional cap per dataset split; 0 means all.")
    parser.add_argument("--sample-manifest", default="", help="Optional JSONL manifest of processed benchmark samples.")
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

    if args.source_mode == "mineru":
        documents, chunks = build_documents_and_chunks_from_mineru(project_root, config_path, datasets, splits, limit)
    else:
        documents, chunks = build_documents_and_chunks(
            project_root,
            config_path,
            datasets,
            splits,
            limit,
            args.include_qa_context,
            sample_manifest=(project_root / args.sample_manifest).resolve() if args.sample_manifest else None,
        )
    doc_count = write_jsonl(document_output, documents)
    chunk_count = write_jsonl(chunk_output, chunks)
    print(f"[parse-chunk] documents={doc_count} -> {document_output}")
    print(f"[parse-chunk] chunks={chunk_count} -> {chunk_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
