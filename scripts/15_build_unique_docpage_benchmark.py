from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def doc_page_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    doc_id = str(metadata.get("ucsf_document_id", ""))
    page_no = str(metadata.get("ucsf_document_page_no", ""))
    return f"{doc_id}|{page_no}"


def primary_question_type(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    value = metadata.get("question_types", [])
    if isinstance(value, list) and value:
        return str(value[0])
    if value not in (None, ""):
        return str(value)
    return "unknown"


def representative_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: (
            0 if primary_question_type(row) != "unknown" else 1,
            len(str(row.get("question", ""))),
            str(row.get("id", "")),
        ),
    )
    return ranked[0]


def stratified_unique_docpage_sample(rows: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    grouped_by_docpage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_by_docpage[doc_page_key(row)].append(row)

    representatives = [representative_row(group) for group in grouped_by_docpage.values()]
    type_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(representatives, key=lambda item: str(item.get("id", ""))):
        type_buckets[primary_question_type(row)].append(row)

    if sample_size >= len(representatives):
        return representatives

    selected: list[dict[str, Any]] = []
    cursor = {bucket: 0 for bucket in type_buckets}
    bucket_order = sorted(type_buckets.keys(), key=lambda key: (-len(type_buckets[key]), key))

    while len(selected) < sample_size:
        progressed = False
        for bucket in bucket_order:
            items = type_buckets[bucket]
            index = cursor[bucket]
            if index >= len(items):
                continue
            selected.append(items[index])
            cursor[bucket] += 1
            progressed = True
            if len(selected) >= sample_size:
                break
        if not progressed:
            break
    return selected


def build_manifest(rows: list[dict[str, Any]], sample_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = stratified_unique_docpage_sample(rows, sample_size)
    type_counts = Counter(primary_question_type(row) for row in selected)
    summary = {
        "num_rows": len(selected),
        "unique_doc_pages": len({doc_page_key(row) for row in selected}),
        "question_type_counts": dict(sorted(type_counts.items())),
    }
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unique-doc-page benchmark manifest from processed DocVQA rows.")
    parser.add_argument("--input", default="data/processed/docvqa/val.jsonl", help="Processed JSONL input.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of unique doc-page samples to select.")
    parser.add_argument("--output", default="outputs/eval/docvqa_val_unique_docpage_100.manifest.jsonl", help="Output manifest JSONL.")
    parser.add_argument("--summary-output", default="outputs/eval/docvqa_val_unique_docpage_100.manifest.summary.json", help="Output summary JSON.")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    selected, summary = build_manifest(rows, args.sample_size)
    write_jsonl(Path(args.output), selected)
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[unique-docpage-benchmark] rows={summary['num_rows']} unique_doc_pages={summary['unique_doc_pages']}")
    print(f"[unique-docpage-benchmark] manifest -> {args.output}")


if __name__ == "__main__":
    main()
