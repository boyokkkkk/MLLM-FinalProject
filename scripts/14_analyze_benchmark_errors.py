from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object JSONL row at {path}:{line_no}, got {type(payload)!r}")
            rows.append(payload)
    return rows


def doc_page_key(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), dict) else {}
    doc_id = str(metadata.get("ucsf_document_id", ""))
    page_no = str(metadata.get("ucsf_document_page_no", ""))
    return f"{doc_id}|{page_no}"


def question_types(sample: dict[str, Any]) -> list[str]:
    metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), dict) else {}
    raw_value = metadata.get("question_types", [])
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    if raw_value in (None, ""):
        return []
    return [str(raw_value)]


def strict_flags(citations: list[dict[str, Any]], sample_id: str) -> list[bool]:
    prefix = f"docvqa/val/{sample_id}"
    flags: list[bool] = []
    for citation in citations:
        source_ref = str(citation.get("source_ref") or citation.get("source") or "")
        flags.append(source_ref.startswith(prefix))
    return flags


def sample_id_from_source_ref(source_ref: str) -> str:
    if not source_ref.startswith("docvqa/val/"):
        return ""
    return source_ref.split("/", 2)[2].split("#", 1)[0]


def docpage_flags(
    citations: list[dict[str, Any]],
    sample_to_docpage: dict[str, str],
    expected_docpage: str,
) -> list[bool]:
    flags: list[bool] = []
    for citation in citations:
        source_ref = str(citation.get("source_ref") or citation.get("source") or "")
        citation_sample_id = sample_id_from_source_ref(source_ref)
        flags.append(sample_to_docpage.get(citation_sample_id, "") == expected_docpage)
    return flags


def first_true_rank(flags: list[bool]) -> int | None:
    for idx, flag in enumerate(flags, start=1):
        if flag:
            return idx
    return None


def classify_sample(
    rag_row: dict[str, Any],
    sample_row: dict[str, Any],
    sample_to_docpage: dict[str, str],
) -> dict[str, Any]:
    sample_id = str(rag_row["sample_id"])
    citations = rag_row.get("citations", [])
    metrics = rag_row.get("metrics", {})
    expected_docpage = sample_to_docpage[sample_id]

    sample_strict_flags = strict_flags(citations, sample_id)
    sample_docpage_flags = docpage_flags(citations, sample_to_docpage, expected_docpage)

    strict_hit = any(sample_strict_flags)
    strict_top1 = bool(sample_strict_flags[0]) if sample_strict_flags else False
    docpage_hit = any(sample_docpage_flags)
    docpage_top1 = bool(sample_docpage_flags[0]) if sample_docpage_flags else False
    strict_rank = first_true_rank(sample_strict_flags)
    docpage_rank = first_true_rank(sample_docpage_flags)

    exact_match = float(metrics.get("exact_match", 0.0))
    answer_contains = float(metrics.get("answer_contains", 0.0))
    token_f1 = float(metrics.get("token_f1", 0.0))
    anls = float(metrics.get("anls", 0.0))

    tags: list[str] = []
    if not strict_hit and docpage_hit:
        tags.append("same_page_false_negative")
    if strict_hit and not strict_top1:
        tags.append("strict_ranking_issue")
    if docpage_hit and not docpage_top1:
        tags.append("docpage_ranking_issue")
    if strict_hit and exact_match < 1.0:
        tags.append("generation_after_strict_hit")
    if docpage_hit and exact_match < 1.0:
        tags.append("generation_after_docpage_hit")
    if exact_match < 1.0 and answer_contains >= 1.0:
        tags.append("normalization_sensitive")
    if not docpage_hit:
        tags.append("true_retrieval_miss")

    if not strict_hit and docpage_hit:
        primary_category = "same_page_false_negative"
        rationale = "top-k hit the same original document page, but only under a different question-level sample_id"
    elif not docpage_hit:
        primary_category = "true_miss"
        rationale = "top-k hit neither the current sample nor the same original document page"
    elif strict_hit and not strict_top1:
        primary_category = "ranking_issue"
        rationale = "the correct sample entered top-k, but was not ranked at top-1"
    elif strict_hit and exact_match < 1.0:
        primary_category = "generation_issue"
        rationale = "the correct sample was retrieved, but the final answer still failed strict EM"
    else:
        primary_category = "clean_hit"
        rationale = "strict top-1 citation is correct and the final answer matches gold"

    top1 = citations[0] if citations else {}
    return {
        "sample_id": sample_id,
        "question": str(rag_row.get("question", "")),
        "question_types": "|".join(question_types(sample_row)),
        "gold_answers": " || ".join(str(item) for item in rag_row.get("answers", [])),
        "pred_answer": str(rag_row.get("answer", "")),
        "doc_page_key": expected_docpage,
        "strict_hit": int(strict_hit),
        "strict_top1": int(strict_top1),
        "strict_first_relevant_rank": strict_rank or "",
        "docpage_hit": int(docpage_hit),
        "docpage_top1": int(docpage_top1),
        "docpage_first_relevant_rank": docpage_rank or "",
        "exact_match": exact_match,
        "answer_contains": answer_contains,
        "token_f1": token_f1,
        "anls": anls,
        "primary_category": primary_category,
        "tags": "|".join(tags),
        "rationale": rationale,
        "top1_source_ref": str(top1.get("source_ref") or top1.get("source") or ""),
        "top1_snippet": str(top1.get("snippet", "")),
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = Counter(str(row["primary_category"]) for row in rows)
    error_category_counts = Counter(str(row["primary_category"]) for row in rows if str(row["primary_category"]) != "clean_hit")
    tag_counts = Counter()
    by_question_type: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        for tag in filter(None, str(row["tags"]).split("|")):
            tag_counts[tag] += 1
        sample_types = [item for item in str(row["question_types"]).split("|") if item]
        if not sample_types:
            sample_types = ["unknown"]
        for sample_type in sample_types:
            by_question_type[sample_type][str(row["primary_category"])] += 1

    return {
        "num_samples": len(rows),
        "category_counts": dict(category_counts),
        "error_category_counts": dict(error_category_counts),
        "tag_counts": dict(tag_counts),
        "by_question_type": {key: dict(counter) for key, counter in sorted(by_question_type.items())},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# DocVQA Val 100 Error Analysis",
        "",
        "## Label Mapping",
        "- `true_miss`: true miss",
        "- `same_page_false_negative`: same-page false negative under strict sample-level evaluation",
        "- `ranking_issue`: the correct sample is in top-k but not top-1",
        "- `generation_issue`: the correct sample is retrieved but answer generation still fails strict EM",
        "- `clean_hit`: strict top-1 citation is correct and the answer also matches gold",
        "",
        "## Classification Rules",
        "- `same_page_false_negative`: strict sample-level miss, but top-k hits the same original `ucsf_document_id + page_no`.",
        "- `ranking_issue`: strict hit exists, but the first strict relevant citation is not rank-1.",
        "- `generation_issue`: strict hit exists and the answer still fails strict EM.",
        "- `true_miss`: top-k hits neither the current sample nor the same original document page.",
        "- `clean_hit`: strict top-1 citation is correct and the answer matches gold.",
        "",
        "## Summary",
        f"- samples: {summary['num_samples']}",
    ]
    for category, count in summary["category_counts"].items():
        lines.append(f"- {category}: {count}")

    lines.extend([
        "",
        "## Error Buckets Only",
    ])
    for category, count in summary["error_category_counts"].items():
        lines.append(f"- {category}: {count}")

    lines.extend([
        "",
        "## Auxiliary Tags",
    ])
    for tag, count in sorted(summary["tag_counts"].items()):
        lines.append(f"- {tag}: {count}")

    lines.extend([
        "",
        "## By Question Type",
        "| question_type | clean_hit | true_miss | same_page_false_negative | ranking_issue | generation_issue |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for sample_type, counts in summary["by_question_type"].items():
        lines.append(
            f"| {sample_type} | {counts.get('clean_hit', 0)} | {counts.get('true_miss', 0)} | {counts.get('same_page_false_negative', 0)} | {counts.get('ranking_issue', 0)} | {counts.get('generation_issue', 0)} |"
        )

    lines.extend([
        "",
        "## Sample Table",
        "| sample_id | category | qtypes | strict_hit | docpage_hit | EM | ANLS | top1_source_ref |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in rows:
        lines.append(
            f"| {row['sample_id']} | {row['primary_category']} | {row['question_types']} | {row['strict_hit']} | {row['docpage_hit']} | {row['exact_match']:.0f} | {row['anls']:.3f} | {row['top1_source_ref']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benchmark errors into miss / same-page / ranking / generation categories.")
    parser.add_argument("--samples", default="data/processed/docvqa/val.jsonl", help="Processed sample JSONL.")
    parser.add_argument("--rag-run", default="outputs/eval/docvqa_val_100_rag_exp5_densefusion_w045.jsonl", help="RAG eval JSONL to analyze.")
    parser.add_argument("--limit", type=int, default=100, help="Number of processed samples to align with the benchmark.")
    parser.add_argument("--output-prefix", default="outputs/eval/docvqa_val_100_error_analysis_exp5", help="Prefix for csv/json/md outputs.")
    args = parser.parse_args()

    sample_rows = read_jsonl(Path(args.samples))[: args.limit]
    rag_rows = read_jsonl(Path(args.rag_run))

    sample_by_id = {str(row["id"]): row for row in sample_rows}
    sample_to_docpage = {str(row["id"]): doc_page_key(row) for row in sample_rows}

    analyzed_rows: list[dict[str, Any]] = []
    for rag_row in rag_rows:
        sample_id = str(rag_row["sample_id"])
        sample_row = sample_by_id.get(sample_id)
        if sample_row is None:
            continue
        analyzed_rows.append(classify_sample(rag_row, sample_row, sample_to_docpage))

    summary = build_summary(analyzed_rows)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    write_csv(output_prefix.with_suffix(".csv"), analyzed_rows)
    output_prefix.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(output_prefix.with_suffix(".md"), analyzed_rows, summary)


if __name__ == "__main__":
    main()
