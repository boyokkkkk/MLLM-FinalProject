from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_index_utils import read_jsonl


SLICE_ORDER = ["logo", "title", "page_number", "handwritten", "chart", "table"]
QUESTION_TYPE_HINTS = {
    "logo": {"image/photo", "image", "photo"},
    "handwritten": {"handwritten"},
    "chart": {"figure/diagram", "chart"},
    "table": {"table/list", "table"},
}


def _sample_doc_page_key(row: dict[str, Any]) -> str:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    return f"{metadata.get('ucsf_document_id', '')}|{metadata.get('ucsf_document_page_no', '')}"


def _sample_question_types(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    raw = metadata.get("question_types", [])
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if raw in (None, ""):
        return []
    return [str(raw)]


def _metric(record: dict[str, Any], key: str) -> float:
    return float((record.get("metrics") or {}).get(key, 0.0))


def _first_citation_source(record: dict[str, Any]) -> str:
    citations = record.get("citations", [])
    if not citations:
        return ""
    top1 = citations[0]
    return str(top1.get("source_ref") or top1.get("source") or "")


def _citation_flags(record: dict[str, Any], sample_to_docpage: dict[str, str]) -> tuple[bool, bool]:
    sample_id = str(record.get("sample_id") or "")
    citations = record.get("citations", [])
    expected_prefix = f"docvqa/val/{sample_id}"
    expected_docpage = sample_to_docpage.get(sample_id, "")

    strict_hit = False
    docpage_hit = False
    for citation in citations:
        source_ref = str(citation.get("source_ref") or citation.get("source") or "")
        if source_ref.startswith(expected_prefix):
            strict_hit = True
        if source_ref.startswith("docvqa/val/"):
            citation_sample_id = source_ref.split("/", 2)[2].split("#", 1)[0]
            if sample_to_docpage.get(citation_sample_id, "") == expected_docpage:
                docpage_hit = True
    return strict_hit, docpage_hit


def _canonical_slices(question: str, raw_types: list[str]) -> list[str]:
    query = (question or "").strip().lower()
    raw_lower = {item.strip().lower() for item in raw_types if item}
    slices: list[str] = []

    if (
        any(marker in query for marker in ("logo", "pack", "brand", "written on the pack", "written within the logo"))
        or raw_lower.intersection(QUESTION_TYPE_HINTS["logo"])
    ):
        slices.append("logo")

    if any(marker in query for marker in ("title", "heading", "subheading", "tagline", "subject", "heading of the page")):
        slices.append("title")

    if "page number" in query or "page no" in query:
        slices.append("page_number")

    if "handwritten" in query or raw_lower.intersection(QUESTION_TYPE_HINTS["handwritten"]):
        slices.append("handwritten")

    if (
        any(marker in query for marker in ("chart", "graph", "x axis", "y axis", "axis", "plot", "diagram", "figure"))
        or raw_lower.intersection(QUESTION_TYPE_HINTS["chart"])
    ):
        slices.append("chart")

    if (
        any(marker in query for marker in ("table", "list", "committee", "attendance", "present", "row", "column"))
        or raw_lower.intersection(QUESTION_TYPE_HINTS["table"])
    ):
        slices.append("table")

    deduped: list[str] = []
    for item in slices:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _is_visual_sensitive(question: str, raw_types: list[str], slices: list[str]) -> bool:
    raw_lower = {item.strip().lower() for item in raw_types if item}
    if slices:
        return True
    return bool(raw_lower.intersection({"image/photo", "figure/diagram", "handwritten", "table/list", "layout"}))


def _mean(records: list[dict[str, Any]], metric_name: str) -> float:
    if not records:
        return 0.0
    return round(sum(_metric(record, metric_name) for record in records) / len(records), 6)


def _parse_run_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        stem, label = spec.split("=", 1)
        return stem.strip(), label.strip()
    stem = spec.strip()
    return stem, stem


def _resolve_run_path(outputs_root: Path, stem_or_path: str) -> Path:
    candidate = Path(stem_or_path)
    if candidate.exists():
        return candidate
    if candidate.suffix:
        return (outputs_root / candidate.name).resolve()
    return (outputs_root / f"{stem_or_path}.jsonl").resolve()


def _build_question_type_rows(
    run_label: str,
    records: list[dict[str, Any]],
    sample_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        meta = sample_meta.get(sample_id, {})
        slices = meta.get("canonical_slices", [])
        for slice_name in slices:
            by_slice[slice_name].append(record)

    rows: list[dict[str, Any]] = []
    for slice_name in SLICE_ORDER:
        items = by_slice.get(slice_name, [])
        rows.append(
            {
                "run": run_label,
                "slice": slice_name,
                "n": len(items),
                "exact_match": _mean(items, "exact_match"),
                "anls": _mean(items, "anls"),
                "token_f1": _mean(items, "token_f1"),
                "citation_accuracy": _mean(items, "citation_accuracy"),
                "hit_at_k": _mean(items, "hit_at_k"),
            }
        )
    return rows


def _build_visual_subset_rows(
    run_label: str,
    records: list[dict[str, Any]],
    sample_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    visual_sensitive = [record for record in records if sample_meta.get(str(record.get("sample_id") or ""), {}).get("visual_sensitive")]
    non_visual_sensitive = [record for record in records if not sample_meta.get(str(record.get("sample_id") or ""), {}).get("visual_sensitive")]
    rows: list[dict[str, Any]] = []
    for subset_name, items in [("visual_sensitive", visual_sensitive), ("non_visual_sensitive", non_visual_sensitive)]:
        rows.append(
            {
                "run": run_label,
                "subset": subset_name,
                "n": len(items),
                "exact_match": _mean(items, "exact_match"),
                "anls": _mean(items, "anls"),
                "token_f1": _mean(items, "token_f1"),
                "citation_accuracy": _mean(items, "citation_accuracy"),
                "hit_at_k": _mean(items, "hit_at_k"),
            }
        )
    return rows


def _build_error_rows(
    run_label: str,
    records: list[dict[str, Any]],
    sample_to_docpage: dict[str, str],
    reference_records: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    retrieval_wrong_page = 0
    same_page_wrong_block = 0
    answer_drift = 0
    visual_noise = 0

    for record in records:
        sample_id = str(record.get("sample_id") or "")
        strict_hit, docpage_hit = _citation_flags(record, sample_to_docpage)
        strict_top1 = bool(_first_citation_source(record).startswith(f"docvqa/val/{sample_id}"))
        exact_match = _metric(record, "exact_match")
        token_f1 = _metric(record, "token_f1")

        if not docpage_hit:
            retrieval_wrong_page += 1
        if docpage_hit and not strict_hit:
            same_page_wrong_block += 1
        if strict_top1 and exact_match < 1.0:
            answer_drift += 1

        if reference_records is not None and sample_id in reference_records:
            ref = reference_records[sample_id]
            ref_strict_hit, ref_docpage_hit = _citation_flags(ref, sample_to_docpage)
            ref_strict_top1 = bool(_first_citation_source(ref).startswith(f"docvqa/val/{sample_id}"))
            ref_exact_match = _metric(ref, "exact_match")
            ref_token_f1 = _metric(ref, "token_f1")
            if ref_docpage_hit and docpage_hit and ref_strict_top1 and strict_top1:
                if exact_match < ref_exact_match or token_f1 + 1e-6 < ref_token_f1:
                    visual_noise += 1

    total = max(1, len(records))
    return [
        {
            "run": run_label,
            "n": len(records),
            "retrieval_wrong_page": retrieval_wrong_page,
            "retrieval_wrong_page_rate": round(retrieval_wrong_page / total, 6),
            "same_page_wrong_block": same_page_wrong_block,
            "same_page_wrong_block_rate": round(same_page_wrong_block / total, 6),
            "answer_drift": answer_drift,
            "answer_drift_rate": round(answer_drift / total, 6),
            "visual_noise_introduced": visual_noise,
            "visual_noise_introduced_rate": round(visual_noise / total, 6),
        }
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_table(path: Path, title: str, headers: list[str], rows: list[list[str]], notes: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    if notes:
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate question-type, visual-sensitive, and error-type ablation tables.")
    parser.add_argument("--manifest", default="outputs/eval/docvqa_val_unique_docpage_100.manifest.jsonl", help="Benchmark manifest JSONL.")
    parser.add_argument("--outputs-root", default="outputs/eval", help="Directory containing eval run jsonl files.")
    parser.add_argument("--run", action="append", required=True, help="Run spec in the form stem=Label or /abs/path.jsonl=Label.")
    parser.add_argument("--reference-run", default="", help="Optional reference run spec used to compute visual_noise_introduced.")
    parser.add_argument("--output-prefix", default="outputs/eval/ablation_tables/docvqa_visual_ablation", help="Output prefix for generated tables.")
    args = parser.parse_args()

    outputs_root = (PROJECT_ROOT / args.outputs_root).resolve()
    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    output_prefix = (PROJECT_ROOT / args.output_prefix).resolve()

    sample_rows = read_jsonl(manifest_path)
    sample_meta: dict[str, dict[str, Any]] = {}
    sample_to_docpage: dict[str, str] = {}
    for row in sample_rows:
        sample_id = str(row.get("id") or "")
        raw_types = _sample_question_types(row)
        canonical_slices = _canonical_slices(str(row.get("question") or ""), raw_types)
        visual_sensitive = _is_visual_sensitive(str(row.get("question") or ""), raw_types, canonical_slices)
        sample_meta[sample_id] = {
            "raw_types": raw_types,
            "canonical_slices": canonical_slices,
            "visual_sensitive": visual_sensitive,
        }
        sample_to_docpage[sample_id] = _sample_doc_page_key(row)

    run_payloads: list[tuple[str, list[dict[str, Any]]]] = []
    for run_spec in args.run:
        run_stem, run_label = _parse_run_spec(run_spec)
        run_path = _resolve_run_path(outputs_root, run_stem)
        run_payloads.append((run_label, read_jsonl(run_path)))

    reference_records: dict[str, dict[str, Any]] | None = None
    if args.reference_run:
        reference_stem, _ = _parse_run_spec(args.reference_run)
        reference_path = _resolve_run_path(outputs_root, reference_stem)
        reference_records = {str(row.get("sample_id") or ""): row for row in read_jsonl(reference_path)}

    question_type_rows: list[dict[str, Any]] = []
    visual_subset_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for run_label, records in run_payloads:
        question_type_rows.extend(_build_question_type_rows(run_label, records, sample_meta))
        visual_subset_rows.extend(_build_visual_subset_rows(run_label, records, sample_meta))
        error_rows.extend(_build_error_rows(run_label, records, sample_to_docpage, reference_records))

    _write_csv(output_prefix.with_name(output_prefix.name + "_question_type_slices").with_suffix(".csv"), question_type_rows)
    _write_csv(output_prefix.with_name(output_prefix.name + "_visual_sensitive_subset").with_suffix(".csv"), visual_subset_rows)
    _write_csv(output_prefix.with_name(output_prefix.name + "_error_types").with_suffix(".csv"), error_rows)

    _write_markdown_table(
        output_prefix.with_name(output_prefix.name + "_question_type_slices").with_suffix(".md"),
        "Question-Type Slice Table",
        ["run", "slice", "n", "EM", "ANLS", "Token-F1", "Citation@1", "Hit@5"],
        [
            [
                row["run"],
                row["slice"],
                _fmt(row["n"]),
                _fmt(row["exact_match"]),
                _fmt(row["anls"]),
                _fmt(row["token_f1"]),
                _fmt(row["citation_accuracy"]),
                _fmt(row["hit_at_k"]),
            ]
            for row in question_type_rows
        ],
        notes=[
            "Slice labels are heuristic and derived from the benchmark question text plus `metadata.question_types`.",
            "A sample may contribute to multiple slices.",
        ],
    )
    _write_markdown_table(
        output_prefix.with_name(output_prefix.name + "_visual_sensitive_subset").with_suffix(".md"),
        "Visual-Sensitive Subset Table",
        ["run", "subset", "n", "EM", "ANLS", "Token-F1", "Citation@1", "Hit@5"],
        [
            [
                row["run"],
                row["subset"],
                _fmt(row["n"]),
                _fmt(row["exact_match"]),
                _fmt(row["anls"]),
                _fmt(row["token_f1"]),
                _fmt(row["citation_accuracy"]),
                _fmt(row["hit_at_k"]),
            ]
            for row in visual_subset_rows
        ],
        notes=[
            "The `visual_sensitive` subset is a heuristic union of logo/title/page-number/handwritten/chart/table oriented questions.",
            "The complementary `non_visual_sensitive` subset is included for contrast.",
        ],
    )
    _write_markdown_table(
        output_prefix.with_name(output_prefix.name + "_error_types").with_suffix(".md"),
        "Error-Type Table",
        [
            "run",
            "n",
            "retrieval_wrong_page",
            "retrieval_wrong_page_rate",
            "same_page_wrong_block",
            "same_page_wrong_block_rate",
            "answer_drift",
            "answer_drift_rate",
            "visual_noise_introduced",
            "visual_noise_introduced_rate",
        ],
        [[_fmt(row[key]) for key in [
            "run",
            "n",
            "retrieval_wrong_page",
            "retrieval_wrong_page_rate",
            "same_page_wrong_block",
            "same_page_wrong_block_rate",
            "answer_drift",
            "answer_drift_rate",
            "visual_noise_introduced",
            "visual_noise_introduced_rate",
        ]] for row in error_rows],
        notes=[
            "`retrieval_wrong_page`: no citation in top-k hits the gold original doc-page.",
            "`same_page_wrong_block`: top-k hits the gold doc-page but not the exact sample-level source.",
            "`answer_drift`: top-1 citation is correct but the final answer still misses strict EM.",
            "`visual_noise_introduced` is measured relative to `--reference-run` when provided.",
        ],
    )

    summary_payload = {
        "question_type_rows": question_type_rows,
        "visual_subset_rows": visual_subset_rows,
        "error_rows": error_rows,
    }
    output_prefix.with_suffix(".json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ablation-tables] question-type -> {output_prefix.with_name(output_prefix.name + '_question_type_slices').with_suffix('.md')}")
    print(f"[ablation-tables] visual-subset -> {output_prefix.with_name(output_prefix.name + '_visual_sensitive_subset').with_suffix('.md')}")
    print(f"[ablation-tables] error-types -> {output_prefix.with_name(output_prefix.name + '_error_types').with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
