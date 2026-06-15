from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_index_utils import read_jsonl


def _parse_run_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        stem, label = spec.split("=", 1)
        return stem.strip(), label.strip()
    value = spec.strip()
    return value, value


def _resolve_run_path(outputs_root: Path, stem_or_path: str) -> Path:
    candidate = Path(stem_or_path)
    if candidate.exists():
        return candidate
    if candidate.suffix:
        return (outputs_root / candidate.name).resolve()
    return (outputs_root / f"{stem_or_path}.jsonl").resolve()


def _metric(record: dict[str, Any], key: str) -> float:
    return float((record.get("metrics") or {}).get(key, 0.0))


def _mean(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return round(sum(_metric(record, key) for record in records) / len(records), 6)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, title: str, headers: list[str], rows: list[list[str]], notes: list[str]) -> None:
    lines = [f"# {title}", ""]
    for note in notes:
        lines.append(f"- {note}")
    lines.extend(["", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate strict-visual-subset and gate-detail tables from existing run jsonl files.")
    parser.add_argument("--strict-subset-manifest", default="outputs/eval/docvqa_val_unique_docpage_100.strict_visual.manifest.jsonl")
    parser.add_argument("--outputs-root", default="outputs/eval")
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output-prefix", default="outputs/eval/ablation_tables/docvqa_strict_visual")
    args = parser.parse_args()

    subset_manifest = (PROJECT_ROOT / args.strict_subset_manifest).resolve()
    outputs_root = (PROJECT_ROOT / args.outputs_root).resolve()
    output_prefix = (PROJECT_ROOT / args.output_prefix).resolve()

    subset_ids = {str(row.get("id") or "") for row in read_jsonl(subset_manifest)}
    rows: list[dict[str, Any]] = []
    for run_spec in args.run:
        stem, label = _parse_run_spec(run_spec)
        run_path = _resolve_run_path(outputs_root, stem)
        records = read_jsonl(run_path)
        subset_records = [record for record in records if str(record.get("sample_id") or "") in subset_ids]
        rows.append(
            {
                "run": label,
                "n": len(subset_records),
                "exact_match": _mean(subset_records, "exact_match"),
                "anls": _mean(subset_records, "anls"),
                "token_f1": _mean(subset_records, "token_f1"),
                "citation_accuracy": _mean(subset_records, "citation_accuracy"),
                "hit_at_k": _mean(subset_records, "hit_at_k"),
            }
        )

    _write_csv(output_prefix.with_suffix(".csv"), rows)
    _write_md(
        output_prefix.with_suffix(".md"),
        "Strict Visual Subset Table",
        ["run", "n", "EM", "ANLS", "Token-F1", "Citation@1", "Hit@5"],
        [[row["run"], str(row["n"]), f"{row['exact_match']:.6f}", f"{row['anls']:.6f}", f"{row['token_f1']:.6f}", f"{row['citation_accuracy']:.6f}", f"{row['hit_at_k']:.6f}"] for row in rows],
        notes=[
            "This subset keeps only questions that clearly require local visual recognition: logo/pack text, title/heading, page number, and handwritten fields.",
            "All rows are recomputed by filtering the per-sample run JSONL files against the strict subset manifest.",
        ],
    )
    print(f"[strict-visual-table] table -> {output_prefix.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
