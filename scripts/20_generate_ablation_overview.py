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


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_summary_path(outputs_root: Path, stem_or_path: str) -> Path:
    candidate = Path(stem_or_path)
    if candidate.exists():
        return candidate
    if candidate.suffix:
        return (outputs_root / candidate.name).resolve()
    return (outputs_root / f"{stem_or_path}.summary.json").resolve()


def _parse_run_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        stem, label = spec.split("=", 1)
        return stem.strip(), label.strip()
    value = spec.strip()
    return value, value


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one overview table for the four ablation groups.")
    parser.add_argument("--outputs-root", default="outputs/eval", help="Directory containing eval outputs.")
    parser.add_argument("--output-prefix", default="outputs/eval/ablation_tables/docvqa_full_ablation_overview", help="Output prefix.")
    args = parser.parse_args()

    outputs_root = (PROJECT_ROOT / args.outputs_root).resolve()
    output_prefix = (PROJECT_ROOT / args.output_prefix).resolve()

    groups: list[tuple[str, str, str]] = [
        ("index", "docvqa_val_unique_docpage_100_ablation_index_page_text=page_text", "page-level text aggregation only"),
        ("index", "docvqa_val_unique_docpage_100_ablation_index_block_text=block_text", "block-level textual chunks only"),
        ("index", "docvqa_val_unique_docpage_100_ablation_index_block_multimodal=block_multimodal", "current multimodal block chunk store"),
        ("retrieval", "docvqa_val_unique_docpage_100_ablation_retrieval_basic=basic", "basic sparse rerank"),
        ("retrieval", "docvqa_val_unique_docpage_100_ablation_retrieval_stronger=stronger", "stronger rerank without QTA/QIA"),
        ("retrieval", "docvqa_val_unique_docpage_100_ablation_retrieval_stronger_qta=stronger_qta", "stronger rerank plus query-type-aware heuristics"),
        ("retrieval", "docvqa_val_unique_docpage_100_ablation_retrieval_stronger_qia=stronger_qia", "stronger rerank plus query-image-aware rerank"),
        ("visual", "docvqa_val_unique_docpage_100_ablation_visual_densefusion=retrieval_densefusion", "visual signal injected at retrieval time"),
        ("visual", "docvqa_val_unique_docpage_100_rag_stronger_qia_base=text_only_rag", "grounding-first rebuilt RAG baseline"),
        ("visual", "docvqa_val_unique_docpage_100_rag_stronger_qia_visualassist_v1=visualassist_always", "generation-time visual assist without gating"),
        ("visual", "docvqa_val_unique_docpage_100_rag_stronger_qia_visualassist_gated_v2=visualassist_gated", "generation-time visual assist with heuristic gating"),
        ("gating", "docvqa_val_unique_docpage_100_rag_stronger_qia_base=text_only_rag", "no generation-time visual assist"),
        ("gating", "docvqa_val_unique_docpage_100_rag_stronger_qia_visualassist_v1=visualassist_always", "visual assist always on"),
        ("gating", "docvqa_val_unique_docpage_100_rag_stronger_qia_visualassist_gated_v2=visualassist_gated", "visual assist only on visual-sensitive questions"),
    ]

    rows: list[dict[str, Any]] = []
    for group_name, run_spec, note in groups:
        stem, label = _parse_run_spec(run_spec)
        payload = _load_summary(_resolve_summary_path(outputs_root, stem))
        overall = payload.get("summary", {}).get("overall", {})
        rows.append(
            {
                "group": group_name,
                "run": label,
                "mode": payload.get("meta", {}).get("mode", ""),
                "hit_at_k": overall.get("hit_at_k"),
                "citation_accuracy": overall.get("citation_accuracy"),
                "exact_match": overall.get("exact_match"),
                "anls": overall.get("anls"),
                "token_f1": overall.get("token_f1"),
                "note": note,
            }
        )

    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Full Ablation Overview",
        "",
        "- `index` and `retrieval` groups are newly rerun on the rebuilt `unique-docpage-100` benchmark.",
        "- `visual` and `gating` groups reuse the stable rebuilt RAG runs already present in `outputs/eval` because the current DashScope VL endpoint rejects the synthetic text-only batch runner configuration.",
        "",
        "| group | run | mode | Hit@5 | Citation@1 | EM | ANLS | Token-F1 | note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['run']} | {row['mode']} | {_fmt(row['hit_at_k'])} | {_fmt(row['citation_accuracy'])} | {_fmt(row['exact_match'])} | {_fmt(row['anls'])} | {_fmt(row['token_f1'])} | {row['note']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ablation-overview] overview -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
