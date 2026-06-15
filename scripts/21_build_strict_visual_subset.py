from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_index_utils import read_jsonl


STRICT_VISUAL_MARKERS = (
    "logo",
    "pack",
    "written on the pack",
    "written within the logo",
    "brand",
    "tagline",
    "title",
    "heading",
    "subheading",
    "subject",
    "page number",
    "page no",
    "handwritten",
)


def _question_types(row: dict[str, Any]) -> list[str]:
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    raw = metadata.get("question_types", [])
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if raw in (None, ""):
        return []
    return [str(raw)]


def _is_strict_visual_sample(row: dict[str, Any]) -> tuple[bool, list[str]]:
    query = str(row.get("question") or "").strip().lower()
    raw_types = {item.strip().lower() for item in _question_types(row) if item}
    reasons: list[str] = []

    for marker in STRICT_VISUAL_MARKERS:
        if marker in query:
            reasons.append(f"query:{marker}")

    if "handwritten" in raw_types:
        reasons.append("type:handwritten")
    if "image/photo" in raw_types and any(marker in query for marker in ("logo", "pack", "brand", "written")):
        reasons.append("type:image_photo+query_visual_text")
    if "layout" in raw_types and any(marker in query for marker in ("title", "heading", "page number", "page no", "tagline", "subject")):
        reasons.append("type:layout+query_title_page")

    return bool(reasons), reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a stricter visual-sensitive benchmark subset manifest.")
    parser.add_argument("--input-manifest", default="outputs/eval/docvqa_val_unique_docpage_100.manifest.jsonl", help="Source manifest JSONL.")
    parser.add_argument("--output-manifest", default="outputs/eval/docvqa_val_unique_docpage_100.strict_visual.manifest.jsonl", help="Subset manifest JSONL.")
    parser.add_argument("--output-summary", default="outputs/eval/docvqa_val_unique_docpage_100.strict_visual.summary.json", help="Subset summary JSON.")
    args = parser.parse_args()

    input_path = (PROJECT_ROOT / args.input_manifest).resolve()
    output_path = (PROJECT_ROOT / args.output_manifest).resolve()
    summary_path = (PROJECT_ROOT / args.output_summary).resolve()

    rows = read_jsonl(input_path)
    selected: list[dict[str, Any]] = []
    reasons_by_id: dict[str, list[str]] = {}
    for row in rows:
        keep, reasons = _is_strict_visual_sample(row)
        if not keep:
            continue
        selected.append(row)
        reasons_by_id[str(row.get("id") or "")] = reasons

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "source_manifest": str(input_path),
        "output_manifest": str(output_path),
        "num_rows": len(selected),
        "sample_ids": [str(row.get("id") or "") for row in selected],
        "reasons_by_id": reasons_by_id,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[strict-visual-subset] rows={len(selected)}")
    print(f"[strict-visual-subset] manifest -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
