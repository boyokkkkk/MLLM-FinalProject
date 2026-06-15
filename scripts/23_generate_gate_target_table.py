from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TARGET_SLICES = ["logo", "title", "page_number", "handwritten"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a compact gate-target slice table from existing ablation table JSON.")
    parser.add_argument("--input-json", default="outputs/eval/ablation_tables/docvqa_visual_assist.json")
    parser.add_argument("--output-md", default="outputs/eval/ablation_tables/docvqa_gate_target_slices.md")
    args = parser.parse_args()

    input_path = (PROJECT_ROOT / args.input_json).resolve()
    output_path = (PROJECT_ROOT / args.output_md).resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("question_type_rows", []) if row.get("slice") in TARGET_SLICES]
    rows.sort(key=lambda item: (TARGET_SLICES.index(str(item["slice"])), str(item["run"])))

    lines = [
        "# Gate-Target Slice Table",
        "",
        "- This table focuses only on the question categories that the generation-time gate is supposed to help: logo, title/heading, page number, and handwritten fields.",
        "- It is extracted from the rebuilt-run slice metrics and is intended to justify the gate design before finer policy variants are rerun.",
        "",
        "| slice | run | n | EM | ANLS | Token-F1 | Citation@1 | Hit@5 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['slice']} | {row['run']} | {row['n']} | {row['exact_match']:.6f} | {row['anls']:.6f} | {row['token_f1']:.6f} | {row['citation_accuracy']:.6f} | {row['hit_at_k']:.6f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[gate-target-table] table -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
