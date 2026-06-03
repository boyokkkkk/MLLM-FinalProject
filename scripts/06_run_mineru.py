from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_index_utils import extract_image_path, read_jsonl, stable_id, write_json


def _resolve_local_path(project_root: Path, dataset: str, split: str, image_value: Any) -> Path | None:
    image_path = extract_image_path(image_value)
    if not image_path:
        return None
    p = Path(image_path)
    candidates = [
        p if p.is_absolute() else project_root / p,
        project_root / "data" / "images" / dataset / split / p.name,
        project_root / "data" / "images" / dataset / split / image_path,
    ]
    return next((x for x in candidates if x and x.exists()), None)


def _item_text(item: dict[str, Any]) -> str:
    for key in ("text", "content", "html", "latex", "table_body", "table_caption", "image_caption"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            joined = " ".join(str(v) for v in value if v)
            if joined.strip():
                return joined.strip()
    return ""


def _item_type(item: dict[str, Any]) -> str:
    raw = str(item.get("type") or item.get("category") or item.get("block_type") or "text").lower()
    if "table" in raw:
        return "table"
    if "equation" in raw or "formula" in raw or "latex" in raw:
        return "formula"
    if "image" in raw or "figure" in raw or "chart" in raw:
        return "figure"
    if "title" in raw:
        return "title"
    return "text"


def _normalize_content_list(payload: Any, default_page_no: int) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("content", "blocks", "items", "spans"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return []

    blocks: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        text = _item_text(item)
        block_type = _item_type(item)
        bbox = item.get("bbox") or item.get("poly") or item.get("position")
        page_val = item.get("page_no", item.get("page", item.get("page_idx", default_page_no)))
        try:
            page_no = int(page_val) + 1 if "page_idx" in item and "page_no" not in item else int(page_val)
        except (TypeError, ValueError):
            page_no = default_page_no
        blocks.append(
            {
                "type": block_type,
                "text": text or f"{block_type} region",
                "bbox": bbox,
                "page_no": page_no,
                "metadata": {"raw_index": idx, "raw_type": item.get("type")},
            }
        )
    return blocks


def _extract_blocks_from_mineru_output(output_dir: Path, default_page_no: int) -> tuple[list[dict[str, Any]], list[str]]:
    json_files = sorted(output_dir.rglob("*.json"))
    priority = sorted(
        json_files,
        key=lambda p: (
            0 if "content_list" in p.name else 1 if "middle" in p.name else 2,
            len(p.parts),
            p.name,
        ),
    )
    raw_files = [str(p) for p in priority]
    for path in priority:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        blocks = _normalize_content_list(payload, default_page_no)
        if blocks:
            return blocks, raw_files

    md_files = sorted(output_dir.rglob("*.md"))
    for path in md_files:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            return [
                {
                    "type": "text",
                    "text": text,
                    "bbox": None,
                    "page_no": default_page_no,
                    "metadata": {"raw_file": str(path), "from_markdown": True},
                }
            ], raw_files + [str(p) for p in md_files]
    return [], raw_files + [str(p) for p in md_files]


def _make_mock_blocks(sample: dict[str, Any], page_no: int) -> list[dict[str, Any]]:
    question = str(sample.get("question", ""))
    answers = sample.get("answers") or []
    if not isinstance(answers, list):
        answers = [answers]
    return [
        {
            "type": "text",
            "text": f"Question: {question}",
            "bbox": [0, 0, 1000, 160],
            "page_no": page_no,
            "metadata": {"mock": True},
        },
        {
            "type": "figure",
            "text": f"Page image associated with question. Expected answer: {'; '.join(str(a) for a in answers)}",
            "bbox": [0, 160, 1000, 1000],
            "page_no": page_no,
            "metadata": {"mock": True},
        },
    ]


def run_mineru_on_sample(
    project_root: Path,
    sample: dict[str, Any],
    image_path: Path,
    output_root: Path,
    backend: str,
    method: str,
    mock: bool,
) -> dict[str, Any]:
    dataset = str(sample.get("dataset", "unknown"))
    split = str(sample.get("split", "unknown"))
    sample_id = str(sample.get("id") or stable_id(dataset, split, image_path))
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    try:
        page_no = int(metadata.get("ucsf_document_page_no") or metadata.get("page_no") or 1)
    except (TypeError, ValueError):
        page_no = 1

    normalized_path = project_root / "data" / "interim" / "mineru" / dataset / f"{sample_id}.json"
    raw_output_dir = output_root / dataset / split / sample_id

    if mock:
        blocks = _make_mock_blocks(sample, page_no)
        raw_files: list[str] = []
    else:
        mineru_bin = shutil.which("mineru") or shutil.which("magic-pdf")
        if not mineru_bin:
            raise RuntimeError("MinerU CLI not found. Install it with: python -m pip install -U 'mineru[core]' or uv pip install -U 'mineru[core]'.")
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [mineru_bin, "-p", str(image_path), "-o", str(raw_output_dir)]
        if Path(mineru_bin).name == "mineru":
            cmd.extend(["-b", backend])
        elif method:
            cmd.extend(["-m", method])
        print(f"[mineru] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(project_root), check=True)
        blocks, raw_files = _extract_blocks_from_mineru_output(raw_output_dir, page_no)
        if not blocks:
            raise RuntimeError(f"MinerU finished but no parseable blocks were found under {raw_output_dir}")

    result = {
        "sample_id": sample_id,
        "dataset": dataset,
        "split": split,
        "source_path": str(image_path),
        "raw_output_dir": str(raw_output_dir),
        "raw_files": raw_files,
        "blocks": blocks,
    }
    write_json(normalized_path, result)
    print(f"[mineru] normalized -> {normalized_path} blocks={len(blocks)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MinerU on local image/PDF samples and normalize blocks for the chunking script.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--datasets", default="docvqa,chartqa", help="Comma-separated datasets.")
    parser.add_argument("--splits", default="val,test", help="Comma-separated splits.")
    parser.add_argument("--limit-per-split", type=int, default=1, help="Max samples per dataset split.")
    parser.add_argument("--output-root", default="data/interim/mineru_raw", help="Raw MinerU output root.")
    parser.add_argument("--backend", default="pipeline", help="MinerU backend, e.g. pipeline.")
    parser.add_argument("--method", default="auto", help="magic-pdf method fallback: auto/ocr/txt.")
    parser.add_argument("--mock", action="store_true", help="Write MinerU-compatible mock blocks without invoking MinerU.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    output_root = project_root / args.output_root
    total = 0
    for dataset in datasets:
        for split in splits:
            rows = read_jsonl(project_root / "data" / "processed" / dataset / f"{split}.jsonl")
            kept = 0
            for sample in rows:
                image_path = _resolve_local_path(project_root, dataset, split, sample.get("image"))
                if image_path is None:
                    continue
                run_mineru_on_sample(project_root, sample, image_path, output_root, args.backend, args.method, args.mock)
                kept += 1
                total += 1
                if args.limit_per_split and kept >= args.limit_per_split:
                    break
    print(f"[mineru] done samples={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
