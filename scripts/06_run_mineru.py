from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_index_utils import extract_image_path, read_jsonl, stable_id, write_json

SUPPORTED_INPUT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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


def _resolve_source_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _iter_input_samples(project_root: Path, input_path: str, dataset: str, split: str, document_id: str | None) -> list[tuple[dict[str, Any], Path]]:
    source = _resolve_source_path(project_root, input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")

    paths: list[Path]
    if source.is_dir():
        paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_SUFFIXES)
    else:
        if source.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            raise ValueError(f"Unsupported input suffix {source.suffix!r}. Supported: {sorted(SUPPORTED_INPUT_SUFFIXES)}")
        paths = [source]

    samples: list[tuple[dict[str, Any], Path]] = []
    for idx, path in enumerate(paths):
        sample_id = document_id if document_id and len(paths) == 1 else path.stem
        sample = {
            "id": sample_id,
            "dataset": dataset,
            "split": split,
            "source_path": str(path),
            "metadata": {
                "input_mode": "raw_document",
                "source_suffix": path.suffix.lower(),
                "source_name": path.name,
                "raw_input_index": idx,
            },
        }
        samples.append((sample, path))
    return samples


def _item_text(item: dict[str, Any]) -> str:
    for key in ("text", "content", "html", "latex", "table_body", "table_caption", "image_caption", "caption"):
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


def _item_image_path(item: dict[str, Any]) -> str | None:
    for key in ("image_path", "img_path", "image", "path"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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
        raw_block_id = item.get("id") or item.get("block_id")
        block_id = str(raw_block_id) if raw_block_id not in (None, "") else f"block-{idx:04d}"
        blocks.append(
            {
                "block_id": block_id,
                "type": block_type,
                "text": text or f"{block_type} region",
                "bbox": bbox,
                "page_no": page_no,
                "image_path": _item_image_path(item),
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
                    "block_id": "block-0000",
                    "type": "text",
                    "text": text,
                    "bbox": None,
                    "page_no": default_page_no,
                    "image_path": None,
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
            "block_id": "block-0000",
            "type": "text",
            "text": f"Question: {question}" if question else "Mock text block",
            "bbox": [0, 0, 1000, 160],
            "page_no": page_no,
            "image_path": None,
            "metadata": {"mock": True},
        },
        {
            "block_id": "block-0001",
            "type": "figure",
            "text": f"Page image associated with question. Expected answer: {'; '.join(str(a) for a in answers)}" if question else "Mock figure block",
            "bbox": [0, 160, 1000, 1000],
            "page_no": page_no,
            "image_path": None,
            "metadata": {"mock": True},
        },
    ]


def run_mineru_on_sample(
    project_root: Path,
    sample: dict[str, Any],
    source_path: Path,
    output_root: Path,
    backend: str,
    method: str,
    mock: bool,
    mineru_model_source: str,
    start_page: int | None = None,
    end_page: int | None = None,
) -> dict[str, Any]:
    dataset = str(sample.get("dataset", "unknown"))
    split = str(sample.get("split", "unknown"))
    sample_id = str(sample.get("id") or stable_id(dataset, split, source_path))
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    try:
        page_no = int(metadata.get("ucsf_document_page_no") or metadata.get("page_no") or 1)
    except (TypeError, ValueError):
        page_no = 1

    document_id = str(sample.get("document_id") or f"doc-{stable_id(dataset, split, sample_id, source_path)}")
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
        cmd = [mineru_bin, "-p", str(source_path), "-o", str(raw_output_dir)]
        if Path(mineru_bin).name == "mineru":
            cmd.extend(["-b", backend])
            if start_page is not None:
                cmd.extend(["-s", str(start_page)])
            if end_page is not None:
                cmd.extend(["-e", str(end_page)])
        elif method:
            cmd.extend(["-m", method])
        env = os.environ.copy()
        if mineru_model_source:
            env["MINERU_MODEL_SOURCE"] = mineru_model_source
        print(f"[mineru] {' '.join(cmd)}")
        print(f"[mineru] MINERU_MODEL_SOURCE={env.get('MINERU_MODEL_SOURCE', '')}")
        subprocess.run(cmd, cwd=str(project_root), check=True, env=env)
        blocks, raw_files = _extract_blocks_from_mineru_output(raw_output_dir, page_no)
        if not blocks:
            raise RuntimeError(f"MinerU finished but no parseable blocks were found under {raw_output_dir}")

    result = {
        "document_id": document_id,
        "sample_id": sample_id,
        "dataset": dataset,
        "split": split,
        "source_path": str(source_path),
        "source_type": source_path.suffix.lower().lstrip(".") or "file",
        "raw_output_dir": str(raw_output_dir),
        "raw_files": raw_files,
        "metadata": metadata,
        "blocks": blocks,
    }
    write_json(normalized_path, result)
    print(f"[mineru] normalized -> {normalized_path} blocks={len(blocks)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MinerU on local image/PDF samples and normalize blocks for the chunking script.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--datasets", default="docvqa,chartqa", help="Comma-separated datasets for processed QA samples.")
    parser.add_argument("--splits", default="val,test", help="Comma-separated splits for processed QA samples.")
    parser.add_argument("--input-path", default="", help="Optional raw PDF/image file or directory. When set, processed QA JSONL is not read.")
    parser.add_argument("--document-id", default="", help="Optional document/sample id for a single --input-path file.")
    parser.add_argument("--limit-per-split", type=int, default=1, help="Max samples per dataset split. 0 means all.")
    parser.add_argument("--output-root", default="data/interim/mineru_raw", help="Raw MinerU output root.")
    parser.add_argument("--backend", default="pipeline", help="MinerU backend, e.g. pipeline.")
    parser.add_argument("--method", default="auto", help="magic-pdf method fallback: auto/ocr/txt.")
    parser.add_argument("--start-page", type=int, default=None, help="Optional zero-based first PDF page for MinerU CLI.")
    parser.add_argument("--end-page", type=int, default=None, help="Optional zero-based last PDF page for MinerU CLI.")
    parser.add_argument("--mock", action="store_true", help="Write MinerU-compatible mock blocks without invoking MinerU.")
    parser.add_argument(
        "--mineru-model-source",
        default=os.getenv("MINERU_MODEL_SOURCE", "modelscope"),
        choices=["modelscope", "huggingface", "local"],
        help="MinerU model source. ModelScope is recommended in China.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    output_root = project_root / args.output_root
    total = 0

    if args.input_path:
        dataset = datasets[0] if datasets else "raw_documents"
        split = splits[0] if splits else "raw"
        for sample, source_path in _iter_input_samples(project_root, args.input_path, dataset, split, args.document_id or None):
            run_mineru_on_sample(
                project_root,
                sample,
                source_path,
                output_root,
                args.backend,
                args.method,
                args.mock,
                args.mineru_model_source,
                args.start_page,
                args.end_page,
            )
            total += 1
            if args.limit_per_split and total >= args.limit_per_split:
                break
        print(f"[mineru] done samples={total}")
        return 0

    for dataset in datasets:
        for split in splits:
            rows = read_jsonl(project_root / "data" / "processed" / dataset / f"{split}.jsonl")
            kept = 0
            for sample in rows:
                source_path = _resolve_local_path(project_root, dataset, split, sample.get("image"))
                if source_path is None:
                    continue
                run_mineru_on_sample(
                    project_root,
                    sample,
                    source_path,
                    output_root,
                    args.backend,
                    args.method,
                    args.mock,
                    args.mineru_model_source,
                    args.start_page,
                    args.end_page,
                )
                kept += 1
                total += 1
                if args.limit_per_split and kept >= args.limit_per_split:
                    break
    print(f"[mineru] done samples={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
