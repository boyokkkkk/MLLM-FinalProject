from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_index_utils import read_jsonl
from src.models.clients import build_embedding_client, build_llm_client
from src.utils.settings import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual descriptor index from region crops using the VLM.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--config", default="configs/datasets.yaml", help="Dataset/index config path.")
    parser.add_argument("--datasets", default="docvqa", help="Comma separated datasets.")
    parser.add_argument("--splits", default="val", help="Comma separated splits.")
    parser.add_argument("--limit-per-split", type=int, default=100, help="Sample cap per split.")
    parser.add_argument("--chunk-types", default="figure,table", help="Visual chunk types to include.")
    parser.add_argument("--max-items", type=int, default=0, help="Optional cap on visual items after filtering.")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent VLM caption requests.")
    parser.add_argument("--embedding-batch-size", type=int, default=8, help="Batch size for descriptor embeddings.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild descriptors even if they already exist.")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _load_target_sample_ids(project_root: Path, datasets: list[str], splits: list[str], limit_per_split: int) -> set[str]:
    sample_ids: set[str] = set()
    for dataset in datasets:
        for split in splits:
            rows = read_jsonl(project_root / "data" / "processed" / dataset / f"{split}.jsonl")
            if limit_per_split > 0:
                rows = rows[:limit_per_split]
            for row in rows:
                sample_ids.add(str(row.get("id")))
    return sample_ids


def _crop_region_to_data_url(source_path: str, bbox: list[int] | None) -> str | None:
    image_path = Path(source_path)
    if not image_path.exists():
        return None
    with Image.open(image_path) as image:
        crop = image.convert("RGB")
        if bbox and len(bbox) == 4:
            left, top, right, bottom = [max(0, int(v)) for v in bbox]
            right = min(right, crop.width)
            bottom = min(bottom, crop.height)
            if right > left and bottom > top:
                crop = crop.crop((left, top, right, bottom))
        buf = BytesIO()
        crop.save(buf, format="PNG")
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _prompt_for_item(item: dict[str, Any], ocr_hint: str) -> str:
    chunk_type = str(item.get("chunk_type", "")).lower()
    if chunk_type == "table":
        return (
            "You are writing a one-line retrieval descriptor for a document table crop. "
            "Describe the table topic, key row/column entities, important dates, years, numeric fields, units, and what question it can answer. "
            f"OCR hint: {ocr_hint or 'N/A'}. "
            "Return exactly one concise line."
        )
    return (
        "You are writing a one-line retrieval descriptor for a document figure or chart crop. "
        "Describe the chart or figure type, key labels, legend terms, years, numbers, units, and what question it can answer. "
        f"OCR hint: {ocr_hint or 'N/A'}. "
        "Return exactly one concise line."
    )


def _clean_descriptor(text: str) -> str:
    value = " ".join((text or "").strip().split())
    return value[:500]


async def _describe_item(
    item: dict[str, Any],
    doc_store: dict[str, dict[str, Any]],
    llm_client,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    source_path = str(item.get("source_path") or "")
    data_url = _crop_region_to_data_url(source_path, item.get("bbox"))
    if not data_url:
        return None

    chunk = doc_store.get(str(item.get("chunk_id") or ""), {})
    ocr_hint = str(chunk.get("text", "")).strip()
    prompt = _prompt_for_item(item, ocr_hint)
    messages = [
        {"role": "system", "content": "You produce concise retrieval descriptors for document image regions."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    async with semaphore:
        try:
            output = await llm_client.chat(messages=messages, temperature=0.0, max_tokens=120)
        except Exception:
            return None

    descriptor = _clean_descriptor(output)
    if not descriptor:
        return None
    return {
        "chunk_id": str(item.get("chunk_id") or ""),
        "sample_id": str(item.get("sample_id") or ""),
        "dataset": str(item.get("dataset") or ""),
        "split": str(item.get("split") or ""),
        "chunk_type": str(item.get("chunk_type") or ""),
        "source_ref": str(item.get("source_ref") or ""),
        "source_path": source_path,
        "image_path": str(item.get("image_path") or ""),
        "page_no": item.get("page_no"),
        "bbox": item.get("bbox"),
        "text": descriptor,
        "ocr_hint": ocr_hint[:500],
    }


async def main_async() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = (project_root / args.config).resolve()
    cfg = _load_yaml(config_path)
    index_cfg = cfg.get("indexing", {})
    datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    chunk_types = {part.strip() for part in args.chunk_types.split(",") if part.strip()}

    sample_ids = _load_target_sample_ids(project_root, datasets, splits, args.limit_per_split)
    visual_path = project_root / index_cfg.get("vision_index_dir", "data/processed/indexes/vision") / "visual_store.json"
    doc_store_path = project_root / index_cfg.get("text_index_dir", "data/processed/indexes/text") / "doc_store.json"
    visual_items = _load_json(visual_path)
    doc_store = _load_json(doc_store_path)
    if not isinstance(visual_items, list) or not isinstance(doc_store, dict):
        raise RuntimeError("Invalid visual_store/doc_store payload.")

    filtered = [
        item for item in visual_items
        if isinstance(item, dict)
        and str(item.get("sample_id") or "") in sample_ids
        and str(item.get("chunk_type") or "") in chunk_types
    ]
    filtered.sort(key=lambda item: (str(item.get("sample_id")), int(item.get("page_no") or 0), str(item.get("chunk_id"))))
    if args.max_items > 0:
        filtered = filtered[: args.max_items]

    metadata_path = settings.retrieval.visual_dense_metadata_path
    vector_path = settings.retrieval.visual_dense_vectors_path
    existing_by_chunk: dict[str, dict[str, Any]] = {}
    if metadata_path.exists() and not args.overwrite:
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing_by_chunk[str(row.get("chunk_id") or "")] = row

    llm_client = build_llm_client(settings.vlm)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    tasks = []
    for item in filtered:
        if not args.overwrite and str(item.get("chunk_id") or "") in existing_by_chunk:
            continue
        tasks.append(_describe_item(item, doc_store, llm_client, semaphore))

    new_rows: list[dict[str, Any]] = []
    total = len(tasks)
    for index, coro in enumerate(asyncio.as_completed(tasks), start=1):
        row = await coro
        if row:
            new_rows.append(row)
        if total:
            print(f"[visual-descriptor] described {index}/{total}")

    merged_rows = list(existing_by_chunk.values()) + new_rows if not args.overwrite else new_rows
    merged_rows.sort(key=lambda row: (str(row.get("sample_id")), int(row.get("page_no") or 0), str(row.get("chunk_id"))))
    _write_jsonl(metadata_path, merged_rows)

    embedding_client = build_embedding_client(settings.text_embedding)
    descriptors = [str(row.get("text", "")) for row in merged_rows]
    vectors: list[list[float]] = []
    batch_size = max(1, args.embedding_batch_size)
    for start in range(0, len(descriptors), batch_size):
        batch = descriptors[start : start + batch_size]
        try:
            batch_vectors = await embedding_client.embed(batch)
        except Exception:
            batch_vectors = []
            for item in batch:
                batch_vectors.extend(await embedding_client.embed([item]))
        vectors.extend(batch_vectors)
        print(f"[visual-descriptor] embedded {min(start + batch_size, len(descriptors))}/{len(descriptors)}")

    _write_json(vector_path, {"vectors": vectors})
    print(f"[visual-descriptor] items={len(merged_rows)}")
    print(f"[visual-descriptor] metadata -> {metadata_path}")
    print(f"[visual-descriptor] vectors -> {vector_path}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
