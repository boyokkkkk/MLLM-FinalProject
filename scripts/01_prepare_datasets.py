from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetSpec:
    name: str
    raw_root: Path
    processed_root: Path
    expected_files: list[str]
    eval_required_files: list[str]


def _infer_image_ext(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return ".gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def _resolve_image_path(image_obj: Any, image_out_dir: Path, sample_id: str) -> str | None:
    if image_obj is None:
        return None

    if isinstance(image_obj, str):
        return image_obj

    if isinstance(image_obj, dict):
        path_val = image_obj.get("path")
        bytes_val = image_obj.get("bytes")
        if path_val:
            return str(path_val)
        if isinstance(bytes_val, (bytes, bytearray)):
            image_out_dir.mkdir(parents=True, exist_ok=True)
            payload = bytes(bytes_val)
            digest = hashlib.sha1(payload).hexdigest()[:12]
            ext = _infer_image_ext(payload)
            filename = f"{sample_id}-{digest}{ext}"
            out = image_out_dir / filename
            if not out.exists():
                out.write_bytes(payload)
            return str(out)
        return None

    # datasets.load_from_disk may decode image columns to PIL image objects.
    # Persist them to files so downstream pipeline has concrete image paths.
    if hasattr(image_obj, "save") and hasattr(image_obj, "mode") and hasattr(image_obj, "size"):
        image_out_dir.mkdir(parents=True, exist_ok=True)
        fmt = str(getattr(image_obj, "format", "") or "").upper()
        ext = ".png" if fmt == "PNG" else ".jpg" if fmt in {"JPEG", "JPG"} else ".png"
        out = image_out_dir / f"{sample_id}{ext}"
        if not out.exists():
            image_obj.save(out)
        return str(out)

    return str(image_obj)


def _safe_read_json(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        rows: list[Any] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _pick_first(d: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _normalize_record(dataset: str, split: str, record: dict[str, Any], idx: int) -> dict[str, Any]:
    qid = _pick_first(record, ["id", "question_id", "qid", "sample_id"], default=f"{dataset}-{split}-{idx}")
    question = _pick_first(record, ["question", "query", "prompt"], default="")
    answers = _ensure_list(_pick_first(record, ["answers", "answer", "label"], default=[]))
    image = _pick_first(record, ["image", "image_path", "img", "img_path"], default=None)

    normalized = {
        "id": str(qid),
        "dataset": dataset,
        "split": split,
        "question": str(question),
        "answers": answers,
        "image": image,
        "evidence": _pick_first(record, ["evidence", "context", "ocr"], default=None),
        "metadata": {
            "raw_keys": sorted(list(record.keys())),
        },
    }
    return normalized


def _normalize_split(spec: DatasetSpec, split: str) -> tuple[int, int]:
    src_json = spec.raw_root / f"{split}.json"
    src_jsonl = spec.raw_root / f"{split}.jsonl"
    if src_json.exists():
        src = src_json
    elif src_jsonl.exists():
        src = src_jsonl
    else:
        raise FileNotFoundError(f"Missing split file: {src_json} or {src_jsonl}")
    dst_dir = spec.processed_root
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{split}.jsonl"

    data = _safe_read_json(src)
    if isinstance(data, dict):
        for key in ["data", "samples", "items", "questions"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON format for {src}. Expected a list or dict containing a list.")

    total = len(data)
    kept = 0
    with dst.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            norm = _normalize_record(spec.name, split, item, idx)
            if not norm["question"]:
                continue
            f.write(json.dumps(norm, ensure_ascii=False) + "\n")
            kept += 1

    return total, kept


def _normalize_split_from_hf(
    spec: DatasetSpec,
    split: str,
    project_root: Path,
    image_export_root: Path,
    show_progress: bool = False,
    progress_every: int = 500,
) -> tuple[int, int]:
    from datasets import load_from_disk

    hf_root = project_root / "data" / "raw_hf" / spec.name
    if not hf_root.exists():
        raise FileNotFoundError(f"HF cache root not found: {hf_root}")

    ds_dict = load_from_disk(str(hf_root))
    split_name = split
    if split == "val" and "validation" in ds_dict:
        split_name = "validation"
    if split_name not in ds_dict:
        raise KeyError(f"Split '{split}' not found in {hf_root}, available={list(ds_dict.keys())}")

    ds = ds_dict[split_name]
    dst_dir = spec.processed_root
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{split}.jsonl"

    total = len(ds)
    kept = 0
    image_out_dir = image_export_root / spec.name / split
    iterator = ds
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(ds, total=total, desc=f"{spec.name}:{split}", unit="sample")
        except Exception:
            iterator = ds

    with dst.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(iterator):
            if not isinstance(item, dict):
                continue
            norm = _normalize_record(spec.name, split, item, idx)
            norm["image"] = _resolve_image_path(item.get("image"), image_out_dir, norm["id"])
            if not norm["question"]:
                continue
            f.write(json.dumps(norm, ensure_ascii=False) + "\n")
            kept += 1
            if show_progress and progress_every > 0 and (idx + 1) % progress_every == 0:
                print(f"[progress] {spec.name}/{split}: {idx + 1}/{total}, kept={kept}")

    return total, kept


def validate_layout(spec: DatasetSpec) -> list[str]:
    return validate_layout_for_mode(spec, mode="train")


def validate_layout_for_mode(spec: DatasetSpec, mode: str) -> list[str]:
    errors: list[str] = []
    if not spec.raw_root.exists():
        errors.append(f"{spec.name}: raw root not found -> {spec.raw_root}")
        return errors

    required = spec.expected_files if mode == "train" else spec.eval_required_files
    for name in required:
        p = spec.raw_root / name
        alt = spec.raw_root / name.replace(".json", ".jsonl")
        if not p.exists() and not alt.exists():
            errors.append(f"{spec.name}: missing file -> {p} or {alt}")
    return errors


def build_specs(project_root: Path) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name="docvqa",
            raw_root=project_root / "data" / "raw" / "docvqa",
            processed_root=project_root / "data" / "processed" / "docvqa",
            expected_files=["val.json", "test.json"],
            eval_required_files=["val.json", "test.json"],
        ),
        DatasetSpec(
            name="chartqa",
            raw_root=project_root / "data" / "raw" / "chartqa",
            processed_root=project_root / "data" / "processed" / "chartqa",
            expected_files=["train.json", "val.json", "test.json"],
            eval_required_files=["val.json", "test.json"],
        ),
    ]


def cmd_validate(project_root: Path, mode: str = "train") -> int:
    specs = build_specs(project_root)
    all_errors: list[str] = []
    for spec in specs:
        all_errors.extend(validate_layout_for_mode(spec, mode=mode))

    if all_errors:
        print("[dataset-validate] FAILED")
        for err in all_errors:
            print(f"- {err}")
        return 1

    print(f"[dataset-validate] OK (mode={mode})")
    return 0


def cmd_prepare(
    project_root: Path,
    mode: str = "train",
    from_hf_cache: bool = False,
    image_export_root: Path | None = None,
    dataset_names: list[str] | None = None,
    show_progress: bool = False,
    progress_every: int = 500,
) -> int:
    specs = build_specs(project_root)
    if dataset_names:
        selected = {x.strip().lower() for x in dataset_names if x.strip()}
        specs = [s for s in specs if s.name in selected]
        if not specs:
            raise ValueError(f"No matching datasets found for --datasets={dataset_names}")
    splits = ["train", "val", "test"] if mode == "train" else ["val", "test"]
    image_export_root = image_export_root or (project_root / "data" / "images")
    for spec in specs:
        source = "raw_hf" if from_hf_cache else "raw_json"
        print(f"[prepare] dataset={spec.name} (mode={mode}, source={source})")
        for split in splits:
            if from_hf_cache:
                total, kept = _normalize_split_from_hf(
                    spec,
                    split,
                    project_root,
                    image_export_root,
                    show_progress=show_progress,
                    progress_every=progress_every,
                )
            else:
                total, kept = _normalize_split(spec, split)
            print(f"  - {split}: total={total}, kept={kept}, out={spec.processed_root / (split + '.jsonl')}")
    print("[prepare] done")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and normalize DocVQA/ChartQA datasets.")
    parser.add_argument(
        "command",
        choices=["validate", "prepare"],
        help="validate: check required raw files; prepare: output normalized jsonl files",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root path (default: current directory).",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="validate mode: train requires train/val/test, eval requires val/test",
    )
    parser.add_argument(
        "--from-hf-cache",
        action="store_true",
        help="Read from data/raw_hf/<dataset> with datasets.load_from_disk and export image paths.",
    )
    parser.add_argument(
        "--image-export-root",
        default="data/images",
        help="Directory to export decoded images when reading from HF cache.",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help='Optional dataset filter, comma-separated. Example: "chartqa" or "docvqa,chartqa".',
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress (tqdm if available, otherwise periodic progress logs).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="When --show-progress is on, print fallback progress every N samples.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    if args.command == "validate":
        return cmd_validate(project_root, mode=args.mode)
    if args.command == "prepare":
        return cmd_prepare(
            project_root,
            mode=args.mode,
            from_hf_cache=bool(args.from_hf_cache),
            image_export_root=(project_root / args.image_export_root).resolve(),
            dataset_names=[x.strip() for x in str(args.datasets).split(",") if x.strip()],
            show_progress=bool(args.show_progress),
            progress_every=int(args.progress_every),
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
