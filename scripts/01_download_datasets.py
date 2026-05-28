from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import yaml


@dataclass
class DownloadFile:
    name: str
    url: str
    output: Path
    sha256: str
    extract: bool


def _make_json_safe(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, bytes):
        # Keep output JSON small and serializable; raw bytes are not needed for our pipeline.
        return None

    if isinstance(obj, list):
        return [_make_json_safe(x) for x in obj]

    if isinstance(obj, tuple):
        return [_make_json_safe(x) for x in obj]

    if isinstance(obj, dict):
        # Hugging Face Image fields are often {"path": ..., "bytes": ...}
        # We preserve path and drop raw bytes to avoid serialization errors and huge files.
        if "bytes" in obj and "path" in obj:
            cleaned = dict(obj)
            cleaned["bytes"] = None
            return {str(k): _make_json_safe(v) for k, v in cleaned.items()}
        return {str(k): _make_json_safe(v) for k, v in obj.items()}

    # Fallback for other non-JSON-native types
    return str(obj)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive_path.suffixes[-2:]).lower()

    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_dir)
        return

    if suffixes in {".tar.gz", ".tgz"} or archive_path.suffix.lower() == ".tar":
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(target_dir)
        return

    raise ValueError(f"Unsupported archive type: {archive_path}")


def download_one(item: DownloadFile, project_root: Path, force: bool, skip_checksum: bool) -> None:
    out_path = project_root / item.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print(f"[skip] exists: {out_path}")
        return

    if not item.url.strip():
        print(f"[warn] empty url for {item.name}, skipped")
        return

    parsed = urlparse(item.url)
    if parsed.scheme in {"http", "https"}:
        tmp_path = out_path.with_suffix(out_path.suffix + ".download")
        print(f"[download] {item.name} -> {out_path}")
        urlretrieve(item.url, tmp_path)
        tmp_path.replace(out_path)
    elif parsed.scheme == "file" or (parsed.scheme == "" and Path(item.url).exists()):
        src = Path(parsed.path if parsed.scheme == "file" else item.url)
        print(f"[copy] {src} -> {out_path}")
        shutil.copy2(src, out_path)
    else:
        raise ValueError(f"Unsupported URL scheme for {item.name}: {item.url}")

    if item.sha256.strip() and not skip_checksum:
        real_hash = compute_sha256(out_path)
        if real_hash.lower() != item.sha256.lower():
            raise ValueError(
                f"sha256 mismatch for {item.name}: expected={item.sha256} actual={real_hash}"
            )
        print(f"[ok] checksum: {item.name}")

    if item.extract:
        extract_dir = out_path.parent
        print(f"[extract] {out_path} -> {extract_dir}")
        extract_archive(out_path, extract_dir)


def load_manifest(path: Path, project_root: Path) -> list[DownloadFile]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    downloads = cfg.get("downloads", {})
    items: list[DownloadFile] = []
    for dataset_name, dataset_cfg in downloads.items():
        if not dataset_cfg.get("enabled", True):
            continue
        for f in dataset_cfg.get("files", []):
            items.append(
                DownloadFile(
                    name=f.get("name", f"{dataset_name}_file"),
                    url=f.get("url", ""),
                    output=Path(f.get("output", "")),
                    sha256=f.get("sha256", ""),
                    extract=bool(f.get("extract", False)),
                )
            )
    return items


def _normalize_hf_split_name(name: str) -> str:
    name = name.lower()
    if name in {"validation", "valid", "dev"}:
        return "val"
    return name


def _hf_export_dataset(
    dataset_id: str,
    target_root: Path,
    force: bool,
    split_map_json: str | None = None,
    cache_root: Path | None = None,
    use_saved_official: bool = False,
    dataset_config: str | None = None,
) -> None:
    from datasets import DatasetDict, load_dataset, load_from_disk

    split_map: dict[str, str] = {}
    if split_map_json:
        split_map = json.loads(split_map_json)

    ds = None
    if use_saved_official:
        if cache_root is None or not cache_root.exists():
            raise FileNotFoundError(f"[hf] official cache not found: {cache_root}")
        print(f"[hf] loading saved official dataset from: {cache_root}")
        ds = load_from_disk(str(cache_root))
    else:
        if dataset_config:
            print(f"[hf] loading dataset from hub: {dataset_id} (config={dataset_config})")
            ds = load_dataset(dataset_id, dataset_config)
        else:
            print(f"[hf] loading dataset from hub: {dataset_id}")
            ds = load_dataset(dataset_id)
        if cache_root is not None:
            if cache_root.exists() and force:
                shutil.rmtree(cache_root)
            if not cache_root.exists():
                cache_root.parent.mkdir(parents=True, exist_ok=True)
                print(f"[hf] save official copy -> {cache_root}")
                ds.save_to_disk(str(cache_root))

    if isinstance(ds, DatasetDict):
        split_items = ds.items()
    elif isinstance(ds, dict):
        split_items = ds.items()
    else:
        raise ValueError(f"Unexpected dataset object for {dataset_id}: {type(ds)}")

    target_root.mkdir(parents=True, exist_ok=True)
    for src_split, dataset in split_items:
        dst_split = split_map.get(src_split, _normalize_hf_split_name(src_split))
        out_path = target_root / f"{dst_split}.json"
        if out_path.exists() and not force:
            print(f"[hf][skip] exists: {out_path}")
            continue
        print(f"[hf] export split {src_split} -> {out_path}")
        records = dataset.to_list()
        records = [_make_json_safe(r) for r in records]
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
    print(f"[hf] done: {dataset_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset downloader with checksum and extraction support.")
    parser.add_argument("--manifest", default="configs/dataset_downloads.yaml", help="Path to download manifest yaml.")
    parser.add_argument("--project-root", default=".", help="Project root path")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--skip-checksum", action="store_true", help="Skip sha256 validation")
    parser.add_argument("--from-hf", action="store_true", help="Download from Hugging Face datasets")
    parser.add_argument(
        "--hf-official-layout",
        action="store_true",
        help="Save official HF dataset directories under data/raw_hf/<dataset> and export raw json from there.",
    )
    parser.add_argument(
        "--hf-export-from-official",
        action="store_true",
        help="Export raw json from existing data/raw_hf/<dataset> without re-downloading.",
    )
    parser.add_argument("--hf-docvqa-id", default="lmms-lab/DocVQA", help="HF dataset id for DocVQA")
    parser.add_argument("--hf-docvqa-config", default="DocVQA", help='HF config name for DocVQA, e.g. "DocVQA"')
    parser.add_argument("--hf-chartqa-id", default="HuggingFaceM4/ChartQA", help="HF dataset id for ChartQA")
    parser.add_argument(
        "--hf-docvqa-split-map",
        default="",
        help='Optional JSON mapping, e.g. {"validation":"val"}',
    )
    parser.add_argument(
        "--hf-chartqa-split-map",
        default="",
        help='Optional JSON mapping, e.g. {"validation":"val"}',
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    if args.from_hf:
        docvqa_official_dir = project_root / "data" / "raw_hf" / "docvqa"
        chartqa_official_dir = project_root / "data" / "raw_hf" / "chartqa"
        _hf_export_dataset(
            dataset_id=args.hf_docvqa_id,
            target_root=project_root / "data" / "raw" / "docvqa",
            force=args.force,
            split_map_json=args.hf_docvqa_split_map or None,
            cache_root=docvqa_official_dir if args.hf_official_layout or args.hf_export_from_official else None,
            use_saved_official=args.hf_export_from_official,
            dataset_config=args.hf_docvqa_config or None,
        )
        _hf_export_dataset(
            dataset_id=args.hf_chartqa_id,
            target_root=project_root / "data" / "raw" / "chartqa",
            force=args.force,
            split_map_json=args.hf_chartqa_split_map or None,
            cache_root=chartqa_official_dir if args.hf_official_layout or args.hf_export_from_official else None,
            use_saved_official=args.hf_export_from_official,
        )
        print("[done] huggingface dataset export finished")
        return 0

    manifest_path = (project_root / args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    items = load_manifest(manifest_path, project_root)
    if not items:
        print("[done] no download items")
        return 0

    for item in items:
        if not str(item.output):
            print(f"[warn] empty output for {item.name}, skipped")
            continue
        download_one(item, project_root=project_root, force=args.force, skip_checksum=args.skip_checksum)

    print("[done] dataset download finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
