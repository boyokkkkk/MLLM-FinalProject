from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def cmd_clean(
    project_root: Path,
    keep_datasets: list[str],
    dry_run: bool = True,
    remove_raw_hf: bool = False,
) -> int:
    keep = {x.strip().lower() for x in keep_datasets if x.strip()}
    if not keep:
        keep = {"docvqa", "chartqa"}

    candidates = [
        project_root / "data" / "processed",
        project_root / "data" / "raw",
        project_root / "data" / "images",
    ]
    removed = 0
    for root in candidates:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name.lower() not in keep:
                if dry_run:
                    print(f"[clean][dry-run] remove dir -> {child}")
                else:
                    shutil.rmtree(child, ignore_errors=False)
                    print(f"[clean] removed dir -> {child}")
                    removed += 1

    raw_hf_root = project_root / "data" / "raw_hf"
    if remove_raw_hf and raw_hf_root.exists():
        if dry_run:
            print(f"[clean][dry-run] remove dir -> {raw_hf_root}")
        else:
            shutil.rmtree(raw_hf_root, ignore_errors=False)
            print(f"[clean] removed dir -> {raw_hf_root}")
            removed += 1

    if dry_run:
        print("[clean] dry-run done")
    else:
        print(f"[clean] done, removed={removed}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean redundant dataset directories under data/*.")
    parser.add_argument("--project-root", default=".", help="Project root path")
    parser.add_argument("--datasets", default="docvqa,chartqa", help='Keep datasets, e.g. "docvqa,chartqa"')
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no deletion")
    parser.add_argument(
        "--remove-raw-hf",
        action="store_true",
        help="Also remove the whole data/raw_hf directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    keep = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    return cmd_clean(
        project_root,
        keep_datasets=keep,
        dry_run=bool(args.dry_run),
        remove_raw_hf=bool(args.remove_raw_hf),
    )


if __name__ == "__main__":
    raise SystemExit(main())
