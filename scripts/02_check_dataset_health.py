from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetSpec:
    name: str
    processed_root: Path


def build_specs(project_root: Path) -> list[DatasetSpec]:
    return [
        DatasetSpec(name="docvqa", processed_root=project_root / "data" / "processed" / "docvqa"),
        DatasetSpec(name="chartqa", processed_root=project_root / "data" / "processed" / "chartqa"),
    ]


def cmd_check(
    project_root: Path,
    mode: str = "eval",
    dataset_names: list[str] | None = None,
    verify_image_open: bool = False,
    sample_image_count: int = 3,
) -> int:
    specs = build_specs(project_root)
    if dataset_names:
        selected = {x.strip().lower() for x in dataset_names if x.strip()}
        specs = [s for s in specs if s.name in selected]
        if not specs:
            raise ValueError(f"No matching datasets found for --datasets={dataset_names}")
    splits = ["train", "val", "test"] if mode == "train" else ["val", "test"]

    all_ok = True
    for spec in specs:
        print(f"[check] dataset={spec.name} (mode={mode})")
        for split in splits:
            p = spec.processed_root / f"{split}.jsonl"
            if not p.exists():
                print(f"  - {split}: MISSING -> {p}")
                all_ok = False
                continue

            rows = 0
            parse_bad = 0
            question_ok = 0
            answers_ok = 0
            image_exists = 0
            samples: list[str] = []
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    rows += 1
                    try:
                        obj = json.loads(line)
                    except Exception:
                        parse_bad += 1
                        continue
                    question_ok += int(bool(obj.get("question")))
                    answers_ok += int(isinstance(obj.get("answers"), list))
                    img = obj.get("image")
                    if isinstance(img, str) and img and os.path.exists(img):
                        image_exists += 1
                        if len(samples) < sample_image_count:
                            samples.append(img)

            print(
                f"  - {split}: rows={rows}, parse_bad={parse_bad}, "
                f"question_ok={question_ok}, answers_list={answers_ok}, image_exists={image_exists}"
            )
            if parse_bad > 0 or question_ok < rows or answers_ok < rows or image_exists < rows:
                all_ok = False

            if verify_image_open and samples:
                try:
                    from PIL import Image

                    for s in samples:
                        with Image.open(s) as im:
                            im.verify()
                    print(f"    image_open_check: OK ({len(samples)} samples)")
                except Exception as e:
                    print(f"    image_open_check: FAILED ({e})")
                    all_ok = False

    if all_ok:
        print("[check] OK")
        return 0
    print("[check] FAILED")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check processed dataset integrity and usability.")
    parser.add_argument("--project-root", default=".", help="Project root path")
    parser.add_argument("--mode", choices=["train", "eval"], default="eval", help="eval checks val/test; train checks train/val/test")
    parser.add_argument("--datasets", default="docvqa,chartqa", help='Comma-separated datasets, e.g. "docvqa,chartqa"')
    parser.add_argument("--verify-image-open", action="store_true", help="Open-verify sample images with PIL")
    parser.add_argument("--sample-image-count", type=int, default=3, help="Number of sample images per split for open-verify")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    return cmd_check(
        project_root=project_root,
        mode=args.mode,
        dataset_names=[x.strip() for x in str(args.datasets).split(",") if x.strip()],
        verify_image_open=bool(args.verify_image_open),
        sample_image_count=int(args.sample_image_count),
    )


if __name__ == "__main__":
    raise SystemExit(main())
