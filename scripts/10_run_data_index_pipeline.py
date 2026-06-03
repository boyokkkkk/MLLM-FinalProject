from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str], cwd: Path, skip: bool = False) -> None:
    print(f"[pipeline] {' '.join(cmd)}")
    if skip:
        print("[pipeline] skipped")
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run download/prepare/parse/index offline data pipeline.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--mode", choices=["train", "eval"], default="eval", help="Dataset prepare mode.")
    parser.add_argument("--datasets", default="docvqa,chartqa", help="Comma-separated datasets.")
    parser.add_argument("--splits", default="val,test", help="Comma-separated splits for parse/chunk.")
    parser.add_argument("--limit-per-split", type=int, default=0, help="Cap samples per split for quick smoke runs.")
    parser.add_argument("--skip-download", action="store_true", help="Do not run dataset download.")
    parser.add_argument("--download-from-hf", action="store_true", help="Use Hugging Face download/export step.")
    parser.add_argument("--hf-official-layout", action="store_true", help="Keep official HF save_to_disk layout.")
    parser.add_argument("--skip-prepare", action="store_true", help="Do not normalize raw datasets.")
    parser.add_argument("--prepare-from-hf-cache", action="store_true", help="Normalize from data/raw_hf and export concrete local image files.")
    parser.add_argument("--image-export-root", default="data/images", help="Directory for images exported from HF cache.")
    parser.add_argument("--run-mllm-smoke", action="store_true", help="Call the configured MLLM on a few random local image QA samples.")
    parser.add_argument("--mllm-dry-run", action="store_true", help="Select samples for MLLM smoke test without calling the API.")
    parser.add_argument("--mllm-num-samples", type=int, default=2, help="Number of MLLM smoke samples.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    py = sys.executable

    download_cmd = [py, "scripts/01_download_datasets.py"]
    if args.download_from_hf:
        download_cmd.append("--from-hf")
    if args.hf_official_layout:
        download_cmd.append("--hf-official-layout")
    run_step(download_cmd, project_root, skip=args.skip_download)

    prepare_cmd = [py, "scripts/01_prepare_datasets.py", "prepare", "--mode", args.mode, "--datasets", args.datasets]
    if args.limit_per_split:
        prepare_cmd.extend(["--limit-per-split", str(args.limit_per_split)])
    if args.prepare_from_hf_cache:
        prepare_cmd.extend(["--from-hf-cache", "--image-export-root", args.image_export_root])
    run_step(prepare_cmd, project_root, skip=args.skip_prepare)

    parse_cmd = [py, "scripts/07_parse_and_chunk.py", "--datasets", args.datasets, "--splits", args.splits]
    if args.limit_per_split:
        parse_cmd.extend(["--limit-per-split", str(args.limit_per_split)])
    run_step(parse_cmd, project_root)
    run_step([py, "scripts/08_build_indexes.py"], project_root)
    smoke_cmd = [py, "scripts/11_smoke_mllm_eval.py", "--datasets", args.datasets, "--splits", args.splits, "--num-samples", str(args.mllm_num_samples)]
    if args.mllm_dry_run:
        smoke_cmd.append("--dry-run")
    run_step(smoke_cmd, project_root, skip=not args.run_mllm_smoke)
    print("[pipeline] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
