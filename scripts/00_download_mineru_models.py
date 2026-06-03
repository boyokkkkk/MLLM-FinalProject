from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MinerU model weights from ModelScope for offline/local parsing.")
    parser.add_argument("--repo", default="OpenDataLab/PDF-Extract-Kit-1.0", help="ModelScope repo id.")
    parser.add_argument("--cache-dir", default="", help="Optional ModelScope cache directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from modelscope import snapshot_download

    kwargs = {}
    if args.cache_dir:
        cache_dir = Path(args.cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = str(cache_dir)
    model_dir = snapshot_download(args.repo, **kwargs)
    print(f"[mineru-models] downloaded -> {model_dir}")
    print("[mineru-models] set MINERU_MODEL_SOURCE=modelscope when running MinerU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
