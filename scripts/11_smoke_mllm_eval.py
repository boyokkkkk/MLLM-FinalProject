from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_index_utils import read_jsonl


def load_env(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def collect_candidates(project_root: Path, datasets: list[str], splits: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for split in splits:
            path = project_root / "data" / "processed" / dataset / f"{split}.jsonl"
            for row in read_jsonl(path):
                image = row.get("image")
                image_path = None
                if isinstance(image, dict):
                    image_path = image.get("path") or image.get("image_path")
                elif isinstance(image, str):
                    image_path = image
                if not image_path:
                    continue
                p = Path(str(image_path))
                candidates = [
                    p if p.is_absolute() else project_root / p,
                    project_root / "data" / "images" / dataset / split / p.name,
                    project_root / "data" / "images" / dataset / split / str(image_path),
                ]
                real_path = next((c for c in candidates if c.exists()), None)
                if real_path is None:
                    continue
                rows.append({**row, "_image_path": str(real_path)})
    return rows


def call_mllm(model: str, base_url: str, sample: dict[str, Any]) -> str:
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY or OPENAI_API_KEY in environment/.env")

    client = OpenAI(api_key=api_key, base_url=base_url)
    image_url = image_to_data_url(Path(sample["_image_path"]))
    question = str(sample.get("question", ""))
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": question},
                ],
            }
        ],
    )
    return completion.choices[0].message.content or ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly smoke-test local image QA samples with an OpenAI-compatible MLLM.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--datasets", default="docvqa,chartqa", help="Comma-separated datasets.")
    parser.add_argument("--splits", default="val,test", help="Comma-separated splits.")
    parser.add_argument("--num-samples", type=int, default=2, help="Number of image QA samples to test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--model", default=os.getenv("MLLM_MODEL", "qwen3.7-plus"), help="Model name.")
    parser.add_argument("--base-url", default=os.getenv("MLLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), help="OpenAI-compatible base URL.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected samples without calling the API.")
    parser.add_argument("--out", default="outputs/mllm_smoke_results.jsonl", help="Output JSONL path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    load_env(project_root)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    candidates = collect_candidates(project_root, datasets, splits)
    if not candidates:
        raise RuntimeError("No local image samples found. Run prepare with --from-hf-cache first.")

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, k=min(args.num_samples, len(candidates)))
    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for sample in selected:
            result = {
                "id": sample.get("id"),
                "dataset": sample.get("dataset"),
                "split": sample.get("split"),
                "question": sample.get("question"),
                "answers": sample.get("answers", []),
                "image_path": sample.get("_image_path"),
                "model": args.model,
            }
            if args.dry_run:
                result["prediction"] = "<dry-run>"
            else:
                result["prediction"] = call_mllm(args.model, args.base_url, sample)
            print(json.dumps(result, ensure_ascii=False))
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"[mllm-smoke] wrote -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
