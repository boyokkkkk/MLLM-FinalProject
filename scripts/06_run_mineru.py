from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_index_utils import extract_image_path, read_jsonl, stable_id, write_json

SUPPORTED_INPUT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_MINERU_API_HOST = "127.0.0.1"
DEFAULT_MINERU_API_PORT = 8000
DEFAULT_MINERU_API_TIMEOUT = 180
DEFAULT_MINERU_NUM_WORKERS = 3


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


def _find_local_executable(project_root: Path, *names: str) -> str | None:
    candidates: list[Path] = []
    script_dir = Path(sys.executable).resolve().parent
    for name in names:
        candidates.extend(
            [
                script_dir / name,
                project_root / ".venv" / "Scripts" / name,
                project_root / ".venv" / "bin" / name,
            ]
        )
    for path in candidates:
        if path.exists():
            return str(path)
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _join_base_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _is_url_ready(url: str, timeout_sec: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            return 200 <= int(getattr(response, "status", 200)) < 500
    except urllib.error.HTTPError as exc:
        return 200 <= int(exc.code) < 500
    except Exception:
        return False


def _wait_for_api(api_url: str, timeout_sec: int) -> None:
    probe_urls = [
        _join_base_url(api_url, "/docs"),
        _join_base_url(api_url, "/openapi.json"),
    ]
    deadline = time.time() + max(timeout_sec, 1)
    while time.time() < deadline:
        if any(_is_url_ready(url) for url in probe_urls):
            return
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for MinerU API at {api_url}")


class MineruApiSession:
    def __init__(
        self,
        project_root: Path,
        api_url: str,
        api_host: str,
        api_port: int,
        startup_timeout: int,
        keep_alive: bool,
        enable_vlm_preload: bool,
        mineru_model_source: str,
    ) -> None:
        self.project_root = project_root
        self.external_api_url = api_url.strip()
        self.api_host = api_host
        self.api_port = api_port
        self.startup_timeout = startup_timeout
        self.keep_alive = keep_alive
        self.enable_vlm_preload = enable_vlm_preload
        self.mineru_model_source = mineru_model_source
        self.process: subprocess.Popen[str] | None = None
        self.started_here = False
        self.base_url = self.external_api_url or f"http://{self.api_host}:{self.api_port}"

    def ensure_started(self) -> str:
        if self.external_api_url:
            print(f"[mineru-api] reuse external service -> {self.base_url}")
            _wait_for_api(self.base_url, self.startup_timeout)
            return self.base_url

        _wait_for_api(self.base_url, timeout_sec=1)
        print(f"[mineru-api] reuse detected local service -> {self.base_url}")
        return self.base_url

    def start_if_needed(self) -> str:
        if self.external_api_url:
            return self.ensure_started()

        try:
            return self.ensure_started()
        except TimeoutError:
            pass

        mineru_api_bin = _find_local_executable(self.project_root, "mineru-api", "mineru-api.exe")
        if not mineru_api_bin:
            raise RuntimeError("mineru-api CLI not found. Install it with: python -m pip install -U 'mineru[core]' or uv pip install -U 'mineru[core]'.")
        env = os.environ.copy()
        if self.mineru_model_source:
            env["MINERU_MODEL_SOURCE"] = self.mineru_model_source
        cmd = [mineru_api_bin, "--host", self.api_host, "--port", str(self.api_port)]
        if self.enable_vlm_preload:
            cmd.extend(["--enable-vlm-preload", "true"])
        print(f"[mineru-api] start -> {' '.join(cmd)}")
        print(f"[mineru-api] MINERU_MODEL_SOURCE={env.get('MINERU_MODEL_SOURCE', '')}")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            env=env,
        )
        self.started_here = True
        try:
            _wait_for_api(self.base_url, self.startup_timeout)
        except Exception:
            self.stop(force=True)
            raise
        print(f"[mineru-api] ready -> {self.base_url}")
        return self.base_url

    def stop(self, force: bool = False) -> None:
        if not self.process:
            return
        if self.keep_alive and not force:
            print(f"[mineru-api] keep alive -> {self.base_url}")
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None


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
    api_url: str | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
    normalized_root: Path | None = None,
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
    normalized_base = normalized_root or (project_root / "data" / "interim" / "mineru")
    normalized_path = normalized_base / dataset / f"{sample_id}.json"
    raw_output_dir = output_root / dataset / split / sample_id

    if mock:
        blocks = _make_mock_blocks(sample, page_no)
        raw_files: list[str] = []
    else:
        mineru_bin = _find_local_executable(project_root, "mineru", "mineru.exe", "magic-pdf", "magic-pdf.exe")
        if not mineru_bin:
            raise RuntimeError("MinerU CLI not found. Install it with: python -m pip install -U 'mineru[core]' or uv pip install -U 'mineru[core]'.")
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [mineru_bin, "-p", str(source_path), "-o", str(raw_output_dir)]
        mineru_prog = Path(mineru_bin).stem.lower()
        if mineru_prog == "mineru":
            if api_url:
                cmd.extend(["--api-url", api_url])
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
    parser.add_argument("--normalized-root", default="data/interim/mineru", help="Normalized MinerU JSON output root.")
    parser.add_argument("--backend", default="pipeline", help="MinerU backend, e.g. pipeline.")
    parser.add_argument("--method", default="auto", help="magic-pdf method fallback: auto/ocr/txt.")
    parser.add_argument("--api-url", default="", help="Reuse an existing MinerU FastAPI service, e.g. http://127.0.0.1:8000.")
    parser.add_argument("--api-host", default=DEFAULT_MINERU_API_HOST, help="Host for auto-started local mineru-api.")
    parser.add_argument("--api-port", type=int, default=DEFAULT_MINERU_API_PORT, help="Port for auto-started local mineru-api.")
    parser.add_argument("--api-startup-timeout", type=int, default=DEFAULT_MINERU_API_TIMEOUT, help="Seconds to wait for mineru-api readiness.")
    parser.add_argument("--keep-api-alive", action="store_true", help="Leave the auto-started mineru-api process running after this script exits.")
    parser.add_argument("--enable-vlm-preload", action="store_true", help="Pass --enable-vlm-preload true when auto-starting mineru-api.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_MINERU_NUM_WORKERS, help="Number of concurrent MinerU client submissions against the persistent mineru-api service.")
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


def _collect_samples_from_args(args: argparse.Namespace, project_root: Path) -> list[tuple[dict[str, Any], Path]]:
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    if args.input_path:
        dataset = datasets[0] if datasets else "raw_documents"
        split = splits[0] if splits else "raw"
        samples = _iter_input_samples(project_root, args.input_path, dataset, split, args.document_id or None)
        if args.limit_per_split:
            return samples[: args.limit_per_split]
        return samples

    tasks: list[tuple[dict[str, Any], Path]] = []
    for dataset in datasets:
        for split in splits:
            rows = read_jsonl(project_root / "data" / "processed" / dataset / f"{split}.jsonl")
            kept = 0
            for sample in rows:
                source_path = _resolve_local_path(project_root, dataset, split, sample.get("image"))
                if source_path is None:
                    continue
                tasks.append((sample, source_path))
                kept += 1
                if args.limit_per_split and kept >= args.limit_per_split:
                    break
    return tasks


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = project_root / args.output_root
    normalized_root = project_root / args.normalized_root
    tasks = _collect_samples_from_args(args, project_root)
    if not tasks:
        print("[mineru] no eligible samples found")
        return 0

    api_session = MineruApiSession(
        project_root=project_root,
        api_url=args.api_url,
        api_host=args.api_host,
        api_port=args.api_port,
        startup_timeout=args.api_startup_timeout,
        keep_alive=args.keep_api_alive,
        enable_vlm_preload=args.enable_vlm_preload,
        mineru_model_source=args.mineru_model_source,
    )
    api_url: str | None = None
    if not args.mock:
        api_url = api_session.start_if_needed()

    num_workers = max(1, min(args.num_workers, len(tasks)))
    print(f"[mineru] tasks={len(tasks)} workers={num_workers} mock={args.mock} api_url={api_url or 'N/A'}")

    try:
        if num_workers == 1:
            for sample, source_path in tasks:
                run_mineru_on_sample(
                    project_root,
                    sample,
                    source_path,
                    output_root,
                    args.backend,
                    args.method,
                    args.mock,
                    args.mineru_model_source,
                    api_url,
                    args.start_page,
                    args.end_page,
                    normalized_root,
                )
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        run_mineru_on_sample,
                        project_root,
                        sample,
                        source_path,
                        output_root,
                        args.backend,
                        args.method,
                        args.mock,
                        args.mineru_model_source,
                        api_url,
                        args.start_page,
                        args.end_page,
                        normalized_root,
                    )
                    for sample, source_path in tasks
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
    finally:
        api_session.stop()

    print(f"[mineru] done samples={len(tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
