from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = "outputs/eval/docvqa_val_unique_docpage_100.manifest.jsonl"
DEFAULT_OUTPUT_ROOT = "outputs/eval"
DEFAULT_ABLATION_INDEX_ROOT = "outputs/eval/ablation_indexes"


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    mode: str
    groups: tuple[str, ...]
    description: str
    cli_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


def _bool_flag(flag: str, value: bool) -> list[str]:
    return [flag if value else flag.replace("--", "--no-", 1)]


def _ablation_experiments(index_root: str) -> list[ExperimentSpec]:
    page_text_chunks = f"{index_root}/page_text/chunks.jsonl"
    page_text_store = f"{index_root}/page_text/doc_store.json"
    page_text_visual = f"{index_root}/page_text/visual_store.json"
    block_text_chunks = f"{index_root}/block_text/chunks.jsonl"
    block_text_store = f"{index_root}/block_text/doc_store.json"
    block_text_visual = f"{index_root}/block_text/visual_store.json"
    block_mm_chunks = f"{index_root}/block_multimodal/chunks.jsonl"
    block_mm_store = f"{index_root}/block_multimodal/doc_store.json"
    block_mm_visual = f"{index_root}/block_multimodal/visual_store.json"

    return [
        ExperimentSpec(
            name="docvqa_val_unique_docpage_100_ablation_index_page_text",
            mode="retrieval",
            groups=("index",),
            description="Page-level text aggregation only.",
            cli_args=[
                "--rerank-profile", "stronger",
                "--query-type-aware-rerank",
                "--query-image-aware-rerank",
                "--metadata-path", page_text_chunks,
                "--sparse-index-path", page_text_store,
                "--visual-index-path", page_text_visual,
            ],
        ),
        ExperimentSpec(
            name="docvqa_val_unique_docpage_100_ablation_index_block_text",
            mode="retrieval",
            groups=("index",),
            description="Block-level textual chunks only.",
            cli_args=[
                "--rerank-profile", "stronger",
                "--query-type-aware-rerank",
                "--query-image-aware-rerank",
                "--metadata-path", block_text_chunks,
                "--sparse-index-path", block_text_store,
                "--visual-index-path", block_text_visual,
            ],
        ),
        ExperimentSpec(
            name="docvqa_val_unique_docpage_100_ablation_index_block_multimodal",
            mode="retrieval",
            groups=("index",),
            description="Current full block-level multimodal chunk store.",
            cli_args=[
                "--rerank-profile", "stronger",
                "--query-type-aware-rerank",
                "--query-image-aware-rerank",
                "--metadata-path", block_mm_chunks,
                "--sparse-index-path", block_mm_store,
                "--visual-index-path", block_mm_visual,
            ],
        ),
        ExperimentSpec(
            name="docvqa_val_unique_docpage_100_ablation_retrieval_basic",
            mode="retrieval",
            groups=("retrieval",),
            description="Basic sparse rerank baseline.",
            cli_args=[
                "--rerank-profile", "basic",
                "--no-query-type-aware-rerank",
                "--no-query-image-aware-rerank",
            ],
        ),
        ExperimentSpec(
            name="docvqa_val_unique_docpage_100_ablation_retrieval_stronger",
            mode="retrieval",
            groups=("retrieval",),
            description="Stronger rerank only.",
            cli_args=[
                "--rerank-profile", "stronger",
                "--no-query-type-aware-rerank",
                "--no-query-image-aware-rerank",
            ],
        ),
        ExperimentSpec(
            name="docvqa_val_unique_docpage_100_ablation_retrieval_stronger_qta",
            mode="retrieval",
            groups=("retrieval",),
            description="Stronger rerank plus query-type-aware heuristics.",
            cli_args=[
                "--rerank-profile", "stronger",
                "--query-type-aware-rerank",
                "--no-query-image-aware-rerank",
            ],
        ),
        ExperimentSpec(
            name="docvqa_val_unique_docpage_100_ablation_retrieval_stronger_qia",
            mode="retrieval",
            groups=("retrieval", "visual", "gating"),
            description="Stronger rerank plus query-image-aware rerank.",
            cli_args=[
                "--rerank-profile", "stronger",
                "--query-type-aware-rerank",
                "--query-image-aware-rerank",
            ],
        ),
        ExperimentSpec(
            name="docvqa_val_unique_docpage_100_ablation_visual_densefusion",
            mode="retrieval",
            groups=("visual",),
            description="Move visual signal into retrieval-time dense fusion.",
            cli_args=[
                "--rerank-profile", "stronger",
                "--query-type-aware-rerank",
                "--no-query-image-aware-rerank",
                "--visual-dense-fusion",
            ],
        ),
        ExperimentSpec(
            name="docvqa_val_unique_docpage_100_ablation_rag_text_only",
            mode="rag",
            groups=("visual",),
            description="RAG without retrieval-time or generation-time visual assistance.",
            cli_args=[
                "--rerank-profile", "stronger",
                "--query-type-aware-rerank",
                "--no-query-image-aware-rerank",
                "--no-generation-visual-assist",
            ],
            env={
                "RETRIEVAL_RERANK_PROFILE": "stronger",
                "RETRIEVAL_QUERY_TYPE_AWARE_RERANK": "true",
                "RETRIEVAL_QUERY_IMAGE_AWARE_RERANK": "false",
                "RETRIEVAL_GENERATION_VISUAL_ASSIST": "false",
            },
        ),
        ExperimentSpec(
            name="docvqa_val_unique_docpage_100_ablation_rag_visualassist_always",
            mode="rag",
            groups=("visual", "gating"),
            description="Generation-time visual assistance always enabled.",
            cli_args=[
                "--rerank-profile", "stronger",
                "--query-type-aware-rerank",
                "--query-image-aware-rerank",
                "--generation-visual-assist",
                "--generation-visual-assist-policy", "always",
            ],
            env={
                "RETRIEVAL_RERANK_PROFILE": "stronger",
                "RETRIEVAL_QUERY_TYPE_AWARE_RERANK": "true",
                "RETRIEVAL_QUERY_IMAGE_AWARE_RERANK": "true",
                "RETRIEVAL_GENERATION_VISUAL_ASSIST": "true",
                "RETRIEVAL_GENERATION_VISUAL_ASSIST_POLICY": "always",
            },
        ),
        ExperimentSpec(
            name="docvqa_val_unique_docpage_100_ablation_rag_visualassist_gated",
            mode="rag",
            groups=("visual", "gating"),
            description="Generation-time visual assistance with heuristic gating.",
            cli_args=[
                "--rerank-profile", "stronger",
                "--query-type-aware-rerank",
                "--query-image-aware-rerank",
                "--generation-visual-assist",
                "--generation-visual-assist-policy", "gated",
            ],
            env={
                "RETRIEVAL_RERANK_PROFILE": "stronger",
                "RETRIEVAL_QUERY_TYPE_AWARE_RERANK": "true",
                "RETRIEVAL_QUERY_IMAGE_AWARE_RERANK": "true",
                "RETRIEVAL_GENERATION_VISUAL_ASSIST": "true",
                "RETRIEVAL_GENERATION_VISUAL_ASSIST_POLICY": "gated",
            },
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four ablation groups for the Doc RAG project.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--config", default="configs/eval.yaml", help="Evaluation config path.")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml", help="Dataset config path.")
    parser.add_argument("--sample-manifest", default=DEFAULT_MANIFEST, help="Benchmark sample manifest.")
    parser.add_argument("--datasets", default="docvqa", help="Datasets to evaluate.")
    parser.add_argument("--splits", default="val", help="Splits to evaluate.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k citations/results.")
    parser.add_argument("--groups", default="index,retrieval,visual,gating", help="Comma-separated ablation groups to run.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter for child processes.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Evaluation output directory.")
    parser.add_argument("--ablation-index-root", default=DEFAULT_ABLATION_INDEX_ROOT, help="Prepared ablation index directory.")
    parser.add_argument("--api-port-base", type=int, default=8030, help="Starting port for temporary API instances.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose summary JSON already exists.")
    parser.add_argument("--summary-stem", default="docvqa_val_unique_docpage_100_ablation_suite", help="Output stem for consolidated summary files.")
    return parser.parse_args()


def _run_command(command: list[str], *, env: dict[str, str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed ({process.returncode}): {' '.join(command)}; see {log_path}")


def _wait_for_health(root_url: str, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    health_url = f"{root_url.rstrip('/')}/health"
    last_error = ""
    while time.time() < deadline:
        try:
            response = httpx.get(health_url, timeout=5.0)
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code}"
        except Exception as exc:  # pragma: no cover - depends on local runtime
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for API health at {health_url}: {last_error}")


def _run_rag_experiment(
    spec: ExperimentSpec,
    *,
    project_root: Path,
    python_bin: str,
    api_port: int,
    command: list[str],
    env: dict[str, str],
    log_dir: Path,
) -> None:
    server_log = log_dir / f"{spec.name}.server.log"
    eval_log = log_dir / f"{spec.name}.eval.log"
    root_url = f"http://127.0.0.1:{api_port}"
    api_base = f"http://127.0.0.1:{api_port}/api/v1"
    server_command = [python_bin, "src/cli.py", "api", "--host", "127.0.0.1", "--port", str(api_port)]

    with server_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            server_command,
            cwd=str(project_root),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    try:
        _wait_for_health(root_url)
        _run_command(command + ["--api-base", api_base], env=env, cwd=project_root, log_path=eval_log)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get("summary", {}).get("overall", {}).get(key)
    if value is None:
        return None
    return float(value)


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}" if value < 1 else f"{value:.3f}"


def _write_summary(output_root: Path, stem: str, rows: list[dict[str, str]]) -> None:
    csv_path = output_root / f"{stem}.csv"
    md_path = output_root / f"{stem}.md"
    json_path = output_root / f"{stem}.json"
    fieldnames = [
        "group",
        "run",
        "mode",
        "status",
        "description",
        "hit_at_k",
        "citation_accuracy",
        "exact_match",
        "anls",
        "token_f1",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Ablation Suite Summary",
        "",
        "| group | run | mode | status | hit@5 | citation@1 | EM | ANLS | Token-F1 | description |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {group} | {run} | {mode} | {status} | {hit_at_k} | {citation_accuracy} | {exact_match} | {anls} | {token_f1} | {description} |".format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    ablation_index_root = args.ablation_index_root.replace("\\", "/").rstrip("/")
    log_dir = output_root / "ablation_logs"

    selected_groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    experiments = [spec for spec in _ablation_experiments(ablation_index_root) if selected_groups.intersection(spec.groups)]
    if not experiments:
        raise RuntimeError(f"No experiments selected for groups={sorted(selected_groups)}")

    prepare_indexes_cmd = [args.python_bin, "scripts/17_prepare_ablation_indexes.py", "--project-root", str(project_root)]
    _run_command(prepare_indexes_cmd, env=os.environ.copy(), cwd=project_root, log_path=log_dir / "prepare_indexes.log")

    rows: list[dict[str, str]] = []
    rag_port = args.api_port_base
    for spec in experiments:
        summary_path = output_root / f"{spec.name}.summary.json"
        status = "skipped"
        if args.skip_existing and summary_path.exists():
            status = "existing"
        else:
            env = os.environ.copy()
            env.update(spec.env)
            command = [
                args.python_bin,
                "scripts/12_run_benchmark_eval.py",
                "--project-root", str(project_root),
                "--config", args.config,
                "--datasets-config", args.datasets_config,
                "--suite", "retrieval_benchmark",
                "--mode", spec.mode,
                "--datasets", args.datasets,
                "--splits", args.splits,
                "--sample-manifest", args.sample_manifest,
                "--top-k", str(args.top_k),
                "--run-name", spec.name,
            ] + spec.cli_args
            if spec.mode == "rag":
                _run_rag_experiment(
                    spec,
                    project_root=project_root,
                    python_bin=args.python_bin,
                    api_port=rag_port,
                    command=command,
                    env=env,
                    log_dir=log_dir,
                )
                rag_port += 1
            else:
                _run_command(command, env=env, cwd=project_root, log_path=log_dir / f"{spec.name}.eval.log")
            status = "ran"

        payload = _load_summary(summary_path)
        for group in spec.groups:
            if group not in selected_groups:
                continue
            rows.append(
                {
                    "group": group,
                    "run": spec.name,
                    "mode": spec.mode,
                    "status": status,
                    "description": spec.description,
                    "hit_at_k": _fmt(_metric(payload, "hit_at_k")),
                    "citation_accuracy": _fmt(_metric(payload, "citation_accuracy")),
                    "exact_match": _fmt(_metric(payload, "exact_match")),
                    "anls": _fmt(_metric(payload, "anls")),
                    "token_f1": _fmt(_metric(payload, "token_f1")),
                }
            )

    _write_summary(output_root, args.summary_stem, rows)
    print(f"[ablation-suite] experiments={len(experiments)} groups={','.join(sorted(selected_groups))}")
    print(f"[ablation-suite] summary -> {output_root / (args.summary_stem + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
