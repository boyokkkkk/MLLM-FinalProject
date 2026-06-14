from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark import (
    collect_eval_samples,
    load_eval_config,
    run_rag_eval,
    run_retrieval_eval,
    write_eval_outputs,
)
from src.utils.settings import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standardized benchmark evaluation for the multimodal Doc RAG project.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--config", default="configs/eval.yaml", help="Evaluation config path.")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml", help="Dataset/index config path.")
    parser.add_argument("--suite", default="retrieval_benchmark", help="Suite name under eval.suites.")
    parser.add_argument("--mode", choices=["retrieval", "rag"], default="", help="Override suite mode.")
    parser.add_argument("--datasets", default="", help="Override comma-separated datasets.")
    parser.add_argument("--splits", default="", help="Override comma-separated splits.")
    parser.add_argument("--sample-manifest", default="", help="Optional JSONL manifest of benchmark samples to evaluate.")
    parser.add_argument("--match-granularity", choices=["sample", "doc_page"], default="sample", help="How to judge retrieval/citation correctness.")
    parser.add_argument("--limit-per-split", type=int, default=-1, help="Override sample cap per split. -1 uses suite/default.")
    parser.add_argument("--top-k", type=int, default=0, help="Override top-k for retrieval/citation metrics.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/api/v1", help="Backend API base URL for rag mode.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature used in rag mode.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens used in rag mode.")
    parser.add_argument("--include-query-images-in-rag", action=argparse.BooleanOptionalAction, default=False, help="Whether rag benchmark requests should upload the query image to the chat API.")
    parser.add_argument("--rerank-profile", choices=["basic", "stronger"], default="", help="Override rerank profile.")
    parser.add_argument("--diversify-results", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable duplicate-aware result diversification.")
    parser.add_argument("--fingerprint-duplicate-penalty", type=float, default=-1.0, help="Override near-duplicate text penalty.")
    parser.add_argument("--docpage-duplicate-penalty", type=float, default=-1.0, help="Override same-doc-page duplicate penalty.")
    parser.add_argument("--query-type-aware-rerank", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable query-type-aware reranking.")
    parser.add_argument("--visual-fusion", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable visual late fusion for retrieval mode.")
    parser.add_argument("--visual-fusion-weight", type=float, default=-1.0, help="Override visual late fusion weight.")
    parser.add_argument("--visual-dense-fusion", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable visual descriptor dense fusion.")
    parser.add_argument("--visual-dense-weight", type=float, default=-1.0, help="Override visual descriptor dense fusion weight.")
    parser.add_argument("--text-fusion-weight", type=float, default=-1.0, help="Override text branch fusion weight.")
    parser.add_argument("--chart-table-specialist", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable chart/table-specialized visual retrieval.")
    parser.add_argument("--chart-table-visual-boost", type=float, default=-1.0, help="Override chart/table visual boost.")
    parser.add_argument("--query-image-aware-rerank", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable query-image-aware reranking in retrieval mode.")
    parser.add_argument("--query-image-pool-size", type=int, default=-1, help="Override query-image rerank pool size.")
    parser.add_argument("--query-image-weight", type=float, default=-1.0, help="Override query-image rerank fusion weight.")
    parser.add_argument("--generation-visual-assist", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable generation-time visual assistance metadata for rag mode.")
    parser.add_argument("--generation-visual-assist-policy", default="", help="Visual-assist policy tag to record and pass through to backend env, e.g. off/always/gated.")
    parser.add_argument("--metadata-path", default="", help="Optional retrieval metadata JSONL override.")
    parser.add_argument("--sparse-index-path", default="", help="Optional sparse doc_store.json override.")
    parser.add_argument("--visual-index-path", default="", help="Optional visual_store.json override.")
    parser.add_argument("--visual-dense-metadata-path", default="", help="Optional visual descriptor metadata JSONL override.")
    parser.add_argument("--visual-dense-vectors-path", default="", help="Optional visual descriptor vectors JSON override.")
    parser.add_argument("--run-name", default="", help="Optional output run name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = (project_root / args.config).resolve()
    eval_cfg = load_eval_config(project_root, config_path)
    suites = eval_cfg.get("suites", {})
    suite = suites.get(args.suite, {})

    mode = args.mode or suite.get("mode", "retrieval")
    datasets = [x.strip() for x in (args.datasets or suite.get("datasets", "")).split(",") if x.strip()]
    splits = [x.strip() for x in (args.splits or suite.get("splits", "")).split(",") if x.strip()]
    top_k = args.top_k or int(suite.get("top_k", eval_cfg.get("default_top_k", 5)))
    limit_per_split = (
        args.limit_per_split
        if args.limit_per_split >= 0
        else int(suite.get("limit_per_split", eval_cfg.get("default_limit_per_split", 0)))
    )

    samples = collect_eval_samples(
        project_root=project_root,
        datasets=datasets,
        splits=splits,
        limit_per_split=limit_per_split,
        sample_manifest=(project_root / args.sample_manifest).resolve() if args.sample_manifest else None,
    )
    if not samples:
        raise RuntimeError("No evaluation samples found. Prepare data first.")

    retrieval_cfg = settings.retrieval.model_copy()
    if args.rerank_profile:
        retrieval_cfg = retrieval_cfg.model_copy(update={"rerank_profile": args.rerank_profile})
    if args.diversify_results is not None:
        retrieval_cfg = retrieval_cfg.model_copy(update={"diversify_results": args.diversify_results})
    if args.fingerprint_duplicate_penalty >= 0.0:
        retrieval_cfg = retrieval_cfg.model_copy(update={"fingerprint_duplicate_penalty": args.fingerprint_duplicate_penalty})
    if args.docpage_duplicate_penalty >= 0.0:
        retrieval_cfg = retrieval_cfg.model_copy(update={"docpage_duplicate_penalty": args.docpage_duplicate_penalty})
    if args.query_type_aware_rerank is not None:
        retrieval_cfg = retrieval_cfg.model_copy(update={"query_type_aware_rerank": args.query_type_aware_rerank})
    if args.visual_fusion is not None:
        retrieval_cfg = retrieval_cfg.model_copy(update={"visual_fusion": args.visual_fusion})
    if args.visual_fusion_weight >= 0.0:
        retrieval_cfg = retrieval_cfg.model_copy(update={"visual_fusion_weight": args.visual_fusion_weight})
    if args.visual_dense_fusion is not None:
        retrieval_cfg = retrieval_cfg.model_copy(update={"visual_dense_fusion": args.visual_dense_fusion})
    if args.visual_dense_weight >= 0.0:
        retrieval_cfg = retrieval_cfg.model_copy(update={"visual_dense_weight": args.visual_dense_weight})
    if args.text_fusion_weight >= 0.0:
        retrieval_cfg = retrieval_cfg.model_copy(update={"text_fusion_weight": args.text_fusion_weight})
    if args.chart_table_specialist is not None:
        retrieval_cfg = retrieval_cfg.model_copy(update={"chart_table_specialist": args.chart_table_specialist})
    if args.chart_table_visual_boost >= 0.0:
        retrieval_cfg = retrieval_cfg.model_copy(update={"chart_table_visual_boost": args.chart_table_visual_boost})
    if args.query_image_aware_rerank is not None:
        retrieval_cfg = retrieval_cfg.model_copy(update={"query_image_aware_rerank": args.query_image_aware_rerank})
    if args.query_image_pool_size > 0:
        retrieval_cfg = retrieval_cfg.model_copy(update={"query_image_pool_size": args.query_image_pool_size})
    if args.query_image_weight >= 0.0:
        retrieval_cfg = retrieval_cfg.model_copy(update={"query_image_weight": args.query_image_weight})
    if args.generation_visual_assist is not None:
        retrieval_cfg = retrieval_cfg.model_copy(update={"generation_visual_assist": args.generation_visual_assist})
    if args.generation_visual_assist_policy:
        retrieval_cfg = retrieval_cfg.model_copy(update={"generation_visual_assist_policy": args.generation_visual_assist_policy})
    if args.metadata_path:
        retrieval_cfg = retrieval_cfg.model_copy(update={"metadata_path": (project_root / args.metadata_path).resolve()})
    if args.sparse_index_path:
        retrieval_cfg = retrieval_cfg.model_copy(update={"sparse_index_path": (project_root / args.sparse_index_path).resolve()})
    if args.visual_index_path:
        retrieval_cfg = retrieval_cfg.model_copy(update={"visual_index_path": (project_root / args.visual_index_path).resolve()})
    if args.visual_dense_metadata_path:
        retrieval_cfg = retrieval_cfg.model_copy(update={"visual_dense_metadata_path": (project_root / args.visual_dense_metadata_path).resolve()})
    if args.visual_dense_vectors_path:
        retrieval_cfg = retrieval_cfg.model_copy(update={"visual_dense_vectors_path": (project_root / args.visual_dense_vectors_path).resolve()})

    if mode == "retrieval":
        records, summary = run_retrieval_eval(
            project_root,
            (project_root / args.datasets_config).resolve(),
            samples,
            top_k,
            retrieval_cfg=retrieval_cfg,
            match_granularity=args.match_granularity,
        )
    else:
        records, summary = run_rag_eval(
            samples=samples,
            top_k=top_k,
            api_base=args.api_base,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            include_query_images=args.include_query_images_in_rag,
            match_granularity=args.match_granularity,
        )

    outputs_root = project_root / eval_cfg.get("outputs_root", "outputs/eval")
    run_name = args.run_name or f"{args.suite}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    meta = {
        "suite": args.suite,
        "mode": mode,
        "datasets": datasets,
        "splits": splits,
        "top_k": top_k,
        "limit_per_split": limit_per_split,
        "sample_manifest": args.sample_manifest or None,
        "match_granularity": args.match_granularity,
        "api_base": args.api_base if mode == "rag" else None,
        "include_query_images_in_rag": args.include_query_images_in_rag if mode == "rag" else None,
        "retrieval_profile": {
            "rerank_profile": retrieval_cfg.rerank_profile,
            "query_type_aware_rerank": retrieval_cfg.query_type_aware_rerank,
            "diversify_results": retrieval_cfg.diversify_results,
            "fingerprint_duplicate_penalty": retrieval_cfg.fingerprint_duplicate_penalty,
            "docpage_duplicate_penalty": retrieval_cfg.docpage_duplicate_penalty,
            "visual_fusion": retrieval_cfg.visual_fusion,
            "visual_fusion_weight": retrieval_cfg.visual_fusion_weight,
            "visual_dense_fusion": retrieval_cfg.visual_dense_fusion,
            "visual_dense_weight": retrieval_cfg.visual_dense_weight,
            "text_fusion_weight": retrieval_cfg.text_fusion_weight,
            "chart_table_specialist": retrieval_cfg.chart_table_specialist,
            "chart_table_visual_boost": retrieval_cfg.chart_table_visual_boost,
            "query_image_aware_rerank": retrieval_cfg.query_image_aware_rerank,
            "query_image_pool_size": retrieval_cfg.query_image_pool_size,
            "query_image_weight": retrieval_cfg.query_image_weight,
            "generation_visual_assist": retrieval_cfg.generation_visual_assist,
            "generation_visual_assist_policy": retrieval_cfg.generation_visual_assist_policy,
            "metadata_path": str(retrieval_cfg.metadata_path),
            "sparse_index_path": str(retrieval_cfg.sparse_index_path),
            "visual_index_path": str(retrieval_cfg.visual_index_path),
        },
    }
    write_eval_outputs(outputs_root, run_name, records, summary, meta)

    print(f"[eval] suite={args.suite} mode={mode} samples={summary['num_samples']}")
    for metric, value in summary["overall"].items():
        print(f"[eval] {metric}={value:.6f}")
    print(f"[eval] outputs -> {outputs_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
