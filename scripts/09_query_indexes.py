from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_index_utils import cosine_sparse, read_yaml, term_frequency, tokenize


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def query_index(project_root: Path, config_path: Path, query: str, top_k: int) -> list[dict[str, Any]]:
    cfg = read_yaml(config_path)
    index_cfg = cfg.get("indexing", {})
    text_dir = project_root / index_cfg.get("text_index_dir", "data/processed/indexes/text")
    doc_store = _load_json(text_dir / "doc_store.json")
    df = _load_json(text_dir / "document_frequency.json")
    n_docs = max(1, len(doc_store))
    idf = {term: math.log((n_docs + 1) / (int(freq) + 1)) + 1.0 for term, freq in df.items()}
    query_tf = term_frequency(tokenize(query))

    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk in doc_store.values():
        score = cosine_sparse(query_tf, chunk.get("tf", {}), idf)
        if score <= 0:
            continue
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [{
        "score": round(score, 6),
        "chunk_id": chunk.get("chunk_id"),
        "sample_id": chunk.get("sample_id"),
        "dataset": chunk.get("dataset"),
        "split": chunk.get("split"),
        "chunk_type": chunk.get("chunk_type"),
        "source_ref": chunk.get("source_ref"),
        "page_no": chunk.get("page_no"),
        "bbox": chunk.get("bbox"),
        "image_path": chunk.get("image_path"),
        "snippet": str(chunk.get("text", ""))[:300],
    } for score, chunk in scored[:top_k]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the local offline document indexes.")
    parser.add_argument("query", help="Natural language query.")
    parser.add_argument("--project-root", default=".", help="Project root path.")
    parser.add_argument("--config", default="configs/datasets.yaml", help="Dataset config path.")
    parser.add_argument("--top-k", type=int, default=0, help="Number of results. 0 uses config default.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of readable text.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = (project_root / args.config).resolve()
    cfg = read_yaml(config_path)
    top_k = int(args.top_k) or int(cfg.get("indexing", {}).get("top_k_default", 5))
    results = query_index(project_root, config_path, args.query, top_k)
    if args.json:
        print(json.dumps({"query": args.query, "results": results}, ensure_ascii=False, indent=2))
        return 0
    print(f"[query] {args.query}")
    if not results:
        print("No hits. Try a query sharing terms with the processed questions/evidence.")
        return 0
    for idx, item in enumerate(results, start=1):
        print(f"{idx}. score={item['score']} {item['source_ref']} ({item['chunk_type']})")
        print(f"   snippet: {item['snippet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
