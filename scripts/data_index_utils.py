from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_id(*parts: Any, length: int = 16) -> str:
    payload = "||".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


def term_frequency(tokens: list[str]) -> dict[str, int]:
    return dict(Counter(tokens))


def cosine_sparse(query_tf: dict[str, float], doc_tf: dict[str, float], idf: dict[str, float]) -> float:
    if not query_tf or not doc_tf:
        return 0.0
    dot = 0.0
    q_norm = 0.0
    d_norm = 0.0
    terms = set(query_tf) | set(doc_tf)
    for term in terms:
        weight = idf.get(term, 1.0)
        qv = query_tf.get(term, 0.0) * weight
        dv = doc_tf.get(term, 0.0) * weight
        dot += qv * dv
        q_norm += qv * qv
        d_norm += dv * dv
    if q_norm <= 0 or d_norm <= 0:
        return 0.0
    return dot / (math.sqrt(q_norm) * math.sqrt(d_norm))


def extract_image_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, dict):
        path = value.get("path") or value.get("image_path")
        return str(path) if path else None
    return str(value)


def source_ref(dataset: str, split: str, sample_id: str, page_no: int | None, figure_id: str | None = None) -> str:
    ref = f"{dataset}/{split}/{sample_id}"
    if page_no is not None:
        ref += f"#page={page_no}"
    if figure_id:
        ref += f"#figure={figure_id}"
    return ref


def split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    step = max(1, max_chars - max(0, overlap))
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += step
    return [c for c in chunks if c]
