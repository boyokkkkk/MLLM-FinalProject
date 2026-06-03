from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import numpy as np

from src.models.clients import build_embedding_client
from src.utils.settings import settings

# 暂时使用docs目录的部分md文件做切块任务
DEFAULT_SOURCES = [
    "README.md",
    "docs/项目框架与分工计划.md",
    "docs/B_检索与推理TODO检查清单.md",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mock retrieval metadata and vectors from local markdown files.")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=DEFAULT_SOURCES,
        help="Workspace-relative source markdown files used to build mock chunks.",
    )
    parser.add_argument(
        "--metadata-path",
        default=str(settings.retrieval.metadata_path),
        help="Output JSONL path for mock chunk metadata.",
    )
    parser.add_argument(
        "--index-path",
        default=str(settings.retrieval.index_path),
        help="Output NPY path for mock chunk vectors.",
    )
    parser.add_argument(
        "--target-chars",
        type=int,
        default=280,
        help="Target characters per chunk when merging paragraphs.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=420,
        help="Maximum characters per chunk.",
    )
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _sanitize_doc_id(source: Path, index: int) -> str:
    stem = source.stem.lower()
    ascii_stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if not ascii_stem:
        ascii_stem = f"mock_doc_{index:03d}"
    return ascii_stem


def _split_markdown(source: Path, doc_id: str, target_chars: int, max_chars: int) -> list[dict]:
    raw_text = _normalize_text(source.read_text(encoding="utf-8"))
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", raw_text) if item.strip()]

    chunks: list[dict] = []
    current_parts: list[str] = []
    current_length = 0
    chunk_index = 1
    char_cursor = 0
    current_section: str | None = None

    for paragraph in paragraphs:
        if paragraph.startswith("#"):
            current_section = paragraph.lstrip("#").strip() or current_section
            continue

        cleaned = paragraph.replace("\n", " ").strip()
        if not cleaned:
            continue

        prefixed = f"{current_section}\n{cleaned}" if current_section else cleaned
        paragraph_length = len(prefixed)

        if current_parts and (current_length + paragraph_length > max_chars or current_length >= target_chars):
            chunk_text = "\n".join(current_parts).strip()
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_p0001_c{chunk_index:04d}",
                    "doc_id": doc_id,
                    "source": source.name,
                    "page": 1,
                    "modality": "text",
                    "section": current_section,
                    "text": chunk_text,
                    "char_start": char_cursor,
                    "char_end": char_cursor + len(chunk_text),
                }
            )
            char_cursor += len(chunk_text)
            chunk_index += 1
            current_parts = []
            current_length = 0

        current_parts.append(prefixed)
        current_length += paragraph_length

    if current_parts:
        chunk_text = "\n".join(current_parts).strip()
        chunks.append(
            {
                "chunk_id": f"{doc_id}_p0001_c{chunk_index:04d}",
                "doc_id": doc_id,
                "source": source.name,
                "page": 1,
                "modality": "text",
                "section": current_section,
                "text": chunk_text,
                "char_start": char_cursor,
                "char_end": char_cursor + len(chunk_text),
            }
        )

    return chunks


async def _embed_chunks(chunk_texts: list[str]) -> list[list[float]]:
    client = build_embedding_client(settings.text_embedding)
    vectors: list[list[float]] = []
    for index, text in enumerate(chunk_texts, start=1):
        try:
            embedded = await client.embed([text])
        except Exception as exc:
            raise RuntimeError(f"Embedding failed for chunk #{index}: {text[:120]!r}") from exc
        vectors.extend(embedded)
    return vectors


async def main_async() -> int:
    args = _parse_args()
    metadata_path = Path(args.metadata_path)
    index_path = Path(args.index_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    for index, source_name in enumerate(args.sources, start=1):
        source_path = Path(source_name)
        if not source_path.is_absolute():
            source_path = Path.cwd() / source_path
        if not source_path.exists():
            raise FileNotFoundError(f"Mock source file not found: {source_path}")
        doc_id = _sanitize_doc_id(source_path, index)
        all_chunks.extend(_split_markdown(source_path, doc_id, args.target_chars, args.max_chars))

    if not all_chunks:
        raise ValueError("No mock chunks were produced.")

    chunk_texts = [item["text"] for item in all_chunks]
    vectors = await _embed_chunks(chunk_texts)
    if len(vectors) != len(all_chunks):
        raise ValueError(f"Embedding count mismatch: {len(vectors)} vectors for {len(all_chunks)} chunks.")

    metadata_payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in all_chunks)
    metadata_path.write_text(metadata_payload + "\n", encoding="utf-8")
    np.save(index_path, np.asarray(vectors, dtype=np.float32))

    print(f"[mock-data] chunks={len(all_chunks)}")
    print(f"[mock-data] metadata={metadata_path}")
    print(f"[mock-data] vectors={index_path}")
    print(f"[mock-data] vector_dim={len(vectors[0]) if vectors else 0}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
