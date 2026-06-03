from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.clients import BaseEmbeddingClient
from src.utils.settings import RetrievalConfig


@dataclass(slots=True)
class Evidence:
    chunk_id: str
    source: str
    page: int | None
    text: str
    snippet: str
    score: float


class BaseTextRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        raise NotImplementedError


class LocalTextRetriever(BaseTextRetriever):
    def __init__(self, embedding_client: BaseEmbeddingClient, config: RetrievalConfig) -> None:
        self.embedding_client = embedding_client
        self.config = config
        self._metadata: list[dict[str, Any]] | None = None
        self._vectors: list[list[float]] | None = None

    async def retrieve(self, query: str, top_k: int | None = None) -> list[Evidence]:
        if not self.config.enable_text_retrieval:
            return []

        normalized_query = query.strip()
        if not normalized_query:
            return []

        metadata, vectors = self._load_resources()
        query_vectors = await self.embedding_client.embed([normalized_query])
        if not query_vectors:
            return []

        query_vector = query_vectors[0]
        limit = top_k or self.config.top_k_text

        ranked: list[tuple[float, dict[str, Any]]] = []
        for item, vector in zip(metadata, vectors, strict=True):
            score = self._cosine_similarity(query_vector, vector)
            if score < self.config.score_threshold:
                continue
            ranked.append((score, item))

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [self._build_evidence(item, score) for score, item in ranked[:limit]]

    def _load_resources(self) -> tuple[list[dict[str, Any]], list[list[float]]]:
        if self._metadata is None:
            self._metadata = self._load_metadata(self.config.metadata_path)
        if self._vectors is None:
            self._vectors = self._load_vectors(self.config.index_path)
        if len(self._metadata) != len(self._vectors):
            raise ValueError(
                "Metadata and vector counts do not match: "
                f"{len(self._metadata)} metadata rows vs {len(self._vectors)} vectors."
            )
        return self._metadata, self._vectors

    def _load_metadata(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Retrieval metadata file not found: {path}")

        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("chunks", payload) if isinstance(payload, dict) else payload
        else:
            raise ValueError(f"Unsupported metadata format: {path.suffix}")

        if not isinstance(rows, list):
            raise ValueError(f"Metadata file must contain a list-like payload: {path}")

        validated: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                raise ValueError(f"Metadata row must be an object: {item!r}")
            self._validate_metadata_row(item, path)
            validated.append(item)
        return validated

    def _load_vectors(self, path: Path) -> list[list[float]]:
        if not path.exists():
            raise FileNotFoundError(f"Retrieval vector file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("vectors", payload) if isinstance(payload, dict) else payload
        elif suffix == ".jsonl":
            rows = [self._extract_vector(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif suffix == ".npy":
            try:
                import numpy as np
            except ImportError as exc:
                raise RuntimeError("Loading .npy vector files requires numpy to be installed.") from exc
            rows = np.load(path, allow_pickle=False).tolist()
        else:
            raise ValueError(f"Unsupported vector format: {path.suffix}")

        if not isinstance(rows, list):
            raise ValueError(f"Vector file must contain a list-like payload: {path}")

        vectors: list[list[float]] = []
        for row in rows:
            vector = self._extract_vector(row)
            if not vector:
                raise ValueError("Vector rows must be non-empty.")
            vectors.append(vector)
        return vectors

    def _extract_vector(self, row: Any) -> list[float]:
        if isinstance(row, dict):
            if "vector" in row:
                row = row["vector"]
            elif "embedding" in row:
                row = row["embedding"]
        if not isinstance(row, list):
            raise ValueError(f"Vector row must be a list of floats: {row!r}")
        return [float(value) for value in row]

    def _validate_metadata_row(self, item: dict[str, Any], path: Path) -> None:
        required_fields = ("chunk_id", "source", "text")
        missing = [field for field in required_fields if not item.get(field)]
        if missing:
            raise ValueError(f"Metadata row in {path} is missing required fields: {', '.join(missing)}")
        page = item.get("page")
        if page is not None and not isinstance(page, int):
            raise ValueError(f"Metadata row page must be int or null: {item!r}")

    def _build_evidence(self, item: dict[str, Any], score: float) -> Evidence:
        text = str(item["text"]).strip()
        snippet = text.replace("\n", " ")[:240]
        return Evidence(
            chunk_id=str(item["chunk_id"]),
            source=str(item["source"]),
            page=item.get("page"),
            text=text,
            snippet=snippet,
            score=score,
        )

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError(f"Vector dimension mismatch: {len(left)} vs {len(right)}")

        dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot_product / (left_norm * right_norm)
