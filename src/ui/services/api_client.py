from __future__ import annotations

import os

import requests

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000/api/v1").rstrip("/")
HEALTH_BASE = API_BASE.rsplit("/api/", 1)[0] if "/api/" in API_BASE else API_BASE


class APIClient:
    @staticmethod
    def get_api_base() -> str:
        return API_BASE

    @staticmethod
    def get_health_base() -> str:
        return HEALTH_BASE

    @staticmethod
    def capabilities() -> list[dict[str, str]]:
        return [
            {
                "endpoint": "/health",
                "purpose": "Backend health and version probe",
                "mapping": "Header status, version pill, service snapshot",
            },
            {
                "endpoint": "/api/v1/chat",
                "purpose": "Grounded multimodal QA with citations",
                "mapping": "Chat workspace, answer panel, evidence timeline",
            },
            {
                "endpoint": "/api/v1/embed/text",
                "purpose": "Text embedding service",
                "mapping": "Backend capability surface for retrieval workflows",
            },
            {
                "endpoint": "/api/v1/embed/vision",
                "purpose": "Vision embedding service",
                "mapping": "Backend capability surface for multimodal workflows",
            },
        ]

    @staticmethod
    def health() -> dict:
        response = requests.get(
            f"{HEALTH_BASE}/health",
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def chat(
        query: str,
        context: list[str] | None = None,
        images: list[str] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> dict:
        payload = {
            "query": query,
            "context": context or [],
            "image_data_urls": images or [],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            f"{API_BASE}/chat",
            json=payload,
            timeout=240,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def embed_text(inputs: list[str]) -> dict:
        response = requests.post(
            f"{API_BASE}/embed/text",
            json={"inputs": inputs},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def embed_vision(inputs: list[str]) -> dict:
        response = requests.post(
            f"{API_BASE}/embed/vision",
            json={"inputs": inputs},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
