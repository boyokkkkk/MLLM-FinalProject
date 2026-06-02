import os
import requests

API_BASE = os.getenv(
    "API_BASE",
    "http://127.0.0.1:8000/api/v1"
)


class APIClient:

    @staticmethod
    def chat(
        query,
        context=None,
        images=None,
        temperature=0.2,
        max_tokens=512,
    ):

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