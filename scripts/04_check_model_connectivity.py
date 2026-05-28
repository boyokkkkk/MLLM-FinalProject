from __future__ import annotations

import argparse
import asyncio
from typing import Any

from src.models.clients import build_embedding_client, build_llm_client
from src.utils.settings import settings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check OpenAI-compatible model connectivity.")
    parser.add_argument(
        "--targets",
        default="chat,text_emb",
        help="Comma separated targets: chat,text_emb,vision_emb",
    )
    parser.add_argument(
        "--chat-query",
        default="请回答: OK",
        help="Probe query for chat target.",
    )
    parser.add_argument(
        "--emb-input",
        default="connectivity check",
        help="Probe text for embedding targets.",
    )
    return parser.parse_args()


async def _check_chat(query: str) -> tuple[bool, str]:
    client = build_llm_client(settings.vlm)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": query},
    ]
    out = await client.chat(messages=messages, temperature=0.0, max_tokens=64)
    if not isinstance(out, str) or not out.strip():
        return False, "empty response"
    return True, out[:120]


async def _check_text_embedding(text: str) -> tuple[bool, str]:
    client = build_embedding_client(settings.text_embedding)
    vectors = await client.embed([text])
    if not vectors or not vectors[0]:
        return False, "empty embedding vectors"
    return True, f"dim={len(vectors[0])}"


async def _check_vision_embedding(text: str) -> tuple[bool, str]:
    client = build_embedding_client(settings.vision_embedding)
    vectors = await client.embed([text])
    if not vectors or not vectors[0]:
        return False, "empty embedding vectors"
    return True, f"dim={len(vectors[0])}"


async def main_async() -> int:
    args = _parse_args()
    targets = {item.strip() for item in args.targets.split(",") if item.strip()}

    checks: list[tuple[str, Any]] = []
    if "chat" in targets:
        checks.append(("chat", _check_chat(args.chat_query)))
    if "text_emb" in targets:
        checks.append(("text_emb", _check_text_embedding(args.emb_input)))
    if "vision_emb" in targets:
        checks.append(("vision_emb", _check_vision_embedding(args.emb_input)))

    if not checks:
        print("[check] no valid targets, allowed: chat,text_emb,vision_emb")
        return 2

    failed = False
    for name, coro in checks:
        try:
            ok, detail = await coro
            if ok:
                print(f"[ok] {name}: {detail}")
            else:
                failed = True
                print(f"[fail] {name}: {detail}")
        except Exception as exc:
            failed = True
            print(f"[fail] {name}: {exc}")

    if failed:
        print("[check] FAILED")
        return 1

    print("[check] OK")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
