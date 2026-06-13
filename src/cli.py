from __future__ import annotations

import argparse

import uvicorn

from src.serving.api import app
from src.utils.settings import settings


def run_api(host: str | None = None, port: int | None = None) -> None:
    uvicorn.run(app, host=host or settings.host, port=port or settings.port)


def run_ui() -> None:
    # The production-facing UI is now mounted directly on the FastAPI app.
    run_api()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multimodal Doc RAG unified runner")
    sub = parser.add_subparsers(dest="command", required=True)

    api = sub.add_parser("api", help="Run FastAPI backend")
    api.add_argument("--host", default=None, help="Override host")
    api.add_argument("--port", type=int, default=None, help="Override port")

    sub.add_parser("ui", help="Run the JS frontend served by FastAPI")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "api":
        run_api(host=args.host, port=args.port)
        return

    if args.command == "ui":
        run_ui()
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
