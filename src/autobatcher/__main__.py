"""CLI entry point: python -m autobatcher serve ..."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autobatcher",
        description="Drop-in OpenAI batch proxy",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser(
        "serve",
        help="Start an OpenAI-compatible HTTP server that batches requests",
    )
    serve.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="Upstream API base URL (default: $OPENAI_BASE_URL or https://api.openai.com/v1)",
    )
    serve.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key for the upstream endpoint (default: $OPENAI_API_KEY)",
    )
    serve.add_argument("--port", type=int, default=8080, help="Listen port (default: 8080)")
    serve.add_argument("--host", default="127.0.0.1", help="Listen host (default: 127.0.0.1)")
    serve.add_argument("--batch-size", type=int, default=1000, help="Max requests per batch (default: 1000)")
    serve.add_argument("--batch-window", type=float, default=10.0, help="Batch window in seconds (default: 10)")
    serve.add_argument("--poll-interval", type=float, default=5.0, help="Poll interval in seconds (default: 5)")
    serve.add_argument("--completion-window", default="24h", choices=["24h", "1h"], help="Batch completion window (default: 24h)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        if not args.api_key:
            print("Error: --api-key or $OPENAI_API_KEY required", file=sys.stderr)
            sys.exit(1)

        from .serve import run_server

        run_server(
            base_url=args.base_url,
            api_key=args.api_key,
            port=args.port,
            host=args.host,
            batch_size=args.batch_size,
            batch_window_seconds=args.batch_window,
            poll_interval_seconds=args.poll_interval,
            completion_window=args.completion_window,
        )


if __name__ == "__main__":
    main()
