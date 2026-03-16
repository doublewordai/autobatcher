"""
HTTP server that exposes BatchOpenAI as an OpenAI-compatible endpoint.

Usage:
    autobatcher serve --base-url https://api.doubleword.ai/v1 --api-key sk-... --port 8080

Clients talk to http://localhost:8080/v1/chat/completions (etc.) and requests
are transparently batched via the batch API.
"""

from __future__ import annotations

import asyncio
import json
import signal
from typing import Any

from aiohttp import web
from loguru import logger

from .client import BatchOpenAI


def create_app(client: BatchOpenAI) -> web.Application:
    """Create the aiohttp application with OpenAI-compatible routes."""

    async def handle_chat_completions(request: web.Request) -> web.Response:
        body = await request.json()
        model = body.pop("model", "")
        messages = body.pop("messages", [])
        result = await client.chat.completions.create(
            model=model, messages=messages, **body
        )
        return web.json_response(result.model_dump(mode="json"))

    async def handle_embeddings(request: web.Request) -> web.Response:
        body = await request.json()
        model = body.pop("model", "")
        input_data = body.pop("input", "")
        result = await client.embeddings.create(
            model=model, input=input_data, **body
        )
        return web.json_response(result.model_dump(mode="json"))

    async def handle_responses(request: web.Request) -> web.Response:
        body = await request.json()
        model = body.pop("model", "")
        input_data = body.pop("input", None)
        result = await client.responses.create(
            model=model, input=input_data, **body
        )
        return web.json_response(result.model_dump(mode="json"))

    async def handle_health(request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def on_cleanup(app: web.Application) -> None:
        await client.close()

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_post("/v1/embeddings", handle_embeddings)
    app.router.add_post("/v1/responses", handle_responses)
    app.router.add_get("/health", handle_health)
    app.on_cleanup.append(on_cleanup)
    return app


def run_server(
    *,
    base_url: str,
    api_key: str,
    port: int = 8080,
    host: str = "127.0.0.1",
    batch_size: int = 1000,
    batch_window_seconds: float = 10.0,
    poll_interval_seconds: float = 5.0,
    completion_window: str = "24h",
) -> None:
    """Start the autobatcher HTTP proxy server."""
    client = BatchOpenAI(
        api_key=api_key,
        base_url=base_url,
        batch_size=batch_size,
        batch_window_seconds=batch_window_seconds,
        poll_interval_seconds=poll_interval_seconds,
        completion_window=completion_window,  # type: ignore[arg-type]
    )

    app = create_app(client)
    logger.info(
        "Starting autobatcher proxy on {}:{} -> {}",
        host, port, base_url,
    )
    web.run_app(app, host=host, port=port, print=None)
