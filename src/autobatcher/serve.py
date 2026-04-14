"""
HTTP server that exposes BatchOpenAI as an OpenAI-compatible endpoint.

Usage:
    autobatcher serve --base-url https://api.doubleword.ai/v1 --api-key sk-... --port 8080

Clients talk to http://localhost:8080/v1/chat/completions (etc.) and requests
are transparently batched via the batch API. Streaming requests are accepted —
the batch is made non-streaming upstream and the complete response is re-wrapped
as an SSE stream for the caller.
"""

from __future__ import annotations

import json
from typing import Any

from aiohttp import web
from loguru import logger

from .client import BatchOpenAI


def _stdout_batch_event_handler(event: dict[str, Any]) -> None:
    """Write structured batch lifecycle events to stdout as JSON lines."""
    print(json.dumps(event, sort_keys=True), flush=True)


def _chat_completion_to_sse(data: dict[str, Any]) -> bytes:
    """Convert a ChatCompletion into SSE bytes (single chunk + [DONE])."""
    chunk = {
        "id": data.get("id", ""),
        "object": "chat.completion.chunk",
        "created": data.get("created", 0),
        "model": data.get("model", ""),
        "choices": [],
    }
    for choice in data.get("choices", []):
        chunk["choices"].append({
            "index": choice.get("index", 0),
            "delta": {
                "role": "assistant",
                "content": choice.get("message", {}).get("content", ""),
            },
            "finish_reason": choice.get("finish_reason"),
        })
    if "usage" in data:
        chunk["usage"] = data["usage"]

    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


def _response_to_sse(data: dict[str, Any]) -> bytes:
    """Convert a Responses API response into SSE bytes.

    Emits: response.created, response.output_item.added,
    response.content_part.added, response.output_text.delta,
    response.output_text.done, response.content_part.done,
    response.output_item.done, response.completed, then [DONE].
    """
    response_id = data.get("id", "")
    events: list[str] = []

    def sse(event_type: str, event_data: dict[str, Any]) -> None:
        events.append(f"event: {event_type}\ndata: {json.dumps(event_data)}\n")

    sse("response.created", {"type": "response.created", "response": data})

    for item in data.get("output", []):
        item_idx = item.get("index", 0)
        sse("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": item_idx,
            "item": item,
        })
        for part_idx, part in enumerate(item.get("content", [])):
            text = part.get("text", "")
            sse("response.content_part.added", {
                "type": "response.content_part.added",
                "output_index": item_idx,
                "content_index": part_idx,
                "part": {"type": "output_text", "text": ""},
            })
            sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "output_index": item_idx,
                "content_index": part_idx,
                "delta": text,
            })
            sse("response.output_text.done", {
                "type": "response.output_text.done",
                "output_index": item_idx,
                "content_index": part_idx,
                "text": text,
            })
            sse("response.content_part.done", {
                "type": "response.content_part.done",
                "output_index": item_idx,
                "content_index": part_idx,
                "part": part,
            })
        sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": item_idx,
            "item": item,
        })

    sse("response.completed", {"type": "response.completed", "response": data})
    events.append("data: [DONE]\n")
    return "\n".join(events).encode()


def _sse_response(body: bytes) -> web.Response:
    return web.Response(
        body=body,
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


def create_app(client: BatchOpenAI) -> web.Application:
    """Create the aiohttp application with OpenAI-compatible routes."""

    async def handle_chat_completions(request: web.Request) -> web.Response:
        body = await request.json()
        model = body.pop("model", "")
        messages = body.pop("messages", [])
        wants_stream = body.pop("stream", False)
        body.pop("stream_options", None)
        result = await client.chat.completions.create(
            model=model, messages=messages, **body
        )
        result_dict = result.model_dump(mode="json")
        if wants_stream:
            return _sse_response(_chat_completion_to_sse(result_dict))
        return web.json_response(result_dict)

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
        wants_stream = body.pop("stream", False)
        body.pop("stream_options", None)
        result = await client.responses.create(
            model=model, input=input_data, **body
        )
        result_dict = result.model_dump(mode="json")
        if wants_stream:
            return _sse_response(_response_to_sse(result_dict))
        return web.json_response(result_dict)

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
    batch_metadata: dict[str, str] | None = None,
    cancel_active_batches_on_close: bool = True,
) -> None:
    """Start the autobatcher HTTP proxy server."""
    client = BatchOpenAI(
        api_key=api_key,
        base_url=base_url,
        batch_size=batch_size,
        batch_window_seconds=batch_window_seconds,
        poll_interval_seconds=poll_interval_seconds,
        completion_window=completion_window,  # type: ignore[arg-type]
        batch_metadata=batch_metadata,
        batch_event_handler=_stdout_batch_event_handler,
        cancel_active_batches_on_close=cancel_active_batches_on_close,
    )

    app = create_app(client)
    logger.info(
        "Starting autobatcher proxy on {}:{} -> {}",
        host, port, base_url,
    )
    web.run_app(app, host=host, port=port, print=None)
