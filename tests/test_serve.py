"""Tests for the HTTP serve module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from autobatcher.serve import create_app
from autobatcher.client import BatchOpenAI


@pytest.fixture
def mock_client():
    """Create a mock BatchOpenAI client."""
    client = MagicMock(spec=BatchOpenAI)
    client.close = AsyncMock()

    # Mock chat.completions.create
    mock_chat_result = MagicMock()
    mock_chat_result.model_dump.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}}],
    }
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=mock_chat_result)

    # Mock embeddings.create
    mock_embed_result = MagicMock()
    mock_embed_result.model_dump.return_value = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
        "model": "text-embedding-3-small",
    }
    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock(return_value=mock_embed_result)

    # Mock responses.create
    mock_resp_result = MagicMock()
    mock_resp_result.model_dump.return_value = {
        "id": "resp-123",
        "output": [{"type": "message", "content": [{"type": "text", "text": "Hi"}]}],
    }
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=mock_resp_result)

    return client


@pytest.fixture
def app(mock_client):
    """Create the aiohttp app with a mock client."""
    return create_app(mock_client)


async def test_health(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/health")
    assert resp.status == 200
    text = await resp.text()
    assert text == "ok"


async def test_chat_completions(aiohttp_client, app, mock_client):
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status == 200
    data = await resp.json()
    assert data["id"] == "chatcmpl-123"
    mock_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o", messages=[{"role": "user", "content": "Hi"}]
    )


async def test_embeddings(aiohttp_client, app, mock_client):
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "hello"},
    )
    assert resp.status == 200
    data = await resp.json()
    assert data["object"] == "list"
    mock_client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-3-small", input="hello"
    )


async def test_responses(aiohttp_client, app, mock_client):
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/responses",
        json={"model": "gpt-4o", "input": "Hello"},
    )
    assert resp.status == 200
    data = await resp.json()
    assert data["id"] == "resp-123"
    mock_client.responses.create.assert_awaited_once_with(
        model="gpt-4o", input="Hello"
    )


async def test_chat_completions_extra_params(aiohttp_client, app, mock_client):
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
            "max_tokens": 100,
        },
    )
    assert resp.status == 200
    mock_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.5,
        max_tokens=100,
    )


async def test_chat_completions_stream(aiohttp_client, app, mock_client):
    """stream: true should return SSE and strip stream from upstream call."""
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert resp.status == 200
    assert resp.content_type == "text/event-stream"

    body = await resp.text()
    assert "data: " in body
    assert "data: [DONE]" in body

    # Parse the SSE chunk
    import json
    lines = [l for l in body.strip().split("\n") if l.startswith("data: ") and l != "data: [DONE]"]
    assert len(lines) == 1
    chunk = json.loads(lines[0][6:])
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["choices"][0]["delta"]["content"] == "Hello!"
    assert chunk["choices"][0]["delta"]["role"] == "assistant"

    # stream/stream_options should NOT be passed to BatchOpenAI
    mock_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
    )


async def test_chat_completions_no_stream(aiohttp_client, app, mock_client):
    """stream: false should return normal JSON."""
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )
    assert resp.status == 200
    assert resp.content_type == "application/json"
    data = await resp.json()
    assert data["id"] == "chatcmpl-123"


async def test_responses_stream(aiohttp_client, app, mock_client):
    """stream: true on responses should return SSE events."""
    client = await aiohttp_client(app)
    resp = await client.post(
        "/v1/responses",
        json={"model": "gpt-4o", "input": "Hello", "stream": True},
    )
    assert resp.status == 200
    assert resp.content_type == "text/event-stream"

    body = await resp.text()
    assert "event: response.created" in body
    assert "event: response.output_text.delta" in body
    assert "event: response.completed" in body
    assert "data: [DONE]" in body

    # stream should NOT be passed to BatchOpenAI
    mock_client.responses.create.assert_awaited_once_with(
        model="gpt-4o", input="Hello",
    )
