from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from autobatcher.client import BatchOpenAI, _ActiveBatch, _PendingRequest

from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse
from openai.types.responses import Response


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_chat_completion(content: str = "Hello!", model: str = "gpt-4o") -> dict:
    """Return a raw dict that passes ChatCompletion.model_validate()."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def make_embedding_response(model: str = "text-embedding-3-small") -> dict:
    """Return a raw dict that passes CreateEmbeddingResponse.model_validate()."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1, 0.2, 0.3],
            }
        ],
        "model": model,
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


def make_response_api_result(model: str = "gpt-4o", output_text: str = "Hello!") -> dict:
    """Return a raw dict that passes Response.model_validate()."""
    return {
        "id": "resp-test123",
        "object": "response",
        "created_at": 1700000000,
        "model": model,
        "output": [
            {
                "type": "message",
                "id": "msg-test123",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": output_text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "status": "completed",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def make_batch_result_line(
    custom_id: str,
    content: str = "Hi",
    *,
    body: dict | None = None,
) -> str:
    """JSONL line for a successful batch result.

    If *body* is given it is used as-is; otherwise a ChatCompletion body is built
    from *content*.
    """
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "request_id": "req-test",
                "body": body if body is not None else make_chat_completion(content),
            },
        }
    )


def make_batch_error_line(custom_id: str, msg: str = "rate_limit") -> str:
    """JSONL line for a failed batch result."""
    return json.dumps(
        {
            "custom_id": custom_id,
            "error": {"code": "server_error", "message": msg},
        }
    )


def make_file_object(file_id: str = "file-abc123") -> MagicMock:
    obj = MagicMock()
    obj.id = file_id
    return obj


def make_batch(
    batch_id: str = "batch-001",
    status: str = "completed",
    output_file_id: str | None = "file-out",
    error_file_id: str | None = None,
) -> MagicMock:
    """Return a mock OpenAI Batch object."""
    obj = MagicMock()
    obj.id = batch_id
    obj.status = status
    obj.output_file_id = output_file_id
    obj.error_file_id = error_file_id
    counts = MagicMock()
    counts.completed = 0
    counts.total = 0
    obj.request_counts = counts
    return obj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_openai() -> AsyncMock:
    """Mocked AsyncOpenAI with files and batches sub-clients."""
    openai = AsyncMock(spec=["files", "batches", "close"])
    openai.files = AsyncMock()
    openai.files.create = AsyncMock(return_value=make_file_object())
    openai.batches = AsyncMock()
    openai.batches.create = AsyncMock(
        return_value=make_batch(status="in_progress", output_file_id=None)
    )
    openai.batches.retrieve = AsyncMock(return_value=make_batch())
    openai.close = AsyncMock()
    return openai


@pytest.fixture()
def client(mock_openai: AsyncMock) -> BatchOpenAI:
    """BatchOpenAI with mocked internals and fast timers.

    Uses __new__ to skip AsyncOpenAI.__init__, then wires up the
    mocked sub-clients and batched proxy classes manually.
    """
    c = BatchOpenAI.__new__(BatchOpenAI)
    # Attributes that replaced the old self._openai composition:
    # The subclass now uses self.files / self.batches directly,
    # so we attach the mocks where the real cached_property would live.
    c.files = mock_openai.files
    c.batches = mock_openai.batches
    c._api_base_url = "https://api.test.com/v1"
    c._api_key_str = "sk-test"
    c._batch_size = 3
    c._batch_window_seconds = 0.05
    c._poll_interval_seconds = 0.05
    c._completion_window = "24h"
    c._http_client = AsyncMock(spec=httpx.AsyncClient)
    c._pending = {}
    c._pending_lock = asyncio.Lock()
    c._window_tasks = {}
    c._active_batches = []
    c._poller_task = None
    # Mock the internal httpx client that AsyncOpenAI.close() would call.
    # Since we skip __init__ via __new__, this isn't set up automatically.
    c._client = AsyncMock()
    c._client.aclose = AsyncMock()
    from autobatcher.client import _BatchedChat, _BatchedEmbeddings, _BatchedResponses
    c.chat = _BatchedChat(c)
    c.embeddings = _BatchedEmbeddings(c)
    c.responses = _BatchedResponses(c)
    return c


def make_active_batch(
    custom_ids: list[str],
    batch_id: str = "batch-001",
    output_file_id: str = "",
    last_offset: int = 0,
    result_types: dict[str, type] | None = None,
) -> _ActiveBatch:
    """Build an _ActiveBatch with futures for the given custom_ids."""
    loop = asyncio.get_event_loop()
    requests = {}
    computed_result_types: dict[str, type] = {}
    for cid in custom_ids:
        fut: asyncio.Future = loop.create_future()
        rt = (result_types or {}).get(cid, ChatCompletion)
        requests[cid] = _PendingRequest(
            custom_id=cid,
            endpoint="/v1/chat/completions",
            result_type=rt,
            params={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            future=fut,
        )
        computed_result_types[cid] = rt
    return _ActiveBatch(
        batch_id=batch_id,
        endpoint="/v1/chat/completions",
        output_file_id=output_file_id,
        error_file_id="",
        requests=requests,
        created_at=time.time(),
        result_types=computed_result_types,
        last_offset=last_offset,
    )
