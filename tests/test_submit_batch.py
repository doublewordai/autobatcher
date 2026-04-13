"""Tests for batch JSONL construction and submission API calls."""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse
from autobatcher.client import BatchOpenAI, _PendingRequest
from tests.conftest import make_batch, make_file_object


def _add_pending(
    client: BatchOpenAI,
    n: int = 2,
    endpoint: str = "/v1/chat/completions",
    result_type: type = ChatCompletion,
) -> list[_PendingRequest]:
    """Add n pending requests to the client (synchronous helper)."""
    loop = asyncio.get_event_loop()
    reqs = []
    for i in range(n):
        fut = loop.create_future()
        req = _PendingRequest(
            custom_id=f"cid-{i}",
            endpoint=endpoint,
            result_type=result_type,
            params={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": f"msg {i}"}],
            },
            future=fut,
        )
        client._pending.append(req)
        reqs.append(req)
    return reqs


class TestSubmitBatch:
    async def test_jsonl_format(self, client: BatchOpenAI) -> None:
        """Each JSONL line must have custom_id, method, url, and body."""
        reqs = _add_pending(client, 2)
        await client._submit_batch()

        file_tuple = client._openai.files.create.call_args.kwargs["file"]
        content = file_tuple[1].getvalue().decode()
        lines = content.strip().split("\n")

        assert len(lines) == 2
        for i, raw in enumerate(lines):
            parsed = json.loads(raw)
            assert parsed["custom_id"] == f"cid-{i}"
            assert parsed["method"] == "POST"
            assert parsed["url"] == "/v1/chat/completions"
            assert "model" in parsed["body"]
            assert "messages" in parsed["body"]

    async def test_files_create_called_with_batch_purpose(
        self, client: BatchOpenAI
    ) -> None:
        """files.create must be called with purpose='batch'."""
        _add_pending(client, 1)
        await client._submit_batch()

        call_kwargs = client._openai.files.create.call_args.kwargs
        assert call_kwargs["purpose"] == "batch"
        # file tuple: (filename, BytesIO, content_type)
        filename = call_kwargs["file"][0]
        assert filename.startswith("batch-")
        assert filename.endswith(".jsonl")

    async def test_batches_create_called_correctly(
        self, client: BatchOpenAI
    ) -> None:
        """batches.create must receive input_file_id, endpoint, completion_window."""
        file_obj = make_file_object("file-xyz")
        client._openai.files.create.return_value = file_obj

        _add_pending(client, 1)
        await client._submit_batch()

        call_kwargs = client._openai.batches.create.call_args.kwargs
        assert call_kwargs["input_file_id"] == "file-xyz"
        assert call_kwargs["endpoint"] == "/v1/chat/completions"
        assert call_kwargs["completion_window"] == "24h"

    async def test_batches_create_passes_through_arbitrary_completion_window(
        self, client: BatchOpenAI
    ) -> None:
        """Nonstandard completion_window values should be passed through unchanged."""
        file_obj = make_file_object("file-xyz")
        client._openai.files.create.return_value = file_obj
        client._completion_window = "72h"

        _add_pending(client, 1)
        await client._submit_batch()

        call_kwargs = client._openai.batches.create.call_args.kwargs
        assert call_kwargs["completion_window"] == "72h"

    async def test_active_batches_populated(self, client: BatchOpenAI) -> None:
        """After submission, _active_batches should contain the batch with correct request map."""
        batch_resp = make_batch(
            batch_id="batch-999",
            status="in_progress",
            output_file_id=None,
        )
        client._openai.batches.create.return_value = batch_resp

        reqs = _add_pending(client, 2)
        await client._submit_batch()

        assert len(client._active_batches) == 1
        ab = client._active_batches[0]
        assert ab.batch_id == "batch-999"
        assert set(ab.requests.keys()) == {"cid-0", "cid-1"}

    async def test_active_batch_has_result_types(self, client: BatchOpenAI) -> None:
        """_ActiveBatch.result_types should map custom_id to the request's result_type."""
        batch_resp = make_batch(batch_id="batch-rt", status="in_progress", output_file_id=None)
        client._openai.batches.create.return_value = batch_resp

        _add_pending(client, 2)
        await client._submit_batch()

        ab = client._active_batches[0]
        assert ab.result_types["cid-0"] is ChatCompletion
        assert ab.result_types["cid-1"] is ChatCompletion

    async def test_poller_task_started(self, client: BatchOpenAI) -> None:
        """A poller task should be started after a successful submission."""
        _add_pending(client, 1)
        await client._submit_batch()

        assert client._poller_task is not None
        assert not client._poller_task.done()
        # Cleanup
        client._poller_task.cancel()
        try:
            await client._poller_task
        except asyncio.CancelledError:
            pass

    async def test_submission_failure_propagates_to_futures(
        self, client: BatchOpenAI
    ) -> None:
        """If files.create raises, all pending futures should get the exception."""
        client._openai.files.create.side_effect = RuntimeError("upload failed")

        reqs = _add_pending(client, 2)
        await client._submit_batch()

        for req in reqs:
            assert req.future.done()
            with pytest.raises(RuntimeError, match="upload failed"):
                req.future.result()

    async def test_empty_pending_is_noop(self, client: BatchOpenAI) -> None:
        """_submit_batch with no pending requests should not call any API."""
        assert len(client._pending) == 0
        await client._submit_batch()

        client._openai.files.create.assert_not_called()
        client._openai.batches.create.assert_not_called()

    async def test_mixed_endpoint_jsonl(self, client: BatchOpenAI) -> None:
        """Requests with different endpoints should produce JSONL with per-request urls."""
        loop = asyncio.get_event_loop()

        # Add a chat completion request
        chat_fut = loop.create_future()
        chat_req = _PendingRequest(
            custom_id="cid-chat",
            endpoint="/v1/chat/completions",
            result_type=ChatCompletion,
            params={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            future=chat_fut,
        )

        # Add an embedding request
        embed_fut = loop.create_future()
        embed_req = _PendingRequest(
            custom_id="cid-embed",
            endpoint="/v1/embeddings",
            result_type=CreateEmbeddingResponse,
            params={"model": "text-embedding-3-small", "input": "hello"},
            future=embed_fut,
        )

        client._pending = [chat_req, embed_req]
        await client._submit_batch()

        file_tuple = client._openai.files.create.call_args.kwargs["file"]
        content = file_tuple[1].getvalue().decode()
        lines = [json.loads(l) for l in content.strip().split("\n")]

        assert lines[0]["url"] == "/v1/chat/completions"
        assert lines[0]["custom_id"] == "cid-chat"
        assert lines[1]["url"] == "/v1/embeddings"
        assert lines[1]["custom_id"] == "cid-embed"

        # Top-level endpoint should be the first request's
        call_kwargs = client._openai.batches.create.call_args.kwargs
        assert call_kwargs["endpoint"] == "/v1/chat/completions"

    async def test_mixed_endpoint_result_types(self, client: BatchOpenAI) -> None:
        """Mixed-endpoint batch should have correct result_types mapping."""
        loop = asyncio.get_event_loop()

        batch_resp = make_batch(batch_id="batch-mix", status="in_progress", output_file_id=None)
        client._openai.batches.create.return_value = batch_resp

        chat_fut = loop.create_future()
        embed_fut = loop.create_future()

        client._pending = [
            _PendingRequest(
                custom_id="cid-c",
                endpoint="/v1/chat/completions",
                result_type=ChatCompletion,
                params={"model": "gpt-4o", "messages": []},
                future=chat_fut,
            ),
            _PendingRequest(
                custom_id="cid-e",
                endpoint="/v1/embeddings",
                result_type=CreateEmbeddingResponse,
                params={"model": "text-embedding-3-small", "input": "hi"},
                future=embed_fut,
            ),
        ]
        await client._submit_batch()

        ab = client._active_batches[0]
        assert ab.result_types["cid-c"] is ChatCompletion
        assert ab.result_types["cid-e"] is CreateEmbeddingResponse
