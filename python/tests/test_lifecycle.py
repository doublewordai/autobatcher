"""Tests for close() and async context manager."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from openai.types.responses import Response

from autobatcher.client import BatchOpenAI, _ActiveBatch
from tests.conftest import make_response_api_result
import time


class TestClose:
    async def test_close_cancels_window_task(self, client: BatchOpenAI) -> None:
        """close() should cancel window timer tasks."""
        task = asyncio.create_task(asyncio.sleep(100))
        client._window_tasks["/v1/chat/completions"] = task
        await client.close()
        await asyncio.sleep(0)
        assert task.cancelled()

    async def test_close_cancels_poller_task(self, client: BatchOpenAI) -> None:
        """close() should cancel the poller task."""
        poller = asyncio.create_task(asyncio.sleep(100))
        client._poller_task = poller
        await client.close()
        await asyncio.sleep(0)
        assert poller.cancelled()
        assert client._poller_task is None

    async def test_close_calls_http_aclose(self, client: BatchOpenAI) -> None:
        """close() should call _http_client.aclose()."""
        await client.close()
        client._http_client.aclose.assert_awaited_once()

    async def test_close_safe_when_no_tasks(self, client: BatchOpenAI) -> None:
        """close() should not raise when no tasks are running."""
        assert not client._window_tasks
        assert client._poller_task is None
        await client.close()  # Should not raise

    async def test_close_cancels_active_flex_poll_and_upstream_response(
        self, client: BatchOpenAI
    ) -> None:
        """Closing must not leave a local poller or paid background response alive."""
        body = make_response_api_result(output_text="")
        body["id"] = "resp-active"
        body["status"] = "queued"
        body["output"] = []
        queued = Response.model_validate(body)
        client._completion_window = None
        client._poll_interval_seconds = 60
        client._responses_api.create = AsyncMock(return_value=queued)
        client._responses_api.cancel = AsyncMock(return_value=queued)

        request = asyncio.create_task(
            client.responses.create(model="test-model", input="hello")
        )
        for _ in range(10):
            if client._responses_api.create.await_count:
                break
            await asyncio.sleep(0)
        assert client._responses_api.create.await_count == 1

        try:
            await client.close()

            assert request.cancelled()
            client._responses_api.cancel.assert_awaited_once_with("resp-active")
        finally:
            if not request.done():
                request.cancel()
                await asyncio.gather(request, return_exceptions=True)

    async def test_close_cancels_active_upstream_batches_when_enabled(
        self, client: BatchOpenAI
    ) -> None:
        """close() should best-effort cancel active upstream batches when configured."""
        events: list[dict] = []
        client._batch_event_handler = events.append
        client._cancel_active_batches_on_close = True
        response_future: asyncio.Future = asyncio.get_event_loop().create_future()
        request = type("Req", (), {})()
        request.future = response_future
        batch = _ActiveBatch(
            batch_id="batch-123",
            endpoint="/v1/chat/completions",
            input_file_id="file-in",
            output_file_id="file-out",
            error_file_id="file-err",
            request_count=1,
            requests={"cid-1": request},
            created_at=time.time(),
            models=("gpt-4o",),
        )
        client._active_batches = [batch]

        await client.close()

        client.batches.cancel.assert_awaited_once_with("batch-123")
        assert response_future.done()
        with pytest.raises(RuntimeError, match="batch batch-123 completed"):
            response_future.result()
        assert [event["event"] for event in events] == [
            "client_closing",
            "batch_cancel_requested",
            "batch_cancelled_upstream",
        ]

    async def test_close_fails_pending_requests(self, client: BatchOpenAI) -> None:
        """close() should fail queued-but-unsubmitted requests."""
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        request = type("Req", (), {})()
        request.future = future
        client._pending["/v1/chat/completions"] = [request]

        await client.close()

        assert future.done()
        with pytest.raises(RuntimeError, match="before pending request on /v1/chat/completions was submitted"):
            future.result()
