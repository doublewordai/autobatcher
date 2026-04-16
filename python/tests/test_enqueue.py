"""Tests for request queuing and batch-trigger logic."""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from openai.types.chat import ChatCompletion
from autobatcher.client import BatchOpenAI
from tests.conftest import make_batch, make_file_object

EP = "/v1/chat/completions"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _enqueue_one(client: BatchOpenAI, content: str = "hi") -> asyncio.Future:
    """Fire-and-forget enqueue; returns the underlying future (don't await it)."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    from autobatcher.client import _PendingRequest
    import uuid as _uuid

    req = _PendingRequest(
        custom_id=str(_uuid.uuid4()),
        endpoint=EP,
        result_type=ChatCompletion,
        params={"model": "gpt-4o", "messages": [{"role": "user", "content": content}]},
        future=future,
    )
    async with client._pending_lock:
        if EP not in client._pending:
            client._pending[EP] = []
        client._pending[EP].append(req)
        if len(client._pending[EP]) == 1:
            client._window_tasks[EP] = asyncio.create_task(
                client._window_timer(EP), name="batch_window_timer"
            )
        if len(client._pending[EP]) >= client._batch_size:
            await client._submit_batch(EP)
    return future


class TestEnqueue:
    async def test_single_request_starts_window_timer(self, client: BatchOpenAI) -> None:
        """First enqueued request should start the window timer task."""
        assert EP not in client._window_tasks
        await _enqueue_one(client)
        assert EP in client._window_tasks
        assert not client._window_tasks[EP].done()
        # Cleanup
        client._window_tasks[EP].cancel()

    async def test_batch_size_triggers_immediate_submit(
        self, client: BatchOpenAI
    ) -> None:
        """Hitting batch_size should call _submit_batch immediately."""
        # batch_size is 3
        for _ in range(3):
            await _enqueue_one(client)

        # files.create should have been called once (batch was submitted)
        client.files.create.assert_called_once()
        assert len(client._pending.get(EP, [])) == 0

    async def test_window_timer_fires_for_undersized_batch(
        self, client: BatchOpenAI
    ) -> None:
        """An undersized batch should still be submitted after the window elapses."""
        await _enqueue_one(client)
        assert len(client._pending.get(EP, [])) == 1

        # Await the window timer task directly instead of guessing a sleep duration
        await asyncio.wait_for(client._window_tasks[EP], timeout=5.0)

        client.files.create.assert_called_once()
        assert len(client._pending.get(EP, [])) == 0

        # Cleanup poller started by _submit_batch
        if client._poller_task and not client._poller_task.done():
            client._poller_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await client._poller_task

    async def test_request_params_passed_correctly(
        self, client: BatchOpenAI
    ) -> None:
        """Model, messages, and kwargs should appear in the JSONL body."""
        from autobatcher.client import _PendingRequest

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        req = _PendingRequest(
            custom_id="test-id-1",
            endpoint=EP,
            result_type=ChatCompletion,
            params={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.5,
            },
            future=future,
        )
        async with client._pending_lock:
            if EP not in client._pending:
                client._pending[EP] = []
            client._pending[EP].append(req)
        await client._submit_batch(EP)

        # Inspect the JSONL content passed to files.create
        call_kwargs = client.files.create.call_args
        file_tuple = call_kwargs.kwargs["file"]
        # file_tuple is (filename, BytesIO, content_type)
        file_bytes = file_tuple[1].getvalue()
        line = json.loads(file_bytes.decode())

        assert line["body"]["model"] == "gpt-4o-mini"
        assert line["body"]["messages"] == [{"role": "user", "content": "test"}]
        assert line["body"]["temperature"] == 0.5
        assert line["method"] == "POST"
        assert line["url"] == "/v1/chat/completions"

    async def test_custom_id_is_valid_uuid(self, client: BatchOpenAI) -> None:
        """custom_id generated during enqueue should be a valid UUID."""
        # Use the real _enqueue_request but don't await the future
        task = asyncio.create_task(
            client._enqueue_request(
                endpoint=EP,
                result_type=ChatCompletion,
                params={"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]},
            )
        )
        # Yield control so the task enqueues (deterministic, no wall-clock wait)
        await asyncio.sleep(0)

        assert len(client._pending.get(EP, [])) == 1
        custom_id = client._pending[EP][0].custom_id
        # Should not raise
        parsed = uuid.UUID(custom_id)
        assert str(parsed) == custom_id

        # Cleanup
        client._window_tasks[EP].cancel()
        client._pending[EP][0].future.set_result(None)
        await asyncio.sleep(0)

    async def test_concurrent_requests_batched_together(
        self, client: BatchOpenAI
    ) -> None:
        """Multiple concurrent requests should land in a single batch."""
        client._batch_size = 10

        # Enqueue 10 requests concurrently
        futures = await asyncio.gather(
            *[_enqueue_one(client, f"msg-{i}") for i in range(10)]
        )

        # All should have been submitted in one batch
        client.files.create.assert_called_once()
        assert len(client._pending.get(EP, [])) == 0
