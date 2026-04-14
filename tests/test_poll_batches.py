"""Tests for the polling loop and batch status transitions."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from autobatcher.client import BatchOpenAI, _ActiveBatch
from tests.conftest import (
    make_active_batch,
    make_batch,
    make_batch_result_line,
    make_chat_completion,
)


def _httpx_response(
    text: str,
    headers: dict[str, str] | None = None,
    status_code: int = 200,
) -> httpx.Response:
    """Build a real httpx.Response for the mock HTTP client."""
    resp = httpx.Response(
        status_code=status_code,
        text=text,
        headers=headers or {},
        request=httpx.Request("GET", "https://api.test.com/v1/files/f/content"),
    )
    return resp


class TestPollBatches:
    async def test_poll_emits_progress_and_completed_events(
        self, client: BatchOpenAI
    ) -> None:
        """Polling should emit structured progress/completed lifecycle events."""
        events: list[dict] = []
        client._batch_event_handler = events.append
        ab = make_active_batch(["id-a"], batch_id="batch-evt", output_file_id="file-out")
        client._active_batches.append(ab)

        in_progress = make_batch(status="in_progress", output_file_id="file-out")
        in_progress.request_counts.completed = 0
        in_progress.request_counts.failed = 0
        in_progress.request_counts.total = 1

        completed = make_batch(status="completed", output_file_id="file-out")
        completed.request_counts.completed = 1
        completed.request_counts.failed = 0
        completed.request_counts.total = 1

        client.batches.retrieve.side_effect = [in_progress, completed]
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(
                make_batch_result_line("id-a", "reply-a"),
                {"X-Incomplete": "false"},
            )
        )

        await asyncio.wait_for(client._poll_batches(), timeout=5.0)

        event_names = [event["event"] for event in events]
        assert event_names == [
            "batch_progress",
            "batch_progress",
            "batch_completed",
        ]
        assert events[0]["counts"] == {"completed": 0, "failed": 0, "total": 1}
        assert events[1]["counts"] == {"completed": 1, "failed": 0, "total": 1}
        assert events[2]["batch_id"] == "batch-evt"

    async def test_completed_batch_resolves_futures(
        self, client: BatchOpenAI
    ) -> None:
        """A completed batch should resolve all futures with ChatCompletion objects."""
        ab = make_active_batch(["id-a", "id-b"])
        client._active_batches.append(ab)

        # batches.retrieve returns completed status
        client.batches.retrieve.return_value = make_batch(
            status="completed", output_file_id="file-out"
        )

        # _fetch_partial_results will be called by _process_completed_batch
        result_lines = "\n".join([
            make_batch_result_line("id-a", "reply-a"),
            make_batch_result_line("id-b", "reply-b"),
        ])
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(result_lines, {"X-Incomplete": "false"})
        )

        # Await the poller directly — it exits when _active_batches is empty
        await asyncio.wait_for(client._poll_batches(), timeout=5.0)

        assert ab.requests["id-a"].future.done()
        assert ab.requests["id-b"].future.done()

        result_a = ab.requests["id-a"].future.result()
        assert result_a.choices[0].message.content == "reply-a"

    async def test_failed_batch_sets_exceptions(
        self, client: BatchOpenAI
    ) -> None:
        """Failed/expired/cancelled batches should set exceptions on all futures."""
        for terminal_status in ("failed", "expired", "cancelled"):
            ab = make_active_batch(["id-1"])
            client._active_batches = [ab]

            client.batches.retrieve.return_value = make_batch(
                status=terminal_status
            )

            await asyncio.wait_for(client._poll_batches(), timeout=5.0)

            assert ab.requests["id-1"].future.done()
            with pytest.raises(Exception, match=terminal_status):
                ab.requests["id-1"].future.result()

    async def test_in_progress_fetches_partial_results(
        self, client: BatchOpenAI
    ) -> None:
        """An in_progress batch with output_file_id should trigger _fetch_partial_results."""
        ab = make_active_batch(["id-x"], output_file_id="file-partial")
        client._active_batches.append(ab)

        # First poll: in_progress with partial results
        # Second poll: completed with remaining results
        retrieve_responses = [
            make_batch(status="in_progress", output_file_id="file-partial"),
            make_batch(status="completed", output_file_id="file-partial"),
        ]
        client.batches.retrieve.side_effect = retrieve_responses

        result_line = make_batch_result_line("id-x", "partial-result")
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(result_line, {"X-Incomplete": "false"})
        )

        await asyncio.wait_for(client._poll_batches(), timeout=5.0)

        assert ab.requests["id-x"].future.done()

    async def test_output_file_id_updated_when_available(
        self, client: BatchOpenAI
    ) -> None:
        """output_file_id should be updated from status when it becomes available."""
        ab = make_active_batch(["id-u"], output_file_id="")
        client._active_batches.append(ab)

        # First poll: in_progress, no output yet
        # Second poll: in_progress, output_file_id now available
        # Third poll: completed
        retrieve_responses = [
            make_batch(status="in_progress", output_file_id=None),
            make_batch(status="in_progress", output_file_id="file-new"),
            make_batch(status="completed", output_file_id="file-new"),
        ]
        client.batches.retrieve.side_effect = retrieve_responses

        result_line = make_batch_result_line("id-u", "done")
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(result_line, {"X-Incomplete": "false"})
        )

        await asyncio.wait_for(client._poll_batches(), timeout=5.0)

        assert ab.output_file_id == "file-new"
        assert ab.requests["id-u"].future.done()

    async def test_poll_error_does_not_crash_loop(
        self, client: BatchOpenAI
    ) -> None:
        """An error polling one batch should not stop the poller from continuing."""
        ab = make_active_batch(["id-e"])
        client._active_batches.append(ab)

        # First poll raises, second poll completes
        retrieve_responses = [
            RuntimeError("network error"),
            make_batch(status="completed", output_file_id="file-out"),
        ]
        call_count = 0

        async def _side_effect(batch_id):
            nonlocal call_count
            resp = retrieve_responses[min(call_count, len(retrieve_responses) - 1)]
            call_count += 1
            if isinstance(resp, Exception):
                raise resp
            return resp

        client.batches.retrieve.side_effect = _side_effect

        result_line = make_batch_result_line("id-e", "recovered")
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(result_line, {"X-Incomplete": "false"})
        )

        await asyncio.wait_for(client._poll_batches(), timeout=5.0)

        assert ab.requests["id-e"].future.done()
        assert ab.requests["id-e"].future.result().choices[0].message.content == "recovered"

    async def test_poller_exits_when_no_active_batches(
        self, client: BatchOpenAI
    ) -> None:
        """Poller should exit naturally when _active_batches is empty."""
        # Start with no batches — poller exits immediately
        assert len(client._active_batches) == 0
        await asyncio.wait_for(client._poll_batches(), timeout=5.0)
