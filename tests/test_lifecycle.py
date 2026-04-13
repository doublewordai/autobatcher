"""Tests for close() and async context manager."""

from __future__ import annotations

import asyncio

import pytest

from autobatcher.client import BatchOpenAI


class TestClose:
    async def test_close_cancels_window_task(self, client: BatchOpenAI) -> None:
        """close() should cancel the window timer task."""
        client._window_task = asyncio.create_task(asyncio.sleep(100))
        await client.close()
        # Let the event loop process the cancellation
        await asyncio.sleep(0)
        assert client._window_task.cancelled()

    async def test_close_cancels_poller_task(self, client: BatchOpenAI) -> None:
        """close() should cancel the poller task."""
        client._poller_task = asyncio.create_task(asyncio.sleep(100))
        await client.close()
        await asyncio.sleep(0)
        assert client._poller_task.cancelled()

    async def test_close_calls_http_aclose(self, client: BatchOpenAI) -> None:
        """close() should call _http_client.aclose()."""
        await client.close()
        client._http_client.aclose.assert_awaited_once()

    async def test_close_calls_openai_close(self, client: BatchOpenAI) -> None:
        """close() should call _openai.close()."""
        await client.close()
        client._openai.close.assert_awaited_once()

    async def test_context_manager_calls_close(self, client: BatchOpenAI) -> None:
        """async with should call close() on exit."""
        async with client:
            pass

        client._http_client.aclose.assert_awaited_once()
        client._openai.close.assert_awaited_once()

    async def test_close_safe_when_no_tasks(self, client: BatchOpenAI) -> None:
        """close() should not raise when no tasks are running."""
        assert client._window_task is None
        assert client._poller_task is None
        await client.close()  # Should not raise
