"""Tests for close() and async context manager."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from autobatcher.client import BatchOpenAI


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
        client._poller_task = asyncio.create_task(asyncio.sleep(100))
        await client.close()
        await asyncio.sleep(0)
        assert client._poller_task.cancelled()

    async def test_close_calls_http_aclose(self, client: BatchOpenAI) -> None:
        """close() should call _http_client.aclose()."""
        await client.close()
        client._http_client.aclose.assert_awaited_once()

    async def test_close_safe_when_no_tasks(self, client: BatchOpenAI) -> None:
        """close() should not raise when no tasks are running."""
        assert not client._window_tasks
        assert client._poller_task is None
        await client.close()  # Should not raise
