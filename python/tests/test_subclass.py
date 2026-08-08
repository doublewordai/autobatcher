"""Tests for AsyncOpenAI subclass behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI

from autobatcher.client import BatchOpenAI


class TestSubclass:
    def test_isinstance_async_openai(self, client: BatchOpenAI) -> None:
        """BatchOpenAI should pass isinstance checks against AsyncOpenAI."""
        assert isinstance(client, AsyncOpenAI)

    def test_issubclass_async_openai(self) -> None:
        """BatchOpenAI should be a subclass of AsyncOpenAI."""
        assert issubclass(BatchOpenAI, AsyncOpenAI)

    def test_inherited_attributes_accessible(self, client: BatchOpenAI) -> None:
        """Non-batched namespaces should be accessible (inherited or mocked)."""
        assert hasattr(client, "files")
        assert hasattr(client, "batches")

    def test_batched_proxies_override_parent(self, client: BatchOpenAI) -> None:
        """chat, embeddings, responses should be our batched proxies, not the parent's."""
        from autobatcher.client import _BatchedChat, _BatchedEmbeddings, _BatchedResponses

        assert isinstance(client.chat, _BatchedChat)
        assert isinstance(client.embeddings, _BatchedEmbeddings)
        assert isinstance(client.responses, _BatchedResponses)

    @pytest.mark.asyncio
    async def test_non_create_response_methods_delegate_to_openai_resource(
        self, client: BatchOpenAI
    ) -> None:
        """Replacing responses.create must not hide retrieve/cancel/delete."""
        expected = object()
        client._responses_api.retrieve = AsyncMock(return_value=expected)

        result = await client.responses.retrieve("resp-existing")

        assert result is expected
        client._responses_api.retrieve.assert_awaited_once_with("resp-existing")

    @pytest.mark.asyncio
    async def test_non_create_chat_completion_methods_delegate(
        self, client: BatchOpenAI
    ) -> None:
        """Stored completion operations must remain available after interception."""
        expected = object()
        retrieve = AsyncMock(return_value=expected)
        client._chat_api = SimpleNamespace(
            completions=SimpleNamespace(retrieve=retrieve)
        )

        result = await client.chat.completions.retrieve("chatcmpl-existing")

        assert result is expected
        retrieve.assert_awaited_once_with("chatcmpl-existing")

    @pytest.mark.asyncio
    async def test_raw_non_create_response_methods_delegate(
        self, client: BatchOpenAI
    ) -> None:
        """The raw-response facade must only override create()."""
        expected = object()
        retrieve = AsyncMock(return_value=expected)
        client._responses_api.with_raw_response = SimpleNamespace(
            retrieve=retrieve
        )

        result = await client.responses.with_raw_response.retrieve("resp-existing")

        assert result is expected
        retrieve.assert_awaited_once_with("resp-existing")
