"""Tests for AsyncOpenAI subclass behavior."""

from __future__ import annotations

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
