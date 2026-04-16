"""Tests for the ``with_raw_response`` accessor on chat / embeddings / responses.

These exist because consumers like ``langchain-openai`` call
``await client.chat.completions.with_raw_response.create(**payload)``
unconditionally on their non-streaming async path, then call ``.parse()`` on
the result. The proxies need to expose the same surface as the openai SDK's
``with_raw_response`` accessor for those consumers to work against
``BatchOpenAI``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse
from openai.types.responses import Response

from autobatcher.client import (
    BatchOpenAI,
    _ChatCompletionsRawResponse,
    _EmbeddingsRawResponse,
    _RawResponseWrapper,
    _ResponsesRawResponse,
)
from tests.conftest import (
    make_chat_completion,
    make_embedding_response,
    make_response_api_result,
)


class TestRawResponseWrapper:
    """The wrapper itself — has the three attributes consumers read."""

    def test_parse_returns_underlying_value(self) -> None:
        sentinel = object()
        wrapper = _RawResponseWrapper(sentinel)
        assert wrapper.parse() is sentinel

    def test_headers_is_empty_dict(self) -> None:
        wrapper = _RawResponseWrapper("anything")
        assert wrapper.headers == {}
        assert isinstance(wrapper.headers, dict)

    def test_http_response_is_none(self) -> None:
        wrapper = _RawResponseWrapper("anything")
        assert wrapper.http_response is None


class TestChatCompletionsWithRawResponse:
    async def test_with_raw_response_returns_accessor(self, client: BatchOpenAI) -> None:
        accessor = client.chat.completions.with_raw_response
        assert isinstance(accessor, _ChatCompletionsRawResponse)

    async def test_create_wraps_parsed_chat_completion(
        self, client: BatchOpenAI
    ) -> None:
        # Bypass the batching path: pretend the enqueued request resolved to a
        # parsed ChatCompletion immediately.
        parsed = ChatCompletion.model_validate(make_chat_completion("hi"))
        client._enqueue_request = AsyncMock(return_value=parsed)  # type: ignore[method-assign]

        raw = await client.chat.completions.with_raw_response.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
        )

        assert isinstance(raw, _RawResponseWrapper)
        assert raw.parse() is parsed
        assert raw.headers == {}
        assert raw.http_response is None

    async def test_create_forwards_kwargs(self, client: BatchOpenAI) -> None:
        captured: dict = {}

        async def fake_enqueue(**kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs.get("params", {}))
            return ChatCompletion.model_validate(make_chat_completion("ok"))

        client._enqueue_request = fake_enqueue  # type: ignore[method-assign]

        await client.chat.completions.with_raw_response.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
        )
        assert captured["model"] == "gpt-4o"
        assert captured["temperature"] == 0.5


class TestEmbeddingsWithRawResponse:
    async def test_with_raw_response_returns_accessor(self, client: BatchOpenAI) -> None:
        accessor = client.embeddings.with_raw_response
        assert isinstance(accessor, _EmbeddingsRawResponse)

    async def test_create_wraps_parsed_embedding_response(
        self, client: BatchOpenAI
    ) -> None:
        parsed = CreateEmbeddingResponse.model_validate(make_embedding_response())
        client._enqueue_request = AsyncMock(return_value=parsed)  # type: ignore[method-assign]

        raw = await client.embeddings.with_raw_response.create(
            model="text-embedding-3-small",
            input="hello",
        )

        assert isinstance(raw, _RawResponseWrapper)
        assert raw.parse() is parsed
        assert raw.headers == {}
        assert raw.http_response is None


class TestResponsesWithRawResponse:
    async def test_with_raw_response_returns_accessor(self, client: BatchOpenAI) -> None:
        accessor = client.responses.with_raw_response
        assert isinstance(accessor, _ResponsesRawResponse)

    async def test_create_wraps_parsed_response(self, client: BatchOpenAI) -> None:
        parsed = Response.model_validate(make_response_api_result())
        client._enqueue_request = AsyncMock(return_value=parsed)  # type: ignore[method-assign]

        raw = await client.responses.with_raw_response.create(
            model="gpt-4o",
            input="hello",
        )

        assert isinstance(raw, _RawResponseWrapper)
        assert raw.parse() is parsed
        assert raw.headers == {}
        assert raw.http_response is None
