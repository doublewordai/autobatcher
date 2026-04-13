"""Tests for httpx-based partial result streaming."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse
from autobatcher.client import BatchOpenAI, _ActiveBatch
from tests.conftest import (
    make_active_batch,
    make_batch_error_line,
    make_batch_result_line,
    make_chat_completion,
    make_embedding_response,
)


def _httpx_response(
    text: str = "",
    headers: dict[str, str] | None = None,
    status_code: int = 200,
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        text=text,
        headers=headers or {},
        request=httpx.Request("GET", "https://api.test.com/v1/files/f/content"),
    )


class TestFetchPartialResults:
    async def test_partial_results_resolve_futures(
        self, client: BatchOpenAI
    ) -> None:
        """Successful partial result lines should resolve the corresponding futures."""
        ab = make_active_batch(["r1", "r2"])
        body = "\n".join([
            make_batch_result_line("r1", "answer-1"),
            make_batch_result_line("r2", "answer-2"),
        ])
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(body, {"X-Incomplete": "false"})
        )

        more = await client._fetch_partial_results(ab, "file-out")

        assert not more  # X-Incomplete: false
        assert ab.requests["r1"].future.done()
        assert ab.requests["r2"].future.done()
        assert ab.requests["r1"].future.result().choices[0].message.content == "answer-1"

    async def test_offset_appended_when_nonzero(
        self, client: BatchOpenAI
    ) -> None:
        """?offset=N should be appended when last_offset > 0."""
        ab = make_active_batch(["r1"], last_offset=42)
        client._http_client.get = AsyncMock(
            return_value=_httpx_response("", {"X-Incomplete": "false"})
        )

        await client._fetch_partial_results(ab, "file-out")

        called_url = client._http_client.get.call_args[0][0]
        assert "?offset=42" in called_url

    async def test_offset_not_appended_when_zero(
        self, client: BatchOpenAI
    ) -> None:
        """No ?offset= when last_offset is 0."""
        ab = make_active_batch(["r1"], last_offset=0)
        client._http_client.get = AsyncMock(
            return_value=_httpx_response("", {"X-Incomplete": "false"})
        )

        await client._fetch_partial_results(ab, "file-out")

        called_url = client._http_client.get.call_args[0][0]
        assert "?offset=" not in called_url

    async def test_x_last_line_updates_offset(
        self, client: BatchOpenAI
    ) -> None:
        """X-Last-Line header should update batch.last_offset."""
        ab = make_active_batch(["r1"])
        body = make_batch_result_line("r1", "ok")
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(
                body, {"X-Incomplete": "true", "X-Last-Line": "7"}
            )
        )

        await client._fetch_partial_results(ab, "file-out")

        assert ab.last_offset == 7

    async def test_error_lines_set_exceptions(
        self, client: BatchOpenAI
    ) -> None:
        """Error JSONL lines should set exceptions on the corresponding futures."""
        ab = make_active_batch(["err1"])
        body = make_batch_error_line("err1", "rate limit exceeded")
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(body, {"X-Incomplete": "false"})
        )

        await client._fetch_partial_results(ab, "file-out")

        assert ab.requests["err1"].future.done()
        with pytest.raises(Exception, match="err1 failed"):
            ab.requests["err1"].future.result()

    async def test_mixed_success_and_error(self, client: BatchOpenAI) -> None:
        """A batch with both success and error lines should handle each correctly."""
        ab = make_active_batch(["ok1", "fail1"])
        body = "\n".join([
            make_batch_result_line("ok1", "good"),
            make_batch_error_line("fail1", "bad input"),
        ])
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(body, {"X-Incomplete": "false"})
        )

        await client._fetch_partial_results(ab, "file-out")

        assert ab.requests["ok1"].future.result().choices[0].message.content == "good"
        with pytest.raises(Exception, match="fail1 failed"):
            ab.requests["fail1"].future.result()

    async def test_http_404_returns_true(self, client: BatchOpenAI) -> None:
        """HTTP 404 (file not ready) should return True (more coming)."""
        ab = make_active_batch(["r1"])

        resp = _httpx_response("Not Found", status_code=404)
        client._http_client.get = AsyncMock(side_effect=httpx.HTTPStatusError(
            "404", request=resp.request, response=resp,
        ))

        result = await client._fetch_partial_results(ab, "file-out")
        assert result is True
        assert not ab.requests["r1"].future.done()

    async def test_other_exception_returns_true(
        self, client: BatchOpenAI
    ) -> None:
        """Non-HTTP exceptions should return True (transient)."""
        ab = make_active_batch(["r1"])
        client._http_client.get = AsyncMock(side_effect=ConnectionError("timeout"))

        result = await client._fetch_partial_results(ab, "file-out")
        assert result is True
        assert not ab.requests["r1"].future.done()

    async def test_already_done_futures_not_overwritten(
        self, client: BatchOpenAI
    ) -> None:
        """If a future is already done, it should not be overwritten."""
        ab = make_active_batch(["r1"])
        # Pre-resolve the future
        original = ChatCompletion.model_validate(make_chat_completion("original"))
        ab.requests["r1"].future.set_result(original)

        body = make_batch_result_line("r1", "new-value")
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(body, {"X-Incomplete": "false"})
        )

        await client._fetch_partial_results(ab, "file-out")

        # Should still have original value
        assert ab.requests["r1"].future.result().choices[0].message.content == "original"

    async def test_x_incomplete_false_returns_false(
        self, client: BatchOpenAI
    ) -> None:
        """X-Incomplete: false should make _fetch_partial_results return False."""
        ab = make_active_batch(["r1"])
        body = make_batch_result_line("r1", "done")
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(body, {"X-Incomplete": "false"})
        )

        result = await client._fetch_partial_results(ab, "file-out")
        assert result is False

    async def test_x_incomplete_true_returns_true(
        self, client: BatchOpenAI
    ) -> None:
        """X-Incomplete: true should make _fetch_partial_results return True."""
        ab = make_active_batch(["r1", "r2"])
        body = make_batch_result_line("r1", "partial")
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(
                body, {"X-Incomplete": "true", "X-Last-Line": "1"}
            )
        )

        result = await client._fetch_partial_results(ab, "file-out")
        assert result is True

    async def test_per_request_result_type_parsing(
        self, client: BatchOpenAI
    ) -> None:
        """Each result should be parsed with its request's result_type."""
        ab = make_active_batch(
            ["chat-1", "embed-1"],
            result_types={
                "chat-1": ChatCompletion,
                "embed-1": CreateEmbeddingResponse,
            },
        )
        body = "\n".join([
            make_batch_result_line("chat-1", "hello"),
            make_batch_result_line("embed-1", body=make_embedding_response()),
        ])
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(body, {"X-Incomplete": "false"})
        )

        await client._fetch_partial_results(ab, "file-out")

        chat_result = ab.requests["chat-1"].future.result()
        assert isinstance(chat_result, ChatCompletion)
        assert chat_result.choices[0].message.content == "hello"

        embed_result = ab.requests["embed-1"].future.result()
        assert isinstance(embed_result, CreateEmbeddingResponse)
        assert len(embed_result.data) == 1
        assert embed_result.data[0].embedding == [0.1, 0.2, 0.3]
