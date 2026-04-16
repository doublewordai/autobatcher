"""Tests for the responses API proxy."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from openai.types.responses import Response
from openai.types.chat import ChatCompletion
from autobatcher.client import BatchOpenAI, _PendingRequest
from tests.conftest import (
    make_active_batch,
    make_batch_error_line,
    make_batch_result_line,
    make_response_api_result,
    make_file_object,
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


class TestResponsesJSONL:
    async def test_jsonl_has_responses_url(self, client: BatchOpenAI) -> None:
        """Response requests should produce JSONL lines with url=/v1/responses."""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        req = _PendingRequest(
            custom_id="resp-1",
            endpoint="/v1/responses",
            result_type=Response,
            params={"model": "gpt-4o", "input": "What is 2+2?"},
            future=fut,
        )
        client._pending[req.endpoint] = [req]
        await client._submit_batch(req.endpoint)

        file_tuple = client.files.create.call_args.kwargs["file"]
        content = file_tuple[1].getvalue().decode()
        line = json.loads(content.strip())

        assert line["url"] == "/v1/responses"
        assert line["method"] == "POST"
        assert line["body"]["model"] == "gpt-4o"
        assert line["body"]["input"] == "What is 2+2?"

    async def test_batches_create_uses_responses_endpoint(
        self, client: BatchOpenAI
    ) -> None:
        """When all requests are responses, top-level endpoint should be /v1/responses."""
        file_obj = make_file_object("file-resp")
        client.files.create.return_value = file_obj

        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        req = _PendingRequest(
            custom_id="resp-2",
            endpoint="/v1/responses",
            result_type=Response,
            params={"model": "gpt-4o", "input": "test"},
            future=fut,
        )
        client._pending[req.endpoint] = [req]
        await client._submit_batch(req.endpoint)

        call_kwargs = client.batches.create.call_args.kwargs
        assert call_kwargs["endpoint"] == "/v1/responses"

    async def test_input_none_omitted_from_params(self, client: BatchOpenAI) -> None:
        """When input=None, it should not appear in the JSONL body."""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()

        # Simulate what _Responses.create does with input=None
        params: dict = {"model": "gpt-4o", "instructions": "Be helpful"}
        # input is None so it's not included — matches _Responses.create behavior

        req = _PendingRequest(
            custom_id="resp-none",
            endpoint="/v1/responses",
            result_type=Response,
            params=params,
            future=fut,
        )
        client._pending[req.endpoint] = [req]
        await client._submit_batch(req.endpoint)

        file_tuple = client.files.create.call_args.kwargs["file"]
        content = file_tuple[1].getvalue().decode()
        line = json.loads(content.strip())

        assert "input" not in line["body"]
        assert line["body"]["model"] == "gpt-4o"


class TestResponsesResultParsing:
    async def test_response_result_parsed_correctly(
        self, client: BatchOpenAI
    ) -> None:
        """Response API results should be parsed as Response."""
        ab = make_active_batch(
            ["resp-r1"],
            result_types={"resp-r1": Response},
        )
        body = make_batch_result_line("resp-r1", body=make_response_api_result(output_text="4"))
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(body, {"X-Incomplete": "false"})
        )

        await client._fetch_partial_results(ab, "file-out")

        result = ab.requests["resp-r1"].future.result()
        assert isinstance(result, Response)
        assert result.output[0].content[0].text == "4"

    async def test_response_error_sets_exception(
        self, client: BatchOpenAI
    ) -> None:
        """Error in a response result should set exception on the future."""
        ab = make_active_batch(
            ["resp-err"],
            result_types={"resp-err": Response},
        )
        body = make_batch_error_line("resp-err", "model not found")
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(body, {"X-Incomplete": "false"})
        )

        await client._fetch_partial_results(ab, "file-out")

        assert ab.requests["resp-err"].future.done()
        with pytest.raises(Exception, match="resp-err failed"):
            ab.requests["resp-err"].future.result()


class TestMixedResponsesAndChat:
    async def test_mixed_batch_parses_both_types(
        self, client: BatchOpenAI
    ) -> None:
        """A batch with both responses and chat completions should parse each correctly."""
        ab = make_active_batch(
            ["chat-1", "resp-1"],
            result_types={
                "chat-1": ChatCompletion,
                "resp-1": Response,
            },
        )
        body = "\n".join([
            make_batch_result_line("chat-1", "hello from chat"),
            make_batch_result_line("resp-1", body=make_response_api_result(output_text="hello from responses")),
        ])
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(body, {"X-Incomplete": "false"})
        )

        await client._fetch_partial_results(ab, "file-out")

        chat_result = ab.requests["chat-1"].future.result()
        assert isinstance(chat_result, ChatCompletion)
        assert chat_result.choices[0].message.content == "hello from chat"

        resp_result = ab.requests["resp-1"].future.result()
        assert isinstance(resp_result, Response)
        assert resp_result.output[0].content[0].text == "hello from responses"
