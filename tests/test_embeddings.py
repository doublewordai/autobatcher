"""Tests for the embeddings proxy."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion
from autobatcher.client import BatchOpenAI, _PendingRequest
from tests.conftest import (
    make_active_batch,
    make_batch,
    make_batch_error_line,
    make_batch_result_line,
    make_embedding_response,
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


class TestEmbeddingsJSONL:
    async def test_jsonl_has_embeddings_url(self, client: BatchOpenAI) -> None:
        """Embedding requests should produce JSONL lines with url=/v1/embeddings."""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        req = _PendingRequest(
            custom_id="emb-1",
            endpoint="/v1/embeddings",
            result_type=CreateEmbeddingResponse,
            params={"model": "text-embedding-3-small", "input": "hello world"},
            future=fut,
        )
        client._pending = [req]
        await client._submit_batch()

        file_tuple = client._openai.files.create.call_args.kwargs["file"]
        content = file_tuple[1].getvalue().decode()
        line = json.loads(content.strip())

        assert line["url"] == "/v1/embeddings"
        assert line["method"] == "POST"
        assert line["body"]["model"] == "text-embedding-3-small"
        assert line["body"]["input"] == "hello world"

    async def test_batches_create_uses_embeddings_endpoint(
        self, client: BatchOpenAI
    ) -> None:
        """When all requests are embeddings, top-level endpoint should be /v1/embeddings."""
        file_obj = make_file_object("file-emb")
        client._openai.files.create.return_value = file_obj

        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        req = _PendingRequest(
            custom_id="emb-2",
            endpoint="/v1/embeddings",
            result_type=CreateEmbeddingResponse,
            params={"model": "text-embedding-3-small", "input": "test"},
            future=fut,
        )
        client._pending = [req]
        await client._submit_batch()

        call_kwargs = client._openai.batches.create.call_args.kwargs
        assert call_kwargs["endpoint"] == "/v1/embeddings"

    async def test_embedding_body_with_list_input(self, client: BatchOpenAI) -> None:
        """Embedding with list input should preserve the list in the JSONL body."""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        req = _PendingRequest(
            custom_id="emb-list",
            endpoint="/v1/embeddings",
            result_type=CreateEmbeddingResponse,
            params={"model": "text-embedding-3-small", "input": ["hello", "world"]},
            future=fut,
        )
        client._pending = [req]
        await client._submit_batch()

        file_tuple = client._openai.files.create.call_args.kwargs["file"]
        content = file_tuple[1].getvalue().decode()
        line = json.loads(content.strip())

        assert line["body"]["input"] == ["hello", "world"]


class TestEmbeddingsResultParsing:
    async def test_embedding_result_parsed_correctly(
        self, client: BatchOpenAI
    ) -> None:
        """Embedding results should be parsed as CreateEmbeddingResponse."""
        ab = make_active_batch(
            ["emb-r1"],
            result_types={"emb-r1": CreateEmbeddingResponse},
        )
        body = make_batch_result_line("emb-r1", body=make_embedding_response())
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(body, {"X-Incomplete": "false"})
        )

        await client._fetch_partial_results(ab, "file-out")

        result = ab.requests["emb-r1"].future.result()
        assert isinstance(result, CreateEmbeddingResponse)
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        assert result.model == "text-embedding-3-small"

    async def test_embedding_error_sets_exception(
        self, client: BatchOpenAI
    ) -> None:
        """Error in an embedding result should set exception on the future."""
        ab = make_active_batch(
            ["emb-err"],
            result_types={"emb-err": CreateEmbeddingResponse},
        )
        body = make_batch_error_line("emb-err", "invalid input")
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(body, {"X-Incomplete": "false"})
        )

        await client._fetch_partial_results(ab, "file-out")

        assert ab.requests["emb-err"].future.done()
        with pytest.raises(Exception, match="emb-err failed"):
            ab.requests["emb-err"].future.result()


class TestMixedEmbeddingsAndChat:
    async def test_mixed_batch_parses_both_types(
        self, client: BatchOpenAI
    ) -> None:
        """A batch with both embeddings and chat completions should parse each correctly."""
        ab = make_active_batch(
            ["chat-1", "emb-1"],
            result_types={
                "chat-1": ChatCompletion,
                "emb-1": CreateEmbeddingResponse,
            },
        )
        body = "\n".join([
            make_batch_result_line("chat-1", "hello"),
            make_batch_result_line("emb-1", body=make_embedding_response()),
        ])
        client._http_client.get = AsyncMock(
            return_value=_httpx_response(body, {"X-Incomplete": "false"})
        )

        await client._fetch_partial_results(ab, "file-out")

        chat_result = ab.requests["chat-1"].future.result()
        assert isinstance(chat_result, ChatCompletion)
        assert chat_result.choices[0].message.content == "hello"

        embed_result = ab.requests["emb-1"].future.result()
        assert isinstance(embed_result, CreateEmbeddingResponse)
        assert embed_result.data[0].embedding == [0.1, 0.2, 0.3]
