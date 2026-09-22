"""Tests for Doubleword flex background submission and polling."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

import autobatcher.client as client_module
from autobatcher import AsyncOpenAI as AutobatcherAsyncOpenAI
from autobatcher.client import BatchOpenAI
from tests.conftest import make_response_api_result


TRANSLATION_CASES = json.loads(
    (Path(__file__).parents[2] / "fixtures/chat_responses_translation.json").read_text()
)


def _response(
    *,
    status: str = "completed",
    output_text: str = "Hello!",
    response_id: str = "resp-test123",
) -> Response:
    body = make_response_api_result(output_text=output_text)
    body["id"] = response_id
    body["status"] = status
    if status in {"queued", "in_progress"}:
        body["output"] = []
    return Response.model_validate(body)


class TestFlexDefaults:
    async def test_both_clients_default_to_flex_mode(self) -> None:
        """Regression: neither public client should batch text by default."""
        clients = [
            BatchOpenAI(api_key="sk-test"),
            AutobatcherAsyncOpenAI(api_key="sk-test"),
        ]
        try:
            assert [client._completion_window for client in clients] == [None, None]
        finally:
            for client in clients:
                await client.close()

    async def test_non_24h_chat_routes_around_batch_queue(
        self, client: BatchOpenAI
    ) -> None:
        """Regression: flex chat must not upload JSONL or create a batch."""
        expected = ChatCompletion.model_validate(
            {
                "id": "chatcmpl-flex",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        execute_flex = AsyncMock(return_value=expected)
        client._execute_flex = execute_flex  # type: ignore[attr-defined]
        client._completion_window = "1h"
        client._batch_size = 1
        client._submit_batch = AsyncMock(
            side_effect=AssertionError("flex chat entered the batch queue")
        )

        result = await client._enqueue_request(
            endpoint="/v1/chat/completions",
            result_type=ChatCompletion,
            params={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert result is expected
        execute_flex.assert_awaited_once()
        client.files.create.assert_not_awaited()


class TestFlexPolling:
    async def test_doubleword_response_extensions_use_extra_body(
        self, client: BatchOpenAI
    ) -> None:
        """Regression: Doubleword-only fields must pass the SDK method boundary."""
        completed = _response(status="completed")
        client._responses_api.create = AsyncMock(return_value=completed)

        await client._execute_flex(
            endpoint="/v1/responses",
            result_type=Response,
            params={
                "model": "test-model",
                "input": "hello",
                "frequency_penalty": 0.2,
                "reasoning_effort": "medium",
                "stop": ["DONE"],
            },
        )

        assert client._responses_api.create.await_args.kwargs["extra_body"] == {
            "frequency_penalty": 0.2,
            "reasoning_effort": "medium",
            "stop": ["DONE"],
        }

    async def test_responses_submit_in_background_and_poll_to_completion(
        self, client: BatchOpenAI
    ) -> None:
        """Regression: inference waits through GET polling, not an open POST."""
        queued = _response(status="queued")
        in_progress = _response(status="in_progress")
        completed = _response(status="completed", output_text="done")
        client._poll_interval_seconds = 0
        client._responses_api.create = AsyncMock(return_value=queued)
        client._responses_api.retrieve = AsyncMock(
            side_effect=[in_progress, completed]
        )

        result = await client._execute_flex(
            endpoint="/v1/responses",
            result_type=Response,
            params={"model": "test-model", "input": "hello", "stream": True},
        )

        assert result is completed
        assert client._responses_api.create.await_args.kwargs == {
            "model": "test-model",
            "input": "hello",
            "stream": False,
            "service_tier": "flex",
            "background": True,
        }
        assert [
            call.args for call in client._responses_api.retrieve.await_args_list
        ] == [("resp-test123",), ("resp-test123",)]
        client.files.create.assert_not_awaited()

    async def test_flex_forwards_per_request_transport_options(
        self, client: BatchOpenAI
    ) -> None:
        """Headers, query parameters, and timeout must reach create and polls."""
        queued = _response(status="queued", response_id="resp-options")
        completed = _response(status="completed", response_id="resp-options")
        client._poll_interval_seconds = 0
        client._responses_api.create = AsyncMock(return_value=queued)
        client._responses_api.retrieve = AsyncMock(return_value=completed)

        await client._execute_flex(
            endpoint="/v1/responses",
            result_type=Response,
            params={
                "model": "test-model",
                "input": "hello",
                "extra_headers": {"X-Route": "tenant-a"},
                "extra_query": {"region": "uk"},
                "timeout": 12.5,
            },
        )

        for awaited in (
            client._responses_api.create.await_args,
            client._responses_api.retrieve.await_args,
        ):
            assert awaited.kwargs["extra_headers"] == {"X-Route": "tenant-a"}
            assert awaited.kwargs["extra_query"] == {"region": "uk"}
            assert awaited.kwargs["timeout"] == 12.5

    async def test_batch_mode_rejects_unusable_transport_options(
        self, client: BatchOpenAI
    ) -> None:
        """Batch JSONL cannot silently discard per-request transport controls."""
        client._completion_window = "24h"

        with pytest.raises(ValueError, match="transport options.*Batch|Batch.*transport"):
            await asyncio.wait_for(
                client._enqueue_request(
                    endpoint="/v1/chat/completions",
                    result_type=ChatCompletion,
                    params={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "timeout": 2.0,
                    },
                ),
                timeout=0.02,
            )

    @pytest.mark.parametrize("status", ["failed", "cancelled"])
    async def test_terminal_non_completed_response_raises(
        self, client: BatchOpenAI, status: str
    ) -> None:
        """Regression: terminal failures must not be returned as successes."""
        terminal = _response(status=status, response_id="resp-terminal")
        client._responses_api.create = AsyncMock(return_value=terminal)

        with pytest.raises(
            RuntimeError, match=rf"resp-terminal.*{status}|{status}.*resp-terminal"
        ):
            await client._execute_flex(
                endpoint="/v1/responses",
                result_type=Response,
                params={"model": "test-model", "input": "hello"},
            )


class TestChatAdapters:
    @pytest.mark.parametrize(
        "case",
        TRANSLATION_CASES,
        ids=[case["name"] for case in TRANSLATION_CASES],
    )
    def test_chat_requests_match_shared_response_fixtures(self, case: dict) -> None:
        """Regression: Python and TypeScript must emit the same typed wire shape."""
        assert client_module._chat_params_to_response(case["chat_request"]) == case[
            "response_request"
        ]

    def test_chat_params_translate_to_responses_fields(self) -> None:
        """Regression: chat-only field names must not leak into Responses."""
        translated = client_module._chat_params_to_response(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 321,
                "response_format": {"type": "json_object"},
                "temperature": 0.4,
                "stream": True,
            }
        )

        assert translated == {
            "model": "test-model",
            "input": [{"role": "user", "content": "hello"}],
            "max_output_tokens": 321,
            "text": {"format": {"type": "json_object"}},
            "temperature": 0.4,
            "stream": False,
        }

    def test_chat_params_reject_multiple_choices(self) -> None:
        """Regression: one Responses result cannot masquerade as n chat choices."""
        with pytest.raises(ValueError, match="n=1"):
            client_module._chat_params_to_response(
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "n": 2,
                }
            )

    def test_chat_params_reject_custom_tools_outside_minimum_sdk_schema(self) -> None:
        """Regression: transforms must stay inside the declared OpenAI 2.x types."""
        with pytest.raises(ValueError, match="custom.*tool|tool.*custom"):
            client_module._chat_params_to_response(
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [
                        {
                            "type": "custom",
                            "custom": {"name": "shell", "format": {"type": "text"}},
                        }
                    ],
                }
            )

    def test_completed_response_converts_to_chat_completion(self) -> None:
        """Regression: flex chat callers still receive ChatCompletion models."""
        response = _response(status="completed", output_text="converted")

        result = client_module._response_to_chat_completion(response)

        assert isinstance(result, ChatCompletion)
        assert result.id == "resp-test123"
        assert result.object == "chat.completion"
        assert result.choices[0].message.content == "converted"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5

    def test_function_call_converts_to_chat_tool_call(self) -> None:
        """Regression: Responses function calls must survive chat adaptation."""
        body = make_response_api_result(output_text="")
        body["output"] = [
            {
                "type": "function_call",
                "id": "fc-test",
                "call_id": "call-test",
                "name": "get_weather",
                "arguments": '{"city":"London"}',
                "status": "completed",
            }
        ]
        response = Response.model_validate(body)

        result = client_module._response_to_chat_completion(response)

        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls is not None
        assert tool_calls[0].id == "call-test"
        assert tool_calls[0].function.name == "get_weather"
        assert tool_calls[0].function.arguments == '{"city":"London"}'
        assert result.choices[0].finish_reason == "tool_calls"

    def test_refusal_service_tier_and_usage_details_are_preserved(self) -> None:
        """Regression: typed response fields must not disappear during adaptation."""
        body = make_response_api_result(output_text="")
        body["service_tier"] = "priority"
        body["output"] = [
            {
                "type": "message",
                "id": "msg-refusal",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "refusal", "refusal": "I cannot help with that."}
                ],
            }
        ]
        body["usage"]["input_tokens_details"]["cached_tokens"] = 7
        body["usage"]["output_tokens_details"]["reasoning_tokens"] = 3
        response = Response.model_validate(body)

        result = client_module._response_to_chat_completion(response)

        message = result.choices[0].message
        assert message.content is None
        assert message.refusal == "I cannot help with that."
        assert result.service_tier == "priority"
        assert result.usage is not None
        assert result.usage.prompt_tokens_details is not None
        assert result.usage.prompt_tokens_details.cached_tokens == 7
        assert result.usage.completion_tokens_details is not None
        assert result.usage.completion_tokens_details.reasoning_tokens == 3

    def test_output_logprobs_are_preserved_in_chat_completion(self) -> None:
        """A requested Responses logprob payload must survive chat adaptation."""
        body = make_response_api_result(output_text="Hello")
        body["output"][0]["content"][0]["logprobs"] = [
            {
                "token": "Hello",
                "bytes": [72, 101, 108, 108, 111],
                "logprob": -0.25,
                "top_logprobs": [
                    {
                        "token": "Hi",
                        "bytes": [72, 105],
                        "logprob": -1.5,
                    }
                ],
            }
        ]
        response = Response.model_validate(body)

        result = client_module._response_to_chat_completion(response)

        logprobs = result.choices[0].logprobs
        assert logprobs is not None
        assert logprobs.content is not None
        assert logprobs.content[0].token == "Hello"
        assert logprobs.content[0].bytes == [72, 101, 108, 108, 111]
        assert logprobs.content[0].logprob == -0.25
        assert logprobs.content[0].top_logprobs[0].token == "Hi"
        assert logprobs.content[0].top_logprobs[0].logprob == -1.5

    def test_function_call_requires_a_real_call_id(self) -> None:
        """Regression: malformed calls must not become placeholder Chat tool IDs."""
        body = make_response_api_result(output_text="")
        body["output"] = [
            {
                "type": "function_call",
                "id": "fc-test",
                "call_id": "",
                "name": "get_weather",
                "arguments": "{}",
                "status": "completed",
            }
        ]
        response = Response.model_validate(body)

        with pytest.raises(ValueError, match="call_id"):
            client_module._response_to_chat_completion(response)


@pytest.mark.parametrize("reason, finish", [("max_output_tokens", "length"), ("content_filter", "content_filter")])
async def test_incomplete_preserves_output_and_usage(reason, finish):
    client = BatchOpenAI(api_key="test", poll_interval_seconds=0)
    terminal = _response(status="incomplete", output_text="partial")
    terminal = Response.model_validate({**terminal.model_dump(), "incomplete_details": {"reason": reason}})
    client._responses_api.create = AsyncMock(return_value=terminal)
    try:
        raw = await client.responses.create(model="test", input="hello")
        assert raw.status == "incomplete"
        assert raw.output[0].content[0].text == "partial"
        chat = await client.chat.completions.create(model="test", messages=[])
        assert chat.choices[0].message.content == "partial"
        assert chat.choices[0].finish_reason == finish
        assert chat.usage.total_tokens == 15
    finally:
        await client.close()

@pytest.mark.parametrize("status", [None, 404, 408, 409, 429, 503])
async def test_poll_recovers_without_resubmitting(status):
    import httpx
    import openai
    client = BatchOpenAI(api_key="test", poll_interval_seconds=0)
    request = httpx.Request("GET", "https://api.test/responses/resp-test123")
    error = (openai.APIConnectionError(request=request) if status is None else
             openai.APIStatusError("temporary", response=httpx.Response(status, request=request), body=None))
    client._responses_api.create = AsyncMock(return_value=_response(status="queued"))
    client._responses_api.retrieve = AsyncMock(side_effect=[error, _response(output_text="recovered")])
    try:
        result = await client.responses.create(model="test", input="hello")
        assert result.output[0].content[0].text == "recovered"
        assert client._responses_api.create.await_count == 1
        assert [call.args[0] for call in client._responses_api.retrieve.await_args_list] == ["resp-test123"] * 2
    finally:
        await client.close()

@pytest.mark.parametrize("status, attempts", [(503, 3), (401, 1)])
async def test_poll_failure_retains_id_and_cause(status, attempts):
    import httpx
    import openai
    client = BatchOpenAI(api_key="test", poll_interval_seconds=0, max_poll_retries=2)
    error = openai.APIStatusError("unavailable", response=httpx.Response(status, request=httpx.Request("GET", "https://api.test")), body=None)
    client._responses_api.create = AsyncMock(return_value=_response(status="queued"))
    client._responses_api.retrieve = AsyncMock(side_effect=error)
    try:
        with pytest.raises(RuntimeError) as caught:
            await client.responses.create(model="test", input="hello")
        assert caught.value.response_id == "resp-test123"
        assert caught.value.__cause__ is error
        assert client._responses_api.retrieve.await_count == attempts
        assert client._responses_api.create.await_count == 1
    finally:
        await client.close()

async def test_flex_http_limit_releases_slots_between_polls():
    client = BatchOpenAI(api_key="test", poll_interval_seconds=0, max_concurrent_requests=1)
    started = asyncio.Event()
    release = asyncio.Event()
    submissions = []
    active = 0
    peak = 0

    async def create(**params):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        submissions.append(params["input"])
        started.set()
        await release.wait()
        active -= 1
        return _response(status="queued", response_id=params["input"])

    async def retrieve(response_id, **options):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        # All submissions can proceed while previous jobs await completion.
        assert len(submissions) == 3
        return _response(response_id=response_id)

    client._responses_api.create = create
    client._responses_api.retrieve = retrieve
    tasks = [asyncio.create_task(client.responses.create(model="test", input=str(i))) for i in range(3)]
    try:
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.sleep(0)
        assert submissions == ["0"]
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), 1)
        assert [r.id for r in results] == ["0", "1", "2"]
        assert peak == 1
    finally:
        release.set()
        await client.close()

async def test_cancelling_a_waiting_submission_does_not_send_it():
    client = BatchOpenAI(api_key="test", poll_interval_seconds=0, max_concurrent_requests=1)
    started, release = asyncio.Event(), asyncio.Event()
    submissions = []
    async def create(**params):
        submissions.append(params["input"])
        started.set()
        await release.wait()
        return _response()
    client._responses_api.create = create
    first = asyncio.create_task(client.responses.create(model="test", input="first"))
    await asyncio.wait_for(started.wait(), 1)
    second = asyncio.create_task(client.responses.create(model="test", input="cancelled"))
    await asyncio.sleep(0)
    second.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await second
        release.set()
        await first
        await client.responses.create(model="test", input="last")
        assert submissions == ["first", "last"]
    finally:
        release.set()
        await client.close()

async def test_poll_backoff_resets_and_honors_retry_after(monkeypatch):
    import httpx
    import openai
    client = BatchOpenAI(api_key="test", poll_interval_seconds=5, max_poll_retries=1)
    delays = []
    async def sleep(delay):
        delays.append(delay)
    monkeypatch.setattr(client_module.asyncio, "sleep", sleep)
    request = httpx.Request("GET", "https://api.test")
    unavailable = openai.APIStatusError("busy", response=httpx.Response(503, request=request), body=None)
    limited = openai.APIStatusError("busy", response=httpx.Response(429, request=request, headers={"Retry-After": "20"}), body=None)
    client._responses_api.create = AsyncMock(return_value=_response(status="queued"))
    client._responses_api.retrieve = AsyncMock(side_effect=[unavailable, _response(status="queued"), limited, _response()])
    try:
        result = await client.responses.create(model="test", input="hello")
        assert result.status == "completed"
        assert len(delays) == 4
        for delay, minimum in zip(delays, [5, 10, 5, 20]):
            assert minimum <= delay <= minimum * 1.2
    finally:
        await client.close()

@pytest.mark.parametrize("option,value", [("max_concurrent_requests", 0), ("max_concurrent_requests", 1.5), ("max_poll_retries", -1), ("max_poll_retries", 1.5)])
def test_invalid_flex_limits_rejected(option, value):
    with pytest.raises(ValueError, match=option):
        BatchOpenAI(api_key="test", **{option: value})

async def test_cancel_preserves_transport_options():
    client = BatchOpenAI(api_key="test", poll_interval_seconds=60)
    submitted = asyncio.Event()
    async def create(**params):
        submitted.set()
        return _response(status="queued")
    client._responses_api.create = create
    client._responses_api.cancel = AsyncMock(return_value=_response(status="cancelled"))
    options = {"extra_headers": {"Authorization": "Bearer tenant"}, "extra_query": {"region": "uk"}, "timeout": 2}
    task = asyncio.create_task(client.responses.create(model="test", input="hello", **options))
    await submitted.wait()
    try:
        await client.close()
        with pytest.raises(asyncio.CancelledError):
            await task
        client._responses_api.cancel.assert_awaited_once_with("resp-test123", **options)
    finally:
        await client.close()
