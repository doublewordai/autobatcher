"""Tests for Doubleword flex background submission and polling."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

import autobatcher.client as client_module
from autobatcher import AsyncOpenAI as AutobatcherAsyncOpenAI
from autobatcher.client import BatchOpenAI
from tests.conftest import make_response_api_result


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

    @pytest.mark.parametrize("status", ["failed", "cancelled", "incomplete"])
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
