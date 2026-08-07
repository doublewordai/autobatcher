"""Typed Chat Completions ↔ Responses adapters.

Request types in openai-python are generated ``TypedDict`` definitions, while
response types are generated Pydantic models.  Keeping the conversion here makes
that public SDK boundary explicit and leaves transport concerns in ``client.py``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, cast

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming,
)
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    Response,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputFileParam,
    ResponseInputImageParam,
    ResponseInputItemParam,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ToolChoiceFunctionParam,
)
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
    ToolChoice as ResponseToolChoice,
)
from openai.types.responses.response_input_param import FunctionCallOutput


def _text_content(content: object, *, parameter: str) -> str:
    """Normalize Chat text or text-part content into one string."""
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable) and not isinstance(content, (bytes, dict)):
        text: list[str] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                raise ValueError(f"{parameter} supports text content only")
            value = part.get("text")
            if not isinstance(value, str):
                raise ValueError(f"{parameter} text content must be a string")
            text.append(value)
        return "".join(text)
    raise ValueError(f"{parameter} content must be text")


def _input_content(
    content: object, *, parameter: str
) -> str | ResponseInputMessageContentListParam:
    """Convert Chat message content into official Responses input content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, Iterable) or isinstance(content, (bytes, dict)):
        raise ValueError(f"{parameter} content must be text or content parts")

    converted: ResponseInputMessageContentListParam = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError(f"{parameter} contains an invalid content part")
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError(f"{parameter} text content must be a string")
            text_part: ResponseInputTextParam = {"type": "input_text", "text": text}
            converted.append(text_part)
        elif part_type == "image_url":
            image = part.get("image_url")
            if not isinstance(image, dict) or not isinstance(image.get("url"), str):
                raise ValueError(f"{parameter} image_url requires a URL")
            image_part: ResponseInputImageParam = {
                "type": "input_image",
                "image_url": image["url"],
                "detail": image.get("detail", "auto"),
            }
            converted.append(image_part)
        elif part_type == "file":
            file = part.get("file")
            if not isinstance(file, dict):
                raise ValueError(f"{parameter} file content is invalid")
            converted_file: ResponseInputFileParam = {"type": "input_file"}
            for key in ("file_data", "file_id", "filename"):
                value = file.get(key)
                if value is not None:
                    converted_file[key] = value
            if len(converted_file) == 1:
                raise ValueError(f"{parameter} file content requires file data or an ID")
            converted.append(converted_file)
        elif part_type == "input_audio":
            raise ValueError("input_audio is not supported by flex Responses translation")
        else:
            raise ValueError(f"{parameter} contains unsupported content type {part_type!r}")
    return converted


def _assistant_items(message: dict[str, Any]) -> list[ResponseInputItemParam]:
    """Convert one assistant history message and its tool calls."""
    if message.get("audio") is not None:
        raise ValueError("assistant audio is not supported by flex Responses translation")
    if message.get("function_call") is not None:
        raise ValueError("deprecated function_call is not supported; use tool_calls")
    if message.get("name") is not None:
        raise ValueError("assistant name is not supported by flex Responses translation")

    items: list[ResponseInputItemParam] = []
    content = message.get("content")
    if content is not None:
        if isinstance(content, str):
            assistant_text = content
        elif isinstance(content, Iterable) and not isinstance(content, (bytes, dict)):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in {"text", "refusal"}:
                    raise ValueError("assistant content supports text and refusal parts only")
                value = part.get("text") if part.get("type") == "text" else part.get("refusal")
                if not isinstance(value, str):
                    raise ValueError("assistant content part must contain text")
                text_parts.append(value)
            assistant_text = "".join(text_parts)
        else:
            raise ValueError("assistant content must be text")

        easy_message: EasyInputMessageParam = {
            "role": "assistant",
            "content": assistant_text,
        }
        items.append(cast(ResponseInputItemParam, easy_message))

    for raw_call in message.get("tool_calls") or []:
        if not isinstance(raw_call, dict):
            raise ValueError("tool_calls contains an invalid call")
        call_type = raw_call.get("type")
        if call_type != "function":
            raise ValueError(f"tool_calls contains unsupported type {call_type!r}")
        call_id = raw_call.get("id")
        function = raw_call.get("function")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("tool_calls function call requires a non-empty id")
        if not isinstance(function, dict):
            raise ValueError("tool_calls function call requires function details")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_calls function call requires a non-empty name")
        if not isinstance(arguments, str):
            raise ValueError("tool_calls function call arguments must be a string")
        function_call: ResponseFunctionToolCallParam = {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        }
        items.append(cast(ResponseInputItemParam, function_call))
    return items


def _message_to_response_items(
    message: ChatCompletionMessageParam,
) -> list[ResponseInputItemParam]:
    raw = cast(dict[str, Any], message)
    role = raw.get("role")

    if role == "assistant":
        return _assistant_items(raw)
    if role == "tool":
        call_id = raw.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("tool message requires a non-empty tool_call_id")
        output = _text_content(raw.get("content"), parameter="tool message")
        function_output: FunctionCallOutput = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }
        return [cast(ResponseInputItemParam, function_output)]
    if role == "function":
        raise ValueError("deprecated function messages are not supported; use tool messages")
    if role not in {"user", "system", "developer"}:
        raise ValueError(f"unsupported chat message role {role!r}")
    if raw.get("name") is not None:
        raise ValueError(f"{role} message name is not supported by flex Responses translation")

    content = _input_content(raw.get("content"), parameter=f"{role} message")
    easy_message: EasyInputMessageParam = {
        "role": cast(Literal["user", "system", "developer"], role),
        "content": content,
    }
    return [cast(ResponseInputItemParam, easy_message)]


def _messages_to_response_input(
    messages: Iterable[ChatCompletionMessageParam],
) -> ResponseInputParam:
    items: ResponseInputParam = []
    for message in messages:
        items.extend(_message_to_response_items(message))
    return items


def _convert_tools(tools: object) -> list[FunctionToolParam]:
    if not isinstance(tools, Iterable) or isinstance(tools, (str, bytes, dict)):
        raise ValueError("tools must be a list")
    converted: list[FunctionToolParam] = []
    for raw_tool in tools:
        if not isinstance(raw_tool, dict):
            raise ValueError("tools contains an invalid definition")
        tool_type = raw_tool.get("type")
        if tool_type == "function":
            function = raw_tool.get("function")
            if not isinstance(function, dict):
                raise ValueError("function tool requires function details")
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("function tool requires a non-empty name")
            parameters = function.get("parameters")
            if parameters is not None and not isinstance(parameters, dict):
                raise ValueError("function tool parameters must be an object")
            strict = function.get("strict", False)
            if strict is not None and not isinstance(strict, bool):
                raise ValueError("function tool strict must be a boolean")
            tool: FunctionToolParam = {
                "type": "function",
                "name": name,
                "parameters": parameters,
                "strict": strict,
            }
            if function.get("description") is not None:
                tool["description"] = function["description"]
            converted.append(tool)
        elif tool_type == "custom":
            raise ValueError(
                "custom tools are not supported by the minimum OpenAI SDK schema"
            )
        else:
            raise ValueError(f"tools contains unsupported type {tool_type!r}")
    return converted


def _convert_tool_choice(choice: object) -> ResponseToolChoice:
    if isinstance(choice, str):
        if choice not in {"none", "auto", "required"}:
            raise ValueError(f"tool_choice contains unsupported mode {choice!r}")
        return choice
    if not isinstance(choice, dict):
        raise ValueError("tool_choice is invalid")
    choice_type = choice.get("type")
    if choice_type == "function":
        function = choice.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("function tool_choice requires a name")
        converted: ToolChoiceFunctionParam = {
            "type": "function",
            "name": function["name"],
        }
        return converted
    if choice_type == "custom":
        raise ValueError(
            "custom tool_choice is not supported by the minimum OpenAI SDK schema"
        )
    raise ValueError(f"tool_choice contains unsupported type {choice_type!r}")


def _convert_response_format(response_format: object) -> dict[str, Any]:
    if not isinstance(response_format, dict):
        raise ValueError("response_format is invalid")
    format_type = response_format.get("type")
    if format_type in {"text", "json_object"}:
        return dict(response_format)
    if format_type == "json_schema":
        schema = response_format.get("json_schema")
        if not isinstance(schema, dict):
            raise ValueError("response_format json_schema requires schema details")
        return {"type": "json_schema", **schema}
    raise ValueError(f"response_format contains unsupported type {format_type!r}")


def _reject_unsupported(params: dict[str, Any]) -> None:
    unsupported = {
        "audio": "audio output",
        "function_call": "deprecated function_call",
        "functions": "deprecated functions",
        "prediction": "prediction",
        "web_search_options": "web_search_options",
    }
    for key, label in unsupported.items():
        if params.get(key) is not None:
            raise ValueError(f"{label} is not supported by flex Responses translation")

    modalities = params.get("modalities")
    if modalities is not None and list(modalities) != ["text"]:
        raise ValueError("modalities supports text output only in flex Responses translation")


def chat_params_to_response(
    params: CompletionCreateParamsNonStreaming,
) -> ResponseCreateParamsNonStreaming:
    """Translate official Chat create parameters into Responses parameters."""
    source = dict(cast(dict[str, Any], params))
    _reject_unsupported(source)

    n = source.get("n", 1)
    if n != 1:
        raise ValueError("Flex inference supports only n=1 for chat completions")
    messages = source.get("messages")
    if messages is None:
        raise ValueError("Chat completions require messages")

    result: dict[str, Any] = {
        "model": source.get("model"),
        "input": _messages_to_response_input(messages),
        "stream": False,
    }

    shared_fields = (
        "metadata",
        "moderation",
        "parallel_tool_calls",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "safety_identifier",
        "store",
        "temperature",
        "top_logprobs",
        "top_p",
        "user",
    )
    for key in shared_fields:
        if source.get(key) is not None:
            result[key] = source[key]

    max_tokens = source.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = source.get("max_tokens")
    if max_tokens is not None:
        result["max_output_tokens"] = max_tokens

    response_format = source.get("response_format")
    verbosity = source.get("verbosity")
    if response_format is not None or verbosity is not None:
        text: dict[str, Any] = {}
        if response_format is not None:
            text["format"] = _convert_response_format(response_format)
        if verbosity is not None:
            text["verbosity"] = verbosity
        result["text"] = text

    reasoning_effort = source.get("reasoning_effort")
    if reasoning_effort is not None:
        result["reasoning"] = {"effort": reasoning_effort}

    if source.get("logprobs") is True:
        result["include"] = ["message.output_text.logprobs"]
    if source.get("tools") is not None:
        result["tools"] = _convert_tools(source["tools"])
    if source.get("tool_choice") is not None:
        result["tool_choice"] = _convert_tool_choice(source["tool_choice"])

    # Doubleword accepts these Chat-compatible extensions on /responses even
    # though openai-python does not currently declare them on Responses.create.
    for key in ("frequency_penalty", "presence_penalty", "seed", "logit_bias", "stop"):
        if source.get(key) is not None:
            result[key] = source[key]

    return cast(ResponseCreateParamsNonStreaming, result)


def response_to_chat_completion(response: Response) -> ChatCompletion:
    """Adapt a completed official Responses object into a ChatCompletion."""
    text_parts: list[str] = []
    refusals: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for item in response.output:
        if isinstance(item, ResponseOutputMessage):
            for part in item.content:
                if isinstance(part, ResponseOutputText):
                    text_parts.append(part.text)
                elif isinstance(part, ResponseOutputRefusal):
                    refusals.append(part.refusal)
        elif isinstance(item, ResponseFunctionToolCall):
            if not item.call_id:
                raise ValueError("Responses function_call requires a non-empty call_id")
            if not item.name:
                raise ValueError("Responses function_call requires a non-empty name")
            tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": item.arguments,
                    },
                }
            )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else None,
        "refusal": "".join(refusals) if refusals else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage: dict[str, Any] | None = None
    if response.usage is not None:
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
            "prompt_tokens_details": {
                "cached_tokens": response.usage.input_tokens_details.cached_tokens,
            },
            "completion_tokens_details": {
                "reasoning_tokens": response.usage.output_tokens_details.reasoning_tokens,
            },
        }

    return ChatCompletion.model_validate(
        {
            "id": response.id,
            "object": "chat.completion",
            "created": int(response.created_at),
            "model": response.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": None,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": usage,
            "service_tier": response.service_tier,
        }
    )
