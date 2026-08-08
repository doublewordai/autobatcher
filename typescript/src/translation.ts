/** Typed Chat Completions ↔ Responses adapters using public OpenAI SDK types. */

import type {
  ChatCompletion,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions";
import type {
  EasyInputMessage,
  Response as OpenAIResponse,
  ResponseCreateParamsNonStreaming,
  ResponseInputItem,
  ResponseInputMessageContentList,
} from "openai/resources/responses/responses";

type ResponseRequest = ResponseCreateParamsNonStreaming &
  Record<string, unknown>;
type ChatFunctionToolCall = Extract<
  NonNullable<
    ChatCompletion["choices"][number]["message"]["tool_calls"]
  >[number],
  { type: "function" }
>;
type ChatChoiceLogprobs = NonNullable<
  ChatCompletion["choices"][number]["logprobs"]
>;

/** Strip transport options and merge extra_body into the API payload. */
export function cleanParams(
  obj: Record<string, unknown>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const extraBody = obj.extra_body;
  for (const [key, value] of Object.entries(obj)) {
    if (
      value !== undefined &&
      key !== "extra_body" &&
      key !== "extra_headers" &&
      key !== "extra_query" &&
      key !== "timeout"
    ) {
      out[key] = value;
    }
  }
  if (
    extraBody &&
    typeof extraBody === "object" &&
    !Array.isArray(extraBody)
  ) {
    Object.assign(out, extraBody);
  }
  return out;
}

function record(value: unknown, parameter: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${parameter} is invalid`);
  }
  return value as Record<string, unknown>;
}

function textContent(content: unknown, parameter: string): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) {
    throw new Error(`${parameter} content must be text`);
  }
  return content
    .map((rawPart) => {
      const part = record(rawPart, `${parameter} content part`);
      if (part.type !== "text" || typeof part.text !== "string") {
        throw new Error(`${parameter} supports text content only`);
      }
      return part.text;
    })
    .join("");
}

function inputContent(
  content: unknown,
  parameter: string,
): EasyInputMessage["content"] {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) {
    throw new Error(`${parameter} content must be text or content parts`);
  }

  const converted: ResponseInputMessageContentList = [];
  for (const rawPart of content) {
    const part = record(rawPart, `${parameter} content part`);
    switch (part.type) {
      case "text": {
        if (typeof part.text !== "string") {
          throw new Error(`${parameter} text content must be a string`);
        }
        converted.push({ type: "input_text", text: part.text });
        break;
      }
      case "image_url": {
        const image = record(part.image_url, `${parameter} image_url`);
        if (typeof image.url !== "string") {
          throw new Error(`${parameter} image_url requires a URL`);
        }
        const detail = image.detail ?? "auto";
        if (detail !== "low" && detail !== "high" && detail !== "auto") {
          throw new Error(`${parameter} image_url detail is invalid`);
        }
        converted.push({
          type: "input_image",
          image_url: image.url,
          detail,
        });
        break;
      }
      case "file": {
        const file = record(part.file, `${parameter} file content`);
        if (
          file.file_data == null &&
          file.file_id == null
        ) {
          throw new Error(
            `${parameter} file content requires file data or an ID`,
          );
        }
        converted.push({
          type: "input_file",
          ...(file.file_data != null ? { file_data: String(file.file_data) } : {}),
          ...(file.file_id != null ? { file_id: String(file.file_id) } : {}),
          ...(file.filename != null ? { filename: String(file.filename) } : {}),
        });
        break;
      }
      case "input_audio":
        throw new Error(
          "input_audio is not supported by flex Responses translation",
        );
      default:
        throw new Error(
          `${parameter} contains unsupported content type ${String(part.type)}`,
        );
    }
  }
  return converted;
}

function assistantItems(
  message: Extract<ChatCompletionMessageParam, { role: "assistant" }>,
): ResponseInputItem[] {
  if (message.audio != null) {
    throw new Error(
      "assistant audio is not supported by flex Responses translation",
    );
  }
  if (message.function_call != null) {
    throw new Error(
      "deprecated function_call is not supported; use tool_calls",
    );
  }
  if (message.name != null) {
    throw new Error(
      "assistant name is not supported by flex Responses translation",
    );
  }

  const items: ResponseInputItem[] = [];
  if (message.content != null) {
    let content: string;
    if (typeof message.content === "string") {
      content = message.content;
    } else {
      content = Array.from(message.content)
        .map((part) => {
          if (part.type === "text") return part.text;
          if (part.type === "refusal") return part.refusal;
          throw new Error(
            "assistant content supports text and refusal parts only",
          );
        })
        .join("");
    }
    items.push({ role: "assistant", content } satisfies ResponseInputItem);
  }

  for (const call of message.tool_calls ?? []) {
    if (call.type !== "function") {
      throw new Error(
        `tool_calls contains unsupported type ${String(call.type)}`,
      );
    }
    if (!call.id) {
      throw new Error("tool_calls function call requires a non-empty id");
    }
    if (!call.function.name) {
      throw new Error("tool_calls function call requires a non-empty name");
    }
    items.push({
      type: "function_call",
      call_id: call.id,
      name: call.function.name,
      arguments: call.function.arguments,
    } satisfies ResponseInputItem);
  }
  return items;
}

function messageToResponseItems(
  message: ChatCompletionMessageParam,
): ResponseInputItem[] {
  switch (message.role) {
    case "assistant":
      return assistantItems(message);
    case "tool": {
      if (!message.tool_call_id) {
        throw new Error("tool message requires a non-empty tool_call_id");
      }
      return [
        {
          type: "function_call_output",
          call_id: message.tool_call_id,
          output: textContent(message.content, "tool message"),
        } satisfies ResponseInputItem,
      ];
    }
    case "function":
      throw new Error(
        "deprecated function messages are not supported; use tool messages",
      );
    case "user":
    case "system":
    case "developer": {
      if (message.name != null) {
        throw new Error(
          `${message.role} message name is not supported by flex Responses translation`,
        );
      }
      return [
        {
          role: message.role,
          content: inputContent(message.content, `${message.role} message`),
        } satisfies ResponseInputItem,
      ];
    }
  }
}

function messagesToResponseInput(
  messages: Iterable<ChatCompletionMessageParam>,
): ResponseInputItem[] {
  const items: ResponseInputItem[] = [];
  for (const message of messages) {
    items.push(...messageToResponseItems(message));
  }
  return items;
}

function convertTools(
  tools: NonNullable<ChatCompletionCreateParamsNonStreaming["tools"]>,
): NonNullable<ResponseCreateParamsNonStreaming["tools"]> {
  return Array.from(tools).map((tool) => {
    if (tool.type === "function") {
      return {
        type: "function",
        name: tool.function.name,
        parameters: tool.function.parameters ?? null,
        strict: tool.function.strict ?? false,
        ...(tool.function.description != null
          ? { description: tool.function.description }
          : {}),
      } satisfies NonNullable<ResponseCreateParamsNonStreaming["tools"]>[number];
    }
    if (tool.type === "custom") {
      throw new Error(
        "custom tools are not supported by the minimum OpenAI SDK schema",
      );
    }
    throw new Error(`tools contains unsupported type ${String((tool as { type?: unknown }).type)}`);
  });
}

function convertToolChoice(
  choice: NonNullable<ChatCompletionCreateParamsNonStreaming["tool_choice"]>,
): NonNullable<ResponseCreateParamsNonStreaming["tool_choice"]> {
  if (typeof choice === "string") return choice;
  if (choice.type === "function") {
    return { type: "function", name: choice.function.name };
  }
  if (choice.type === "custom") {
    throw new Error(
      "custom tool_choice is not supported by the minimum OpenAI SDK schema",
    );
  }
  throw new Error(`tool_choice contains unsupported type ${choice.type}`);
}

function convertResponseFormat(
  responseFormat: NonNullable<
    ChatCompletionCreateParamsNonStreaming["response_format"]
  >,
): NonNullable<
  NonNullable<ResponseCreateParamsNonStreaming["text"]>["format"]
> {
  if (responseFormat.type === "json_schema") {
    const { name, description, schema, strict } = responseFormat.json_schema;
    if (schema == null) {
      throw new Error("response_format json_schema requires schema details");
    }
    return {
      type: "json_schema",
      name,
      schema,
      ...(description != null ? { description } : {}),
      ...(strict != null ? { strict } : {}),
    };
  }
  return responseFormat;
}

function rejectUnsupported(
  params: ChatCompletionCreateParamsNonStreaming & Record<string, unknown>,
): void {
  const unsupported: Array<[keyof typeof params, string]> = [
    ["audio", "audio output"],
    ["function_call", "deprecated function_call"],
    ["functions", "deprecated functions"],
    ["prediction", "prediction"],
    ["web_search_options", "web_search_options"],
  ];
  for (const [key, label] of unsupported) {
    if (params[key] != null) {
      throw new Error(
        `${label} is not supported by flex Responses translation`,
      );
    }
  }
  if (
    params.modalities != null &&
    (params.modalities.length !== 1 || params.modalities[0] !== "text")
  ) {
    throw new Error(
      "modalities supports text output only in flex Responses translation",
    );
  }
}

export function chatParamsToResponse(
  params: ChatCompletionCreateParamsNonStreaming | Record<string, unknown>,
): ResponseRequest {
  const source = cleanParams(params as Record<string, unknown>) as unknown as
    ChatCompletionCreateParamsNonStreaming & Record<string, unknown>;
  rejectUnsupported(source);

  if ((source.n ?? 1) !== 1) {
    throw new Error("Flex inference supports only n=1 for chat completions");
  }
  if (source.messages == null) {
    throw new Error("Chat completions require messages");
  }

  const result: Record<string, unknown> = {
    model: source.model,
    input: messagesToResponseInput(source.messages),
    stream: false,
  };

  const sharedFields = [
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
  ] as const;
  for (const key of sharedFields) {
    if (source[key] != null) result[key] = source[key];
  }

  const maxOutputTokens =
    source.max_completion_tokens ?? source.max_tokens;
  if (maxOutputTokens != null) result.max_output_tokens = maxOutputTokens;

  if (source.response_format != null || source.verbosity != null) {
    result.text = {
      ...(source.response_format != null
        ? { format: convertResponseFormat(source.response_format) }
        : {}),
      ...(source.verbosity != null ? { verbosity: source.verbosity } : {}),
    };
  }
  if (source.reasoning_effort != null) {
    result.reasoning = { effort: source.reasoning_effort };
  }
  if (source.logprobs === true) {
    result.include = ["message.output_text.logprobs"];
  }
  if (source.tools != null) result.tools = convertTools(source.tools);
  if (source.tool_choice != null) {
    result.tool_choice = convertToolChoice(source.tool_choice);
  }

  for (const key of [
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "logit_bias",
    "stop",
  ] as const) {
    if (source[key] != null) result[key] = source[key];
  }

  return result as ResponseRequest;
}

export function responseToChatCompletion(
  response: OpenAIResponse,
): ChatCompletion {
  const textParts: string[] = [];
  const refusals: string[] = [];
  const toolCalls: ChatFunctionToolCall[] = [];
  const contentLogprobs: NonNullable<ChatChoiceLogprobs["content"]> = [];

  for (const item of response.output) {
    switch (item.type) {
      case "message":
        for (const part of item.content) {
          if (part.type === "output_text") {
            textParts.push(part.text);
            for (const logprob of part.logprobs ?? []) {
              contentLogprobs.push({
                token: logprob.token,
                bytes: logprob.bytes,
                logprob: logprob.logprob,
                top_logprobs: logprob.top_logprobs.map((top) => ({
                  token: top.token,
                  bytes: top.bytes,
                  logprob: top.logprob,
                })),
              });
            }
          } else if (part.type === "refusal") {
            refusals.push(part.refusal);
          }
        }
        break;
      case "function_call":
        if (!item.call_id) {
          throw new Error(
            "Responses function_call requires a non-empty call_id",
          );
        }
        if (!item.name) {
          throw new Error(
            "Responses function_call requires a non-empty name",
          );
        }
        toolCalls.push({
          id: item.call_id,
          type: "function",
          function: { name: item.name, arguments: item.arguments },
        });
        break;
      default:
        break;
    }
  }

  const message: ChatCompletion["choices"][number]["message"] = {
    role: "assistant",
    content: textParts.length > 0 ? textParts.join("") : null,
    refusal: refusals.length > 0 ? refusals.join("") : null,
    ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
  };

  return {
    id: response.id,
    object: "chat.completion",
    created: response.created_at,
    model: response.model,
    choices: [
      {
        index: 0,
        message,
        logprobs:
          contentLogprobs.length > 0
            ? { content: contentLogprobs, refusal: null }
            : null,
        finish_reason: toolCalls.length > 0 ? "tool_calls" : "stop",
      },
    ],
    usage: response.usage
      ? {
          prompt_tokens: response.usage.input_tokens,
          completion_tokens: response.usage.output_tokens,
          total_tokens: response.usage.total_tokens,
          prompt_tokens_details: {
            cached_tokens: response.usage.input_tokens_details.cached_tokens,
          },
          completion_tokens_details: {
            reasoning_tokens:
              response.usage.output_tokens_details.reasoning_tokens,
          },
        }
      : undefined,
    service_tier: response.service_tier,
  };
}
