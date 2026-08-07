import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type { Response } from "openai/resources/responses/responses";

import {
  AsyncOpenAI,
  BatchOpenAI,
  chatParamsToResponse,
  responseToChatCompletion,
} from "../src/client.ts";

interface TranslationCase {
  name: string;
  chat_request: Record<string, unknown>;
  response_request: Record<string, unknown>;
}

const translationCases = JSON.parse(
  readFileSync(
    new URL("../../fixtures/chat_responses_translation.json", import.meta.url),
    "utf8",
  ),
) as TranslationCase[];

function makeResponse(
  status: Response["status"] = "completed",
  outputText = "hello",
): Response {
  return {
    id: "resp-test",
    object: "response",
    created_at: 1_700_000_000,
    completed_at: status === "completed" ? 1_700_000_001 : null,
    status,
    model: "test-model",
    output_text: outputText,
    output:
      status === "completed"
        ? [
            {
              id: "msg-test",
              type: "message",
              role: "assistant",
              status: "completed",
              content: [
                {
                  type: "output_text",
                  text: outputText,
                  annotations: [],
                },
              ],
            },
          ]
        : [],
    error: null,
    incomplete_details: null,
    instructions: null,
    metadata: null,
    parallel_tool_calls: true,
    temperature: null,
    tool_choice: "auto",
    tools: [],
    top_p: null,
    usage: {
      input_tokens: 10,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens: 5,
      output_tokens_details: { reasoning_tokens: 0 },
      total_tokens: 15,
    },
  } as Response;
}

function makeClient(options: ConstructorParameters<typeof BatchOpenAI>[0] = {}) {
  return new BatchOpenAI({
    apiKey: "sk-test",
    baseURL: "https://api.test/v1",
    pollIntervalSeconds: 0,
    ...options,
  });
}

test("both public clients default text generation to flex", async () => {
  const batch = makeClient();
  const asyncClient = new AsyncOpenAI({
    apiKey: "sk-test",
    baseURL: "https://api.test/v1",
  });
  try {
    assert.equal((batch as any)._completionWindow, undefined);
    assert.equal((asyncClient as any)._completionWindow, undefined);
  } finally {
    await batch.close();
    await asyncClient.close();
  }
});

test("chat params translate to Responses API fields", () => {
  assert.deepEqual(
    chatParamsToResponse({
      model: "test-model",
      messages: [{ role: "user", content: "hello" }],
      max_completion_tokens: 321,
      response_format: { type: "json_object" },
      temperature: 0.4,
      stream: true,
    }),
    {
      model: "test-model",
      input: [{ role: "user", content: "hello" }],
      max_output_tokens: 321,
      text: { format: { type: "json_object" } },
      temperature: 0.4,
      stream: false,
    },
  );
});

for (const translationCase of translationCases) {
  test(`shared translation fixture: ${translationCase.name}`, () => {
    assert.deepEqual(
      chatParamsToResponse(translationCase.chat_request),
      translationCase.response_request,
    );
  });
}

test("chat params reject multiple choices", () => {
  assert.throws(
    () =>
      chatParamsToResponse({
        model: "test-model",
        messages: [{ role: "user", content: "hello" }],
        n: 2,
      }),
    /n=1/,
  );
});

test("chat params reject custom tools outside the minimum SDK schema", () => {
  assert.throws(
    () =>
      chatParamsToResponse({
        model: "test-model",
        messages: [{ role: "user", content: "hello" }],
        tools: [
          {
            type: "custom",
            custom: { name: "shell", format: { type: "text" } },
          },
        ],
      }),
    /custom.*tool|tool.*custom/,
  );
});

test("completed response converts to a chat completion", () => {
  const completion = responseToChatCompletion(makeResponse());

  assert.equal(completion.id, "resp-test");
  assert.equal(completion.object, "chat.completion");
  assert.equal(completion.choices[0]?.message.content, "hello");
  assert.equal(completion.choices[0]?.finish_reason, "stop");
  assert.equal(completion.usage?.prompt_tokens, 10);
  assert.equal(completion.usage?.completion_tokens, 5);
});

test("function calls convert to chat tool calls", () => {
  const response = makeResponse();
  response.output = [
    {
      type: "function_call",
      id: "fc-test",
      call_id: "call-test",
      name: "get_weather",
      arguments: '{"city":"London"}',
      status: "completed",
    },
  ];

  const completion = responseToChatCompletion(response);

  assert.equal(completion.choices[0]?.message.tool_calls?.[0]?.id, "call-test");
  assert.equal(
    completion.choices[0]?.message.tool_calls?.[0]?.function.name,
    "get_weather",
  );
  assert.equal(completion.choices[0]?.finish_reason, "tool_calls");
});

test("refusal, service tier, and usage details survive chat adaptation", () => {
  const response = makeResponse();
  response.service_tier = "priority";
  response.output = [
    {
      id: "msg-refusal",
      type: "message",
      role: "assistant",
      status: "completed",
      content: [
        { type: "refusal", refusal: "I cannot help with that." },
      ],
    },
  ];
  response.usage = {
    input_tokens: 10,
    input_tokens_details: { cached_tokens: 7 },
    output_tokens: 5,
    output_tokens_details: { reasoning_tokens: 3 },
    total_tokens: 15,
  };

  const completion = responseToChatCompletion(response);

  assert.equal(completion.choices[0]?.message.content, null);
  assert.equal(
    completion.choices[0]?.message.refusal,
    "I cannot help with that.",
  );
  assert.equal(completion.service_tier, "priority");
  assert.equal(completion.usage?.prompt_tokens_details?.cached_tokens, 7);
  assert.equal(completion.usage?.completion_tokens_details?.reasoning_tokens, 3);
});

test("function calls require a real call ID", () => {
  const response = makeResponse();
  response.output = [
    {
      type: "function_call",
      id: "fc-test",
      call_id: "",
      name: "get_weather",
      arguments: "{}",
      status: "completed",
    },
  ];

  assert.throws(() => responseToChatCompletion(response), /call_id/);
});

test("flex responses submit in background and poll to completion", async () => {
  const client = makeClient();
  const createBodies: Array<Record<string, unknown>> = [];
  const retrieveIDs: string[] = [];
  const queued = makeResponse("queued", "");
  const inProgress = makeResponse("in_progress", "");
  const completed = makeResponse("completed", "done");
  const results = [inProgress, completed];
  (client as any)._responses = {
    create: async (body: Record<string, unknown>) => {
      createBodies.push(body);
      return queued;
    },
    retrieve: async (id: string) => {
      retrieveIDs.push(id);
      return results.shift();
    },
  };

  try {
    const result = await (client as any)._executeFlex(
      "/v1/responses",
      { model: "test-model", input: "hello", stream: true },
    );

    assert.equal(result, completed);
    assert.deepEqual(createBodies, [
      {
        model: "test-model",
        input: "hello",
        stream: false,
        service_tier: "flex",
        background: true,
      },
    ]);
    assert.deepEqual(retrieveIDs, ["resp-test", "resp-test"]);
  } finally {
    await client.close();
  }
});

for (const status of ["failed", "cancelled", "incomplete"] as const) {
  test(`terminal ${status} flex response rejects`, async () => {
    const client = makeClient();
    const terminal = makeResponse(status);
    (client as any)._responses = {
      create: async () => terminal,
      retrieve: async () => terminal,
    };
    try {
      await assert.rejects(
        (client as any)._executeFlex("/v1/responses", {
          model: "test-model",
          input: "hello",
        }),
        new RegExp(`resp-test.*${status}|${status}.*resp-test`),
      );
    } finally {
      await client.close();
    }
  });
}

test("embeddings always route through a 24h batch", async () => {
  const client = makeClient();
  const calls: Array<[string, Record<string, unknown>]> = [];
  (client as any)._enqueueBatch = async (
    endpoint: string,
    body: Record<string, unknown>,
  ) => {
    calls.push([endpoint, body]);
    return { object: "list", data: [], model: "test", usage: {} };
  };
  (client as any)._executeFlex = async () => {
    throw new Error("embeddings entered flex inference");
  };

  try {
    await client._enqueue("/v1/embeddings", {
      model: "embedding-model",
      input: "hello",
    });
    assert.deepEqual(calls, [
      ["/v1/embeddings", { model: "embedding-model", input: "hello" }],
    ]);
  } finally {
    await client.close();
  }
});

test("non-24h chat routes through flex without entering the batch queue", async () => {
  const client = makeClient({ completionWindow: "1h" });
  const calls: string[] = [];
  (client as any)._executeFlex = async (endpoint: string) => {
    calls.push(endpoint);
    return { id: "chatcmpl-flex" };
  };
  (client as any)._enqueueBatch = async () => {
    throw new Error("flex chat entered the batch queue");
  };

  try {
    const result = await client._enqueue("/v1/chat/completions", {
      model: "test-model",
      messages: [{ role: "user", content: "hello" }],
    });
    assert.deepEqual(result, { id: "chatcmpl-flex" });
    assert.deepEqual(calls, ["/v1/chat/completions"]);
  } finally {
    await client.close();
  }
});

test("explicit 24h text routes through the batch path", async () => {
  const client = makeClient({ completionWindow: "24h" });
  const calls: string[] = [];
  (client as any)._enqueueBatch = async (endpoint: string) => {
    calls.push(endpoint);
    return { id: "chatcmpl-batch" };
  };
  (client as any)._executeFlex = async () => {
    throw new Error("24h text entered flex inference");
  };

  try {
    await client._enqueue("/v1/chat/completions", {
      model: "test-model",
      messages: [{ role: "user", content: "hello" }],
    });
    assert.deepEqual(calls, ["/v1/chat/completions"]);
  } finally {
    await client.close();
  }
});
