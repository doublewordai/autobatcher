import assert from "node:assert/strict";
import { once } from "node:events";
import { readFileSync } from "node:fs";
import { request as httpRequest } from "node:http";
import test from "node:test";

import type { Response } from "openai/resources/responses/responses";

import {
  AsyncOpenAI,
  BatchOpenAI,
  chatParamsToResponse,
  responseToChatCompletion,
} from "../src/client.ts";
import { serve } from "../src/serve.ts";

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

function captureRejectedBatchSubmissions(client: BatchOpenAI) {
  const uploads: string[] = [];
  const endpoints: string[] = [];
  (client as any)._files = {
    create: async ({ file }: { file: File }) => {
      uploads.push(await file.text());
      return { id: `file-${uploads.length}` };
    },
  };
  (client as any)._batches = {
    create: async ({ endpoint }: { endpoint: string }) => {
      endpoints.push(endpoint);
      throw new Error("stop after capturing batch submission");
    },
  };
  return { uploads, endpoints };
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

test("non-create Responses methods remain delegated to the OpenAI resource", async () => {
  const client = makeClient();
  const expected = makeResponse();
  const retrieved: string[] = [];
  (client as any)._responses.retrieve = async (id: string) => {
    retrieved.push(id);
    return expected;
  };

  try {
    const result = await client.responses.retrieve("resp-existing");
    assert.equal(result, expected);
    assert.deepEqual(retrieved, ["resp-existing"]);
  } finally {
    await client.close();
  }
});

test("stored Chat Completion methods remain delegated", async () => {
  const requested: string[] = [];
  const client = makeClient({
    fetch: async (url) => {
      requested.push(String(url));
      return new globalThis.Response(
        JSON.stringify({
          id: "chatcmpl-existing",
          object: "chat.completion",
          created: 1_700_000_000,
          model: "test-model",
          choices: [],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    },
  });

  try {
    const result = await client.chat.completions.retrieve("chatcmpl-existing");
    assert.equal(result.id, "chatcmpl-existing");
    assert.equal(requested.length, 1);
    assert.match(requested[0], /chat\/completions\/chatcmpl-existing/);
  } finally {
    await client.close();
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

test("output logprobs survive chat adaptation", () => {
  const response = makeResponse();
  const message = response.output[0];
  assert.equal(message?.type, "message");
  if (message?.type !== "message" || message.content[0]?.type !== "output_text") {
    throw new Error("test response is not an output text message");
  }
  message.content[0].logprobs = [
    {
      token: "hello",
      bytes: [104, 101, 108, 108, 111],
      logprob: -0.25,
      top_logprobs: [
        { token: "hi", bytes: [104, 105], logprob: -1.5 },
      ],
    },
  ];

  const completion = responseToChatCompletion(response);

  assert.deepEqual(completion.choices[0]?.logprobs, {
    content: [
      {
        token: "hello",
        bytes: [104, 101, 108, 108, 111],
        logprob: -0.25,
        top_logprobs: [
          { token: "hi", bytes: [104, 105], logprob: -1.5 },
        ],
      },
    ],
    refusal: null,
  });
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

test("flex forwards per-request transport options to create and polls", async () => {
  const client = makeClient();
  const calls: Array<{ operation: string; options: unknown }> = [];
  const queued = makeResponse("queued", "");
  const completed = makeResponse("completed", "done");
  (client as any)._responses = {
    create: async (_body: unknown, options: unknown) => {
      calls.push({ operation: "create", options });
      return queued;
    },
    retrieve: async (_id: string, _query: unknown, options: unknown) => {
      calls.push({ operation: "retrieve", options });
      return completed;
    },
  };
  const options = {
    timeout: 12_500,
    headers: { "X-Route": "tenant-a" },
    query: { region: "uk" },
  };

  try {
    await client.responses.create(
      { model: "test-model", input: "hello" },
      options,
    );
    assert.deepEqual(calls.map((call) => call.operation), [
      "create",
      "retrieve",
    ]);
    for (const call of calls) {
      const forwarded = call.options as typeof options & { signal: AbortSignal };
      assert.equal(forwarded.timeout, options.timeout);
      assert.deepEqual(forwarded.headers, options.headers);
      assert.deepEqual(forwarded.query, options.query);
      assert.equal(forwarded.signal instanceof AbortSignal, true);
    }
  } finally {
    await client.close();
  }
});

test("24h batch mode rejects per-request transport options", async () => {
  const client = makeClient({ completionWindow: "24h", batchSize: 1 });
  captureRejectedBatchSubmissions(client);

  try {
    await assert.rejects(
      client.responses.create(
        { model: "test-model", input: "hello" },
        { timeout: 100 },
      ),
      /transport options.*Batch|Batch.*transport/,
    );
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

test("24h queues submit separate batches for separate endpoints", async () => {
  const client = makeClient({
    completionWindow: "24h",
    batchSize: 10,
    batchWindowSeconds: 60,
  });
  const captured = captureRejectedBatchSubmissions(client);
  const requests = Promise.allSettled([
    client._enqueue("/v1/chat/completions", {
      model: "test-model",
      messages: [{ role: "user", content: "hello" }],
    }),
    client._enqueue("/v1/embeddings", {
      model: "embedding-model",
      input: "hello",
    }),
  ]);

  await client.close();
  await requests;

  assert.deepEqual(captured.endpoints.sort(), [
    "/v1/chat/completions",
    "/v1/embeddings",
  ]);
  const uploadedEndpoints = captured.uploads
    .flatMap((upload) => upload.split("\n"))
    .map((line) => JSON.parse(line).url)
    .sort();
  assert.deepEqual(uploadedEndpoints, [
    "/v1/chat/completions",
    "/v1/embeddings",
  ]);
});

test("24h text batches force non-streaming JSONL bodies", async () => {
  const client = makeClient({
    completionWindow: "24h",
    batchSize: 10,
    batchWindowSeconds: 60,
  });
  const captured = captureRejectedBatchSubmissions(client);
  const request = client._enqueue("/v1/responses", {
    model: "test-model",
    input: "hello",
    stream: true,
    stream_options: { include_usage: true },
  });

  await client.close();
  await Promise.allSettled([request]);

  const line = JSON.parse(captured.uploads[0]);
  assert.equal(line.body.stream, false);
  assert.equal("stream_options" in line.body, false);
});

test("close aborts active flex polling and cancels the upstream response", async () => {
  const client = makeClient({ pollIntervalSeconds: 0.01 });
  const queued = makeResponse("queued", "");
  const completed = makeResponse("completed", "done");
  const cancelled: string[] = [];
  let allowCompletion = false;
  (client as any)._responses = {
    create: async () => queued,
    retrieve: async () => (allowCompletion ? completed : queued),
    cancel: async (id: string) => {
      cancelled.push(id);
      return { ...queued, status: "cancelled" };
    },
  };

  const request = client.responses
    .create({ model: "test-model", input: "hello" })
    .then(
      () => "resolved",
      () => "rejected",
    );
  await new Promise((resolve) => setTimeout(resolve, 0));

  try {
    await client.close();
    const outcome = await Promise.race([
      request,
      new Promise<string>((resolve) =>
        setTimeout(() => resolve("still-polling"), 25),
      ),
    ]);

    assert.equal(outcome, "rejected");
    assert.deepEqual(cancelled, ["resp-test"]);
  } finally {
    allowCompletion = true;
    await request;
  }
});

test("HTTP proxy rejects oversized request bodies before parsing", async () => {
  const { server, close } = serve({
    apiKey: "sk-test",
    baseURL: "https://api.test/v1",
    host: "127.0.0.1",
    port: 0,
  });
  await once(server, "listening");
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("test server did not expose a TCP address");
  }
  const body = "x".repeat(1024 * 1024 + 1);

  try {
    const status = await new Promise<number | undefined>((resolve, reject) => {
      const request = httpRequest(
        {
          host: "127.0.0.1",
          port: address.port,
          path: "/v1/responses",
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Connection: "close",
          },
        },
        (response) => {
          response.resume();
          response.on("end", () => resolve(response.statusCode));
        },
      );
      request.on("error", reject);
      request.end(body);
    });

    assert.equal(status, 413);
  } finally {
    await close();
  }
});
