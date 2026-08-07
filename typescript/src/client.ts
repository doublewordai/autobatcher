/**
 * BatchOpenAI – an OpenAI subclass that intercepts chat.completions.create(),
 * embeddings.create(), and responses.create(). Text uses background flex
 * polling by default; embeddings and explicit 24-hour calls use the Batch API.
 *
 * Flex calls submit immediately and poll by response ID without holding the
 * submission connection open. Batch calls retain queue, JSONL, upload, poll,
 * and result-distribution behavior.
 *
 * This mirrors the Python `autobatcher.BatchOpenAI` class.
 */

import OpenAI from "openai";
import type { ClientOptions as OpenAIClientOptions } from "openai";
import type {
  ChatCompletion,
  ChatCompletionCreateParamsNonStreaming,
} from "openai/resources/chat/completions";
import type {
  CreateEmbeddingResponse,
  EmbeddingCreateParams,
} from "openai/resources/embeddings";
import type {
  Response as OpenAIResponse,
  ResponseCreateParamsNonStreaming,
} from "openai/resources/responses/responses";
/** Runtime-agnostic UUID — works in Node, Deno, Bun, and Cloudflare Workers. */
const uuid = (): string =>
  (globalThis.crypto as unknown as { randomUUID(): string }).randomUUID();

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BatchOpenAIOptions extends OpenAIClientOptions {
  /** Maximum requests per batch before auto-flush (default 1000). */
  batchSize?: number;
  /** Seconds to wait before flushing a partial batch (default 10). */
  batchWindowSeconds?: number;
  /** Seconds between flex or batch poll ticks (default 5). */
  pollIntervalSeconds?: number;
  /** Set to "24h" for batch inference; otherwise use flex polling (default). */
  completionWindow?: string;
}

interface PendingRequest {
  customId: string;
  endpoint: string;
  body: Record<string, unknown>;
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
}

interface BatchLineResult {
  custom_id: string;
  response?: {
    status_code: number;
    request_id?: string;
    body: Record<string, unknown>;
  };
  error?: {
    code: string;
    message: string;
  };
}

// ---------------------------------------------------------------------------
// Proxy resource classes
// ---------------------------------------------------------------------------

class BatchedCompletions {
  constructor(private client: BatchOpenAI) {}

  create(
    body: ChatCompletionCreateParamsNonStreaming,
  ): Promise<ChatCompletion> {
    return this.client._enqueue(
      "/v1/chat/completions",
      body as unknown as Record<string, unknown>,
    ) as Promise<ChatCompletion>;
  }
}

class BatchedChat {
  completions: BatchedCompletions;

  constructor(client: BatchOpenAI) {
    this.completions = new BatchedCompletions(client);
  }
}

class BatchedEmbeddings {
  constructor(private client: BatchOpenAI) {}

  create(body: EmbeddingCreateParams): Promise<CreateEmbeddingResponse> {
    return this.client._enqueue(
      "/v1/embeddings",
      body as unknown as Record<string, unknown>,
    ) as Promise<CreateEmbeddingResponse>;
  }
}

class BatchedResponses {
  constructor(private client: BatchOpenAI) {}

  create(body: ResponseCreateParamsNonStreaming): Promise<OpenAIResponse> {
    return this.client._enqueue(
      "/v1/responses",
      body as unknown as Record<string, unknown>,
    ) as Promise<OpenAIResponse>;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Strip undefined values so JSON.stringify produces clean output. */
function cleanParams(obj: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const extraBody = obj.extra_body;
  for (const [k, v] of Object.entries(obj)) {
    if (
      v !== undefined &&
      k !== "extra_body" &&
      k !== "extra_headers" &&
      k !== "extra_query" &&
      k !== "timeout"
    ) {
      out[k] = v;
    }
  }
  if (extraBody && typeof extraBody === "object" && !Array.isArray(extraBody)) {
    Object.assign(out, extraBody);
  }
  return out;
}

export function chatParamsToResponse(
  params: Record<string, unknown>,
): Record<string, unknown> {
  const translated = cleanParams(params);
  const n = translated.n ?? 1;
  delete translated.n;
  if (n !== 1) {
    throw new Error("Flex inference supports only n=1 for chat completions");
  }

  const messages = translated.messages;
  delete translated.messages;
  if (!messages) {
    throw new Error("Chat completions require messages");
  }
  translated.input = messages;

  const maxCompletionTokens = translated.max_completion_tokens;
  const legacyMaxTokens = translated.max_tokens;
  delete translated.max_completion_tokens;
  delete translated.max_tokens;
  if (maxCompletionTokens !== undefined) {
    translated.max_output_tokens = maxCompletionTokens;
  } else if (legacyMaxTokens !== undefined) {
    translated.max_output_tokens = legacyMaxTokens;
  }

  const responseFormat = translated.response_format;
  delete translated.response_format;
  if (responseFormat !== undefined) {
    translated.text = { format: responseFormat };
  }

  delete translated.stream_options;
  translated.stream = false;
  return translated;
}

export function responseToChatCompletion(
  response: OpenAIResponse,
): ChatCompletion {
  const textParts: string[] = [];
  const toolCalls: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }> = [];

  for (const rawItem of response.output) {
    const item = rawItem as unknown as Record<string, unknown>;
    if (item.type === "message") {
      const content = item.content;
      if (typeof content === "string") {
        textParts.push(content);
      } else if (Array.isArray(content)) {
        for (const rawPart of content) {
          const part = rawPart as Record<string, unknown>;
          if (part.type === "output_text" && typeof part.text === "string") {
            textParts.push(part.text);
          }
        }
      }

      const messageToolCalls = item.tool_calls;
      if (Array.isArray(messageToolCalls)) {
        for (const rawCall of messageToolCalls) {
          const call = rawCall as Record<string, unknown>;
          const fn = (call.function ?? {}) as Record<string, unknown>;
          toolCalls.push({
            id: String(call.id ?? call.call_id),
            type: "function",
            function: {
              name: String(fn.name ?? call.name),
              arguments: String(fn.arguments ?? call.arguments ?? "{}"),
            },
          });
        }
      }
    } else if (item.type === "function_call") {
      toolCalls.push({
        id: String(item.call_id ?? item.id),
        type: "function",
        function: {
          name: String(item.name),
          arguments: String(item.arguments ?? "{}"),
        },
      });
    }
  }

  const message: ChatCompletion["choices"][number]["message"] = {
    role: "assistant",
    content: textParts.length > 0 ? textParts.join("") : null,
    refusal: null,
  };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;

  return {
    id: response.id,
    object: "chat.completion",
    created: response.created_at,
    model: response.model,
    choices: [
      {
        index: 0,
        message,
        logprobs: null,
        finish_reason: toolCalls.length > 0 ? "tool_calls" : "stop",
      },
    ],
    usage: response.usage
      ? {
          prompt_tokens: response.usage.input_tokens,
          completion_tokens: response.usage.output_tokens,
          total_tokens: response.usage.total_tokens,
        }
      : undefined,
    service_tier: "flex",
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getHeader(headers: unknown, name: string): string | null {
  if (!headers || typeof headers !== "object") return null;
  const source = headers as Record<string, unknown> & { get?: unknown };
  if (typeof source.get === "function") {
    return source.get.call(headers, name) as string | null;
  }
  const raw = source[name] ?? source[name.toLowerCase()];
  if (raw == null) return null;
  return Array.isArray(raw) ? raw.join(", ") : String(raw);
}

/** Extract retry delay in seconds from a Retry-After header, defaulting to 60s. */
function parseRetryAfter(headers: unknown, defaultSeconds = 60): number {
  const raw = getHeader(headers, "retry-after");
  if (raw != null) {
    const parsed = Number(raw);
    if (!Number.isNaN(parsed)) return Math.max(parsed, 1);
  }
  return defaultSeconds;
}

// ---------------------------------------------------------------------------
// BatchOpenAI
// ---------------------------------------------------------------------------

export class BatchOpenAI extends OpenAI {
  private readonly _batchSize: number;
  private readonly _batchWindowSeconds: number;
  private readonly _pollIntervalSeconds: number;
  private readonly _completionWindow: string | undefined;

  private _pending: PendingRequest[] = [];
  private _windowTimer: ReturnType<typeof setTimeout> | null = null;
  private _inflightBatches: Promise<void>[] = [];
  private _closed = false;

  /** The parent's files resource, saved before we shadow anything. */
  private readonly _files: OpenAI["files"];
  /** The parent's batches resource, saved before we shadow anything. */
  private readonly _batches: OpenAI["batches"];
  /** The parent's responses resource used for flex submission and polling. */
  private readonly _responses: OpenAI["responses"];

  constructor(options: BatchOpenAIOptions = {}) {
    const { batchSize, batchWindowSeconds, pollIntervalSeconds, completionWindow, ...openaiOpts } = options;
    super(openaiOpts);

    this._batchSize = batchSize ?? 1000;
    this._batchWindowSeconds = batchWindowSeconds ?? 10;
    this._pollIntervalSeconds = pollIntervalSeconds ?? 5;
    this._completionWindow = completionWindow;

    // Save references to the parent's real resources before overwriting.
    this._files = this.files;
    this._batches = this.batches;
    this._responses = this.responses;

    // Shadow intercepted resources with routing proxies.
    (this as Record<string, unknown>).chat = new BatchedChat(this);
    (this as Record<string, unknown>).embeddings = new BatchedEmbeddings(this);
    (this as Record<string, unknown>).responses = new BatchedResponses(this);
  }

  // -----------------------------------------------------------------------
  // Internal: queue a request
  // -----------------------------------------------------------------------

  /** @internal – called by the proxy resource classes. */
  _enqueue(
    endpoint: string,
    params: Record<string, unknown>,
  ): Promise<unknown> {
    if (this._closed) {
      return Promise.reject(
        new Error("BatchOpenAI is closed; cannot accept new requests"),
      );
    }

    if (endpoint !== "/v1/embeddings" && this._completionWindow !== "24h") {
      return this._executeFlex(endpoint, params);
    }

    return this._enqueueBatch(endpoint, params);
  }

  private _enqueueBatch(
    endpoint: string,
    params: Record<string, unknown>,
  ): Promise<unknown> {
    return new Promise<unknown>((resolve, reject) => {
      this._pending.push({
        customId: uuid(),
        endpoint,
        body: cleanParams(params),
        resolve,
        reject,
      });

      if (this._pending.length >= this._batchSize) {
        this._scheduleFlush();
      } else if (!this._windowTimer) {
        this._windowTimer = setTimeout(() => {
          this._windowTimer = null;
          this._scheduleFlush();
        }, this._batchWindowSeconds * 1000);
      }
    });
  }

  private async _executeFlex(
    endpoint: string,
    params: Record<string, unknown>,
  ): Promise<OpenAIResponse | ChatCompletion> {
    const submitParams =
      endpoint === "/v1/chat/completions"
        ? chatParamsToResponse(params)
        : cleanParams(params);
    if (endpoint !== "/v1/chat/completions" && endpoint !== "/v1/responses") {
      throw new Error(`Flex inference is not supported for ${endpoint}`);
    }

    delete submitParams.stream_options;
    submitParams.stream = false;
    submitParams.service_tier = "flex";
    submitParams.background = true;

    let response = await this._responses.create(
      submitParams as unknown as ResponseCreateParamsNonStreaming,
    );
    while (response.status === "queued" || response.status === "in_progress") {
      await sleep(this._pollIntervalSeconds * 1000);
      response = await this._responses.retrieve(response.id);
    }

    if (response.status !== "completed") {
      const detail = response.error?.message ?? "no error details";
      throw new Error(
        `Flex response ${response.id} reached terminal status ${response.status}: ${detail}`,
      );
    }

    return endpoint === "/v1/chat/completions"
      ? responseToChatCompletion(response)
      : response;
  }

  private _scheduleFlush(): void {
    if (this._pending.length === 0) return;
    const p = this._flush();
    this._inflightBatches.push(p);
    p.finally(() => {
      const idx = this._inflightBatches.indexOf(p);
      if (idx >= 0) this._inflightBatches.splice(idx, 1);
    });
  }

  // -----------------------------------------------------------------------
  // Flush: submit batch, poll, distribute results
  // -----------------------------------------------------------------------

  private async _flush(): Promise<void> {
    if (this._pending.length === 0) return;

    const batch = this._pending.splice(0, this._pending.length);

    if (this._windowTimer) {
      clearTimeout(this._windowTimer);
      this._windowTimer = null;
    }

    // Determine the endpoint for this batch (use the first request's endpoint).
    const batchEndpoint = batch[0].endpoint;

    try {
      // 1. Build JSONL
      const jsonl = batch
        .map((r) =>
          JSON.stringify({
            custom_id: r.customId,
            method: "POST",
            url: r.endpoint,
            body: r.body,
          }),
        )
        .join("\n");

      // 2. Upload file via the parent's (real) files resource (retry on rate limit).
      let file: Awaited<ReturnType<typeof this._files.create>>;
      while (true) {
        try {
          file = await this._files.create({
            file: new File([jsonl], "batch.jsonl", { type: "application/jsonl" }),
            purpose: "batch" as "assistants", // Cast needed; the SDK types don't include "batch" but the API accepts it.
          });
          break;
        } catch (err) {
          if (err instanceof OpenAI.RateLimitError) {
            const retryAfter = parseRetryAfter(err.headers);
            console.info(
              `[autobatcher] Rate limited uploading batch file, retrying in ${retryAfter}s (Retry-After: ${getHeader(err.headers, "retry-after") ?? "not set"})`,
            );
            await sleep(retryAfter * 1000);
            continue;
          }
          throw err;
        }
      }

      // 3. Create batch via the parent's (real) batches resource (retry on rate limit).
      let batchJob: Awaited<ReturnType<typeof this._batches.create>>;
      while (true) {
        try {
          batchJob = await this._batches.create({
            input_file_id: file.id,
            endpoint: batchEndpoint as "/v1/chat/completions",
            completion_window: "24h",
          });
          break;
        } catch (err) {
          if (err instanceof OpenAI.RateLimitError) {
            const retryAfter = parseRetryAfter(err.headers);
            console.info(
              `[autobatcher] Rate limited creating batch, retrying in ${retryAfter}s (Retry-After: ${getHeader(err.headers, "retry-after") ?? "not set"})`,
            );
            await sleep(retryAfter * 1000);
            continue;
          }
          throw err;
        }
      }

      // 4. Poll until terminal state.
      const outputFileId = await this._pollBatch(batchJob.id);

      // 5. Fetch results with partial-result support.
      const results = await this._fetchResults(outputFileId);

      // 6. Distribute results.
      const resultMap = new Map<string, BatchLineResult>();
      for (const line of results) {
        resultMap.set(line.custom_id, line);
      }

      for (const req of batch) {
        const result = resultMap.get(req.customId);
        if (!result) {
          req.reject(new Error(`No result returned for request ${req.customId}`));
        } else if (result.error) {
          req.reject(new Error(`Batch request failed: [${result.error.code}] ${result.error.message}`));
        } else if (result.response) {
          if (result.response.status_code >= 400) {
            req.reject(new Error(`Batch request returned status ${result.response.status_code}: ${JSON.stringify(result.response.body)}`));
          } else {
            req.resolve(result.response.body);
          }
        } else {
          req.reject(new Error(`Unexpected result shape for request ${req.customId}`));
        }
      }
    } catch (err) {
      for (const req of batch) {
        req.reject(err);
      }
    }
  }

  // -----------------------------------------------------------------------
  // Poll & fetch
  // -----------------------------------------------------------------------

  private async _pollBatch(batchId: string): Promise<string> {
    while (true) {
      const batch = await this._batches.retrieve(batchId);

      switch (batch.status) {
        case "completed": {
          if (!batch.output_file_id) {
            throw new Error("Batch completed but no output_file_id present");
          }
          return batch.output_file_id;
        }
        case "failed":
        case "expired":
        case "cancelled":
        case "cancelling": {
          const errMsg =
            batch.errors?.data
              ?.map((e) => `[${e.code}] ${e.message}`)
              .join("; ") ?? "unknown error";
          throw new Error(`Batch ${batchId} reached terminal state "${batch.status}": ${errMsg}`);
        }
        default:
          // in_progress, validating, finalizing
          break;
      }

      await sleep(this._pollIntervalSeconds * 1000);
    }
  }

  /**
   * Fetch results from the output file, supporting Doubleword's partial-result
   * protocol (X-Incomplete / X-Last-Line headers with ?offset= query param).
   */
  private async _fetchResults(outputFileId: string): Promise<BatchLineResult[]> {
    const results: BatchLineResult[] = [];
    let offset: string | null = null;

    while (true) {
      // Use the raw file content endpoint. The SDK's files.content() returns
      // a Response-like object, but we need access to custom headers, so we
      // make a direct fetch using the client's configuration.
      const contentUrl: string = offset
        ? `${this.baseURL}/files/${outputFileId}/content?offset=${encodeURIComponent(offset)}`
        : `${this.baseURL}/files/${outputFileId}/content`;

      const res: Response = await fetch(contentUrl, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
        },
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`File content fetch failed (${res.status}): ${text}`);
      }

      const text = await res.text();
      const lines = text.split("\n").filter((l: string) => l.trim().length > 0);
      for (const line of lines) {
        results.push(JSON.parse(line) as BatchLineResult);
      }

      const incomplete = res.headers.get("X-Incomplete");
      const lastLine: string | null = res.headers.get("X-Last-Line");

      if (incomplete === "true" && lastLine) {
        offset = lastLine;
        await sleep(1000);
      } else {
        break;
      }
    }

    return results;
  }

  // -----------------------------------------------------------------------
  // Lifecycle
  // -----------------------------------------------------------------------

  /**
   * Flush any remaining pending requests and wait for all in-flight batches
   * to complete. After calling close(), no new requests are accepted.
   */
  async close(): Promise<void> {
    this._closed = true;

    if (this._windowTimer) {
      clearTimeout(this._windowTimer);
      this._windowTimer = null;
    }

    if (this._pending.length > 0) {
      this._scheduleFlush();
    }

    await Promise.all(this._inflightBatches);
  }
}

/** Compatibility alias for the default flex-inference client. */
export class AsyncOpenAI extends BatchOpenAI {
  constructor(options: BatchOpenAIOptions = {}) {
    super(options);
  }
}
