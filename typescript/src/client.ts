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
import {
  chatParamsToResponse,
  cleanParams,
  responseToChatCompletion,
} from "./translation.js";

type OpenAIRequestOptions = NonNullable<
  Parameters<OpenAI["responses"]["create"]>[1]
>;

export { chatParamsToResponse, responseToChatCompletion } from "./translation.js";
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

interface ActiveFlexOperation {
  controller: AbortController;
  responseId?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function abortError(signal: AbortSignal): Error {
  if (signal.reason instanceof Error) return signal.reason;
  const error = new Error("The operation was aborted");
  error.name = "AbortError";
  return error;
}

function sleep(ms: number, signal?: AbortSignal | null): Promise<void> {
  if (signal?.aborted) return Promise.reject(abortError(signal));
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortError(signal!));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function overrideCreate<T extends object, Args extends unknown[], Result>(
  resource: T,
  create: (...args: Args) => Result,
): T {
  return new Proxy(resource, {
    get(target, property) {
      if (property === "create") return create;
      const value = Reflect.get(target, property, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}

function hasRequestOptions(options: OpenAIRequestOptions | undefined): boolean {
  return (
    options != null &&
    Object.values(options).some((value) => value !== undefined)
  );
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

  private _pending = new Map<string, PendingRequest[]>();
  private _windowTimers = new Map<
    string,
    ReturnType<typeof setTimeout>
  >();
  private _inflightBatches = new Set<Promise<void>>();
  private _activeFlex = new Map<Promise<unknown>, ActiveFlexOperation>();
  private _closed = false;

  /** The parent's files resource, saved before we shadow anything. */
  private readonly _files: OpenAI["files"];
  /** The parent's batches resource, saved before we shadow anything. */
  private readonly _batches: OpenAI["batches"];
  /** The parent's responses resource used for flex submission and polling. */
  private readonly _responses: OpenAI["responses"];
  /** Parent resources retained for all non-create operations. */
  private readonly _chat: OpenAI["chat"];
  private readonly _embeddings: OpenAI["embeddings"];

  constructor(options: BatchOpenAIOptions = {}) {
    const { batchSize, batchWindowSeconds, pollIntervalSeconds, completionWindow, ...openaiOpts } = options;
    super(openaiOpts);

    this._batchSize = batchSize ?? 1000;
    this._batchWindowSeconds = batchWindowSeconds ?? 10;
    this._pollIntervalSeconds = pollIntervalSeconds ?? 5;
    this._completionWindow = completionWindow;

    // Save references to the parent's real resources before overwriting.
    this._chat = this.chat;
    this._embeddings = this.embeddings;
    this._files = this.files;
    this._batches = this.batches;
    this._responses = this.responses;

    // Shadow intercepted resources with routing proxies.
    const completions = overrideCreate(
      this._chat.completions,
      (
        body: ChatCompletionCreateParamsNonStreaming,
        requestOptions?: OpenAIRequestOptions,
      ) =>
        this._enqueue(
          "/v1/chat/completions",
          body as unknown as Record<string, unknown>,
          requestOptions,
        ) as Promise<ChatCompletion>,
    );
    (this as Record<string, unknown>).chat = new Proxy(this._chat, {
      get(target, property) {
        if (property === "completions") return completions;
        const value = Reflect.get(target, property, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
    (this as Record<string, unknown>).embeddings = overrideCreate(
      this._embeddings,
      (body: EmbeddingCreateParams, requestOptions?: OpenAIRequestOptions) =>
        this._enqueue(
          "/v1/embeddings",
          body as unknown as Record<string, unknown>,
          requestOptions,
        ) as Promise<CreateEmbeddingResponse>,
    );
    (this as Record<string, unknown>).responses = overrideCreate(
      this._responses,
      (
        body: ResponseCreateParamsNonStreaming,
        options?: OpenAIRequestOptions,
      ) =>
        this._enqueue(
          "/v1/responses",
          body as unknown as Record<string, unknown>,
          options,
        ) as Promise<OpenAIResponse>,
    );
  }

  // -----------------------------------------------------------------------
  // Internal: queue a request
  // -----------------------------------------------------------------------

  /** @internal – called by the proxy resource classes. */
  _enqueue(
    endpoint: string,
    params: Record<string, unknown>,
    options?: OpenAIRequestOptions,
  ): Promise<unknown> {
    if (this._closed) {
      return Promise.reject(
        new Error("BatchOpenAI is closed; cannot accept new requests"),
      );
    }

    if (endpoint !== "/v1/embeddings" && this._completionWindow !== "24h") {
      return this._startFlex(endpoint, params, options);
    }

    if (hasRequestOptions(options)) {
      return Promise.reject(
        new Error(
          "Per-request transport options cannot be represented by the Batch API",
        ),
      );
    }

    return this._enqueueBatch(endpoint, params);
  }

  private _startFlex(
    endpoint: string,
    params: Record<string, unknown>,
    options?: OpenAIRequestOptions,
  ): Promise<OpenAIResponse | ChatCompletion> {
    const operation: ActiveFlexOperation = {
      controller: new AbortController(),
    };
    const callerSignal = options?.signal;
    const abortFromCaller = () =>
      operation.controller.abort(callerSignal?.reason);
    if (callerSignal?.aborted) abortFromCaller();
    else callerSignal?.addEventListener("abort", abortFromCaller, { once: true });

    const requestOptions = {
      ...options,
      signal: operation.controller.signal,
    } as OpenAIRequestOptions;
    const promise = this._executeFlex(
      endpoint,
      params,
      requestOptions,
      operation,
    );
    this._activeFlex.set(promise, operation);
    const cleanup = () => {
      callerSignal?.removeEventListener("abort", abortFromCaller);
      this._activeFlex.delete(promise);
    };
    promise.then(cleanup, cleanup);
    return promise;
  }

  private _enqueueBatch(
    endpoint: string,
    params: Record<string, unknown>,
  ): Promise<unknown> {
    return new Promise<unknown>((resolve, reject) => {
      const pending = this._pending.get(endpoint) ?? [];
      pending.push({
        customId: uuid(),
        endpoint,
        body: cleanParams(params),
        resolve,
        reject,
      });
      this._pending.set(endpoint, pending);

      if (pending.length >= this._batchSize) {
        this._scheduleFlush(endpoint);
      } else if (!this._windowTimers.has(endpoint)) {
        const timer = setTimeout(() => {
          this._windowTimers.delete(endpoint);
          this._scheduleFlush(endpoint);
        }, this._batchWindowSeconds * 1000);
        this._windowTimers.set(endpoint, timer);
      }
    });
  }

  private async _executeFlex(
    endpoint: string,
    params: Record<string, unknown>,
    options?: OpenAIRequestOptions,
    operation?: ActiveFlexOperation,
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

    try {
      let response = await this._responses.create(
        submitParams as unknown as ResponseCreateParamsNonStreaming,
        options,
      );
      if (operation) operation.responseId = response.id;
      while (response.status === "queued" || response.status === "in_progress") {
        await sleep(this._pollIntervalSeconds * 1000, options?.signal);
        response = await this._responses.retrieve(response.id, undefined, options);
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
    } catch (error) {
      if (options?.signal?.aborted && operation?.responseId) {
        try {
          await this._responses.cancel(operation.responseId);
        } catch {
          // Preserve the caller's cancellation error if upstream cleanup fails.
        }
      }
      throw error;
    }
  }

  private _scheduleFlush(endpoint: string): void {
    if ((this._pending.get(endpoint)?.length ?? 0) === 0) return;
    const promise = this._flush(endpoint);
    this._inflightBatches.add(promise);
    const remove = () => this._inflightBatches.delete(promise);
    promise.then(remove, remove);
  }

  // -----------------------------------------------------------------------
  // Flush: submit batch, poll, distribute results
  // -----------------------------------------------------------------------

  private async _flush(endpoint: string): Promise<void> {
    const pending = this._pending.get(endpoint);
    if (!pending?.length) return;

    const batch = pending.splice(0, pending.length);
    this._pending.delete(endpoint);

    const timer = this._windowTimers.get(endpoint);
    if (timer) clearTimeout(timer);
    this._windowTimers.delete(endpoint);

    const batchEndpoint = endpoint;

    try {
      // 1. Build JSONL
      const jsonl = batch
        .map((r) =>
          JSON.stringify({
            custom_id: r.customId,
            method: "POST",
            url: r.endpoint,
            body: (() => {
              const body = { ...r.body };
              delete body.stream_options;
              if (r.endpoint !== "/v1/embeddings") body.stream = false;
              return body;
            })(),
          }),
        )
        .join("\n");

      // 2. Upload file via the parent's (real) files resource (retry on rate limit).
      let file: Awaited<ReturnType<OpenAI["files"]["create"]>>;
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
      let batchJob: Awaited<ReturnType<OpenAI["batches"]["create"]>>;
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

    const activeFlex = Array.from(this._activeFlex.entries());
    for (const [, operation] of activeFlex) {
      operation.controller.abort(new Error("BatchOpenAI closed"));
    }
    await Promise.allSettled(activeFlex.map(([promise]) => promise));

    for (const timer of this._windowTimers.values()) clearTimeout(timer);
    this._windowTimers.clear();

    for (const endpoint of Array.from(this._pending.keys())) {
      this._scheduleFlush(endpoint);
    }

    await Promise.all(Array.from(this._inflightBatches));
  }
}

/** Compatibility alias for the default flex-inference client. */
export class AsyncOpenAI extends BatchOpenAI {
  constructor(options: BatchOpenAIOptions = {}) {
    super(options);
  }
}
