/**
 * BatchOpenAI – an OpenAI subclass that intercepts chat.completions.create(),
 * embeddings.create(), and responses.create() and routes them through the
 * OpenAI-compatible Batch API.
 *
 * Concurrent calls are collected into a queue, serialised as JSONL, uploaded
 * as a batch input file, and polled until results are available. Each original
 * caller's Promise is resolved with the corresponding result.
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
/** Runtime-agnostic UUID — works in Node, Deno, Bun, and Cloudflare Workers. */
const uuid = (): string =>
  (globalThis.crypto as unknown as { randomUUID(): string }).randomUUID();

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BatchOpenAIOptions extends OpenAIClientOptions {
  /**
   * Scheduling mode:
   * - `"async"` (default) — async inference, faster turnaround (1h completion window)
   * - `"batch"` — batch inference, maximum cost savings (24h completion window)
   */
  mode?: "async" | "batch";
  /** Maximum requests per batch before auto-flush (default 1000). */
  batchSize?: number;
  /** Seconds to wait before flushing a partial batch (default 10). */
  batchWindowSeconds?: number;
  /** Seconds between poll ticks when waiting for batch completion (default 5). */
  pollIntervalSeconds?: number;
  /** @internal Explicit completion window override. Prefer `mode` instead. */
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Strip undefined values so JSON.stringify produces clean output. */
function cleanParams(obj: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (v !== undefined) {
      out[k] = v;
    }
  }
  return out;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// BatchOpenAI
// ---------------------------------------------------------------------------

export class BatchOpenAI extends OpenAI {
  private readonly _batchSize: number;
  private readonly _batchWindowSeconds: number;
  private readonly _pollIntervalSeconds: number;
  private readonly _completionWindow: string;
  private readonly _batchMetadata: Record<string, string>;

  private _pending: PendingRequest[] = [];
  private _windowTimer: ReturnType<typeof setTimeout> | null = null;
  private _inflightBatches: Promise<void>[] = [];
  private _closed = false;

  /** The parent's files resource, saved before we shadow anything. */
  private readonly _files: OpenAI["files"];
  /** The parent's batches resource, saved before we shadow anything. */
  private readonly _batches: OpenAI["batches"];

  constructor(options: BatchOpenAIOptions = {}) {
    const { mode, batchSize, batchWindowSeconds, pollIntervalSeconds, completionWindow, ...openaiOpts } = options;
    super(openaiOpts);

    const resolvedMode = mode ?? "async";
    this._batchSize = batchSize ?? 1000;
    this._batchWindowSeconds = batchWindowSeconds ?? 10;
    this._pollIntervalSeconds = pollIntervalSeconds ?? 5;
    this._completionWindow = completionWindow ?? (resolvedMode === "batch" ? "24h" : "1h");
    this._batchMetadata = { scheduling: resolvedMode };

    // Save references to the parent's real resources before overwriting.
    this._files = this.files;
    this._batches = this.batches;

    // Shadow the parent's chat and embeddings with our batching proxies.
    (this as Record<string, unknown>).chat = new BatchedChat(this);
    (this as Record<string, unknown>).embeddings = new BatchedEmbeddings(this);
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

      // 2. Upload file via the parent's (real) files resource.
      const file = await this._files.create({
        file: new File([jsonl], "batch.jsonl", { type: "application/jsonl" }),
        purpose: "batch" as "assistants", // Cast needed; the SDK types don't include "batch" but the API accepts it.
      });

      // 3. Create batch via the parent's (real) batches resource.
      const batchJob = await this._batches.create({
        input_file_id: file.id,
        endpoint: batchEndpoint as "/v1/chat/completions",
        completion_window: this._completionWindow as "24h",
        metadata: this._batchMetadata,
      });

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
