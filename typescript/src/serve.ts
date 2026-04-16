/**
 * autobatcher serve — local OpenAI-compatible HTTP proxy that transparently
 * batches incoming requests via BatchOpenAI.
 *
 * Usage:
 *   npx autobatcher serve --base-url https://api.doubleword.ai/v1 --api-key sk-...
 */

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { BatchOpenAI, type BatchOpenAIOptions } from "./client.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ServeOptions {
  /** Upstream API base URL. */
  baseURL: string;
  /** Upstream API key. */
  apiKey: string;
  /** Host to bind (default "127.0.0.1"). */
  host?: string;
  /** Port to listen on (default 8080). */
  port?: number;
  /** Scheduling mode: "async" (default) or "batch". */
  mode?: "async" | "batch";
  /** BatchOpenAI options forwarded to the client. */
  batchSize?: number;
  batchWindowSeconds?: number;
  pollIntervalSeconds?: number;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const BATCHED_ROUTES = new Set([
  "/v1/chat/completions",
  "/v1/embeddings",
  "/v1/responses",
]);

function jsonResponse(res: ServerResponse, status: number, body: unknown): void {
  const json = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(json),
  });
  res.end(json);
}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
  }
  return Buffer.concat(chunks).toString("utf-8");
}

function log(event: string, data: Record<string, unknown> = {}): void {
  const entry = {
    source: "autobatcher",
    event,
    ts: Date.now() / 1000,
    ...data,
  };
  process.stdout.write(JSON.stringify(entry) + "\n");
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/**
 * Start an OpenAI-compatible HTTP proxy that batches requests.
 * Returns the server instance and a close function.
 */
export function serve(options: ServeOptions): {
  server: ReturnType<typeof createServer>;
  close: () => Promise<void>;
} {
  const client = new BatchOpenAI({
    apiKey: options.apiKey,
    baseURL: options.baseURL,
    mode: options.mode,
    batchSize: options.batchSize,
    batchWindowSeconds: options.batchWindowSeconds,
    pollIntervalSeconds: options.pollIntervalSeconds,
  });

  const host = options.host ?? "127.0.0.1";
  const port = options.port ?? 8080;

  const server = createServer(async (req, res) => {
    const url = req.url ?? "/";
    const method = req.method ?? "GET";

    // Health check
    if (url === "/health" && method === "GET") {
      jsonResponse(res, 200, { status: "ok" });
      return;
    }

    // Only accept POST to batched routes
    if (method !== "POST" || !BATCHED_ROUTES.has(url)) {
      jsonResponse(res, 404, {
        error: { message: `Route not found: ${method} ${url}`, type: "invalid_request_error" },
      });
      return;
    }

    try {
      const body = await readBody(req);
      const params = JSON.parse(body) as Record<string, unknown>;

      // Strip streaming — batch doesn't support it
      delete params.stream;
      delete params.stream_options;

      let result: unknown;
      switch (url) {
        case "/v1/chat/completions":
          result = await client.chat.completions.create(params as unknown as Parameters<typeof client.chat.completions.create>[0]);
          break;
        case "/v1/embeddings":
          result = await client.embeddings.create(params as unknown as Parameters<typeof client.embeddings.create>[0]);
          break;
        case "/v1/responses":
          // Responses API — enqueue directly
          result = await client._enqueue("/v1/responses", params);
          break;
        default:
          jsonResponse(res, 404, { error: { message: "Not found" } });
          return;
      }

      jsonResponse(res, 200, result);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log("request_error", { url, error: message });
      jsonResponse(res, 500, {
        error: { message, type: "server_error" },
      });
    }
  });

  server.listen(port, host, () => {
    log("server_started", { host, port, baseURL: options.baseURL });
  });

  const closeFn = async (): Promise<void> => {
    log("server_closing");
    await client.close();
    await new Promise<void>((resolve, reject) => {
      server.close((err) => (err ? reject(err) : resolve()));
    });
    log("server_closed");
  };

  return { server, close: closeFn };
}
