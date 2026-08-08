/**
 * autobatcher serve — local OpenAI-compatible HTTP proxy for flex and batch
 * inference via BatchOpenAI.
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
  /** BatchOpenAI options forwarded to the client. */
  batchSize?: number;
  batchWindowSeconds?: number;
  pollIntervalSeconds?: number;
  completionWindow?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const BATCHED_ROUTES = new Set([
  "/v1/chat/completions",
  "/v1/embeddings",
  "/v1/responses",
]);
const MAX_REQUEST_BODY_BYTES = 1024 * 1024;

class PayloadTooLargeError extends Error {}

function jsonResponse(res: ServerResponse, status: number, body: unknown): void {
  const json = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(json),
  });
  res.end(json);
}

async function readBody(req: IncomingMessage): Promise<string> {
  const declaredLength = Number(req.headers["content-length"] ?? 0);
  let tooLarge =
    Number.isFinite(declaredLength) &&
    declaredLength > MAX_REQUEST_BODY_BYTES;

  const chunks: Buffer[] = [];
  let totalBytes = 0;
  for await (const chunk of req) {
    const buffer = typeof chunk === "string" ? Buffer.from(chunk) : chunk;
    totalBytes += buffer.byteLength;
    if (totalBytes > MAX_REQUEST_BODY_BYTES) {
      tooLarge = true;
      chunks.length = 0;
    }
    if (!tooLarge) {
      chunks.push(buffer);
    }
  }
  if (tooLarge) {
    throw new PayloadTooLargeError("Request body exceeds the 1 MiB limit");
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
 * Start an OpenAI-compatible HTTP proxy using flex polling by default.
 * Returns the server instance and a close function.
 */
export function serve(options: ServeOptions): {
  server: ReturnType<typeof createServer>;
  close: () => Promise<void>;
} {
  const client = new BatchOpenAI({
    apiKey: options.apiKey,
    baseURL: options.baseURL,
    batchSize: options.batchSize,
    batchWindowSeconds: options.batchWindowSeconds,
    pollIntervalSeconds: options.pollIntervalSeconds,
    completionWindow: options.completionWindow,
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

    // Only accept POST to intercepted inference routes
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
          result = await client.responses.create(params as unknown as Parameters<typeof client.responses.create>[0]);
          break;
        default:
          jsonResponse(res, 404, { error: { message: "Not found" } });
          return;
      }

      jsonResponse(res, 200, result);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log("request_error", { url, error: message });
      const status = err instanceof PayloadTooLargeError ? 413 : 500;
      jsonResponse(res, status, {
        error: {
          message,
          type: status === 413 ? "invalid_request_error" : "server_error",
        },
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
