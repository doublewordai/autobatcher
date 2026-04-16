#!/usr/bin/env node

/**
 * CLI entry point for `autobatcher serve`.
 *
 * Usage:
 *   npx autobatcher serve \
 *     --base-url https://api.doubleword.ai/v1 \
 *     --api-key sk-... \
 *     --port 8080
 */

import { parseArgs } from "node:util";
import { serve } from "./serve.js";

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    "base-url": { type: "string" },
    "api-key": { type: "string" },
    host: { type: "string", default: "127.0.0.1" },
    port: { type: "string", default: "8080" },
    "batch-size": { type: "string", default: "1000" },
    "batch-window": { type: "string", default: "10" },
    "poll-interval": { type: "string", default: "5" },
    "completion-window": { type: "string", default: "24h" },
    help: { type: "boolean", short: "h" },
  },
});

const command = positionals[0];

if (values.help || !command) {
  console.log(`
autobatcher — OpenAI-compatible HTTP proxy with transparent batching

Usage:
  autobatcher serve [options]

Options:
  --base-url <url>            Upstream API base URL (required)
  --api-key <key>             Upstream API key (required)
  --host <host>               Host to bind (default: 127.0.0.1)
  --port <port>               Port to listen on (default: 8080)
  --batch-size <n>            Max requests per batch (default: 1000)
  --batch-window <seconds>    Batch collection window (default: 10)
  --poll-interval <seconds>   Polling interval (default: 5)
  --completion-window <window> Batch deadline: "24h" or "1h" (default: 24h)
  -h, --help                  Show this help
`.trim());
  process.exit(values.help ? 0 : 1);
}

if (command !== "serve") {
  console.error(`Unknown command: ${command}. Use "autobatcher serve".`);
  process.exit(1);
}

const baseURL = values["base-url"];
const apiKey = values["api-key"] ?? process.env.OPENAI_API_KEY;

if (!baseURL) {
  console.error("Error: --base-url is required");
  process.exit(1);
}
if (!apiKey) {
  console.error("Error: --api-key is required (or set OPENAI_API_KEY)");
  process.exit(1);
}

const { close } = serve({
  baseURL,
  apiKey,
  host: values.host,
  port: parseInt(values.port!, 10),
  batchSize: parseInt(values["batch-size"]!, 10),
  batchWindowSeconds: parseInt(values["batch-window"]!, 10),
  pollIntervalSeconds: parseInt(values["poll-interval"]!, 10),
  completionWindow: values["completion-window"],
});

// Graceful shutdown
for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.on(signal, () => {
    close().then(() => process.exit(0));
  });
}
