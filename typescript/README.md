# autobatcher (TypeScript)

Drop-in [`OpenAI`](https://www.npmjs.com/package/openai) client that
uses Doubleword flex background inference and polling by default, with explicit
24-hour batch inference. It is designed for the
[Doubleword Inference API](https://docs.doubleword.ai/inference-api/autobatcher).

`BatchOpenAI` is a subclass of `OpenAI` — it passes `instanceof` checks and
works anywhere the standard client is accepted. Chat and Responses calls submit
immediately with `service_tier: "flex"` and `background: true`, then poll by
response ID. Embeddings always use 24-hour batches.

## Installation

```bash
npm install autobatcher openai
```

## Usage

### Chat completions

```typescript
import { BatchOpenAI } from "autobatcher";

const client = new BatchOpenAI({
  apiKey: "sk-...", // or set OPENAI_API_KEY env var
  baseURL: "https://api.doubleword.ai/v1",
});

const response = await client.chat.completions.create({
  model: "Qwen/Qwen3.5-35B-A3B-FP8",
  messages: [{ role: "user", content: "What is 2+2?" }],
});
console.log(response.choices[0].message.content);

await client.close();
```

### Embeddings

```typescript
const response = await client.embeddings.create({
  model: "Qwen/Qwen3-Embedding-8B",
  input: "Hello, world!",
});
console.log(response.data[0].embedding.slice(0, 5));
```

### Responses API

```typescript
const response = await client.responses.create({
  model: "Qwen/Qwen3.5-35B-A3B-FP8",
  input: "Explain quantum computing in one sentence.",
});
console.log(response.output_text);
```

### Parallel requests

Concurrent text requests are submitted independently and queued by Doubleword:

```typescript
const prompts = ["What is 1+1?", "What is 2+2?", "What is 3+3?"];

// All requests use background flex inference and poll independently
const results = await Promise.all(
  prompts.map((prompt) =>
    client.chat.completions.create({
      model: "Qwen/Qwen3.5-35B-A3B-FP8",
      messages: [{ role: "user", content: prompt }],
    })
  )
);

for (const r of results) {
  console.log(r.choices[0].message.content);
}

await client.close();
```

## Serve mode

`autobatcher serve` runs a local OpenAI-compatible HTTP proxy that routes
incoming requests. Useful for transparently handling traffic from tools that
support a custom `baseURL` — evaluation frameworks, benchmark runners, or any
OpenAI SDK consumer.

```bash
npx autobatcher serve \
  --base-url https://api.doubleword.ai/v1 \
  --api-key "$DOUBLEWORD_API_KEY" \
  --port 8080
```

Then point any OpenAI-compatible client at the proxy:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=dummy
```

Use your real credential for the proxy's upstream `--api-key`. The downstream
client uses a dummy key because it is only talking to the local proxy.

Supported proxy routes:

| Route | Default upstream behavior |
|-------|---------------------------|
| `POST /v1/chat/completions` | flex Responses polling, converted back to chat |
| `POST /v1/embeddings` | 24-hour batch |
| `POST /v1/responses` | flex Responses polling |
| `GET /health` | local healthcheck |

Pass `--mode batch` to use 24-hour batches for text generation as well.

The proxy emits structured JSON lifecycle events to stdout for log collection:

```json
{"source":"autobatcher","event":"server_started","ts":1776163751.821,"host":"127.0.0.1","port":8080}
```

### Programmatic usage

You can also start the server programmatically:

```typescript
import { serve } from "autobatcher";

const { server, close } = serve({
  baseURL: "https://api.doubleword.ai/v1",
  apiKey: "sk-...",
  port: 8080,
  pollIntervalSeconds: 5,
});

// Later: gracefully shut down
await close();
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `apiKey` | env var | OpenAI / Doubleword API key (falls back to `OPENAI_API_KEY`) |
| `baseURL` | provider default | API base URL |
| `batchSize` | `1000` | Submit batch when this many requests are queued |
| `batchWindowSeconds` | `10` | Submit batch after this many seconds |
| `pollIntervalSeconds` | `5` | How often to poll flex responses or batch completion |
| `completionWindow` | unset | Set to `"24h"` for text-generation batches |

### Completion window

An omitted value (and legacy `"1h"`) selects Doubleword flex polling for text
generation. Exactly `"24h"` selects the Batch API. Embeddings always use a
24-hour batch, irrespective of this setting.

## Supported endpoints

| Endpoint | Return type |
|----------|-------------|
| `client.chat.completions.create()` | `ChatCompletion` |
| `client.embeddings.create()` | `CreateEmbeddingResponse` |
| `client.responses.create()` | `Response` |

All other methods on the client (e.g. `client.models.list()`,
`client.files.create()`) pass through to the underlying OpenAI client
unchanged — only the endpoints above are intercepted.

## Limitations

- Not suitable for real-time or interactive use cases. Flex work has
  minutes-scale latency and 24-hour batches may take longer.
- Streaming is not supported. Requests with `stream: true` will have streaming
  stripped and results returned as a complete response.
- Flex polling is Doubleword-only. OpenAI users must select `"24h"` for batch
  inference or use the upstream OpenAI client for realtime inference.
- No automatic escalation to realtime if flex or batch inference is delayed.

## License

MIT
