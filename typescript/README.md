# autobatcher (TypeScript)

Drop-in [`OpenAI`](https://www.npmjs.com/package/openai) client that
transparently batches requests via the
[Batch API](https://docs.doubleword.ai/inference-api/autobatcher). Designed for
the [Doubleword Inference API](https://docs.doubleword.ai) where batch pricing
saves up to 90%.

`BatchOpenAI` is a subclass of `OpenAI` — it passes `instanceof` checks and
works anywhere the standard client is accepted. The only difference:
`chat.completions.create()` and `embeddings.create()` calls are collected into
a queue and submitted as batch jobs instead of making individual HTTP requests.

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
  model: "Qwen/Qwen3-30B-A3B",
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

### Parallel requests

The real power comes when you have many requests:

```typescript
const prompts = ["What is 1+1?", "What is 2+2?", "What is 3+3?"];

// All requests are batched together automatically
const results = await Promise.all(
  prompts.map((prompt) =>
    client.chat.completions.create({
      model: "Qwen/Qwen3-30B-A3B",
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

`autobatcher serve` runs a local OpenAI-compatible HTTP proxy that batches
incoming requests. Useful for transparently batching traffic from tools that
support a custom `baseURL` — evaluation frameworks, benchmark runners, or any
OpenAI SDK consumer.

```bash
npx autobatcher serve \
  --base-url https://api.doubleword.ai/v1 \
  --api-key "$DOUBLEWORD_API_KEY" \
  --port 8080 \
  --batch-size 1024 \
  --batch-window 60 \
  --mode batch
```

Then point any OpenAI-compatible client at the proxy:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=dummy
```

Use your real credential for the proxy's upstream `--api-key`. The downstream
client uses a dummy key because it is only talking to the local proxy.

Supported proxy routes:

| Route | Upstream batched endpoint |
|-------|--------------------------|
| `POST /v1/chat/completions` | `/v1/chat/completions` |
| `POST /v1/embeddings` | `/v1/embeddings` |
| `POST /v1/responses` | `/v1/responses` |
| `GET /health` | local healthcheck |

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
  batchSize: 1024,
  mode: "batch",
});

// Later: gracefully shut down
await close();
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `apiKey` | env var | OpenAI / Doubleword API key (falls back to `OPENAI_API_KEY`) |
| `baseURL` | provider default | API base URL |
| `mode` | `"async"` | Scheduling mode (see below) |
| `batchSize` | `1000` | Submit batch when this many requests are queued |
| `batchWindowSeconds` | `10` | Submit batch after this many seconds |
| `pollIntervalSeconds` | `5` | How often to poll for batch completion |

### Mode

The `mode` parameter controls scheduling and pricing:

- **`"async"`** (default) — async inference. Requests are processed as soon as
  possible with faster turnaround, still significantly cheaper than real-time.
  Supported by the [Doubleword Inference API](https://docs.doubleword.ai) only.
- **`"batch"`** — batch inference. Maximum cost savings (up to 90% with the
  [Doubleword Inference API](https://docs.doubleword.ai), 50% with OpenAI).
  Use for background jobs like evals, data processing, or bulk extraction
  where latency doesn't matter. This is the only mode OpenAI supports.

## Supported endpoints

| Endpoint | Return type |
|----------|-------------|
| `client.chat.completions.create()` | `ChatCompletion` |
| `client.embeddings.create()` | `CreateEmbeddingResponse` |

All other methods on the client (e.g. `client.models.list()`,
`client.files.create()`) pass through to the underlying OpenAI client
unchanged — only the endpoints above are intercepted for batching.

## Limitations

- Not suitable for real-time or interactive use cases — batch mode adds latency
  from the collection window and polling cycle.
- Streaming is not supported. Requests with `stream: true` will have streaming
  stripped and results returned as a complete response.
- OpenAI only supports `mode: "batch"` (24h completion window). Async mode is a
  [Doubleword Inference API](https://docs.doubleword.ai) feature.
- No automatic escalation to real-time if the completion window elapses — the
  batch will be marked as expired.
- Responses API batching (`client.responses.create()`) is available via the
  serve proxy but not yet via the `BatchOpenAI` class directly.

## License

MIT
