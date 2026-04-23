# autobatcher

Drop-in OpenAI client replacement that transparently batches requests via the
Batch API. Available for [Python](#python) and [TypeScript](#typescript).

This library is designed for use with the [Doubleword Inference API](https://docs.doubleword.ai/inference-api/autobatcher).
Support for OpenAI's batch API or other compatible APIs is best effort — if you experience any issues, please open an issue.

| Language | Package | Install |
|----------|---------|---------|
| Python | [`autobatcher`](https://pypi.org/project/autobatcher/) | `pip install autobatcher` |
| TypeScript | [`autobatcher`](https://www.npmjs.com/package/autobatcher) | `npm install autobatcher` |

## Why?

Batch APIs offer significant cost savings — up to 90% with the
[Doubleword Inference API](https://docs.doubleword.ai) (OpenAI offers 50% off
with their batch API) — but they require you to restructure your
code around file uploads and polling. **autobatcher** lets you keep your
existing async code while getting batch pricing automatically.

## Clients

autobatcher exports two clients:

- **`AsyncOpenAI`** — for async inference (up to 50% off with Doubleword).
  Requests are prioritised ahead of batch but are not real-time. Ideal for
  agentic workflows, background jobs, and development. **Doubleword only** —
  OpenAI does not offer an async tier; use `AsyncOpenAI` from the `openai`
  package instead for real-time OpenAI requests.
- **`BatchOpenAI`** — for batch inference (up to 90% off with Doubleword, 50%
  off with OpenAI). Designed for bulk workloads with less time pressure,
  offering the best price. Works with both Doubleword and OpenAI.

```python
# Async inference (Doubleword only)
from autobatcher import AsyncOpenAI
client = AsyncOpenAI(
    api_key="sk-...",
    base_url="https://api.doubleword.ai/v1",
)

# Batch inference (Doubleword or OpenAI)
from autobatcher import BatchOpenAI
client = BatchOpenAI(
    api_key="sk-...",
    base_url="https://api.doubleword.ai/v1",
)

# Same interface for both — just like the OpenAI SDK
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

```typescript
// Async inference (Doubleword only)
import { AsyncOpenAI } from "autobatcher";
const client = new AsyncOpenAI({
  apiKey: "sk-...",
  baseURL: "https://api.doubleword.ai/v1",
});

// Batch inference (Doubleword or OpenAI)
import { BatchOpenAI } from "autobatcher";
const client = new BatchOpenAI({
  apiKey: "sk-...",
  baseURL: "https://api.doubleword.ai/v1",
});

// Same interface for both — just like the OpenAI SDK
const response = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello!" }],
});
```

## How it works

1. Requests are collected over a configurable time window (default: 10 seconds)
2. When the window closes or batch size is reached, requests are submitted as a batch
3. Results are polled and returned to waiting callers as they complete
4. Your code sees normal response objects — no API changes needed

Different request types (chat completions, embeddings, responses) can be mixed
in a single batch — each result is parsed with the correct type automatically.

## Configuration

| Parameter | Python | TypeScript | Default | Description |
|-----------|--------|------------|---------|-------------|
| API key | `api_key` | `apiKey` | env var | OpenAI / Doubleword API key |
| Base URL | `base_url` | `baseURL` | provider default | API base URL |
| Batch size | `batch_size` | `batchSize` | `1000` | Submit batch when this many requests are queued |
| Batch window | `batch_window_seconds` | `batchWindowSeconds` | `10.0` | Submit batch after this many seconds |
| Poll interval | `poll_interval_seconds` | `pollIntervalSeconds` | `5.0` | How often to poll for batch completion |
| Batch metadata | `batch_metadata` | — | `None` | Optional metadata attached to each batch (Python only) |

## Supported endpoints

| Endpoint | Method | Return type |
|----------|--------|-------------|
| `client.chat.completions.create()` | Chat completions | `ChatCompletion` |
| `client.embeddings.create()` | Embeddings | `CreateEmbeddingResponse` |
| `client.responses.create()` | Responses API | `Response` |

All other methods on the client (e.g. `client.models.list()`,
`client.files.create()`) pass through to the underlying OpenAI client
unchanged — only the endpoints above are intercepted for batching.

## Serve mode

Both SDKs include a local OpenAI-compatible HTTP proxy that batches incoming
requests. Useful for transparently batching traffic from tools that support a
custom `base_url` — evaluation frameworks, benchmark runners, or any OpenAI SDK
consumer.

```bash
# Python (defaults to --mode async)
autobatcher serve \
  --base-url https://api.doubleword.ai/v1 \
  --api-key "$DOUBLEWORD_API_KEY" \
  --port 8080

# TypeScript (batch mode)
npx autobatcher serve \
  --base-url https://api.doubleword.ai/v1 \
  --api-key "$DOUBLEWORD_API_KEY" \
  --mode batch \
  --port 8080
```

The `--mode` flag controls the inference tier:

- `--mode async` (default) — async inference, higher priority than batch
- `--mode batch` — batch inference, best price for bulk workloads

Then point any OpenAI-compatible client at the proxy:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=dummy
```

Supported proxy routes:

| Route | Upstream batched endpoint |
|-------|--------------------------|
| `POST /v1/chat/completions` | `/v1/chat/completions` |
| `POST /v1/embeddings` | `/v1/embeddings` |
| `POST /v1/responses` | `/v1/responses` |
| `GET /health` | local healthcheck |

The proxy emits structured JSON lifecycle events to stdout for log collection.
The Python version additionally supports batch metadata stamping and configurable
shutdown behaviour — see the [Python README](python/README.md) for full details.

## Limitations

- Not suitable for real-time or interactive use cases due to latency from the
  collection window and polling cycle.
- Streaming is not supported. Requests that would normally stream are forced to
  non-streaming; the proxy can re-wrap results as SSE for consuming clients.
- `AsyncOpenAI` is Doubleword-only. OpenAI does not offer an async tier — use
  `BatchOpenAI` for OpenAI batch workloads, or `AsyncOpenAI` from the `openai`
  package for real-time OpenAI requests.

## Python

Full documentation: [`python/README.md`](python/README.md)

```bash
pip install autobatcher
```

Both `AsyncOpenAI` and `BatchOpenAI` are subclasses of `openai.AsyncOpenAI` —
they pass `isinstance` checks and work anywhere the async client is accepted
(LangChain, LlamaIndex, PydanticAI, OpenAI Agents SDK, etc.).

```python
from autobatcher import AsyncOpenAI

async with AsyncOpenAI(
    api_key="sk-...",
    base_url="https://api.doubleword.ai/v1",
) as client:
    results = await asyncio.gather(*[
        client.chat.completions.create(
            model="Qwen/Qwen3.5-35B-A3B-FP8",
            messages=[{"role": "user", "content": prompt}],
        )
        for prompt in prompts
    ])
```

## TypeScript

Full documentation: [`typescript/README.md`](typescript/README.md)

```bash
npm install autobatcher openai
```

Both `AsyncOpenAI` and `BatchOpenAI` are subclasses of `OpenAI` — they pass
`instanceof` checks and work anywhere the standard client is accepted.

```typescript
import { AsyncOpenAI } from "autobatcher";

const client = new AsyncOpenAI({
  apiKey: "sk-...",
  baseURL: "https://api.doubleword.ai/v1",
});

const [a, b, c] = await Promise.all([
  client.chat.completions.create({ model: "Qwen/Qwen3.5-35B-A3B-FP8", messages: [{ role: "user", content: "What is 1+1?" }] }),
  client.chat.completions.create({ model: "Qwen/Qwen3.5-35B-A3B-FP8", messages: [{ role: "user", content: "What is 2+2?" }] }),
  client.chat.completions.create({ model: "Qwen/Qwen3.5-35B-A3B-FP8", messages: [{ role: "user", content: "What is 3+3?" }] }),
]);

await client.close();
```

## License

MIT
