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
with their 24-hour batch window) — but they require you to restructure your
code around file uploads and polling. **autobatcher** lets you keep your
existing async code while getting batch pricing automatically.

```python
# Before: regular async calls (full price)
from openai import AsyncOpenAI
client = AsyncOpenAI()

# After: batched calls (up to 90% off with Doubleword Inference API)
from autobatcher import BatchOpenAI
client = BatchOpenAI(base_url="https://api.doubleword.ai/v1")

# Same interface, same code
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

```typescript
// Before: regular calls (full price)
import OpenAI from "openai";
const client = new OpenAI();

// After: batched calls (up to 90% off with Doubleword Inference API)
import { BatchOpenAI } from "autobatcher";
const client = new BatchOpenAI({ baseURL: "https://api.doubleword.ai/v1" });

// Same interface, same code
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
| Mode | `mode` | `mode` | `"async"` | Scheduling mode (see below) |
| Batch size | `batch_size` | `batchSize` | `1000` | Submit batch when this many requests are queued |
| Batch window | `batch_window_seconds` | `batchWindowSeconds` | `10.0` | Submit batch after this many seconds |
| Poll interval | `poll_interval_seconds` | `pollIntervalSeconds` | `5.0` | How often to poll for batch completion |

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
# Python
autobatcher serve \
  --base-url https://api.doubleword.ai/v1 \
  --api-key "$DOUBLEWORD_API_KEY" \
  --port 8080

# TypeScript
npx autobatcher serve \
  --base-url https://api.doubleword.ai/v1 \
  --api-key "$DOUBLEWORD_API_KEY" \
  --port 8080
```

Use `--mode batch` for maximum cost savings on background workloads.

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

- Not suitable for real-time or interactive use cases — batch mode adds latency
  from the collection window and polling cycle.
- Streaming is not supported. Requests that would normally stream are forced to
  non-streaming; the proxy can re-wrap results as SSE for consuming clients.
- OpenAI only supports `mode: "batch"` (24h completion window). Async mode is a
  [Doubleword Inference API](https://docs.doubleword.ai) feature.
- No automatic escalation to real-time if the completion window elapses — the
  batch will be marked as expired.

## Python

Full documentation: [`python/README.md`](python/README.md)

```bash
pip install autobatcher
```

`BatchOpenAI` is a subclass of `AsyncOpenAI` — it passes `isinstance` checks
and works anywhere the async client is accepted (LangChain, LlamaIndex,
PydanticAI, OpenAI Agents SDK, etc.).

```python
from autobatcher import BatchOpenAI

async with BatchOpenAI(
    api_key="sk-...",
    base_url="https://api.doubleword.ai/v1",
) as client:
    results = await asyncio.gather(*[
        client.chat.completions.create(
            model="Qwen/Qwen3-30B-A3B",
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

`BatchOpenAI` is a subclass of `OpenAI` — it passes `instanceof` checks and
works anywhere the standard client is accepted.

```typescript
import { BatchOpenAI } from "autobatcher";

const client = new BatchOpenAI({
  apiKey: "sk-...",
  baseURL: "https://api.doubleword.ai/v1",
});

const [a, b, c] = await Promise.all([
  client.chat.completions.create({ model: "Qwen/Qwen3-30B-A3B", messages: [{ role: "user", content: "What is 1+1?" }] }),
  client.chat.completions.create({ model: "Qwen/Qwen3-30B-A3B", messages: [{ role: "user", content: "What is 2+2?" }] }),
  client.chat.completions.create({ model: "Qwen/Qwen3-30B-A3B", messages: [{ role: "user", content: "What is 3+3?" }] }),
]);

await client.close();
```

## License

MIT
