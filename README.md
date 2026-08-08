# autobatcher

Drop-in OpenAI client replacement that runs text generation through
Doubleword's flex async-inference polling API by default, with 24-hour batch
inference available explicitly. Available for [Python](#python) and
[TypeScript](#typescript).

This library is designed for use with the [Doubleword Inference API](https://docs.doubleword.ai/inference-api/autobatcher).
Support for OpenAI's batch API or other compatible APIs is best effort — if you experience any issues, please open an issue.

| Language | Package | Install |
|----------|---------|---------|
| Python | [`autobatcher`](https://pypi.org/project/autobatcher/) | `pip install autobatcher` |
| TypeScript | [`autobatcher`](https://www.npmjs.com/package/autobatcher) | `npm install autobatcher` |

## Why?

Async and batch inference reduce cost, but normally require applications to
manage background response IDs, polling, JSONL files, uploads, and result
matching. **autobatcher** keeps the familiar OpenAI client interface while
handling those lifecycles internally.

## Clients

autobatcher exports `AsyncOpenAI` and `BatchOpenAI` as compatible names for the
same default behavior. Both use Doubleword flex polling unless
`completion_window="24h"` / `completionWindow: "24h"` is supplied. Embeddings
always use 24-hour batches because they do not have a flex tier.

```python
# Async inference (Doubleword only)
from autobatcher import AsyncOpenAI
client = AsyncOpenAI(
    api_key="sk-...",
    base_url="https://api.doubleword.ai/v1",
)

# Explicit 24-hour batch inference
from autobatcher import BatchOpenAI
client = BatchOpenAI(
    api_key="sk-...",
    base_url="https://api.doubleword.ai/v1",
    completion_window="24h",
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

// Explicit 24-hour batch inference
import { BatchOpenAI } from "autobatcher";
const client = new BatchOpenAI({
  apiKey: "sk-...",
  baseURL: "https://api.doubleword.ai/v1",
  completionWindow: "24h",
});

// Same interface for both — just like the OpenAI SDK
const response = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello!" }],
});
```

## How it works

1. Default chat-completion and Responses calls are submitted immediately to
   `/v1/responses` with `service_tier: "flex"` and `background: true`.
2. Autobatcher polls the response ID using short retrieval requests, so the
   submission connection is not held open during inference.
3. Chat callers receive a normal `ChatCompletion`; Responses callers receive a
   normal `Response`.
4. Embeddings, and all endpoints in explicit 24-hour mode, use the existing
   queue → JSONL → upload → batch → poll → result lifecycle.

## Configuration

| Parameter | Python | TypeScript | Default | Description |
|-----------|--------|------------|---------|-------------|
| API key | `api_key` | `apiKey` | env var | OpenAI / Doubleword API key |
| Base URL | `base_url` | `baseURL` | provider default | API base URL |
| Batch size | `batch_size` | `batchSize` | `1000` | Submit batch when this many requests are queued |
| Batch window | `batch_window_seconds` | `batchWindowSeconds` | `10.0` | Submit batch after this many seconds |
| Poll interval | `poll_interval_seconds` | `pollIntervalSeconds` | `5.0` | How often to poll flex responses or batch completion |
| Completion window | `completion_window` | `completionWindow` | unset | Set exactly `"24h"` to batch text generation; any other value uses flex |
| Batch metadata | `batch_metadata` | — | `None` | Optional metadata attached to each batch (Python only) |

## Supported endpoints

| Endpoint | Method | Return type |
|----------|--------|-------------|
| `client.chat.completions.create()` | Chat completions | `ChatCompletion` |
| `client.embeddings.create()` | Embeddings | `CreateEmbeddingResponse` |
| `client.responses.create()` | Responses API | `Response` |

All other methods on the client (e.g. `client.models.list()`,
`client.files.create()`) pass through to the underlying OpenAI client
unchanged — only the endpoints above are intercepted.

## Serve mode

Both SDKs include a local OpenAI-compatible HTTP proxy that routes incoming
requests. Useful for transparently handling traffic from tools that support a
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

- `--mode async` (default) — flex background inference with polling
- `--mode batch` — batch inference, best price for bulk workloads

Then point any OpenAI-compatible client at the proxy:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=dummy
```

Supported proxy routes:

| Route | Default upstream behavior |
|-------|---------------------------|
| `POST /v1/chat/completions` | flex Responses polling, converted back to chat |
| `POST /v1/embeddings` | 24-hour batch |
| `POST /v1/responses` | flex Responses polling |
| `GET /health` | local healthcheck |

The proxy emits structured JSON lifecycle events to stdout for log collection.
The Python version additionally supports batch metadata stamping and configurable
shutdown behaviour — see the [Python README](python/README.md) for full details.

## Limitations

- Not suitable for real-time or interactive use cases. Flex work has
  minutes-scale latency and 24-hour batches may take longer.
- Streaming is not supported. Requests that would normally stream are forced to
  non-streaming; the proxy can re-wrap results as SSE for consuming clients.
- Per-request headers, query parameters, timeouts, and abort signals are
  supported for flex calls. Batch calls reject transport options because they
  cannot be represented in Batch API JSONL.
- HTTP proxy request bodies are limited to 1 MiB.
- Default flex polling is Doubleword-only. For OpenAI batch workloads, specify
  a 24-hour completion window; for realtime OpenAI requests, use the upstream
  OpenAI client directly.

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
