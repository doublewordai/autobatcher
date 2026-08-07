# autobatcher (Python)

Drop-in replacement for `AsyncOpenAI` that uses Doubleword flex background
inference and polling by default, with explicit 24-hour batch inference. It is
designed for the [Doubleword Inference API](https://docs.doubleword.ai/inference-api/autobatcher).

## Why?

Async and batch inference reduce cost, but usually require applications to
manage background response IDs, polling, JSONL uploads, and result matching.
**autobatcher** handles those lifecycles behind the familiar OpenAI interface.

```python
# Before: regular async calls (full price)
from openai import AsyncOpenAI
client = AsyncOpenAI()

# After: flex calls with background polling
from autobatcher import AsyncOpenAI
client = AsyncOpenAI(base_url="https://api.doubleword.ai/v1")

# Same interface, same code
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## How it works

1. Chat-completion and Responses calls are submitted immediately to the
   Responses API with `service_tier="flex"` and `background=True`.
2. The returned response ID is polled with separate retrieval requests until
   completion; no inference connection remains open.
3. Chat results are converted back to `ChatCompletion`; Responses results stay
   native `Response` objects.
4. Embeddings always use 24-hour batches. Passing
   `completion_window="24h"` puts text generation on the same batch lifecycle.

## Installation

```bash
pip install autobatcher
```

## Usage

### Chat completions

```python
import asyncio
from autobatcher import BatchOpenAI

async def main():
    client = BatchOpenAI(
        api_key="sk-...",  # or set OPENAI_API_KEY env var
    )

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )
    print(response.choices[0].message.content)

    await client.close()

asyncio.run(main())
```

### Embeddings

```python
async def embed(client: BatchOpenAI):
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input="Hello, world!",
    )
    print(response.data[0].embedding[:5])
```

### Responses API

```python
async def respond(client: BatchOpenAI):
    response = await client.responses.create(
        model="gpt-4o",
        input="Explain quantum computing in one sentence.",
    )
    print(response.output[0].content[0].text)
```

### Parallel requests

Concurrent text requests are submitted independently and queued by Doubleword:

```python
async def process_many(prompts: list[str]) -> list[str]:
    client = BatchOpenAI()

    async def get_response(prompt: str) -> str:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    # All requests use background flex inference and poll independently
    results = await asyncio.gather(*[get_response(p) for p in prompts])

    await client.close()
    return results
```

### Mixed endpoint routing

Chat uses flex polling while embeddings continue through a 24-hour batch:

```python
async def mixed(client: BatchOpenAI):
    chat, embedding = await asyncio.gather(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
        ),
        client.embeddings.create(
            model="text-embedding-3-small",
            input="Hello!",
        ),
    )
```

### Context manager

```python
async with BatchOpenAI() as client:
    response = await client.chat.completions.create(...)
```

## Serve mode

`autobatcher serve` runs a local OpenAI-compatible HTTP proxy. This is useful
when you want to transparently route traffic from tools that already support an
OpenAI-style `base_url`, such as evaluation frameworks, SDK consumers, or local
benchmark runners.

```bash
autobatcher serve \
  --base-url https://api.doubleword.ai/v1 \
  --api-key "$DOUBLEWORD_API_KEY" \
  --host 127.0.0.1 \
  --port 8080 \
  --poll-interval 10
```

Then point your OpenAI-compatible client at the proxy:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=dummy
```

Use your real Doubleword credential for the proxy's upstream `--api-key`. The
downstream client still uses a dummy `OPENAI_API_KEY` because it is only talking
to the local OpenAI-compatible proxy.

Supported proxy routes:

| Route | Default upstream behavior |
|-------|---------------------------|
| `/v1/chat/completions` | flex Responses polling, converted back to chat |
| `/v1/embeddings` | 24-hour batch |
| `/v1/responses` | flex Responses polling |
| `/health` | local healthcheck |

Pass `--mode batch` to use 24-hour batches for text generation as well.

### Batch lifecycle events

In `serve` mode, autobatcher emits structured JSON lines to stdout for batch
lifecycle events. These are intended for log collection systems such as
Kubernetes logs, Loki, or Cloud Logging.

Example event:

```json
{
  "batch_id": "batch_123",
  "completion_window": "24h",
  "endpoint": "/v1/chat/completions",
  "event": "batch_submitted",
  "input_file_id": "file_123",
  "metadata": {
    "benchmark_id": "bench-2026-04-14",
    "github_run_id": "24393857047"
  },
  "models": ["Qwen/Qwen3.5-397B-A17B-FP8"],
  "request_count": 872,
  "source": "autobatcher",
  "ts": 1776163751.821
}
```

Emitted events currently include:

- `batch_submitted`
- `batch_progress`
- `batch_completed`
- `batch_terminal`
- `batch_cancel_requested`
- `batch_cancelled_upstream`
- `batch_cancel_failed`
- `client_closing`

### Batch metadata

You can stamp correlation metadata onto every upstream batch:

```bash
autobatcher serve \
  --base-url https://api.doubleword.ai/v1 \
  --api-key "$DOUBLEWORD_API_KEY" \
  --batch-metadata benchmark_id=bench-2026-04-14 \
  --batch-metadata github_run_id=24393857047 \
  --batch-metadata k8s_job=perf-1234
```

This metadata is passed through to the upstream `batches.create(...)` call and
is also included in the emitted lifecycle events.

### Shutdown behavior

By default, `serve` mode best-effort cancels any still-active upstream batches
when the proxy shuts down. This is useful for short-lived pods or CI jobs where
the proxy lifetime should own the batch lifetime.

If you want upstream batches to continue running after the proxy exits, use:

```bash
autobatcher serve --keep-active-batches-on-close
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | `None` | OpenAI API key (falls back to `OPENAI_API_KEY` env var) |
| `base_url` | `None` | API base URL (for proxies or compatible APIs) |
| `batch_size` | `1000` | Submit batch when this many requests are queued |
| `batch_window_seconds` | `10.0` | Submit batch after this many seconds |
| `poll_interval_seconds` | `5.0` | How often to poll flex responses or batch completion |
| `completion_window` | `None` | Set to `"24h"` for text-generation batches |
| `batch_metadata` | `None` | Optional metadata attached to each upstream batch |
| `cancel_active_batches_on_close` | `False` | Best-effort cancel active upstream batches when closing the client |

### Completion window

An omitted value (and legacy `"1h"`) selects Doubleword flex polling for text
generation. Exactly `"24h"` selects the Batch API. Embeddings always use a
24-hour batch, irrespective of this setting.

## Supported endpoints

| Endpoint | Method | Return type |
|----------|--------|-------------|
| `client.chat.completions.create()` | Chat completions | `ChatCompletion` |
| `client.embeddings.create()` | Embeddings | `CreateEmbeddingResponse` |
| `client.responses.create()` | Responses API | `Response` |

## Limitations

- Not suitable for real-time or interactive use cases. Flex work has
  minutes-scale latency and 24-hour batches may take longer.
- Streaming is not supported. Requests that would normally stream are forced to
  non-streaming; the serve proxy can re-wrap results as SSE for consuming clients.
- Flex polling is Doubleword-only. OpenAI users must select `"24h"` for batch
  inference or use the upstream OpenAI client for realtime inference.
- No automatic escalation to realtime if flex or batch inference is delayed.

## License

MIT
