# autobatcher

Drop-in replacement for `AsyncOpenAI` that transparently batches requests. This library is designed for use with the [Doubleword Batch API](https://docs.doubleword.ai/batches/getting-started-with-batched-api). Support for OpenAI's batch API or other compatible APIs is best effort. If you experience any issues, please open an issue.
 
## Why?

Batch LLM APIs offers 50% cost savings (and specialist inference providers like Doubleword offer 80%+ savings), but these APIs you to restructure your code around file uploads and polling. **autobatcher** lets you keep your existing async code while getting batch pricing automatically.

```python
# Before: regular async calls (full price)
from openai import AsyncOpenAI
client = AsyncOpenAI()

# After: batched calls (50% off)
from autobatcher import BatchOpenAI
client = BatchOpenAI()

# Same interface, same code
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## How it works

1. Requests are collected over a configurable time window (default: 10 seconds)
2. When the window closes or batch size is reached, requests are submitted as a batch
3. Results are polled and returned to waiting callers as they complete
4. Your code sees normal response objects (`ChatCompletion`, `CreateEmbeddingResponse`, `Response`)

Different request types (chat completions, embeddings, responses) can be mixed
in a single batch — each result is parsed with the correct type automatically.

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

The real power comes when you have many requests:

```python
async def process_many(prompts: list[str]) -> list[str]:
    client = BatchOpenAI(batch_size=500, batch_window_seconds=5.0)

    async def get_response(prompt: str) -> str:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    # All requests are batched together automatically
    results = await asyncio.gather(*[get_response(p) for p in prompts])

    await client.close()
    return results
```

### Mixed batching

Different request types are automatically mixed into the same batch:

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
when you want to transparently batch traffic from tools that already support an
OpenAI-style `base_url`, such as evaluation frameworks, SDK consumers, or local
benchmark runners.

```bash
autobatcher serve \
  --base-url https://api.doubleword.ai/v1 \
  --api-key "$DOUBLEWORD_API_KEY" \
  --host 127.0.0.1 \
  --port 8080 \
  --batch-size 1024 \
  --batch-window 60 \
  --poll-interval 10 \
  --completion-window 24h
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

| Route | Upstream batched endpoint |
|-------|----------------------------|
| `/v1/chat/completions` | `/v1/chat/completions` |
| `/v1/embeddings` | `/v1/embeddings` |
| `/v1/responses` | `/v1/responses` |
| `/health` | local healthcheck |

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
| `poll_interval_seconds` | `5.0` | How often to poll for batch completion |
| `completion_window` | `"24h"` | Batch completion window passed through to the upstream API |
| `batch_metadata` | `None` | Optional metadata attached to each upstream batch |
| `cancel_active_batches_on_close` | `False` | Best-effort cancel active upstream batches when closing the client |

## Supported endpoints

| Endpoint | Method | Return type |
|----------|--------|-------------|
| `client.chat.completions.create()` | Chat completions | `ChatCompletion` |
| `client.embeddings.create()` | Embeddings | `CreateEmbeddingResponse` |
| `client.responses.create()` | Responses API | `Response` |

## Limitations

- Batch API has a 24-hour completion window by default. 1hr SLAs is also offered with Doubleword.
- No escalations when the completion window elapses
- Not suitable for real-time/interactive use cases

## License

MIT
