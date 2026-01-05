# autobatcher

Drop-in replacement for `AsyncOpenAI` that transparently batches requests using
OpenAI's (or compatible) [Batch
API](https://platform.openai.com/docs/guides/batch).

## Why?

Batch LLM APIs (like OpenAI's) offers 50% cost savings, but requires you to
restructure your code around file uploads and polling. **autobatcher** lets you
keep your existing async code while getting batch pricing automatically.

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

1. Requests are collected over a configurable time window (default: 1 second)
2. When the window closes or batch size is reached, requests are submitted as a batch
3. Results are polled and returned to waiting callers as they complete
4. Your code sees normal `ChatCompletion` responses

## Installation

```bash
pip install autobatcher
```

## Usage

```python
import asyncio
from autobatcher import BatchOpenAI

async def main():
    client = BatchOpenAI(
        api_key="sk-...",  # or set OPENAI_API_KEY env var
        batch_size=100,              # submit batch when this many requests queued
        batch_window_seconds=1.0,    # or after this many seconds
        poll_interval_seconds=5.0,   # how often to check for results
    )

    # Use exactly like AsyncOpenAI
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )
    print(response.choices[0].message.content)

    await client.close()

asyncio.run(main())
```

### Parallel requests

The real power comes when you have many requests:

```python
async def process_many(prompts: list[str]) -> list[str]:
    client = BatchOpenAI(batch_size=50, batch_window_seconds=2.0)

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

### Context manager

```python
async with BatchOpenAI() as client:
    response = await client.chat.completions.create(...)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | `None` | OpenAI API key (falls back to `OPENAI_API_KEY` env var) |
| `base_url` | `None` | API base URL (for proxies or compatible APIs) |
| `batch_size` | `100` | Submit batch when this many requests are queued |
| `batch_window_seconds` | `1.0` | Submit batch after this many seconds |
| `poll_interval_seconds` | `5.0` | How often to poll for batch completion |
| `completion_window` | `"24h"` | Batch completion window (`"24h"` or `"1h"`) |

## Limitations

- Only `chat.completions.create` is supported for now
- Batch API has a 24-hour completion window by default
- No escalations when the completion window elapses
- Not suitable for real-time/interactive use cases
- This library is designed or use with the [Doubleword batched
API](https://docs.doubleword.ai/batches/getting-started-with-batched-api).
Support for OpenAI's batch API or other compatible APIs is best effort. If you
experience any issues, please open an issue.

## License

MIT
