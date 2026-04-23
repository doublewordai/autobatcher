"""
Autobatcher: Drop-in AsyncOpenAI replacement that transparently batches requests.

Usage:
    from autobatcher import BatchOpenAI   # 24h batch inference (default)
    from autobatcher import AsyncOpenAI   # 1h async inference

    client = BatchOpenAI(api_key="...")
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
"""

from .client import AsyncOpenAI, BatchOpenAI

__version__ = "0.10.0"
__all__ = ["AsyncOpenAI", "BatchOpenAI"]
