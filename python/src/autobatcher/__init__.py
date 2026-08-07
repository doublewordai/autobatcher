"""
Autobatcher: Drop-in AsyncOpenAI replacement for flex and batch inference.

Usage:
    from autobatcher import BatchOpenAI   # flex polling by default
    from autobatcher import AsyncOpenAI   # equivalent compatibility name

    client = BatchOpenAI(api_key="...")
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
"""

from .client import AsyncOpenAI, BatchOpenAI

__version__ = "0.10.0"
__all__ = ["AsyncOpenAI", "BatchOpenAI"]
