"""
Autobatcher: Drop-in AsyncOpenAI replacement that transparently batches requests.

Usage:
    from autobatcher import BatchOpenAI

    client = BatchOpenAI(api_key="...")
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
"""

from .client import BatchOpenAI

__version__ = "0.1.0"
__all__ = ["BatchOpenAI"]
