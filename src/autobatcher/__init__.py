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

from .client import BatchOpenAI, ProgressCallback, default_progress_callback, progress_bar_callback

__version__ = "0.1.1"
__all__ = ["BatchOpenAI", "ProgressCallback", "default_progress_callback", "progress_bar_callback"]
