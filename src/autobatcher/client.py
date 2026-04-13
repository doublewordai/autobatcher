"""
BatchOpenAI: A drop-in replacement for AsyncOpenAI that uses the batch API.

Collects requests over a time window or until a size threshold, submits them
as a batch, polls for results, and returns them to waiting callers.

Subclasses AsyncOpenAI so it passes isinstance checks and provides full
access to non-batched endpoints (models, files, etc.) out of the box.
"""

from __future__ import annotations

import asyncio
import json
import io
import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse


# ---------------------------------------------------------------------------
# Batch endpoint registry
# ---------------------------------------------------------------------------

BATCH_ENDPOINTS: dict[str, str] = {
    "chat_completions": "/v1/chat/completions",
    "embeddings": "/v1/embeddings",
    "completions": "/v1/completions",
}


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _PendingRequest:
    """A request waiting to be batched."""

    custom_id: str
    endpoint: str  # e.g. "/v1/chat/completions"
    params: dict[str, Any]
    future: asyncio.Future[Any]


@dataclass
class _ActiveBatch:
    """A batch that has been submitted and is being polled."""

    batch_id: str
    endpoint: str
    output_file_id: str
    error_file_id: str
    requests: dict[str, _PendingRequest]  # custom_id -> request
    created_at: float
    last_offset: int = 0  # Track offset for partial result streaming


# ---------------------------------------------------------------------------
# Proxy classes that intercept .create() and route to batching
# ---------------------------------------------------------------------------

class _BatchedChatCompletions:
    """Proxy for chat.completions that batches create() calls."""

    def __init__(self, client: BatchOpenAI):
        self._client = client

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        return await self._client._enqueue_request(
            endpoint=BATCH_ENDPOINTS["chat_completions"],
            model=model,
            messages=messages,
            **kwargs,
        )


class _BatchedChat:
    """Proxy for chat namespace."""

    def __init__(self, client: BatchOpenAI):
        self.completions = _BatchedChatCompletions(client)


class _BatchedEmbeddings:
    """Proxy for embeddings that batches create() calls."""

    def __init__(self, client: BatchOpenAI):
        self._client = client

    async def create(
        self,
        *,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        return await self._client._enqueue_request(
            endpoint=BATCH_ENDPOINTS["embeddings"],
            model=model,
            input=input,
            **kwargs,
        )


class _BatchedCompletions:
    """Proxy for completions that batches create() calls."""

    def __init__(self, client: BatchOpenAI):
        self._client = client

    async def create(
        self,
        *,
        model: str,
        prompt: str | list[str],
        **kwargs: Any,
    ) -> Any:
        return await self._client._enqueue_request(
            endpoint=BATCH_ENDPOINTS["completions"],
            model=model,
            prompt=prompt,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Response validators per endpoint
# ---------------------------------------------------------------------------

def _validate_response(endpoint: str, body: dict[str, Any]) -> Any:
    """Parse a batch result body into the correct OpenAI response type."""
    if endpoint == BATCH_ENDPOINTS["chat_completions"]:
        return ChatCompletion.model_validate(body)
    elif endpoint == BATCH_ENDPOINTS["embeddings"]:
        return CreateEmbeddingResponse.model_validate(body)
    else:
        # For /v1/completions or unknown endpoints, return raw dict
        return body


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class BatchOpenAI(AsyncOpenAI):
    """
    Drop-in replacement for AsyncOpenAI that uses the batch API.

    Requests are collected and submitted as batches based on size and time
    thresholds. Results are polled and returned to waiting callers.

    Subclasses AsyncOpenAI, so it passes isinstance checks and provides
    full access to non-batched endpoints (models, files, etc.).

    Usage:
        client = BatchOpenAI(
            api_key="...",
            base_url="https://api.doubleword.ai/v1",
            batch_size=100,
            batch_window_seconds=1.0,
        )

        # Use exactly like AsyncOpenAI
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
        )

        # Embeddings are also batched
        embeddings = await client.embeddings.create(
            model="text-embedding-3-small",
            input="Hello world",
        )
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 1000,
        batch_window_seconds: float = 10.0,
        poll_interval_seconds: float = 5.0,
        completion_window: Literal["24h", "1h"] = "24h",
        **openai_kwargs: Any,
    ):
        """
        Initialize BatchOpenAI.

        Args:
            api_key: API key for the OpenAI-compatible endpoint
            base_url: Base URL for the API (e.g., "https://api.doubleword.ai/v1")
            batch_size: Submit batch when this many requests are queued
            batch_window_seconds: Submit batch after this many seconds, even if size not reached
            poll_interval_seconds: How often to poll for batch completion
            completion_window: Batch completion window ("24h" or "1h")
            **openai_kwargs: Additional arguments passed to AsyncOpenAI
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            **openai_kwargs,
        )

        self._api_base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._api_key_str = api_key
        self._batch_size = batch_size
        self._batch_window_seconds = batch_window_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._completion_window = completion_window

        # HTTP client for raw requests (needed for partial result streaming headers)
        self._http_client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(60.0),
        )

        # Request collection — keyed by endpoint so different types batch separately
        self._pending: dict[str, list[_PendingRequest]] = {}
        self._pending_lock = asyncio.Lock()
        self._window_tasks: dict[str, asyncio.Task[None]] = {}

        # Active batches being polled
        self._active_batches: list[_ActiveBatch] = []
        self._poller_task: asyncio.Task[None] | None = None

        # Override the chat, embeddings, and completions namespaces with batched proxies
        self.chat = _BatchedChat(self)  # type: ignore[assignment]
        self.embeddings = _BatchedEmbeddings(self)  # type: ignore[assignment]
        self.completions = _BatchedCompletions(self)  # type: ignore[assignment]

        logger.debug("Initialized with batch_size={}, window={}s", batch_size, batch_window_seconds)

    async def _enqueue_request(
        self,
        endpoint: str,
        **params: Any,
    ) -> Any:
        """Add a request to the pending queue and return when result is ready."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        request = _PendingRequest(
            custom_id=str(uuid.uuid4()),
            endpoint=endpoint,
            params=params,
            future=future,
        )

        async with self._pending_lock:
            if endpoint not in self._pending:
                self._pending[endpoint] = []

            self._pending[endpoint].append(request)
            pending_count = len(self._pending[endpoint])

            # Start window timer if this is the first request for this endpoint
            if pending_count == 1:
                logger.debug("Starting {}s batch window timer for {}", self._batch_window_seconds, endpoint)
                self._window_tasks[endpoint] = asyncio.create_task(
                    self._window_timer(endpoint),
                    name=f"batch_window_timer_{endpoint}"
                )

            # Check if we've hit the size threshold
            if pending_count >= self._batch_size:
                logger.debug("Batch size {} reached for {}", self._batch_size, endpoint)
                await self._submit_batch(endpoint)

        return await future

    async def _window_timer(self, endpoint: str) -> None:
        """Timer that triggers batch submission after the window elapses."""
        try:
            await asyncio.sleep(self._batch_window_seconds)
            async with self._pending_lock:
                if self._pending.get(endpoint):
                    await self._submit_batch(endpoint)
        except asyncio.CancelledError:
            logger.debug("Window timer cancelled for {}", endpoint)
            raise
        except Exception as e:
            logger.error("Window timer error for {}: {}", endpoint, e)
            for req in self._pending.get(endpoint, []):
                if not req.future.done():
                    req.future.set_exception(e)
            raise

    async def _submit_batch(self, endpoint: str) -> None:
        """Submit all pending requests for an endpoint as a batch."""
        requests = self._pending.get(endpoint, [])
        if not requests:
            return

        # Cancel the window timer if running
        current_task = asyncio.current_task()
        window_task = self._window_tasks.get(endpoint)
        if window_task and not window_task.done() and window_task is not current_task:
            window_task.cancel()
        self._window_tasks.pop(endpoint, None)

        # Take all pending requests
        self._pending[endpoint] = []

        # Create JSONL content
        lines = []
        for req in requests:
            line = {
                "custom_id": req.custom_id,
                "method": "POST",
                "url": req.endpoint,
                "body": req.params,
            }
            lines.append(json.dumps(line))
        content = "\n".join(lines)

        try:
            # Upload the batch file
            file_obj = io.BytesIO(content.encode("utf-8"))
            filename = f"batch-{uuid.uuid4()}.jsonl"

            file_response = await self.files.create(
                file=(filename, file_obj, "application/jsonl"),
                purpose="batch",
            )
            logger.debug("Uploaded batch file: {}", file_response.id)

            # Create the batch
            batch_response = await self.batches.create(
                input_file_id=file_response.id,
                endpoint=endpoint,
                completion_window=self._completion_window,
            )
            logger.info("Submitted batch {} with {} {} requests", batch_response.id, len(requests), endpoint)

            # Track the active batch
            active_batch = _ActiveBatch(
                batch_id=batch_response.id,
                endpoint=endpoint,
                output_file_id=batch_response.output_file_id or "",
                error_file_id=batch_response.error_file_id or "",
                requests={req.custom_id: req for req in requests},
                created_at=time.time(),
            )
            self._active_batches.append(active_batch)

            # Start the poller if not running
            if self._poller_task is None or self._poller_task.done():
                self._poller_task = asyncio.create_task(
                    self._poll_batches(),
                    name="batch_poller"
                )

        except Exception as e:
            logger.error("Batch submission failed: {}", e)
            for req in requests:
                if not req.future.done():
                    req.future.set_exception(e)

    async def _poll_batches(self) -> None:
        """Poll active batches for completion and distribute results."""
        logger.debug("Poller started with {} active batches", len(self._active_batches))

        while self._active_batches:
            await asyncio.sleep(self._poll_interval_seconds)

            completed_indices = []

            for i, batch in enumerate(self._active_batches):
                try:
                    status = await self.batches.retrieve(batch.batch_id)
                    counts = status.request_counts
                    logger.debug(
                        "Batch {} status: {} (completed={}/{})",
                        batch.batch_id[:12], status.status,
                        counts.completed if counts else 0,
                        counts.total if counts else 0
                    )

                    # Update output_file_id if it becomes available
                    if status.output_file_id and not batch.output_file_id:
                        batch.output_file_id = status.output_file_id

                    if status.status == "completed":
                        await self._process_completed_batch(batch, status.output_file_id)
                        completed_indices.append(i)
                        logger.info("Batch {} completed", batch.batch_id)
                    elif status.status in ("failed", "expired", "cancelled"):
                        logger.error("Batch {} {}", batch.batch_id, status.status)
                        error = Exception(f"Batch {batch.batch_id} {status.status}")
                        for req in batch.requests.values():
                            if not req.future.done():
                                req.future.set_exception(error)
                        completed_indices.append(i)
                    elif status.status in ("in_progress", "validating", "finalizing"):
                        # Fetch partial results if output file is available
                        if batch.output_file_id:
                            await self._fetch_partial_results(batch, batch.output_file_id)

                except Exception as e:
                    logger.error("Error polling batch {}: {}", batch.batch_id, e)

            # Remove completed batches (in reverse order to preserve indices)
            for i in reversed(completed_indices):
                self._active_batches.pop(i)

        logger.debug("Poller finished")

    async def _fetch_partial_results(self, batch: _ActiveBatch, output_file_id: str) -> bool:
        """
        Fetch partial results from an in-progress batch and resolve available futures.

        Uses the Doubleword API's partial result streaming:
        - X-Incomplete header indicates if more results are coming
        - X-Last-Line header tracks progress for resumption
        - ?offset= query param fetches only new results

        Returns True if there are more results to fetch, False if complete.
        """
        url = f"{self._api_base_url}/files/{output_file_id}/content"
        if batch.last_offset > 0:
            url = f"{url}?offset={batch.last_offset}"

        try:
            response = await self._http_client.get(url)
            response.raise_for_status()

            is_incomplete = response.headers.get("X-Incomplete", "").lower() == "true"
            last_line = response.headers.get("X-Last-Line")

            text = response.text
            if not text.strip():
                return is_incomplete

            # Parse each line and resolve the corresponding future
            resolved = 0
            for line in text.strip().split("\n"):
                if not line:
                    continue

                result = json.loads(line)
                custom_id = result.get("custom_id")

                response_data = result.get("response", {})
                error_data = result.get("error")

                if custom_id in batch.requests:
                    req = batch.requests[custom_id]
                    if not req.future.done():
                        if error_data:
                            req.future.set_exception(
                                Exception(f"Request {custom_id} failed: {error_data}")
                            )
                        else:
                            response_body = response_data.get("body", {})
                            validated = _validate_response(batch.endpoint, response_body)
                            req.future.set_result(validated)
                        resolved += 1

            # Update offset for next fetch
            if last_line:
                batch.last_offset = int(last_line)

            if resolved > 0:
                pending = sum(1 for req in batch.requests.values() if not req.future.done())
                logger.debug("Resolved {} partial results, {} pending", resolved, pending)

            return is_incomplete

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return True
            logger.debug("HTTP error fetching partial results: {}", e)
            return True
        except Exception as e:
            logger.debug("Error fetching partial results: {}", e)
            return True

    async def _process_completed_batch(
        self, batch: _ActiveBatch, output_file_id: str | None
    ) -> None:
        """Fetch any remaining results and ensure all futures are resolved."""
        if not output_file_id:
            logger.error("Batch {} completed but no output file", batch.batch_id)
            error = Exception(f"Batch {batch.batch_id} completed but no output file")
            for req in batch.requests.values():
                if not req.future.done():
                    req.future.set_exception(error)
            return

        try:
            await self._fetch_partial_results(batch, output_file_id)

            for req in batch.requests.values():
                if not req.future.done():
                    logger.warning("No result for request {}", req.custom_id)
                    req.future.set_exception(
                        Exception(f"No result for request {req.custom_id}")
                    )

        except Exception as e:
            logger.error("Error processing batch results: {}", e)
            for req in batch.requests.values():
                if not req.future.done():
                    req.future.set_exception(e)

    async def close(self) -> None:
        """Close the client and cancel any pending operations."""
        for task in self._window_tasks.values():
            if not task.done():
                task.cancel()
        if self._poller_task and not self._poller_task.done():
            self._poller_task.cancel()
        await self._http_client.aclose()
        await super().close()
