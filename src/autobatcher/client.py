"""
BatchOpenAI: A drop-in replacement for AsyncOpenAI that uses the batch API.

Collects requests over a time window or until a size threshold, submits them
as a batch, polls for results, and returns them to waiting callers.
"""

from __future__ import annotations

import asyncio
import json
import io
import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeVar

import httpx
from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse
from openai.types.responses import Response


BatchEndpoint = Literal[
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/responses",
]


class _Validatable(Protocol):
    """Protocol for pydantic-style model classes with model_validate."""

    @classmethod
    def model_validate(cls, obj: Any) -> Any: ...


V = TypeVar("V", bound=_Validatable)


@dataclass
class _PendingRequest(Generic[V]):
    """A request waiting to be batched."""

    custom_id: str
    endpoint: BatchEndpoint
    result_type: type[V]
    params: dict[str, Any]
    future: asyncio.Future[V]


class _RawResponseWrapper(Generic[V]):
    """Stand-in for the openai SDK's ``LegacyAPIResponse`` for batch results.

    The openai SDK exposes ``client.chat.completions.with_raw_response.create(...)``
    which returns a wrapper object whose ``.parse()`` method yields the parsed
    body and whose ``.headers`` / ``.http_response`` expose the underlying HTTP
    metadata. Consumers like ``langchain-openai`` use this surface
    unconditionally on their non-streaming async path:

    .. code-block:: python

        raw_response = await self.async_client.with_raw_response.create(**payload)
        response = raw_response.parse()

    Batch results don't have meaningful per-request HTTP metadata (everything
    comes back through the batch poll), so ``.headers`` is an empty dict and
    ``.http_response`` is ``None``. The ``.parse()`` method returns the parsed
    result that the underlying ``.create()`` already produced.
    """

    def __init__(self, parsed: V) -> None:
        self._parsed = parsed
        self.headers: dict[str, str] = {}
        self.http_response: Any = None

    def parse(self) -> V:
        return self._parsed


class _ChatCompletionsRawResponse:
    """``with_raw_response`` accessor for :class:`_ChatCompletions`."""

    def __init__(self, completions: "_ChatCompletions") -> None:
        self._completions = completions

    async def create(self, **kwargs: Any) -> _RawResponseWrapper[ChatCompletion]:
        result = await self._completions.create(**kwargs)
        return _RawResponseWrapper(result)


class _EmbeddingsRawResponse:
    """``with_raw_response`` accessor for :class:`_Embeddings`."""

    def __init__(self, embeddings: "_Embeddings") -> None:
        self._embeddings = embeddings

    async def create(
        self, **kwargs: Any
    ) -> _RawResponseWrapper[CreateEmbeddingResponse]:
        result = await self._embeddings.create(**kwargs)
        return _RawResponseWrapper(result)


class _ResponsesRawResponse:
    """``with_raw_response`` accessor for :class:`_Responses`."""

    def __init__(self, responses: "_Responses") -> None:
        self._responses = responses

    async def create(self, **kwargs: Any) -> _RawResponseWrapper[Response]:
        result = await self._responses.create(**kwargs)
        return _RawResponseWrapper(result)


@dataclass
class _ActiveBatch:
    """A batch that has been submitted and is being polled."""

    batch_id: str
    output_file_id: str
    error_file_id: str
    requests: dict[str, _PendingRequest[Any]]  # custom_id -> request
    created_at: float
    result_types: dict[str, type[_Validatable]] = field(default_factory=dict)  # custom_id -> result type
    last_offset: int = 0  # Track offset for partial result streaming


class _ChatCompletions:
    """Proxy for chat.completions that batches requests."""

    def __init__(self, client: BatchOpenAI):
        self._client = client

    @property
    def with_raw_response(self) -> _ChatCompletionsRawResponse:
        """Mimic the openai SDK's ``with_raw_response`` accessor.

        Returns an object whose ``.create()`` awaits the parent ``.create()``
        and wraps the result in a :class:`_RawResponseWrapper` exposing
        ``.parse()``, ``.headers``, and ``.http_response``. Required for
        consumers like ``langchain-openai`` that read response headers via
        the legacy raw-response surface.
        """
        return _ChatCompletionsRawResponse(self)

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion:
        """
        Create a chat completion. The request is queued and batched.

        Returns when the batch completes and results are available.
        """
        return await self._client._enqueue_request(
            endpoint="/v1/chat/completions",
            result_type=ChatCompletion,
            params={"model": model, "messages": messages, **kwargs},
        )


class _Chat:
    """Proxy for chat namespace."""

    def __init__(self, client: BatchOpenAI):
        self.completions = _ChatCompletions(client)


class _Embeddings:
    """Proxy for embeddings that batches requests."""

    def __init__(self, client: BatchOpenAI):
        self._client = client

    @property
    def with_raw_response(self) -> _EmbeddingsRawResponse:
        """Mimic the openai SDK's ``with_raw_response`` accessor.

        See :meth:`_ChatCompletions.with_raw_response` for the rationale.
        """
        return _EmbeddingsRawResponse(self)

    async def create(
        self,
        *,
        input: Any,
        model: str,
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        """
        Create embeddings. The request is queued and batched.

        Returns when the batch completes and results are available.
        """
        return await self._client._enqueue_request(
            endpoint="/v1/embeddings",
            result_type=CreateEmbeddingResponse,
            params={"input": input, "model": model, **kwargs},
        )


class _Responses:
    """Proxy for responses API that batches requests."""

    def __init__(self, client: BatchOpenAI):
        self._client = client

    @property
    def with_raw_response(self) -> _ResponsesRawResponse:
        """Mimic the openai SDK's ``with_raw_response`` accessor.

        See :meth:`_ChatCompletions.with_raw_response` for the rationale.
        """
        return _ResponsesRawResponse(self)

    async def create(
        self,
        *,
        model: str,
        input: Any = None,
        **kwargs: Any,
    ) -> Response:
        """
        Create a response. The request is queued and batched.

        Returns when the batch completes and results are available.
        """
        params: dict[str, Any] = {"model": model, **kwargs}
        if input is not None:
            params["input"] = input
        return await self._client._enqueue_request(
            endpoint="/v1/responses",
            result_type=Response,
            params=params,
        )


class BatchOpenAI:
    """
    Drop-in replacement for AsyncOpenAI that uses the batch API.

    Requests are collected and submitted as batches based on size and time
    thresholds. Results are polled and returned to waiting callers.

    Usage:
        client = BatchOpenAI(
            api_key="...",
            base_url="https://api.doubleword.ai/v1",
            batch_size=1000,
            batch_window_seconds=10.0,
        )

        # Use exactly like AsyncOpenAI
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello!"}],
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
        self._openai = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            **openai_kwargs,
        )
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._api_key = api_key
        self._batch_size = batch_size
        self._batch_window_seconds = batch_window_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._completion_window = completion_window

        # HTTP client for raw requests (needed to access response headers for partial results)
        self._http_client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(60.0),
        )

        # Request collection
        self._pending: list[_PendingRequest] = []
        self._pending_lock = asyncio.Lock()
        self._window_task: asyncio.Task[None] | None = None

        # Active batches being polled
        self._active_batches: list[_ActiveBatch] = []
        self._poller_task: asyncio.Task[None] | None = None

        # Public interface matching AsyncOpenAI
        self.chat = _Chat(self)
        self.embeddings = _Embeddings(self)
        self.responses = _Responses(self)

        logger.debug("Initialized with batch_size={}, window={}s", batch_size, batch_window_seconds)

    async def _enqueue_request(
        self,
        *,
        endpoint: BatchEndpoint,
        result_type: type[V],
        params: dict[str, Any],
    ) -> V:
        """Add a request to the pending queue and return when result is ready."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[V] = loop.create_future()

        request = _PendingRequest(
            custom_id=str(uuid.uuid4()),
            endpoint=endpoint,
            result_type=result_type,
            params=params,
            future=future,
        )

        async with self._pending_lock:
            self._pending.append(request)
            pending_count = len(self._pending)

            # Start window timer if this is the first request
            if pending_count == 1:
                logger.debug("Starting {}s batch window timer", self._batch_window_seconds)
                self._window_task = asyncio.create_task(
                    self._window_timer(),
                    name="batch_window_timer"
                )

            # Check if we've hit the size threshold
            if pending_count >= self._batch_size:
                logger.debug("Batch size {} reached", self._batch_size)
                await self._submit_batch()

        return await future

    async def _window_timer(self) -> None:
        """Timer that triggers batch submission after the window elapses."""
        try:
            await asyncio.sleep(self._batch_window_seconds)
            async with self._pending_lock:
                if self._pending:
                    await self._submit_batch()
        except asyncio.CancelledError:
            logger.debug("Window timer cancelled")
            raise
        except Exception as e:
            logger.error("Window timer error: {}", e)
            # Fail all pending futures
            for req in self._pending:
                if not req.future.done():
                    req.future.set_exception(e)
            raise

    async def _submit_batch(self) -> None:
        """Submit all pending requests as a batch."""
        if not self._pending:
            return

        # Cancel the window timer if running (but not if we ARE the window timer)
        current_task = asyncio.current_task()
        if self._window_task and not self._window_task.done() and self._window_task is not current_task:
            self._window_task.cancel()
        self._window_task = None

        # Take all pending requests
        requests = self._pending
        self._pending = []

        # Create JSONL content — each line uses the request's own endpoint
        lines = []
        for req in requests:
            # Force non-streaming: batch results are polled, not streamed
            body = {**req.params, "stream": False}
            line = {
                "custom_id": req.custom_id,
                "method": "POST",
                "url": req.endpoint,
                "body": body,
            }
            lines.append(json.dumps(line))
        content = "\n".join(lines)

        # Use the first request's endpoint for the top-level batches.create() call
        top_level_endpoint = requests[0].endpoint

        try:
            # Upload the batch file using BytesIO
            file_obj = io.BytesIO(content.encode("utf-8"))
            filename = f"batch-{uuid.uuid4()}.jsonl"

            file_response = await self._openai.files.create(
                file=(filename, file_obj, "application/jsonl"),
                purpose="batch",
            )
            logger.debug("Uploaded batch file: {}", file_response.id)

            # Create the batch.
            # The openai SDK types `completion_window` as Literal["24h"], but
            # Doubleword supports a "1h" extension. Suppress the type error;
            # the runtime accepts both values.
            batch_response = await self._openai.batches.create(
                input_file_id=file_response.id,
                endpoint=top_level_endpoint,
                completion_window=self._completion_window,  # ty: ignore[invalid-argument-type]
            )
            logger.info("Submitted batch {} with {} requests", batch_response.id, len(requests))

            # Track the active batch
            active_batch = _ActiveBatch(
                batch_id=batch_response.id,
                output_file_id=batch_response.output_file_id or "",
                error_file_id=batch_response.error_file_id or "",
                requests={req.custom_id: req for req in requests},
                created_at=time.time(),
                result_types={req.custom_id: req.result_type for req in requests},
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
            # If batch submission fails, fail all waiting requests
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
                    status = await self._openai.batches.retrieve(batch.batch_id)
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
        url = f"{self._base_url}/files/{output_file_id}/content"
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

                # Handle both success and error responses
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
                            result_type = batch.result_types.get(custom_id, ChatCompletion)
                            parsed = result_type.model_validate(response_body)
                            req.future.set_result(parsed)
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
                # File not ready yet, this is normal for early polling
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
            # Fetch any remaining results using the partial results mechanism
            # This continues from where we left off (using batch.last_offset)
            await self._fetch_partial_results(batch, output_file_id)

            # Handle any requests that didn't get results
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
        if self._window_task and not self._window_task.done():
            self._window_task.cancel()
        if self._poller_task and not self._poller_task.done():
            self._poller_task.cancel()
        await self._http_client.aclose()
        await self._openai.close()

    async def __aenter__(self) -> BatchOpenAI:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
