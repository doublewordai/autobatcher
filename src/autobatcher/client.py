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
from collections.abc import Callable
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
BatchEventHandler = Callable[[dict[str, Any]], None]


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

        raw_response = await client.chat.completions.with_raw_response.create(**payload)
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
    """``with_raw_response`` accessor for :class:`_BatchedChatCompletions`."""

    def __init__(self, completions: _BatchedChatCompletions) -> None:
        self._completions = completions

    async def create(self, **kwargs: Any) -> _RawResponseWrapper[ChatCompletion]:
        result = await self._completions.create(**kwargs)
        return _RawResponseWrapper(result)


class _EmbeddingsRawResponse:
    """``with_raw_response`` accessor for :class:`_BatchedEmbeddings`."""

    def __init__(self, embeddings: _BatchedEmbeddings) -> None:
        self._embeddings = embeddings

    async def create(
        self, **kwargs: Any
    ) -> _RawResponseWrapper[CreateEmbeddingResponse]:
        result = await self._embeddings.create(**kwargs)
        return _RawResponseWrapper(result)


class _ResponsesRawResponse:
    """``with_raw_response`` accessor for :class:`_BatchedResponses`."""

    def __init__(self, responses: _BatchedResponses) -> None:
        self._responses = responses

    async def create(self, **kwargs: Any) -> _RawResponseWrapper[Response]:
        result = await self._responses.create(**kwargs)
        return _RawResponseWrapper(result)


@dataclass
class _ActiveBatch:
    """A batch that has been submitted and is being polled."""

    batch_id: str
    endpoint: str
    input_file_id: str
    output_file_id: str
    error_file_id: str
    request_count: int
    requests: dict[str, _PendingRequest[Any]]  # custom_id -> request
    created_at: float
    models: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
    result_types: dict[str, type[_Validatable]] = field(default_factory=dict)  # custom_id -> result type
    last_offset: int = 0  # Track offset for partial result streaming
    last_status: str | None = None
    last_completed_count: int = -1
    last_failed_count: int = -1
    last_total_count: int = -1


# ---------------------------------------------------------------------------
# Proxy classes that intercept .create() and route to batching
# ---------------------------------------------------------------------------

class _BatchedChatCompletions:
    """Proxy for chat.completions that batches create() calls."""

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
        return await self._client._enqueue_request(
            endpoint="/v1/chat/completions",
            result_type=ChatCompletion,
            params={"model": model, "messages": messages, **kwargs},
        )


class _BatchedChat:
    """Proxy for chat namespace."""

    def __init__(self, client: BatchOpenAI):
        self.completions = _BatchedChatCompletions(client)


class _BatchedEmbeddings:
    """Proxy for embeddings that batches requests."""

    def __init__(self, client: BatchOpenAI):
        self._client = client

    @property
    def with_raw_response(self) -> _EmbeddingsRawResponse:
        """Mimic the openai SDK's ``with_raw_response`` accessor.

        See :meth:`_BatchedChatCompletions.with_raw_response` for the rationale.
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


class _BatchedResponses:
    """Proxy for responses API that batches requests."""

    def __init__(self, client: BatchOpenAI):
        self._client = client

    @property
    def with_raw_response(self) -> _ResponsesRawResponse:
        """Mimic the openai SDK's ``with_raw_response`` accessor.

        See :meth:`_BatchedChatCompletions.with_raw_response` for the rationale.
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
            batch_size=1000,
            batch_window_seconds=10.0,
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
        completion_window: str = "24h",
        batch_metadata: dict[str, str] | None = None,
        batch_event_handler: BatchEventHandler | None = None,
        cancel_active_batches_on_close: bool = False,
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
            completion_window: Batch completion window passed through to the upstream API
            batch_metadata: Optional metadata attached to upstream batches
            batch_event_handler: Optional callback for structured batch lifecycle events
            cancel_active_batches_on_close: Best-effort cancel for in-flight upstream batches on close()
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
        self._batch_metadata = dict(batch_metadata or {})
        self._batch_event_handler = batch_event_handler
        self._cancel_active_batches_on_close = cancel_active_batches_on_close
        self._closed = False

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

        # Override namespaces with batched proxies.
        # Write to __dict__ directly to shadow the parent's cached_property
        # descriptors without triggering type-checker read-only errors.
        self.__dict__["chat"] = _BatchedChat(self)
        self.__dict__["embeddings"] = _BatchedEmbeddings(self)
        self.__dict__["responses"] = _BatchedResponses(self)

        logger.debug("Initialized with batch_size={}, window={}s", batch_size, batch_window_seconds)

    def _emit_batch_event(self, event: str, **payload: Any) -> None:
        """Emit a structured batch lifecycle event to the configured handler."""
        if self._batch_event_handler is None:
            return

        body = {
            "source": "autobatcher",
            "event": event,
            "ts": time.time(),
            **payload,
        }

        try:
            self._batch_event_handler(body)
        except Exception as exc:
            logger.warning("Batch event handler failed for {}: {}", event, exc)

    async def _enqueue_request(
        self,
        *,
        endpoint: BatchEndpoint,
        result_type: type[V],
        params: dict[str, Any],
    ) -> V:
        """Add a request to the pending queue and return when result is ready."""
        if self._closed:
            raise RuntimeError("BatchOpenAI is closed")

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
        models = tuple(sorted({
            model for req in requests for model in [req.params.get("model")] if isinstance(model, str)
        }))

        try:
            # Upload the batch file
            file_obj = io.BytesIO(content.encode("utf-8"))
            filename = f"batch-{uuid.uuid4()}.jsonl"

            file_response = await self.files.create(
                file=(filename, file_obj, "application/jsonl"),
                purpose="batch",
            )
            logger.debug("Uploaded batch file: {}", file_response.id)

            # Create the batch.
            # The openai SDK types `completion_window` narrowly, but some
            # OpenAI-compatible providers accept additional values. Pass the
            # caller-provided string through unchanged.
            batch_create_kwargs: dict[str, Any] = {
                "input_file_id": file_response.id,
                "endpoint": top_level_endpoint,
                "completion_window": self._completion_window,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            }
            if self._batch_metadata:
                batch_create_kwargs["metadata"] = self._batch_metadata

            batch_response = await self.batches.create(**batch_create_kwargs)
            logger.info("Submitted batch {} with {} {} requests", batch_response.id, len(requests), endpoint)

            # Track the active batch
            active_batch = _ActiveBatch(
                batch_id=batch_response.id,
                endpoint=top_level_endpoint,
                input_file_id=file_response.id,
                output_file_id=batch_response.output_file_id or "",
                error_file_id=batch_response.error_file_id or "",
                request_count=len(requests),
                requests={req.custom_id: req for req in requests},
                created_at=time.time(),
                models=models,
                metadata=dict(self._batch_metadata),
                result_types={req.custom_id: req.result_type for req in requests},
            )
            self._active_batches.append(active_batch)
            self._emit_batch_event(
                "batch_submitted",
                batch_id=batch_response.id,
                endpoint=top_level_endpoint,
                input_file_id=file_response.id,
                output_file_id=batch_response.output_file_id,
                error_file_id=batch_response.error_file_id,
                request_count=len(requests),
                models=list(models),
                completion_window=self._completion_window,
                metadata=dict(self._batch_metadata),
            )

            # Start the poller if not running
            if self._poller_task is None or self._poller_task.done():
                self._poller_task = asyncio.create_task(
                    self._poll_batches(),
                    name="batch_poller"
                )

        except Exception as e:
            logger.error("Batch submission failed: {}", e)
            self._emit_batch_event(
                "batch_submission_failed",
                endpoint=top_level_endpoint,
                request_count=len(requests),
                models=list(models),
                completion_window=self._completion_window,
                metadata=dict(self._batch_metadata),
                error=str(e),
            )
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
                    completed_count = counts.completed if counts else 0
                    failed_count = getattr(counts, "failed", 0) if counts else 0
                    total_count = counts.total if counts else batch.request_count
                    logger.debug(
                        "Batch {} status: {} (completed={}/{})",
                        batch.batch_id[:12], status.status,
                        completed_count,
                        total_count
                    )

                    # Update output_file_id if it becomes available
                    if status.output_file_id and not batch.output_file_id:
                        batch.output_file_id = status.output_file_id
                    if status.error_file_id and not batch.error_file_id:
                        batch.error_file_id = status.error_file_id

                    if (
                        status.status != batch.last_status
                        or completed_count != batch.last_completed_count
                        or failed_count != batch.last_failed_count
                        or total_count != batch.last_total_count
                    ):
                        self._emit_batch_event(
                            "batch_progress",
                            batch_id=batch.batch_id,
                            endpoint=batch.endpoint,
                            input_file_id=batch.input_file_id,
                            output_file_id=batch.output_file_id or None,
                            error_file_id=batch.error_file_id or None,
                            request_count=batch.request_count,
                            counts={
                                "completed": completed_count,
                                "failed": failed_count,
                                "total": total_count,
                            },
                            status=status.status,
                            models=list(batch.models),
                            completion_window=self._completion_window,
                            metadata=dict(batch.metadata),
                            elapsed_seconds=round(time.time() - batch.created_at, 3),
                        )
                        batch.last_status = status.status
                        batch.last_completed_count = completed_count
                        batch.last_failed_count = failed_count
                        batch.last_total_count = total_count

                    if status.status == "completed":
                        await self._process_completed_batch(batch, status.output_file_id)
                        completed_indices.append(i)
                        logger.info("Batch {} completed", batch.batch_id)
                        self._emit_batch_event(
                            "batch_completed",
                            batch_id=batch.batch_id,
                            endpoint=batch.endpoint,
                            input_file_id=batch.input_file_id,
                            output_file_id=batch.output_file_id or None,
                            error_file_id=batch.error_file_id or None,
                            request_count=batch.request_count,
                            counts={
                                "completed": completed_count,
                                "failed": failed_count,
                                "total": total_count,
                            },
                            models=list(batch.models),
                            completion_window=self._completion_window,
                            metadata=dict(batch.metadata),
                            elapsed_seconds=round(time.time() - batch.created_at, 3),
                        )
                    elif status.status in ("failed", "expired", "cancelled"):
                        logger.error("Batch {} {}", batch.batch_id, status.status)
                        error = Exception(f"Batch {batch.batch_id} {status.status}")
                        for req in batch.requests.values():
                            if not req.future.done():
                                req.future.set_exception(error)
                        completed_indices.append(i)
                        self._emit_batch_event(
                            "batch_terminal",
                            batch_id=batch.batch_id,
                            endpoint=batch.endpoint,
                            input_file_id=batch.input_file_id,
                            output_file_id=batch.output_file_id or None,
                            error_file_id=batch.error_file_id or None,
                            request_count=batch.request_count,
                            counts={
                                "completed": completed_count,
                                "failed": failed_count,
                                "total": total_count,
                            },
                            status=status.status,
                            models=list(batch.models),
                            completion_window=self._completion_window,
                            metadata=dict(batch.metadata),
                            elapsed_seconds=round(time.time() - batch.created_at, 3),
                        )
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
        if self._closed:
            return
        self._closed = True

        pending_count = sum(len(reqs) for reqs in self._pending.values())
        active_batch_ids = [batch.batch_id for batch in self._active_batches]
        self._emit_batch_event(
            "client_closing",
            pending_request_count=pending_count,
            active_batch_ids=active_batch_ids,
            cancel_active_batches_on_close=self._cancel_active_batches_on_close,
        )

        for task in self._window_tasks.values():
            if not task.done():
                task.cancel()
        if self._window_tasks:
            await asyncio.gather(*self._window_tasks.values(), return_exceptions=True)
        self._window_tasks.clear()

        if self._poller_task and not self._poller_task.done():
            self._poller_task.cancel()
            await asyncio.gather(self._poller_task, return_exceptions=True)
        self._poller_task = None

        for endpoint, requests in self._pending.items():
            for req in requests:
                if not req.future.done():
                    req.future.set_exception(
                        RuntimeError(
                            f"BatchOpenAI closed before pending request on {endpoint} was submitted"
                        )
                    )
        self._pending.clear()

        if self._cancel_active_batches_on_close:
            for batch in list(self._active_batches):
                self._emit_batch_event(
                    "batch_cancel_requested",
                    batch_id=batch.batch_id,
                    endpoint=batch.endpoint,
                    input_file_id=batch.input_file_id,
                    output_file_id=batch.output_file_id or None,
                    error_file_id=batch.error_file_id or None,
                    request_count=batch.request_count,
                    models=list(batch.models),
                    completion_window=self._completion_window,
                    metadata=dict(batch.metadata),
                )
                try:
                    await self.batches.cancel(batch.batch_id)
                except Exception as exc:
                    logger.warning("Failed to cancel upstream batch {} during close: {}", batch.batch_id, exc)
                    self._emit_batch_event(
                        "batch_cancel_failed",
                        batch_id=batch.batch_id,
                        endpoint=batch.endpoint,
                        error=str(exc),
                    )
                else:
                    self._emit_batch_event(
                        "batch_cancelled_upstream",
                        batch_id=batch.batch_id,
                        endpoint=batch.endpoint,
                    )

        for batch in self._active_batches:
            for req in batch.requests.values():
                if not req.future.done():
                    req.future.set_exception(
                        RuntimeError(
                            f"BatchOpenAI closed before batch {batch.batch_id} completed"
                        )
                    )
        self._active_batches.clear()

        await self._http_client.aclose()
        await super().close()
