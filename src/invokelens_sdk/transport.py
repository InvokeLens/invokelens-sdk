"""Async batched transport for sending telemetry events."""

import logging
import queue
import threading
import time
import atexit
from typing import Optional

from .schema import TelemetryEvent

logger = logging.getLogger("invokelens_sdk.transport")

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0


class EventTransport:
    """Sends telemetry events to the ingestion endpoint.

    Uses a background thread with a buffered queue to avoid blocking the caller.
    Supports HTTP (default) and EventBridge modes.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        mode: str = "http",
        event_bus_name: Optional[str] = None,
        api_key: str = "",
        batch_size: int = 10,
        flush_interval_seconds: float = 5.0,
        max_queue_size: int = 1000,
    ):
        self.endpoint_url = endpoint_url
        self.mode = mode
        self.event_bus_name = event_bus_name
        self.api_key = api_key
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        atexit.register(self.shutdown)

    def send(self, event: TelemetryEvent):
        """Enqueue event for async delivery. Non-blocking, drops if full."""
        try:
            self._queue.put_nowait(event.model_dump_json())
        except queue.Full:
            logger.warning("InvokeLens event queue full, dropping event")

    def _run(self):
        """Background worker that batches and flushes events."""
        batch: list[str] = []
        while not self._shutdown.is_set():
            try:
                item = self._queue.get(timeout=self.flush_interval)
                batch.append(item)
                if len(batch) >= self.batch_size:
                    self._flush(batch)
                    batch = []
            except queue.Empty:
                if batch:
                    self._flush(batch)
                    batch = []

        # Drain remaining items on shutdown
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._flush(batch)

    def _flush(self, batch: list[str]):
        """Send a batch of events."""
        if self.mode == "http":
            self._flush_http(batch)
        elif self.mode == "eventbridge":
            self._flush_eventbridge(batch)

    def _flush_http(self, batch: list[str]):
        """Send batch via HTTP with retry and exponential backoff."""
        import httpx

        backoff = INITIAL_BACKOFF_SECONDS
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = httpx.post(
                    f"{self.endpoint_url}/v1/ingest",
                    json={"events": batch},
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=10.0,
                )
                if response.status_code < 400:
                    return  # Success

                if 400 <= response.status_code < 500:
                    # Client error (auth, validation) — do not retry
                    logger.warning(
                        "InvokeLens ingest rejected (HTTP %d): %s. Not retrying.",
                        response.status_code,
                        response.text[:200],
                    )
                    return

                # 5xx — server error, retry
                logger.warning(
                    "InvokeLens ingest server error (HTTP %d), attempt %d/%d",
                    response.status_code,
                    attempt + 1,
                    MAX_RETRIES + 1,
                )
            except Exception as exc:
                # Network error, timeout, DNS failure, etc.
                exc_type = type(exc).__name__
                logger.warning(
                    "InvokeLens ingest %s, attempt %d/%d",
                    exc_type,
                    attempt + 1,
                    MAX_RETRIES + 1,
                )

            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER

        logger.error(
            "InvokeLens ingest failed after %d attempts. Dropping %d events.",
            MAX_RETRIES + 1,
            len(batch),
        )

    def _flush_eventbridge(self, batch: list[str]):
        """Send batch via EventBridge."""
        import boto3

        try:
            client = boto3.client("events")
            entries = [
                {
                    "Source": "invokelens.sdk",
                    "DetailType": "InvocationTelemetry",
                    "Detail": event_json,
                    "EventBusName": self.event_bus_name or "invokelens-bus",
                }
                for event_json in batch
            ]
            response = client.put_events(Entries=entries)
            failed = response.get("FailedEntryCount", 0)
            if failed > 0:
                logger.warning(
                    "InvokeLens EventBridge: %d/%d entries failed",
                    failed,
                    len(entries),
                )
        except Exception:
            logger.warning(
                "InvokeLens EventBridge delivery failed",
                exc_info=True,
            )

    def shutdown(self):
        """Flush remaining events and stop the worker thread."""
        self._shutdown.set()
        self._worker.join(timeout=10)
