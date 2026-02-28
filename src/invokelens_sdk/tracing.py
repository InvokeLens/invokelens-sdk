"""Span model and trace context for invocation-level tracing."""

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .cost import estimate_cost

logger = logging.getLogger("invokelens_sdk.tracing")

MAX_SPANS_PER_TRACE = 100
MAX_IO_LENGTH = 2000

SpanType = Literal["llm", "tool", "chain", "retrieval", "guardrail", "custom"]


def _truncate(value: Optional[str], max_len: int = MAX_IO_LENGTH) -> Optional[str]:
    """Truncate a string to max_len, appending '...[truncated]' if needed."""
    if value is None:
        return None
    value = str(value)
    if len(value) <= max_len:
        return value
    return value[: max_len - 14] + "...[truncated]"


class Span(BaseModel):
    """A single span representing one step in an agent invocation."""

    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    span_type: SpanType = "custom"
    name: str = ""
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ended_at: Optional[str] = None
    duration_ms: int = 0

    input: Optional[str] = None
    output: Optional[str] = None

    status: Literal["OK", "ERROR"] = "OK"
    error: Optional[str] = None

    model_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class TraceContext:
    """Collects spans for a single invocation.

    Thread-safe: multiple spans can be started/ended concurrently.
    The active stack tracks parent-child nesting per thread of execution,
    but since most agent invocations are single-threaded, we use a simple
    shared stack protected by a lock.
    """

    def __init__(self) -> None:
        self._spans: list[Span] = []
        self._active_stack: list[str] = []
        self._lock = threading.Lock()
        self._start_mono = time.monotonic()

    def start_span(
        self,
        name: str,
        span_type: SpanType = "custom",
        input: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Span:
        """Create and register a new span. Parents to the current top-of-stack."""
        with self._lock:
            if len(self._spans) >= MAX_SPANS_PER_TRACE:
                logger.debug("MAX_SPANS_PER_TRACE (%d) reached, dropping span %s",
                             MAX_SPANS_PER_TRACE, name)
                # Return a detached span that won't be recorded
                return Span(name=name, span_type=span_type,
                            input=_truncate(input), model_id=model_id)

            parent_id = self._active_stack[-1] if self._active_stack else None
            span = Span(
                name=name,
                span_type=span_type,
                parent_span_id=parent_id,
                input=_truncate(input),
                model_id=model_id,
            )
            self._spans.append(span)
            self._active_stack.append(span.span_id)
        return span

    def end_span(
        self,
        span: Span,
        output: Optional[str] = None,
        status: Literal["OK", "ERROR"] = "OK",
        error: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model_id: Optional[str] = None,
    ) -> None:
        """Finalize a span with output, status, tokens, and cost."""
        now = datetime.now(timezone.utc)
        span.ended_at = now.isoformat()
        span.output = _truncate(output)
        span.status = status
        span.error = error
        span.input_tokens = input_tokens
        span.output_tokens = output_tokens

        if model_id:
            span.model_id = model_id

        # Compute duration
        try:
            started = datetime.fromisoformat(span.started_at)
            span.duration_ms = int((now - started).total_seconds() * 1000)
        except (ValueError, TypeError):
            span.duration_ms = 0

        # Compute cost if we have a model and tokens
        if span.model_id and (input_tokens or output_tokens):
            span.estimated_cost_usd = estimate_cost(
                span.model_id, input_tokens, output_tokens
            )

        with self._lock:
            if self._active_stack and self._active_stack[-1] == span.span_id:
                self._active_stack.pop()

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType = "custom",
        input: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        """Context manager for creating and auto-ending a span.

        Usage:
            with trace.span("call_llm", span_type="llm") as s:
                result = llm.invoke(prompt)
                s.output = result
                s.input_tokens = 100
                s.output_tokens = 50
        """
        s = self.start_span(name, span_type=span_type, input=input, model_id=model_id)
        try:
            yield s
        except Exception as exc:
            self.end_span(
                s,
                output=None,
                status="ERROR",
                error=str(exc),
                input_tokens=s.input_tokens,
                output_tokens=s.output_tokens,
                model_id=s.model_id,
            )
            raise
        else:
            self.end_span(
                s,
                output=s.output,
                status=s.status,
                error=s.error,
                input_tokens=s.input_tokens,
                output_tokens=s.output_tokens,
                model_id=s.model_id,
            )

    def to_dicts(self) -> list[dict]:
        """Serialize all spans to dicts for transport."""
        with self._lock:
            return [span.model_dump() for span in self._spans]

    @property
    def spans(self) -> list[Span]:
        """Return a copy of the spans list."""
        with self._lock:
            return list(self._spans)
