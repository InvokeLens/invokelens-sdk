"""Tests for the tracing module: Span, TraceContext, _truncate."""

import time
from invokelens_sdk.tracing import (
    Span,
    TraceContext,
    _truncate,
    MAX_SPANS_PER_TRACE,
    MAX_IO_LENGTH,
)


# ── _truncate ────────────────────────────────────────────


def test_truncate_none():
    assert _truncate(None) is None


def test_truncate_short_string():
    assert _truncate("hello") == "hello"


def test_truncate_exact_limit():
    s = "x" * MAX_IO_LENGTH
    assert _truncate(s) == s


def test_truncate_long_string():
    s = "x" * (MAX_IO_LENGTH + 100)
    result = _truncate(s)
    assert len(result) == MAX_IO_LENGTH
    assert result.endswith("...[truncated]")


def test_truncate_custom_limit():
    s = "a" * 100
    result = _truncate(s, max_len=50)
    assert len(result) == 50
    assert result.endswith("...[truncated]")


# ── Span defaults ────────────────────────────────────────


def test_span_defaults():
    span = Span(name="test-span")
    assert span.name == "test-span"
    assert span.span_type == "custom"
    assert span.status == "OK"
    assert span.parent_span_id is None
    assert span.duration_ms == 0
    assert span.input_tokens == 0
    assert span.output_tokens == 0
    assert span.estimated_cost_usd == 0.0
    assert span.span_id  # auto-generated UUID


def test_span_serialization():
    span = Span(name="llm-call", span_type="llm", model_id="claude-3")
    d = span.model_dump()
    assert d["name"] == "llm-call"
    assert d["span_type"] == "llm"
    assert d["model_id"] == "claude-3"
    assert isinstance(d["span_id"], str)


# ── TraceContext basics ──────────────────────────────────


def test_start_and_end_span():
    ctx = TraceContext()
    span = ctx.start_span("test", span_type="tool")
    assert span.name == "test"
    assert span.span_type == "tool"

    ctx.end_span(span, output="result", status="OK")
    assert span.ended_at is not None
    assert span.output == "result"
    assert span.status == "OK"
    assert span.duration_ms >= 0


def test_nested_parent_tracking():
    ctx = TraceContext()
    parent = ctx.start_span("parent", span_type="chain")
    child = ctx.start_span("child", span_type="llm")

    assert child.parent_span_id == parent.span_id

    grandchild = ctx.start_span("grandchild", span_type="tool")
    assert grandchild.parent_span_id == child.span_id

    # End in reverse order
    ctx.end_span(grandchild)
    ctx.end_span(child)
    ctx.end_span(parent)

    spans = ctx.spans
    assert len(spans) == 3


def test_context_manager_success():
    ctx = TraceContext()
    with ctx.span("my-tool", span_type="tool") as s:
        s.output = "result data"

    spans = ctx.spans
    assert len(spans) == 1
    assert spans[0].name == "my-tool"
    assert spans[0].status == "OK"
    assert spans[0].output == "result data"
    assert spans[0].ended_at is not None


def test_context_manager_error():
    ctx = TraceContext()
    try:
        with ctx.span("failing", span_type="tool") as s:
            raise ValueError("boom")
    except ValueError:
        pass

    spans = ctx.spans
    assert len(spans) == 1
    assert spans[0].status == "ERROR"
    assert spans[0].error == "boom"


def test_to_dicts():
    ctx = TraceContext()
    with ctx.span("step1", span_type="llm") as s:
        s.output = "done"

    dicts = ctx.to_dicts()
    assert len(dicts) == 1
    assert isinstance(dicts[0], dict)
    assert dicts[0]["name"] == "step1"
    assert dicts[0]["span_type"] == "llm"


# ── MAX_SPANS enforcement ───────────────────────────────


def test_max_spans_per_trace():
    ctx = TraceContext()
    for i in range(MAX_SPANS_PER_TRACE + 10):
        span = ctx.start_span(f"span-{i}")
        ctx.end_span(span)

    # Only MAX_SPANS_PER_TRACE should be recorded
    assert len(ctx.spans) == MAX_SPANS_PER_TRACE


def test_max_spans_overflow_returns_detached_span():
    """Spans beyond the limit are returned but not stored."""
    ctx = TraceContext()
    for i in range(MAX_SPANS_PER_TRACE):
        span = ctx.start_span(f"span-{i}")
        ctx.end_span(span)

    overflow = ctx.start_span("overflow")
    assert overflow.name == "overflow"  # Still returns a valid span
    assert len(ctx.spans) == MAX_SPANS_PER_TRACE  # But not recorded


# ── IO truncation in spans ───────────────────────────────


def test_start_span_truncates_input():
    ctx = TraceContext()
    long_input = "x" * (MAX_IO_LENGTH + 500)
    span = ctx.start_span("test", input=long_input)
    assert len(span.input) == MAX_IO_LENGTH


def test_end_span_truncates_output():
    ctx = TraceContext()
    span = ctx.start_span("test")
    long_output = "y" * (MAX_IO_LENGTH + 500)
    ctx.end_span(span, output=long_output)
    assert len(span.output) == MAX_IO_LENGTH


# ── Cost estimation in spans ─────────────────────────────


def test_end_span_computes_cost():
    ctx = TraceContext()
    span = ctx.start_span("llm-call", span_type="llm", model_id="anthropic.claude-3-haiku")
    ctx.end_span(span, input_tokens=1000, output_tokens=500, model_id="anthropic.claude-3-haiku")
    assert span.estimated_cost_usd > 0


def test_end_span_no_cost_without_model():
    ctx = TraceContext()
    span = ctx.start_span("step")
    ctx.end_span(span, input_tokens=0, output_tokens=0)
    assert span.estimated_cost_usd == 0.0
