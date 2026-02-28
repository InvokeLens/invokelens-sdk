"""Tests for the @observe decorator."""

import pytest
from unittest.mock import MagicMock, patch
from invokelens_sdk.decorators import ObserveDecorator
from invokelens_sdk.transport import EventTransport
from invokelens_sdk.tracing import TraceContext
from invokelens_sdk.exceptions import AgentBlockedError


def test_observe_captures_success():
    """Test that the decorator captures a successful invocation."""
    transport = MagicMock(spec=EventTransport)

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        agent_name="Test Bot",
        model_id="anthropic.claude-3-haiku",
        api_key="test-key",
        sdk_version="0.1.0",
    )

    @decorator
    def my_function():
        return {"usage": {"inputTokens": 100, "outputTokens": 200}}

    result = my_function()

    assert result == {"usage": {"inputTokens": 100, "outputTokens": 200}}
    transport.send.assert_called_once()

    event = transport.send.call_args[0][0]
    assert event.agent_id == "test-agent"
    assert event.agent_name == "Test Bot"
    assert event.status == "SUCCESS"
    assert event.input_tokens == 100
    assert event.output_tokens == 200
    assert event.estimated_cost_usd > 0
    assert event.duration_ms >= 0
    assert event.sdk_version == "0.1.0"


def test_observe_captures_failure():
    """Test that the decorator captures a failed invocation."""
    transport = MagicMock(spec=EventTransport)

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        model_id="anthropic.claude-3-sonnet",
        api_key="test-key",
        sdk_version="0.1.0",
    )

    @decorator
    def failing_function():
        raise ValueError("Something went wrong")

    try:
        failing_function()
    except ValueError:
        pass

    transport.send.assert_called_once()
    event = transport.send.call_args[0][0]
    assert event.status == "FAILURE"
    assert event.error_type == "ValueError"
    assert "Something went wrong" in event.error_message


def test_observe_never_crashes_on_transport_error():
    """Test that transport errors don't bubble up to user code."""
    transport = MagicMock(spec=EventTransport)
    transport.send.side_effect = Exception("Network error")

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        model_id="unknown",
        api_key="test-key",
        sdk_version="0.1.0",
    )

    @decorator
    def my_function():
        return "ok"

    # Should not raise
    result = my_function()
    assert result == "ok"


def test_observe_preserves_function_metadata():
    """Test that functools.wraps preserves function name and docstring."""
    transport = MagicMock(spec=EventTransport)

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        model_id="unknown",
        api_key="test-key",
        sdk_version="0.1.0",
    )

    @decorator
    def my_documented_function():
        """This is my docstring."""
        return 42

    assert my_documented_function.__name__ == "my_documented_function"
    assert my_documented_function.__doc__ == "This is my docstring."


def test_blocked_agent_raises_error():
    """A blocked agent should raise AgentBlockedError before calling func."""
    transport = MagicMock(spec=EventTransport)
    status_checker = MagicMock()
    status_checker.is_blocked.return_value = (True, "manual")

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="blocked-agent",
        model_id="unknown",
        api_key="test-key",
        sdk_version="0.1.0",
        status_checker=status_checker,
    )

    call_count = 0

    @decorator
    def my_function():
        nonlocal call_count
        call_count += 1
        return "should not reach here"

    with pytest.raises(AgentBlockedError) as exc_info:
        my_function()

    assert "blocked-agent" in str(exc_info.value)
    assert call_count == 0  # Function was never called
    transport.send.assert_not_called()  # No telemetry — blocked before try/finally


def test_status_check_failure_allows_invocation():
    """Status check errors should fail open — invocation proceeds."""
    transport = MagicMock(spec=EventTransport)
    status_checker = MagicMock()
    status_checker.is_blocked.side_effect = Exception("Network timeout")

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        model_id="unknown",
        api_key="test-key",
        sdk_version="0.1.0",
        status_checker=status_checker,
    )

    @decorator
    def my_function():
        return "ok"

    result = my_function()
    assert result == "ok"


def test_no_status_checker_backward_compat():
    """Without status_checker, decorator works as before — no pre-check."""
    transport = MagicMock(spec=EventTransport)

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        model_id="unknown",
        api_key="test-key",
        sdk_version="0.1.0",
        # No status_checker
    )

    @decorator
    def my_function():
        return "ok"

    result = my_function()
    assert result == "ok"
    transport.send.assert_called_once()


def test_observe_injects_trace_context():
    """Function with 'trace' param should receive a TraceContext and spans attached to event."""
    transport = MagicMock(spec=EventTransport)
    received_trace = None

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        model_id="anthropic.claude-3-haiku",
        api_key="test-key",
        sdk_version="0.1.0",
    )

    @decorator
    def my_function(query: str, trace: TraceContext = None):
        nonlocal received_trace
        received_trace = trace
        with trace.span("search", span_type="tool") as s:
            s.output = "found it"
        return {"usage": {"inputTokens": 10, "outputTokens": 20}}

    result = my_function("hello")

    assert received_trace is not None
    assert isinstance(received_trace, TraceContext)

    # Event should have spans
    event = transport.send.call_args[0][0]
    assert len(event.spans) >= 2  # root span + tool span
    assert event.tools_called == ["search"]


def test_observe_works_without_trace_parameter():
    """Function without 'trace' param still works — gets root span only."""
    transport = MagicMock(spec=EventTransport)

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        model_id="unknown",
        api_key="test-key",
        sdk_version="0.1.0",
    )

    @decorator
    def my_function():
        return "ok"

    result = my_function()
    assert result == "ok"

    event = transport.send.call_args[0][0]
    # Should have at least the root span
    assert len(event.spans) >= 1
    assert event.spans[0]["name"] == "my_function"
    assert event.spans[0]["span_type"] == "chain"


def test_observe_captures_prompt_from_kwargs():
    """Function with 'prompt' param should have fingerprint in event."""
    transport = MagicMock(spec=EventTransport)

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        model_id="unknown",
        api_key="test-key",
        sdk_version="0.1.0",
    )

    @decorator
    def my_function(prompt: str):
        return "ok"

    my_function(prompt="Hello, how are you?")

    event = transport.send.call_args[0][0]
    assert event.prompt_fingerprint is not None
    assert event.prompt_fingerprint["char_count"] == 19
    assert event.prompt_fingerprint["word_count"] == 4
    assert event.prompt_fingerprint["prompt_hash"]  # non-empty
    assert event.prompt_summary == "Hello, how are you?"


def test_observe_captures_prompt_from_positional():
    """First positional string arg should be captured as prompt."""
    transport = MagicMock(spec=EventTransport)

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        model_id="unknown",
        api_key="test-key",
        sdk_version="0.1.0",
    )

    @decorator
    def my_function(text: str, count: int = 5):
        return "ok"

    my_function("What is the meaning of life?")

    event = transport.send.call_args[0][0]
    assert event.prompt_fingerprint is not None
    assert event.prompt_summary == "What is the meaning of life?"


def test_observe_no_prompt_backward_compat():
    """Function without prompt → no fingerprint, no crash."""
    transport = MagicMock(spec=EventTransport)

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        model_id="unknown",
        api_key="test-key",
        sdk_version="0.1.0",
    )

    @decorator
    def my_function(x: int, y: int):
        return x + y

    result = my_function(3, 4)
    assert result == 7

    event = transport.send.call_args[0][0]
    assert event.prompt_fingerprint is None
    assert event.prompt_summary is None
