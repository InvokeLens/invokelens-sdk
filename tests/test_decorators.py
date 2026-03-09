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


def test_bedrock_agent_id_auto_resolves_model():
    """bedrock_agent_id triggers GetAgent call to resolve model_id."""
    transport = MagicMock(spec=EventTransport)

    mock_bedrock_client = MagicMock()
    mock_bedrock_client.get_agent.return_value = {
        "agent": {
            "agentId": "ABCDEFGHIJ",
            "agentName": "Test Agent",
            "foundationModel": "amazon.nova-micro-v1:0",
        }
    }

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        agent_name="Test Bot",
        api_key="test-key",
        sdk_version="0.1.0",
        bedrock_agent_id="ABCDEFGHIJ",
        bedrock_region="us-east-1",
    )

    mock_session = MagicMock()
    mock_session.client.return_value = mock_bedrock_client

    with patch("boto3.Session", return_value=mock_session):

        @decorator
        def my_function():
            return "Hello from the agent"

        result = my_function()

    assert result == "Hello from the agent"
    transport.send.assert_called_once()

    event = transport.send.call_args[0][0]
    assert event.model_id == "amazon.nova-micro-v1:0"
    mock_bedrock_client.get_agent.assert_called_once_with(agentId="ABCDEFGHIJ")


def test_bedrock_agent_id_caches_model():
    """GetAgent is only called once — second invocation uses cached result."""
    transport = MagicMock(spec=EventTransport)

    mock_bedrock_client = MagicMock()
    mock_bedrock_client.get_agent.return_value = {
        "agent": {"foundationModel": "anthropic.claude-3-haiku"}
    }

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        api_key="test-key",
        sdk_version="0.1.0",
        bedrock_agent_id="XYZXYZXYZ",
    )

    mock_session = MagicMock()
    mock_session.client.return_value = mock_bedrock_client

    with patch("boto3.Session", return_value=mock_session):

        @decorator
        def my_function():
            return "ok"

        my_function()
        my_function()

    # GetAgent should only have been called once
    mock_bedrock_client.get_agent.assert_called_once()
    assert transport.send.call_count == 2

    # Both events should have the resolved model
    for call in transport.send.call_args_list:
        assert call[0][0].model_id == "anthropic.claude-3-haiku"


def test_bedrock_agent_id_fails_open():
    """If GetAgent fails, invocation proceeds with 'unknown' model."""
    transport = MagicMock(spec=EventTransport)

    mock_bedrock_client = MagicMock()
    mock_bedrock_client.get_agent.side_effect = Exception("Access denied")

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        api_key="test-key",
        sdk_version="0.1.0",
        bedrock_agent_id="NOPERM",
    )

    mock_session = MagicMock()
    mock_session.client.return_value = mock_bedrock_client

    with patch("boto3.Session", return_value=mock_session):

        @decorator
        def my_function():
            return "ok"

        result = my_function()

    assert result == "ok"
    event = transport.send.call_args[0][0]
    assert event.model_id == "unknown"


def test_explicit_model_id_takes_precedence_over_bedrock_agent_id():
    """If both model_id and bedrock_agent_id are provided, explicit wins."""
    transport = MagicMock(spec=EventTransport)

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        model_id="my-custom-model",
        api_key="test-key",
        sdk_version="0.1.0",
        bedrock_agent_id="ABCDEFGHIJ",
    )

    @decorator
    def my_function():
        return "ok"

    result = my_function()

    assert result == "ok"
    event = transport.send.call_args[0][0]
    # Explicit model_id is set, so bedrock_agent_id resolution shouldn't override
    assert event.model_id == "my-custom-model"


def test_unknown_model_logs_warning(caplog):
    """When model_id falls back to 'unknown', a warning is logged."""
    import logging

    transport = MagicMock(spec=EventTransport)

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        api_key="test-key",
        sdk_version="0.1.0",
        # No model_id, no bedrock_agent_id
    )

    @decorator
    def my_function():
        return "ok"

    with caplog.at_level(logging.WARNING, logger="invokelens_sdk.decorators"):
        my_function()

    assert any("model_id is 'unknown'" in msg for msg in caplog.messages)
    assert any("bedrock_agent_id" in msg for msg in caplog.messages)


def test_session_id_included_in_event():
    """session_id is forwarded to the telemetry event."""
    transport = MagicMock(spec=EventTransport)

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        model_id="anthropic.claude-3-haiku",
        api_key="test-key",
        sdk_version="0.2.1",
        session_id="sess-abc-123",
    )

    @decorator
    def my_function():
        return "hello"

    my_function()

    transport.send.assert_called_once()
    event = transport.send.call_args[0][0]
    assert event.session_id == "sess-abc-123"


def test_session_id_none_by_default():
    """When session_id is not provided, it defaults to None."""
    transport = MagicMock(spec=EventTransport)

    decorator = ObserveDecorator(
        transport=transport,
        agent_id="test-agent",
        model_id="anthropic.claude-3-haiku",
        api_key="test-key",
        sdk_version="0.2.1",
    )

    @decorator
    def my_function():
        return "hello"

    my_function()

    event = transport.send.call_args[0][0]
    assert event.session_id is None
