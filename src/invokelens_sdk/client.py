"""Main entry point for the InvokeLens SDK."""

import functools
from typing import Optional

from .decorators import ObserveDecorator
from .transport import EventTransport
from .config import SDKConfig
from .status import AgentStatusChecker
from .tracing import TraceContext
from ._version import __version__


class InvokeLensClient:
    """Client for sending agent telemetry to the InvokeLens platform.

    Usage:
        client = InvokeLensClient(api_key="il_live_abc123")

        @client.observe(agent_id="my-agent", agent_name="Support Bot")
        def ask_agent(prompt: str):
            response = bedrock.invoke_agent(...)
            return response

        # On app shutdown:
        client.shutdown()
    """

    def __init__(
        self,
        api_key: str,
        endpoint_url: Optional[str] = None,
        transport_mode: str = "http",
        event_bus_name: Optional[str] = None,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        enable_kill_switch: bool = True,
        status_check_ttl: float = 10.0,
    ):
        self.api_key = api_key
        self.config = SDKConfig(
            api_key=api_key,
            endpoint_url=endpoint_url or "https://api.invokelens.com",
            transport_mode=transport_mode,
            event_bus_name=event_bus_name,
        )
        self._transport = EventTransport(
            endpoint_url=self.config.endpoint_url,
            mode=transport_mode,
            event_bus_name=event_bus_name,
            api_key=api_key,
            batch_size=batch_size,
            flush_interval_seconds=flush_interval,
        )

        self._status_checker = None
        if enable_kill_switch:
            self._status_checker = AgentStatusChecker(
                endpoint_url=self.config.endpoint_url,
                api_key=api_key,
                ttl_seconds=status_check_ttl,
            )

    def observe(
        self,
        agent_id: str,
        agent_name: Optional[str] = None,
        model_id: Optional[str] = None,
        bedrock_agent_id: Optional[str] = None,
        bedrock_agent_alias_id: Optional[str] = None,
        bedrock_region: Optional[str] = None,
        boto3_session=None,
    ):
        """Decorator that wraps a function and emits telemetry.

        Args:
            agent_id: Unique identifier for the agent in InvokeLens.
            agent_name: Human-readable name for the agent.
            model_id: Bedrock model ID. If omitted, the SDK will try to
                auto-detect it from the response or resolve it via the
                Bedrock GetAgent API (requires ``bedrock_agent_id``).
            bedrock_agent_id: The Bedrock agent ID (e.g. ``"4SRDERSZRC"``).
                When provided, the SDK calls ``GetAgent`` once to auto-resolve
                the ``model_id``. Requires ``bedrock-agent:GetAgent`` IAM
                permission.
            bedrock_agent_alias_id: Optional Bedrock agent alias ID.
            bedrock_region: AWS region for the Bedrock GetAgent call.
                Defaults to ``AWS_DEFAULT_REGION`` or ``us-east-1``.
            boto3_session: Optional ``boto3.Session`` for the GetAgent call.
                Useful for local development with named profiles. In production
                (Lambda, ECS, EC2), this is not needed — the SDK uses the
                default IAM role credentials automatically.
        """
        return ObserveDecorator(
            transport=self._transport,
            agent_id=agent_id,
            agent_name=agent_name,
            model_id=model_id,
            api_key=self.api_key,
            sdk_version=__version__,
            status_checker=self._status_checker,
            bedrock_agent_id=bedrock_agent_id,
            bedrock_agent_alias_id=bedrock_agent_alias_id,
            bedrock_region=bedrock_region,
            boto3_session=boto3_session,
        )

    def trace_tool(self, name: Optional[str] = None, span_type: str = "tool"):
        """Decorator for tool functions that creates a span around the call.

        The decorated function must accept a 'trace' keyword argument
        (injected by @observe). If no trace context is present, the function
        runs without tracing.

        Usage:
            @client.trace_tool(name="web_search")
            def search(query: str, trace: TraceContext = None):
                return do_search(query)
        """
        def decorator(func):
            tool_name = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                trace = kwargs.get("trace")
                if not isinstance(trace, TraceContext):
                    return func(*args, **kwargs)

                with trace.span(tool_name, span_type=span_type) as s:
                    result = func(*args, **kwargs)
                    s.output = str(result)[:2000] if result is not None else None
                    return result

            return wrapper
        return decorator

    def shutdown(self):
        """Flush pending events and clean up. Call on app exit."""
        self._transport.shutdown()
