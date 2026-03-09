"""Decorator for wrapping Bedrock calls with telemetry."""

import functools
import inspect
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional, Callable

from .schema import TelemetryEvent
from .transport import EventTransport
from .cost import estimate_cost
from .exceptions import AgentBlockedError, PolicyViolationError
from .status import _rate_tracker
from .tracing import TraceContext
from .fingerprint import compute_fingerprint

logger = logging.getLogger("invokelens_sdk.decorators")


class ObserveDecorator:
    """Wraps a function that calls Bedrock and emits telemetry."""

    def __init__(
        self,
        transport: EventTransport,
        agent_id: str,
        agent_name: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: str = "",
        sdk_version: str = "",
        status_checker=None,
        bedrock_agent_id: Optional[str] = None,
        bedrock_agent_alias_id: Optional[str] = None,
        bedrock_region: Optional[str] = None,
        boto3_session=None,
    ):
        self.transport = transport
        self.agent_id = agent_id
        self.agent_name = agent_name
        self._explicit_model_id = model_id
        self.model_id = model_id or "unknown"
        self.api_key = api_key
        self.sdk_version = sdk_version
        self.status_checker = status_checker
        self.bedrock_agent_id = bedrock_agent_id
        self.bedrock_agent_alias_id = bedrock_agent_alias_id
        self.bedrock_region = bedrock_region or os.environ.get(
            "AWS_DEFAULT_REGION", "us-east-1"
        )
        self._boto3_session = boto3_session
        self._resolved_model_id: Optional[str] = None
        self._model_resolve_attempted = False

    def _resolve_model_from_bedrock(self) -> Optional[str]:
        """One-time lookup of model ID from Bedrock GetAgent API. Result is cached."""
        if self._model_resolve_attempted:
            return self._resolved_model_id
        self._model_resolve_attempted = True

        if not self.bedrock_agent_id:
            return None

        try:
            import boto3

            session = self._boto3_session or boto3.Session()
            bedrock_client = session.client(
                "bedrock-agent", region_name=self.bedrock_region
            )
            response = bedrock_client.get_agent(agentId=self.bedrock_agent_id)
            agent = response.get("agent", {})
            model_id = agent.get("foundationModel")
            if model_id:
                self._resolved_model_id = model_id
                self.model_id = model_id
                logger.info(
                    "Auto-resolved model_id=%s from Bedrock agent %s",
                    model_id,
                    self.bedrock_agent_id,
                )
                return model_id
        except ImportError:
            logger.debug(
                "boto3 not available — cannot auto-resolve model from Bedrock agent"
            )
        except Exception:
            logger.debug(
                "Failed to resolve model from Bedrock agent %s",
                self.bedrock_agent_id,
                exc_info=True,
            )
        return None

    @staticmethod
    def _extract_prompt_text(
        args: tuple, kwargs: dict, sig: inspect.Signature
    ) -> Optional[str]:
        """Try to extract prompt text from function arguments."""
        # Check kwargs for common prompt parameter names
        for key in ("prompt", "input_text", "query", "message"):
            if key in kwargs and isinstance(kwargs[key], str):
                return kwargs[key]

        # Check if any positional arg maps to a prompt-like parameter
        params = list(sig.parameters.keys())
        for key in ("prompt", "input_text", "query", "message"):
            if key in params:
                idx = params.index(key)
                if idx < len(args) and isinstance(args[idx], str):
                    return args[idx]

        # Fallback: first positional string argument
        for a in args:
            if isinstance(a, str):
                return a

        return None

    def __call__(self, func: Callable) -> Callable:
        # Check at decoration time whether the function accepts a 'trace' param
        sig = inspect.signature(func)
        accepts_trace = "trace" in sig.parameters

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Pre-invocation kill switch check
            if self.status_checker is not None:
                try:
                    blocked, reason = self.status_checker.is_blocked(self.agent_id)
                    if blocked:
                        raise AgentBlockedError(self.agent_id, reason)
                except AgentBlockedError:
                    raise
                except Exception:
                    pass  # Fail open: status check error should not block invocation

            # Pre-invocation policy enforcement
            if self.status_checker is not None:
                try:
                    policies = self.status_checker.get_policies(self.agent_id)
                    violation = self._evaluate_pre_invocation_policies(policies)
                    if violation:
                        raise PolicyViolationError(
                            agent_id=self.agent_id,
                            policy_id=violation["policy_id"],
                            policy_type=violation["policy_type"],
                            message=violation["message"],
                        )
                except PolicyViolationError:
                    raise
                except Exception:
                    pass  # Fail open: policy evaluation error should not block invocation

            # Create trace context and root span
            trace = TraceContext()
            root_span = trace.start_span(
                name=func.__name__, span_type="chain"
            )

            # Inject trace if the function accepts it
            if accepts_trace:
                kwargs["trace"] = trace

            started_at = datetime.now(timezone.utc)
            start_time = time.monotonic()
            status = "SUCCESS"
            error_message = None
            error_type = None
            result = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "FAILURE"
                error_message = str(e)[:500]
                error_type = type(e).__name__
                raise
            finally:
                ended_at = datetime.now(timezone.utc)
                duration_ms = int((time.monotonic() - start_time) * 1000)

                input_tokens, output_tokens = self._extract_tokens(result)

                # Model ID resolution chain:
                # 1. Use explicit model_id if provided at decoration time
                # 2. Try extracting from response dict
                # 3. Try auto-resolving from Bedrock GetAgent API
                # 4. Fall back to "unknown" and warn
                if self._explicit_model_id:
                    model_id = self._explicit_model_id
                else:
                    model_id = self._extract_model_id(result)
                    if not model_id and self.bedrock_agent_id:
                        model_id = self._resolve_model_from_bedrock()
                    if not model_id:
                        model_id = self.model_id
                if model_id == "unknown":
                    logger.warning(
                        "model_id is 'unknown' for agent '%s'. "
                        "Pass model_id or bedrock_agent_id to @client.observe() "
                        "for accurate model tracking and cost estimation.",
                        self.agent_id,
                    )

                # End root span
                trace.end_span(
                    root_span,
                    status="ERROR" if status != "SUCCESS" else "OK",
                    error=error_message,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_id=model_id,
                )

                # Extract Bedrock trace data if present
                self._extract_bedrock_trace(result, trace)

                # Collect tool names from spans
                tool_names = [
                    s.name for s in trace.spans if s.span_type == "tool"
                ]

                # Prompt fingerprinting for drift detection
                prompt_text = self._extract_prompt_text(args, kwargs, sig)
                prompt_fingerprint = None
                prompt_summary = None
                if prompt_text:
                    try:
                        prompt_fingerprint = compute_fingerprint(prompt_text)
                        prompt_summary = prompt_text[:500]
                    except Exception:
                        pass

                event = TelemetryEvent(
                    event_type=f"invocation.{'completed' if status == 'SUCCESS' else 'failed'}",
                    api_key=self.api_key,
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    model_id=model_id,
                    region=self._detect_region(),
                    started_at=started_at.isoformat(),
                    ended_at=ended_at.isoformat(),
                    duration_ms=duration_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    estimated_cost_usd=estimate_cost(
                        model_id, input_tokens, output_tokens
                    ),
                    status=status,
                    error_message=error_message,
                    error_type=error_type,
                    tools_called=tool_names,
                    prompt_summary=prompt_summary,
                    prompt_fingerprint=prompt_fingerprint,
                    spans=trace.to_dicts(),
                    sdk_version=self.sdk_version,
                )

                # Record invocation for rate limiting
                try:
                    _rate_tracker.record_invocation(self.agent_id)
                except Exception:
                    pass

                # Fire-and-forget: never let telemetry crash the user's code
                try:
                    self.transport.send(event)
                except Exception:
                    pass

        return wrapper

    def _evaluate_pre_invocation_policies(
        self, policies: list[dict]
    ) -> Optional[dict]:
        """Evaluate pre-invocation policies. Returns violation dict or None."""
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        for policy in policies:
            policy_type = policy.get("policy_type", "")
            conditions = policy.get("conditions", {})
            enforcement = policy.get("enforcement", "BLOCK")

            if enforcement != "BLOCK":
                continue  # LOG-only policies don't block

            violation_msg = None

            if policy_type == "COST_CAP":
                max_cost = conditions.get("max_cost_usd", float("inf"))
                estimated = self._estimate_typical_cost()
                if estimated > max_cost:
                    violation_msg = (
                        f"Estimated invocation cost ${estimated:.4f} "
                        f"exceeds cap ${max_cost:.4f}"
                    )

            elif policy_type == "TOKEN_LIMIT":
                max_tokens = conditions.get("max_tokens", float("inf"))
                estimated_input = conditions.get("estimated_input_tokens", 0)
                if estimated_input and estimated_input > max_tokens:
                    violation_msg = (
                        f"Estimated tokens {estimated_input} exceeds limit {max_tokens}"
                    )

            elif policy_type == "RATE_LIMIT":
                max_invocations = conditions.get("max_invocations", float("inf"))
                window_minutes = conditions.get("window_minutes", 60)
                window_seconds = window_minutes * 60
                count = _rate_tracker.count_in_window(self.agent_id, window_seconds)
                if count >= max_invocations:
                    violation_msg = (
                        f"Rate limit exceeded: {count} invocations "
                        f"in last {window_minutes} minutes (limit: {max_invocations})"
                    )

            elif policy_type == "TIME_RESTRICTION":
                allowed_hours = conditions.get("allowed_hours_utc", [0, 24])
                allowed_start = allowed_hours[0] if len(allowed_hours) > 0 else 0
                allowed_end = allowed_hours[1] if len(allowed_hours) > 1 else 24
                current_hour = _dt.now(_tz.utc).hour
                if not (allowed_start <= current_hour < allowed_end):
                    violation_msg = (
                        f"Invocation outside allowed hours: "
                        f"current={current_hour}:00 UTC, "
                        f"allowed={allowed_start}:00-{allowed_end}:00 UTC"
                    )

            if violation_msg:
                return {
                    "policy_id": policy.get("policy_id", "unknown"),
                    "policy_type": policy_type,
                    "message": violation_msg,
                }

        return None

    def _estimate_typical_cost(self) -> float:
        """Rough estimate of a typical invocation cost for this model."""
        # Assume a moderate invocation: 500 input, 200 output tokens
        return estimate_cost(self.model_id, 500, 200)

    def _extract_bedrock_trace(self, response, trace: TraceContext) -> None:
        """Best-effort extraction of Bedrock InvokeAgent trace data into spans."""
        if response is None or not isinstance(response, dict):
            return
        try:
            orch_trace = response.get("trace", {}).get("orchestrationTrace", {})
            if not orch_trace:
                return

            # Model invocation steps -> llm spans
            for step in orch_trace.get("modelInvocationInput", []):
                if isinstance(step, dict):
                    with trace.span(
                        name=step.get("type", "llm_call"),
                        span_type="llm",
                        input=str(step.get("text", ""))[:2000],
                        model_id=step.get("foundationModel"),
                    ) as s:
                        s.output = str(step.get("rawResponse", {}).get("content", ""))[:2000]

            # Action group invocations -> tool spans
            for inv in orch_trace.get("invocationInput", []):
                if isinstance(inv, dict):
                    action_input = inv.get("actionGroupInvocationInput", {})
                    if action_input:
                        with trace.span(
                            name=action_input.get("actionGroupName", "action_group"),
                            span_type="tool",
                            input=str(action_input.get("apiPath", "")),
                        ) as s:
                            s.output = str(action_input.get("verb", ""))
        except Exception:
            logger.debug("Failed to extract Bedrock trace data", exc_info=True)

    def _extract_tokens(self, response) -> tuple[int, int]:
        """Best-effort extraction of token counts from Bedrock responses."""
        if response is None:
            return 0, 0
        try:
            if isinstance(response, dict):
                # Bedrock InvokeModel response
                usage = response.get("usage", {})
                if usage:
                    return (
                        usage.get("inputTokens", usage.get("input_tokens", 0)),
                        usage.get("outputTokens", usage.get("output_tokens", 0)),
                    )
                # Check ResponseMetadata
                metadata = response.get("ResponseMetadata", {})
                usage = metadata.get("usage", {})
                if usage:
                    return (
                        usage.get("inputTokens", 0),
                        usage.get("outputTokens", 0),
                    )
        except Exception:
            pass
        return 0, 0

    def _extract_model_id(self, response) -> Optional[str]:
        """Try to extract model ID from a Bedrock response."""
        if response is None:
            return None
        try:
            if isinstance(response, dict):
                return response.get("modelId") or response.get("model_id")
        except Exception:
            pass
        return None

    def _detect_region(self) -> str:
        return os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
