"""Telemetry event schema emitted by the SDK."""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, timezone
import uuid


class TelemetryEvent(BaseModel):
    """The canonical event emitted by the SDK."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Literal[
        "invocation.started",
        "invocation.completed",
        "invocation.failed",
    ]
    event_version: str = "1.0"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Identity
    api_key: str
    agent_id: str
    agent_name: Optional[str] = None
    invocation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Model
    model_id: str = "unknown"
    region: str = "us-east-1"

    # Timing
    started_at: str
    ended_at: Optional[str] = None
    duration_ms: int = 0

    # Tokens
    input_tokens: int = 0
    output_tokens: int = 0

    # Cost (SDK computes best-effort estimate)
    estimated_cost_usd: float = 0.0

    # Status
    status: Literal["SUCCESS", "FAILURE", "TIMEOUT"] = "SUCCESS"
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    # Tools
    tools_called: list[str] = []

    # Content (truncated for transport)
    prompt_summary: Optional[str] = None
    response_summary: Optional[str] = None

    # Prompt drift detection
    prompt_fingerprint: Optional[dict] = None

    # Trace spans
    spans: list[dict] = []

    # Custom
    tags: dict[str, str] = {}
    sdk_version: str = ""
