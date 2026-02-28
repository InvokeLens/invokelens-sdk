"""Custom exceptions for the InvokeLens SDK."""


class AgentBlockedError(Exception):
    """Raised when an agent is blocked by the kill switch.

    The SDK raises this BEFORE making any Bedrock call, so there is
    zero cost incurred.
    """

    def __init__(self, agent_id: str, reason: str | None = None):
        self.agent_id = agent_id
        self.reason = reason or "Agent is blocked"
        super().__init__(
            f"Agent '{agent_id}' is blocked: {self.reason}. "
            f"Unblock via the InvokeLens dashboard."
        )


class PolicyViolationError(Exception):
    """Raised when a pre-invocation guardrail policy check fails.

    Contains details about which policy was violated so the caller
    can make informed decisions (retry, escalate, log, etc.).
    """

    def __init__(
        self,
        agent_id: str,
        policy_id: str,
        policy_type: str,
        message: str,
    ):
        self.agent_id = agent_id
        self.policy_id = policy_id
        self.policy_type = policy_type
        self.violation_message = message
        super().__init__(
            f"Policy violation for agent '{agent_id}': "
            f"[{policy_type}] {message} (policy_id={policy_id})"
        )
