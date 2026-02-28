"""Agent status checker with TTL-based in-memory cache.

Extended to also cache and return guardrail policies returned by the
backend's ``GET /agents/<id>/status`` endpoint.
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("invokelens_sdk.status")

DEFAULT_STATUS_TTL_SECONDS = 10.0


class _CacheEntry:
    __slots__ = ("status", "blocked_reason", "policies", "expires_at")

    def __init__(
        self,
        status: str,
        blocked_reason: Optional[str],
        policies: list,
        ttl: float,
    ):
        self.status = status
        self.blocked_reason = blocked_reason
        self.policies = policies
        self.expires_at = time.monotonic() + ttl


class _RateLimitTracker:
    """Thread-safe in-memory invocation counter for RATE_LIMIT policy enforcement.

    Tracks timestamps of recent invocations per agent to support
    sliding-window rate limiting.
    """

    def __init__(self):
        self._counts: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def record_invocation(self, agent_id: str):
        """Record an invocation timestamp."""
        now = time.monotonic()
        with self._lock:
            if agent_id not in self._counts:
                self._counts[agent_id] = []
            self._counts[agent_id].append(now)

    def count_in_window(self, agent_id: str, window_seconds: float) -> int:
        """Count invocations within the last *window_seconds*."""
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            timestamps = self._counts.get(agent_id, [])
            # Prune old entries
            timestamps = [t for t in timestamps if t > cutoff]
            self._counts[agent_id] = timestamps
            return len(timestamps)


# Module-level singleton — shared across all decorators.
_rate_tracker = _RateLimitTracker()


class AgentStatusChecker:
    """Checks whether an agent is blocked, with in-memory TTL cache.

    Design principles:
    - Cache hits are O(1) dict lookup, ~0ms overhead.
    - Cache misses make a single HTTP GET to /agents/<id>/status.
    - On ANY network/parse error, the checker returns ACTIVE (fail open).
    - Thread-safe: multiple decorated functions can check concurrently.
    - Policies returned by the status endpoint are cached alongside status.
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        ttl_seconds: float = DEFAULT_STATUS_TTL_SECONDS,
    ):
        self._endpoint_url = endpoint_url.rstrip("/")
        self._api_key = api_key
        self._ttl = ttl_seconds
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    def is_blocked(self, agent_id: str) -> tuple[bool, Optional[str]]:
        """Check if agent is blocked.

        Returns (is_blocked, blocked_reason).
        On any error, returns (False, None) — fail open.
        """
        entry = self._cache.get(agent_id)
        if entry and time.monotonic() < entry.expires_at:
            return entry.status == "BLOCKED", entry.blocked_reason

        try:
            self._fetch_and_cache(agent_id)
        except Exception:
            logger.debug("Status check failed for %s, allowing invocation", agent_id)
            return False, None

        entry = self._cache.get(agent_id)
        if entry:
            return entry.status == "BLOCKED", entry.blocked_reason
        return False, None

    def get_policies(self, agent_id: str) -> list[dict]:
        """Return cached policies for an agent.

        Returns an empty list on cache miss or error (fail open).
        """
        entry = self._cache.get(agent_id)
        if entry and time.monotonic() < entry.expires_at:
            return entry.policies

        # Trigger a fetch to populate the cache
        try:
            self._fetch_and_cache(agent_id)
            entry = self._cache.get(agent_id)
            return entry.policies if entry else []
        except Exception:
            return []

    def _fetch_and_cache(self, agent_id: str):
        """Fetch status + policies from backend and update cache."""
        status, reason, policies = self._fetch_status(agent_id)
        with self._lock:
            self._cache[agent_id] = _CacheEntry(status, reason, policies, self._ttl)

    def _fetch_status(self, agent_id: str) -> tuple[str, Optional[str], list]:
        """Fetch agent status and policies from the backend."""
        import httpx

        response = httpx.get(
            f"{self._endpoint_url}/agents/{agent_id}/status",
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=2.0,
        )
        if response.status_code != 200:
            return "ACTIVE", None, []

        data = response.json()
        return (
            data.get("status", "ACTIVE"),
            data.get("blocked_reason"),
            data.get("policies", []),
        )

    def invalidate(self, agent_id: str):
        """Remove a cached entry."""
        with self._lock:
            self._cache.pop(agent_id, None)

    def clear_cache(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
