"""Tests for AgentStatusChecker with TTL cache."""

import time
from unittest.mock import patch, MagicMock
from invokelens_sdk.status import AgentStatusChecker


def _make_checker(ttl=10.0):
    """Create a checker with stubbed endpoint."""
    return AgentStatusChecker(
        endpoint_url="http://localhost:3001",
        api_key="test-key",
        ttl_seconds=ttl,
    )


def test_active_agent_returns_not_blocked():
    """Active agent should return (False, None)."""
    checker = _make_checker()

    with patch.object(checker, "_fetch_status", return_value=("ACTIVE", None)):
        blocked, reason = checker.is_blocked("agent-1")

    assert blocked is False
    assert reason is None


def test_blocked_agent_returns_blocked():
    """Blocked agent should return (True, reason)."""
    checker = _make_checker()

    with patch.object(checker, "_fetch_status", return_value=("BLOCKED", "manual")):
        blocked, reason = checker.is_blocked("agent-1")

    assert blocked is True
    assert reason == "manual"


def test_cache_hit_skips_network_call():
    """Cached status should not trigger another fetch."""
    checker = _make_checker(ttl=60.0)

    with patch.object(checker, "_fetch_status", return_value=("BLOCKED", "manual")) as mock_fetch:
        checker.is_blocked("agent-1")
        checker.is_blocked("agent-1")
        checker.is_blocked("agent-1")

    assert mock_fetch.call_count == 1


def test_cache_expiry_triggers_refetch():
    """Expired cache entry should trigger a new fetch."""
    checker = _make_checker(ttl=0.05)  # 50ms TTL

    with patch.object(checker, "_fetch_status", return_value=("ACTIVE", None)) as mock_fetch:
        checker.is_blocked("agent-1")
        time.sleep(0.1)  # Wait for cache to expire
        checker.is_blocked("agent-1")

    assert mock_fetch.call_count == 2


def test_network_error_fails_open():
    """Network errors should return (False, None) — fail open."""
    checker = _make_checker()

    with patch.object(checker, "_fetch_status", side_effect=Exception("connection refused")):
        blocked, reason = checker.is_blocked("agent-1")

    assert blocked is False
    assert reason is None


def test_invalidate_forces_refetch():
    """Invalidating a cache entry should force a refetch on next call."""
    checker = _make_checker(ttl=60.0)

    with patch.object(checker, "_fetch_status", return_value=("ACTIVE", None)) as mock_fetch:
        checker.is_blocked("agent-1")
        checker.invalidate("agent-1")
        checker.is_blocked("agent-1")

    assert mock_fetch.call_count == 2


def test_clear_cache():
    """clear_cache should remove all entries."""
    checker = _make_checker(ttl=60.0)

    with patch.object(checker, "_fetch_status", return_value=("ACTIVE", None)) as mock_fetch:
        checker.is_blocked("agent-1")
        checker.is_blocked("agent-2")
        checker.clear_cache()
        checker.is_blocked("agent-1")
        checker.is_blocked("agent-2")

    assert mock_fetch.call_count == 4


def test_separate_agents_cached_independently():
    """Different agents should have independent cache entries."""
    checker = _make_checker(ttl=60.0)

    def fake_fetch(agent_id):
        if agent_id == "agent-blocked":
            return ("BLOCKED", "auto:rule-1")
        return ("ACTIVE", None)

    with patch.object(checker, "_fetch_status", side_effect=fake_fetch):
        blocked1, _ = checker.is_blocked("agent-blocked")
        blocked2, _ = checker.is_blocked("agent-ok")

    assert blocked1 is True
    assert blocked2 is False
