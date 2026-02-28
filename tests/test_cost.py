"""Tests for cost estimation."""

from invokelens_sdk.cost import estimate_cost, set_custom_pricing


def test_known_model_cost():
    """Test cost for a known model."""
    cost = estimate_cost("anthropic.claude-3-haiku", input_tokens=1000, output_tokens=1000)
    # haiku: $0.00025/1K input + $0.00125/1K output = $0.0015
    assert abs(cost - 0.0015) < 0.0001


def test_unknown_model_uses_default():
    """Test that unknown models use default pricing."""
    cost = estimate_cost("unknown-model", input_tokens=1000, output_tokens=1000)
    # default: $0.003/1K input + $0.015/1K output = $0.018
    assert abs(cost - 0.018) < 0.001


def test_custom_pricing_override():
    """Test custom pricing override."""
    set_custom_pricing("my-custom-model", input_per_1k=0.001, output_per_1k=0.002)
    cost = estimate_cost("my-custom-model", input_tokens=2000, output_tokens=500)
    # 2K * 0.001 + 0.5K * 0.002 = 0.002 + 0.001 = 0.003
    assert abs(cost - 0.003) < 0.0001


def test_zero_tokens():
    """Test cost with zero tokens."""
    cost = estimate_cost("anthropic.claude-3-sonnet", input_tokens=0, output_tokens=0)
    assert cost == 0.0
