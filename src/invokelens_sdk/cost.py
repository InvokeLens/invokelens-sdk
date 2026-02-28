"""Bedrock model pricing lookup for cost estimation."""

# Prices in USD per 1,000 tokens (approximate on-demand rates)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "anthropic.claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "anthropic.claude-3-opus": {"input": 0.015, "output": 0.075},
    "amazon.titan-text-lite-v1": {"input": 0.0003, "output": 0.0004},
    "amazon.titan-text-express-v1": {"input": 0.0008, "output": 0.0016},
    "meta.llama3-70b-instruct-v1": {"input": 0.00265, "output": 0.0035},
    "meta.llama3-8b-instruct-v1": {"input": 0.0003, "output": 0.0006},
    "mistral.mistral-large": {"input": 0.004, "output": 0.012},
    "mistral.mistral-small": {"input": 0.001, "output": 0.003},
    "cohere.command-r-plus-v1": {"input": 0.003, "output": 0.015},
    "_default": {"input": 0.003, "output": 0.015},
}

_custom_pricing: dict[str, dict[str, float]] = {}


def set_custom_pricing(model_id: str, input_per_1k: float, output_per_1k: float):
    """Override pricing for a specific model."""
    _custom_pricing[model_id] = {"input": input_per_1k, "output": output_per_1k}


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a given invocation."""
    pricing = (
        _custom_pricing.get(model_id)
        or MODEL_PRICING.get(model_id)
        or MODEL_PRICING["_default"]
    )
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return round(input_cost + output_cost, 8)
