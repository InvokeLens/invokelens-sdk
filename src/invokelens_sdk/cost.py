"""Bedrock model pricing — pulls live rates from the AWS Pricing API.

On first use, queries the AWS Pricing API for on-demand Bedrock token prices
in the customer's region. Results are cached for the lifetime of the process.
Falls back to hardcoded defaults only if the API call fails.
"""

import logging
from typing import Optional

logger = logging.getLogger("invokelens_sdk.cost")

# Hardcoded fallback — only used when the AWS Pricing API is unreachable.
_FALLBACK_PRICING: dict[str, dict[str, float]] = {
    "amazon.nova-micro-v1:0": {"input": 0.000035, "output": 0.00014},
    "amazon.nova-lite-v1:0": {"input": 0.00006, "output": 0.00024},
    "amazon.nova-pro-v1:0": {"input": 0.0008, "output": 0.0032},
    "anthropic.claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-5-haiku": {"input": 0.0008, "output": 0.004},
    "anthropic.claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "anthropic.claude-3-opus": {"input": 0.015, "output": 0.075},
    "amazon.titan-text-lite-v1": {"input": 0.0003, "output": 0.0004},
    "amazon.titan-text-express-v1": {"input": 0.0008, "output": 0.0016},
    "amazon.titan-text-premier-v1:0": {"input": 0.0005, "output": 0.0015},
    "meta.llama3-70b-instruct-v1": {"input": 0.00265, "output": 0.0035},
    "meta.llama3-8b-instruct-v1": {"input": 0.0003, "output": 0.0006},
    "mistral.mistral-large": {"input": 0.004, "output": 0.012},
    "mistral.mistral-small": {"input": 0.001, "output": 0.003},
    "cohere.command-r-plus-v1": {"input": 0.003, "output": 0.015},
    "_default": {"input": 0.003, "output": 0.015},
}

# Live pricing cache — populated on first call to _ensure_pricing_loaded()
_live_pricing: dict[str, dict[str, float]] = {}
_pricing_loaded = False
_custom_pricing: dict[str, dict[str, float]] = {}


def set_custom_pricing(model_id: str, input_per_1k: float, output_per_1k: float):
    """Override pricing for a specific model."""
    _custom_pricing[model_id] = {"input": input_per_1k, "output": output_per_1k}


def _fetch_bedrock_pricing(region: str = "us-east-1") -> dict[str, dict[str, float]]:
    """Fetch on-demand Bedrock token pricing from the AWS Pricing API.

    Returns a dict mapping usagetype prefixes to {input, output} rates per 1K tokens.
    The Pricing API is always in us-east-1 but we filter by the customer's region.
    """
    import boto3

    # AWS Pricing API is only available in us-east-1 and ap-south-1
    client = boto3.client("pricing", region_name="us-east-1")

    # Map region to Pricing API location code prefix (e.g. us-east-1 → USE1)
    region_prefix_map = {
        "us-east-1": "USE1",
        "us-west-2": "USW2",
        "eu-west-1": "EU",
        "eu-central-1": "EUC1",
        "ap-northeast-1": "APN1",
        "ap-southeast-1": "APS1",
        "ap-southeast-2": "APS2",
    }
    region_prefix = region_prefix_map.get(region, "USE1")

    pricing: dict[str, dict[str, float]] = {}
    next_token: Optional[str] = None

    while True:
        kwargs = {
            "ServiceCode": "AmazonBedrock",
            "Filters": [
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
            ],
            "MaxResults": 100,
        }
        if next_token:
            kwargs["NextToken"] = next_token

        response = client.get_products(**kwargs)

        for item in response.get("PriceList", []):
            import json
            parsed = json.loads(item)
            attrs = parsed.get("product", {}).get("attributes", {})
            inf_type = attrs.get("inferenceType", "").lower()
            usagetype = attrs.get("usagetype", "")

            # Only on-demand text token pricing (input/output)
            if "token" not in inf_type or "batch" in inf_type.lower():
                continue
            if "cache" in inf_type or "storage" in inf_type:
                continue
            if "provisioned" in usagetype.lower() or "customization" in usagetype.lower():
                continue

            # Extract the model name from usagetype: USE1-NovaMicro-input-tokens → NovaMicro
            # Format: {REGION}-{ModelKey}-{type}-tokens[-suffix]
            parts = usagetype.split("-", 1)
            if len(parts) < 2:
                continue
            rest = parts[1]  # e.g. "NovaMicro-input-tokens-custom-model"

            # Get price
            price = 0.0
            terms = parsed.get("terms", {}).get("OnDemand", {})
            for term in terms.values():
                for dim in term.get("priceDimensions", {}).values():
                    price = float(dim.get("pricePerUnit", {}).get("USD", "0"))

            if price <= 0:
                continue

            # Determine if input or output
            is_input = "input" in inf_type.lower()
            is_output = "output" in inf_type.lower()

            # Extract model key — everything before -input or -output
            model_key = rest
            for separator in ["-input-token", "-output-token"]:
                if separator in rest.lower():
                    idx = rest.lower().index(separator)
                    model_key = rest[:idx]
                    break

            if model_key not in pricing:
                pricing[model_key] = {"input": 0.0, "output": 0.0}

            if is_input:
                pricing[model_key]["input"] = price
            elif is_output:
                pricing[model_key]["output"] = price

        next_token = response.get("NextToken")
        if not next_token:
            break

    return pricing


# Maps AWS Pricing API model keys to Bedrock model IDs
_MODEL_KEY_TO_ID: dict[str, str] = {
    "NovaMicro": "amazon.nova-micro-v1:0",
    "NovaMicro1.0": "amazon.nova-micro-v1:0",
    "NovaLite": "amazon.nova-lite-v1:0",
    "NovaLite1.0": "amazon.nova-lite-v1:0",
    "NovaPro": "amazon.nova-pro-v1:0",
    "NovaPro1.0": "amazon.nova-pro-v1:0",
    "Nova2.0Micro": "amazon.nova-2-micro-v1:0",
    "Nova2.0Lite": "amazon.nova-2-lite-v1:0",
    "Nova2.0Pro": "amazon.nova-2-pro-v1:0",
    "Claude3.5Sonnet": "anthropic.claude-3-5-sonnet",
    "Claude3.5Haiku": "anthropic.claude-3-5-haiku",
    "Claude3.7Sonnet": "anthropic.claude-3-7-sonnet",
    "Claude3Sonnet": "anthropic.claude-3-sonnet",
    "Claude3Haiku": "anthropic.claude-3-haiku",
    "Claude3Opus": "anthropic.claude-3-opus",
    "Claude4Sonnet": "anthropic.claude-4-sonnet",
    "Claude4Opus": "anthropic.claude-4-opus",
    "TitanTextLite": "amazon.titan-text-lite-v1",
    "TitanTextExpress": "amazon.titan-text-express-v1",
    "TitanTextPremier": "amazon.titan-text-premier-v1:0",
    "Llama3.170BInstruct": "meta.llama3-1-70b-instruct-v1:0",
    "Llama3.18BInstruct": "meta.llama3-1-8b-instruct-v1:0",
    "Llama3.1405BInstruct": "meta.llama3-1-405b-instruct-v1:0",
    "Llama370BInstruct": "meta.llama3-70b-instruct-v1",
    "Llama38BInstruct": "meta.llama3-8b-instruct-v1",
    "MistralLarge": "mistral.mistral-large",
    "MistralSmall": "mistral.mistral-small",
    "CommandRPlus": "cohere.command-r-plus-v1",
    "CommandR": "cohere.command-r-v1",
}


def _ensure_pricing_loaded(region: str = "us-east-1") -> None:
    """Load pricing from AWS Pricing API (once)."""
    global _pricing_loaded, _live_pricing
    if _pricing_loaded:
        return

    _pricing_loaded = True  # Mark as attempted even if it fails

    try:
        raw = _fetch_bedrock_pricing(region)
        for key, rates in raw.items():
            # Map the API key (e.g. "NovaMicro") to model ID (e.g. "amazon.nova-micro-v1:0")
            model_id = _MODEL_KEY_TO_ID.get(key)
            if model_id and rates["input"] > 0 and rates["output"] > 0:
                _live_pricing[model_id] = rates

            # Also store under the raw key for unmapped models
            # so prefix matching can still find them
            if rates["input"] > 0 and rates["output"] > 0:
                _live_pricing[key] = rates

        if _live_pricing:
            logger.info(
                "Loaded live Bedrock pricing for %d models from AWS Pricing API",
                len(_live_pricing),
            )
        else:
            logger.debug("AWS Pricing API returned no usable pricing data")
    except Exception:
        logger.debug("Failed to fetch live pricing, using fallback", exc_info=True)


def _lookup_pricing(model_id: str) -> dict[str, float]:
    """Find pricing for a model — checks custom, live API, then fallback."""
    # Custom overrides always win
    if model_id in _custom_pricing:
        return _custom_pricing[model_id]

    # Try live pricing (exact match)
    if model_id in _live_pricing:
        return _live_pricing[model_id]

    # Try fallback (exact match)
    if model_id in _FALLBACK_PRICING:
        return _FALLBACK_PRICING[model_id]

    # Prefix match across all sources
    all_pricing = {**_FALLBACK_PRICING, **_live_pricing, **_custom_pricing}
    best_match = ""
    for key in all_pricing:
        if key == "_default":
            continue
        if model_id.startswith(key) and len(key) > len(best_match):
            best_match = key
    if best_match:
        return all_pricing[best_match]

    return _FALLBACK_PRICING["_default"]


def estimate_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    region: str = "us-east-1",
) -> float:
    """Estimate cost in USD for a given invocation.

    On first call, fetches live pricing from the AWS Pricing API.
    Falls back to hardcoded rates if the API is unavailable.
    """
    _ensure_pricing_loaded(region)
    pricing = _lookup_pricing(model_id)
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return round(input_cost + output_cost, 8)
