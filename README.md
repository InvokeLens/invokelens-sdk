# InvokeLens SDK

**Observability & Guardrails for AWS Bedrock Agents**

InvokeLens captures telemetry from your AWS Bedrock agent invocations — cost, latency, token usage, tool calls, and errors — and sends it to the InvokeLens platform for monitoring, alerting, and analysis.

## Installation

```bash
pip install invokelens-sdk
```

## Quick Start

```python
from invokelens_sdk import InvokeLensClient

# Initialize the client
client = InvokeLensClient(
    api_key="your-api-key",
    endpoint_url="https://api.invokelens.com/v1",
)

# Decorate your Bedrock agent function
@client.observe(
    agent_id="my-agent",
    agent_name="Customer Support Bot",
    bedrock_agent_id="ABCDEFGHIJ",  # auto-resolves model ID
)
def invoke_agent(prompt: str):
    import boto3
    bedrock = boto3.client("bedrock-agent-runtime")
    response = bedrock.invoke_agent(
        agentId="ABCDEFGHIJ",
        agentAliasId="TSTALIASID",
        sessionId="session-123",
        inputText=prompt,
    )
    return response

# Call your function as normal — telemetry is captured automatically
result = invoke_agent("What is the status of order #1234?")

# Flush remaining events on shutdown
client.shutdown()
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | *(required)* | Your InvokeLens API key |
| `endpoint_url` | `https://api.invokelens.com` | InvokeLens ingest endpoint URL |
| `transport_mode` | `"http"` | Transport backend: `"http"` or `"eventbridge"` |
| `event_bus_name` | `None` | EventBridge bus name (required if transport_mode is `"eventbridge"`) |
| `batch_size` | `10` | Number of events per batch flush |
| `flush_interval` | `5.0` | Seconds between automatic flushes |

## What Gets Captured

The `@client.observe()` decorator automatically captures:

- **Timing** — invocation start, end, and duration
- **Token usage** — input and output token counts (auto-detected from Bedrock response)
- **Model ID** — which Bedrock model was used (auto-detected)
- **Cost estimate** — computed from token usage and model pricing
- **Status** — SUCCESS, FAILURE, or TIMEOUT
- **Errors** — exception type and message (truncated to 500 chars)
- **Tool calls** — names of tools invoked during execution

### Model Detection

The SDK automatically resolves the model ID using this chain:

1. **Response extraction** — reads `modelId` from the Bedrock response dict (works for `invoke_model`)
2. **Bedrock GetAgent API** — if you provide `bedrock_agent_id`, the SDK calls `GetAgent` once and caches the model (works for `invoke_agent`)
3. **Explicit override** — you can always pass `model_id` directly
4. **Warning** — if none of the above produce a model ID, the SDK logs a warning

**Recommended for `invoke_agent` users** — pass your Bedrock agent ID and the model is resolved automatically:

```python
@client.observe(
    agent_id="my-agent",
    agent_name="Customer Support Bot",
    bedrock_agent_id="ABCDEFGHIJ",   # SDK auto-resolves model via GetAgent
)
def invoke_agent(prompt: str):
    ...
```

Or set the model explicitly:

```python
@client.observe(
    agent_id="my-agent",
    agent_name="Customer Support Bot",
    model_id="anthropic.claude-3-sonnet",  # manual override
)
def invoke_agent(prompt: str):
    ...
```

### `@client.observe()` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agent_id` | *(required)* | Unique identifier for the agent in InvokeLens |
| `agent_name` | `None` | Human-readable name for the agent |
| `model_id` | auto-detected | Bedrock model ID (e.g. `"anthropic.claude-3-sonnet"`) |
| `bedrock_agent_id` | `None` | Bedrock agent ID — enables auto model resolution via `GetAgent` |
| `bedrock_agent_alias_id` | `None` | Bedrock agent alias ID |
| `bedrock_region` | `AWS_DEFAULT_REGION` | AWS region for the `GetAgent` call |

## Cost Estimation

The SDK includes built-in pricing for common Bedrock models:

```python
from invokelens_sdk import estimate_cost

cost = estimate_cost(
    model_id="anthropic.claude-3-sonnet",
    input_tokens=1000,
    output_tokens=500,
)
print(f"Estimated cost: ${cost:.4f}")
```

## Transport Modes

### HTTP (Default)

Sends batched events to the InvokeLens API via HTTPS. Includes automatic retry with exponential backoff (3 attempts).

### EventBridge

Publishes events to an Amazon EventBridge bus. Useful for AWS-native architectures where you want to process events through EventBridge rules.

```python
client = InvokeLensClient(
    api_key="your-api-key",
    transport_mode="eventbridge",
    event_bus_name="invokelens-events",
)
```

## Requirements

- Python 3.11+
- `pydantic` >= 2.0
- `httpx` >= 0.24.0
- `boto3` >= 1.28.0 *(optional — needed for EventBridge transport and `bedrock_agent_id` auto-resolution)*

## License

MIT License. See [LICENSE](LICENSE) for details.
