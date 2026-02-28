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
    endpoint_url="https://your-invokelens-api.com/v1",
)

# Decorate your Bedrock agent function
@client.observe(agent_id="my-agent", agent_name="Customer Support Bot")
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

### Optional Fields

You can enrich events with additional context:

```python
@client.observe(
    agent_id="my-agent",
    agent_name="Customer Support Bot",
    model_id="anthropic.claude-3-sonnet",  # override auto-detection
)
def invoke_agent(prompt: str):
    ...
```

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
- `boto3` >= 1.28.0
- `pydantic` >= 2.0
- `httpx` >= 0.24.0

## License

MIT License. See [LICENSE](LICENSE) for details.
