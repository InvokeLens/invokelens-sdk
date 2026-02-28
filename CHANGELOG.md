# Changelog

All notable changes to the InvokeLens SDK will be documented in this file.

## [0.1.0] - 2026-02-24

### Added
- Core telemetry event schema with Pydantic validation
- `@trace_invocation` decorator for automatic Bedrock agent tracing
- HTTP and EventBridge transport backends
- Automatic model detection from Bedrock API calls
- Built-in cost estimation for Bedrock models
- Configurable batch flushing (size + interval)
- Retry logic with exponential backoff for HTTP transport
- Tool call tracking
- Error capture with truncation
- Session and user ID support
- Custom tags for metadata
- Queue-based async event processing
