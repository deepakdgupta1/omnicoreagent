# OmniServe Cookbook

Production-ready API server examples for OmniCoreAgent and DeepAgent.

## Examples

| Example | Description |
|---------|-------------|
| [python_api.py](./python_api.py) | Full Python API with all config options |
| [cli_agent.py](./cli_agent.py) | Agent file for CLI deployment |
| [resilience_patterns.py](./resilience_patterns.py) | Retry, circuit breaker, metrics |

---

## Quick Start

### Option 1: Python API (Direct)
```bash
uv run python cookbook/omniserve/python_api.py
```

### Option 2: CLI (Zero-code deployment)
```bash
# Quickstart with defaults
omniserve quickstart --provider gemini --model gemini-2.0-flash

# Run with your agent file
omniserve run --agent cookbook/omniserve/cli_agent.py --port 8000 --auth-token secret --rate-limit 100
```

---

## Environment Variables

All config via `OMNISERVE_*` prefix. **Environment variables ALWAYS override code values.**

```python
# Code says port=8000
config = OmniServeConfig(port=8000)

# But OMNISERVE_PORT=9000 in .env
# Result: port will be 9000 (env wins!)
```

| Variable | Description | Default |
|----------|-------------|---------|
| `OMNISERVE_HOST` | Host to bind to | `0.0.0.0` |
| `OMNISERVE_PORT` | Port to bind to | `8000` |
| `OMNISERVE_WORKERS` | Worker processes | `1` |
| `OMNISERVE_API_PREFIX` | API path prefix | `""` |
| `OMNISERVE_ENABLE_DOCS` | Swagger UI | `true` |
| `OMNISERVE_ENABLE_REDOC` | ReDoc | `true` |
| `OMNISERVE_CORS_ENABLED` | Enable CORS | `true` |
| `OMNISERVE_CORS_ORIGINS` | Allowed origins | `*` |
| `OMNISERVE_AUTH_ENABLED` | Enable auth | `false` |
| `OMNISERVE_AUTH_TOKEN` | Bearer token | — |
| `OMNISERVE_RATE_LIMIT_ENABLED` | Rate limiting | `false` |
| `OMNISERVE_RATE_LIMIT_REQUESTS` | Requests/window | `100` |
| `OMNISERVE_RATE_LIMIT_WINDOW` | Window seconds | `60` |
| `OMNISERVE_REQUEST_LOGGING` | Log requests | `true` |
| `OMNISERVE_LOG_LEVEL` | Log level | `INFO` |
| `OMNISERVE_REQUEST_TIMEOUT` | Timeout seconds | `300` |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/run` | SSE streaming response |
| POST | `/run/sync` | JSON response |
| GET | `/health` | Health check |
| GET | `/ready` | Readiness check |
| GET | `/prometheus` | Prometheus metrics |
| GET | `/tools` | List available tools |
| GET | `/metrics` | Agent usage metrics |
