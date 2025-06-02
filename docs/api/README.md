# LLM Evaluation Platform API Documentation

## Overview

The LLM Evaluation Platform provides a comprehensive REST API for managing LLM traces, evaluations, experiments, and analytics. This API enables you to integrate evaluation capabilities into your LLM-powered applications.

## Base URL

```
http://localhost:8000/api
```

## Authentication

All API endpoints require authentication using a bearer token:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     http://localhost:8000/api/traces
```

## Rate Limits

The API implements rate limiting to ensure fair usage:

- **Default**: 1000 requests/minute
- **Authentication**: 10 requests/minute
- **Upload**: 100 requests/hour
- **Export**: 50 requests/hour
- **Evaluations**: 500 requests/hour

Rate limit headers are included in responses:
- `X-Rate-Limit`: Maximum requests allowed
- `X-Rate-Limit-Remaining`: Remaining requests in current window
- `X-Rate-Limit-Reset`: Unix timestamp when limit resets

## Response Format

All API responses follow a consistent format:

### Success Response
```json
{
  "data": { ... },
  "message": "Success message",
  "timestamp": "2025-01-27T00:00:00Z"
}
```

### Error Response
```json
{
  "detail": "Error description",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-01-27T00:00:00Z"
}
```

## Performance Headers

All responses include performance monitoring headers:
- `X-Process-Time`: Request processing time in seconds
- `X-Request-ID`: Unique request identifier for tracing

## Endpoints

### Traces

#### Create Trace
```http
POST /api/traces
```

Log a new LLM interaction for evaluation.

**Request Body:**
```json
{
  "user_input": "What is the capital of France?",
  "model_output": "The capital of France is Paris.",
  "model_name": "gpt-4",
  "system_prompt": "You are a helpful assistant.",
  "session_id": "session_123",
  "metadata": {
    "temperature": 0.7,
    "max_tokens": 150
  },
  "latency_ms": 1200,
  "token_count": {
    "input": 25,
    "output": 12
  },
  "cost_usd": 0.0024
}
```

**Response:**
```json
{
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Trace created successfully"
}
```

#### Get Traces
```http
GET /api/traces?limit=100&offset=0&model_name=gpt-4&session_id=session_123
```

Retrieve traces with optional filtering and pagination.

**Query Parameters:**
- `limit` (integer, 1-1000): Number of traces to return (default: 100)
- `offset` (integer): Number of traces to skip (default: 0)
- `model_name` (string): Filter by model name
- `session_id` (string): Filter by session ID

**Response:**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2025-01-27T00:00:00Z",
    "user_input": "What is the capital of France?",
    "model_output": "The capital of France is Paris.",
    "model_name": "gpt-4",
    "system_prompt": "You are a helpful assistant.",
    "session_id": "session_123",
    "metadata": { ... },
    "latency_ms": 1200,
    "token_count": { ... },
    "cost_usd": 0.0024,
    "status": "completed",
    "tags": []
  }
]
```

#### Get Single Trace
```http
GET /api/traces/{trace_id}
```

Retrieve detailed information about a specific trace.

**Response:** Same format as individual trace in the list above.

#### Search Traces
```http
POST /api/traces/search
```

Advanced trace search with complex filtering.

**Request Body:**
```json
{
  "model_name": "gpt-4",
  "session_id": "session_123",
  "tag_filters": {
    "topic": ["finance", "healthcare"],
    "scenario": ["production"]
  },
  "start_date": "2025-01-01T00:00:00Z",
  "end_date": "2025-01-27T23:59:59Z"
}
```

#### Get Trace Statistics
```http
GET /api/traces/stats/summary
```

Get summary statistics about traces.

**Response:**
```json
{
  "total_traces": 1500,
  "unique_models": 5,
  "unique_sessions": 120,
  "average_latency_ms": 850.5,
  "total_cost_usd": 12.45,
  "average_cost_usd": 0.0083
}
```

### Experiments (A/B Testing)

#### Create Experiment
```http
POST /api/experiments
```

Create a new A/B test experiment.

**Request Body:**
```json
{
  "name": "Model Comparison Test",
  "description": "Compare GPT-4 vs Claude performance",
  "variants": [
    {
      "name": "control",
      "config": { "model": "gpt-4", "temperature": 0.7 },
      "traffic_percentage": 50
    },
    {
      "name": "treatment",
      "config": { "model": "claude-3", "temperature": 0.7 },
      "traffic_percentage": 50
    }
  ],
  "metrics": [
    {
      "name": "response_quality",
      "type": "numeric",
      "description": "1-5 rating of response quality"
    }
  ],
  "target_sample_size": 1000
}
```

#### Get Experiments
```http
GET /api/experiments
```

List all experiments with their current status.

#### Get Experiment Results
```http
GET /api/experiments/{experiment_id}/results
```

Get statistical analysis of experiment results.

**Response:**
```json
{
  "experiment_id": "exp_123",
  "status": "running",
  "participants": 856,
  "variants": [
    {
      "name": "control",
      "participants": 428,
      "metrics": {
        "response_quality": {
          "mean": 4.2,
          "std": 0.8,
          "confidence_interval": [4.1, 4.3]
        }
      }
    },
    {
      "name": "treatment", 
      "participants": 428,
      "metrics": {
        "response_quality": {
          "mean": 4.5,
          "std": 0.7,
          "confidence_interval": [4.4, 4.6]
        }
      }
    }
  ],
  "statistical_tests": {
    "response_quality": {
      "test_type": "two_sample_ttest",
      "p_value": 0.003,
      "effect_size": 0.42,
      "significant": true
    }
  }
}
```

### Evaluations

#### Create Evaluation
```http
POST /api/evaluations
```

Create a human or automated evaluation for a trace.

**Request Body:**
```json
{
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "evaluator_type": "human",
  "score": 4.5,
  "label": "accepted",
  "critique": "Good response, accurate and helpful.",
  "metadata": {
    "criteria": ["accuracy", "helpfulness", "clarity"]
  }
}
```

#### Get Evaluations
```http
GET /api/evaluations?trace_id=550e8400-e29b-41d4-a716-446655440000
```

Retrieve evaluations for traces.

### Testing

#### Run Test Suite
```http
POST /api/tests/run
```

Execute a test suite against your LLM.

**Request Body:**
```json
{
  "test_suite_id": "suite_123",
  "model_config": {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 150
  }
}
```

#### Get Test Results
```http
GET /api/tests/results/{run_id}
```

Get results from a test run.

### Data Export

#### Export Data
```http
POST /api/export
```

Export traces and evaluations in various formats.

**Request Body:**
```json
{
  "format": "csv",
  "filters": {
    "start_date": "2025-01-01T00:00:00Z",
    "end_date": "2025-01-27T23:59:59Z",
    "model_name": "gpt-4"
  },
  "include_evaluations": true
}
```

## Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Request format or parameters are invalid |
| `UNAUTHORIZED` | Authentication required or invalid |
| `FORBIDDEN` | Insufficient permissions |
| `NOT_FOUND` | Requested resource not found |
| `RATE_LIMITED` | Rate limit exceeded |
| `INTERNAL_ERROR` | Server error occurred |

## SDKs and Examples

### Python SDK

```python
from llm_eval_platform import EvalClient

client = EvalClient(
    base_url="http://localhost:8000",
    api_key="your_api_key"
)

# Log a trace
trace = client.traces.create(
    user_input="What is machine learning?",
    model_output="Machine learning is...",
    model_name="gpt-4"
)

# Get traces
traces = client.traces.list(limit=50, model_name="gpt-4")

# Create evaluation
evaluation = client.evaluations.create(
    trace_id=trace.id,
    score=4.5,
    label="accepted"
)
```

### JavaScript SDK

```javascript
import { EvalClient } from '@llm-eval-platform/sdk';

const client = new EvalClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your_api_key'
});

// Log a trace
const trace = await client.traces.create({
  userInput: 'What is machine learning?',
  modelOutput: 'Machine learning is...',
  modelName: 'gpt-4'
});

// Get traces
const traces = await client.traces.list({
  limit: 50,
  modelName: 'gpt-4'
});
```

### cURL Examples

```bash
# Create a trace
curl -X POST http://localhost:8000/api/traces \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "What is the capital of France?",
    "model_output": "The capital of France is Paris.",
    "model_name": "gpt-4"
  }'

# Get traces
curl -H "Authorization: Bearer YOUR_TOKEN" \
     "http://localhost:8000/api/traces?limit=10&model_name=gpt-4"

# Create evaluation
curl -X POST http://localhost:8000/api/evaluations \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "550e8400-e29b-41d4-a716-446655440000",
    "evaluator_type": "human",
    "score": 4.5,
    "label": "accepted"
  }'
```

## Webhooks

The platform supports webhooks for real-time notifications:

### Webhook Events

- `trace.created`: New trace logged
- `evaluation.created`: New evaluation added
- `experiment.completed`: A/B test completed
- `test.completed`: Test suite execution finished

### Webhook Payload

```json
{
  "event": "trace.created",
  "timestamp": "2025-01-27T00:00:00Z",
  "data": {
    "trace_id": "550e8400-e29b-41d4-a716-446655440000",
    "model_name": "gpt-4",
    "session_id": "session_123"
  }
}
```

## Support

For API support and questions:
- Documentation: [https://docs.llm-eval-platform.com](https://docs.llm-eval-platform.com)
- GitHub Issues: [https://github.com/your-org/llm-eval-platform/issues](https://github.com/your-org/llm-eval-platform/issues)
- Email: support@llm-eval-platform.com 