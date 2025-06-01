# External Tool Integration API Guide

The LLM Evaluation Platform provides a comprehensive External Tool Integration API that allows third-party tools and applications to integrate with the evaluation system. This guide covers authentication, available endpoints, SDKs, and best practices.

## Table of Contents

1. [Authentication](#authentication)
2. [Rate Limiting](#rate-limiting)
3. [API Endpoints](#api-endpoints)
4. [SDKs](#sdks)
5. [Examples](#examples)
6. [Error Handling](#error-handling)
7. [Best Practices](#best-practices)

## Authentication

The External API uses API key-based authentication. You'll need to create an API key through the web interface first.

### API Key Management

#### Creating an API Key

```bash
curl -X POST "http://localhost:8000/api/external/api-keys" \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Integration Key",
    "tier": "premium",
    "description": "API key for my data science workflow"
  }'
```

#### Using API Keys

Include your API key in requests using either method:

**X-API-Key Header (Recommended):**
```bash
curl -H "X-API-Key: llm-eval-your-api-key-here" \
  "http://localhost:8000/api/external/health"
```

**Authorization Header:**
```bash
curl -H "Authorization: Bearer llm-eval-your-api-key-here" \
  "http://localhost:8000/api/external/health"
```

## Rate Limiting

Rate limits vary by tier:

- **Free**: 100 requests/hour
- **Premium**: 1,000 requests/hour  
- **Enterprise**: 10,000 requests/hour

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Your rate limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: When the limit resets

When rate limited, you'll receive a 429 status with `Retry-After` header.

## API Endpoints

### Health Check

Check API status and availability.

```http
GET /api/external/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "External Integration API",
  "version": "v1.0.0",
  "timestamp": "2025-01-27T00:00:00.000Z",
  "endpoints": [...],
  "authentication": "API Key required",
  "rate_limiting": "Varies by tier and endpoint"
}
```

### Available Models

Get information about available evaluation models and capabilities.

```http
GET /api/external/models
```

**Response:**
```json
{
  "evaluator_models": [
    {
      "id": "gpt-4",
      "name": "GPT-4",
      "provider": "OpenAI",
      "capabilities": ["text-evaluation", "reasoning", "scoring"],
      "cost_per_1k_tokens": 0.03,
      "max_tokens": 8192,
      "recommended_for": ["high-quality-evaluation", "complex-reasoning"]
    }
  ],
  "evaluation_criteria": ["relevance", "coherence", "accuracy", "safety"],
  "supported_formats": ["text", "json", "structured"],
  "api_version": "v1"
}
```

### Create Evaluation

Evaluate a model response against user input.

```http
POST /api/external/evaluations
```

**Request Body:**
```json
{
  "user_input": "What is the capital of France?",
  "model_output": "The capital of France is Paris.",
  "model_name": "gpt-4",
  "criteria": ["accuracy", "completeness"],
  "system_prompt": "You are a helpful assistant.",
  "session_id": "session-123",
  "context": {"language": "en"},
  "metadata": {"experiment": "geography-qa"}
}
```

**Response:**
```json
{
  "evaluation_id": "eval-456",
  "trace_id": "trace-789",
  "overall_score": 0.95,
  "criteria_scores": {
    "accuracy": 1.0,
    "completeness": 0.9
  },
  "reasoning": "The response correctly identifies Paris as the capital of France...",
  "confidence": 0.98,
  "evaluator_model": "external-api-v1",
  "evaluation_time_ms": 1500,
  "cost_usd": 0.002
}
```

### List Evaluations

Retrieve evaluations with optional filtering.

```http
GET /api/external/evaluations?limit=50&offset=0&model_name=gpt-4&min_score=0.8
```

**Query Parameters:**
- `limit` (1-500): Maximum results to return
- `offset` (≥0): Number of results to skip
- `model_name`: Filter by model name
- `min_score` (0-1): Minimum evaluation score

### Submit Trace

Submit a trace for future evaluation or analysis.

```http
POST /api/external/traces
```

**Request Body:**
```json
{
  "user_input": "Explain quantum computing",
  "model_output": "Quantum computing is a type of computation...",
  "model_name": "claude-3",
  "system_prompt": "Be thorough in your explanations",
  "session_id": "session-456",
  "latency_ms": 2500,
  "cost_usd": 0.01,
  "metadata": {"topic": "technology"}
}
```

### List Traces

Retrieve traces with advanced filtering.

```http
GET /api/external/traces?model_names=gpt-4,claude-3&date_from=2025-01-01&has_evaluation=true
```

**Query Parameters:**
- `model_names`: Comma-separated model names
- `date_from`/`date_to`: ISO date range
- `min_score`/`max_score`: Score range filter
- `session_ids`: Comma-separated session IDs
- `has_evaluation`: Filter by evaluation presence
- `limit`/`offset`: Pagination

### Batch Operations

Process multiple items in a single request.

```http
POST /api/external/batch
```

**Request Body:**
```json
{
  "operation": "evaluate",
  "items": [
    {
      "user_input": "Question 1",
      "model_output": "Answer 1",
      "model_name": "gpt-4"
    },
    {
      "user_input": "Question 2", 
      "model_output": "Answer 2",
      "model_name": "gpt-4"
    }
  ],
  "options": {
    "batch_size": 10
  },
  "callback_url": "https://your-app.com/webhook"
}
```

**Response:**
```json
{
  "batch_id": "batch-123",
  "status": "queued",
  "total_items": 2,
  "completed_items": 0,
  "failed_items": 0,
  "estimated_completion": "2025-01-27T10:15:00.000Z",
  "results_url": "/api/external/batch/batch-123/results"
}
```

### Usage Statistics

Get API usage statistics for your API key.

```http
GET /api/external/usage
```

**Response:**
```json
{
  "api_key_id": "key-123",
  "current_period_usage": 45,
  "rate_limit": 1000,
  "usage_remaining": 955,
  "reset_time": "2025-01-27T11:00:00.000Z",
  "total_usage_today": 245,
  "total_usage_month": 8750,
  "top_endpoints": [
    {"endpoint": "/api/external/evaluations", "count": 120},
    {"endpoint": "/api/external/traces", "count": 80}
  ],
  "average_response_time_ms": 850.5
}
```

## SDKs

### Python SDK

Install and use the Python SDK:

```python
from backend.api.external_sdk import LLMEvaluationClient

# Initialize client
client = LLMEvaluationClient(api_key="your-api-key-here")

# Quick evaluation
result = client.evaluate_text(
    user_input="What is machine learning?",
    model_output="Machine learning is a subset of AI...",
    model_name="gpt-4",
    criteria=["accuracy", "clarity"]
)

print(f"Score: {result['overall_score']}")
print(f"Reasoning: {result['reasoning']}")
```

**Async Version:**
```python
from backend.api.external_sdk import AsyncLLMEvaluationClient

async def main():
    async with AsyncLLMEvaluationClient("your-api-key") as client:
        # Evaluate multiple responses concurrently
        evaluations = [...]  # List of EvaluationRequest objects
        
        async for result in client.stream_evaluations(evaluations, max_concurrent=5):
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Evaluation {result['evaluation_id']}: {result['overall_score']}")

# Run async function
import asyncio
asyncio.run(main())
```

### JavaScript SDK

Use in Node.js or browser:

```javascript
const { LLMEvaluationClient } = require('./sdk/javascript/llm-eval-sdk');

// Initialize client
const client = new LLMEvaluationClient('your-api-key-here');

// Evaluate a response
async function evaluateResponse() {
  try {
    const result = await client.evaluateText(
      'What is the capital of Spain?',
      'The capital of Spain is Madrid.',
      'gpt-4',
      ['accuracy', 'completeness']
    );
    
    console.log(`Score: ${result.overall_score}`);
    console.log(`Reasoning: ${result.reasoning}`);
  } catch (error) {
    if (error instanceof RateLimitError) {
      console.log(`Rate limited. Retry after ${error.retryAfter} seconds`);
    } else {
      console.error('Error:', error.message);
    }
  }
}

evaluateResponse();
```

**Batch Processing:**
```javascript
const evaluations = [
  {
    user_input: 'Question 1',
    model_output: 'Answer 1', 
    model_name: 'gpt-4'
  },
  // ... more evaluations
];

// Process in batch
const batchResult = await client.bulkEvaluate(evaluations, 10);
console.log(`Batch ID: ${batchResult.batch_id}`);

// Or stream individual results
for await (const result of client.streamEvaluations(evaluations, 3)) {
  console.log(`Result: ${result.overall_score || result.error}`);
}
```

## Examples

### Data Science Workflow

```python
import pandas as pd
from backend.api.external_sdk import LLMEvaluationClient

# Initialize client
client = LLMEvaluationClient("your-api-key")

# Load your evaluation dataset
df = pd.read_csv("model_outputs.csv")

# Evaluate each response
results = []
for _, row in df.iterrows():
    try:
        result = client.evaluate_text(
            user_input=row['question'],
            model_output=row['answer'],
            model_name=row['model'],
            criteria=['accuracy', 'helpfulness', 'safety']
        )
        results.append({
            'id': row['id'],
            'score': result['overall_score'],
            'reasoning': result['reasoning'],
            'confidence': result['confidence']
        })
    except Exception as e:
        print(f"Error evaluating {row['id']}: {e}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("evaluation_results.csv", index=False)
```

### CI/CD Integration

```yaml
# .github/workflows/evaluate-model.yml
name: Evaluate Model Performance

on:
  pull_request:
    paths: ['models/**', 'data/**']

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: |
          pip install requests pandas
          
      - name: Run evaluation
        env:
          LLM_EVAL_API_KEY: ${{ secrets.LLM_EVAL_API_KEY }}
        run: |
          python scripts/evaluate_pr_changes.py
```

### Real-time Monitoring

```javascript
// monitor-model-quality.js
const { LLMEvaluationClient } = require('./sdk/javascript/llm-eval-sdk');

class ModelQualityMonitor {
  constructor(apiKey) {
    this.client = new LLMEvaluationClient(apiKey);
    this.scoreThreshold = 0.8;
  }
  
  async evaluateProduction(userInput, modelOutput, modelName) {
    // Submit trace for logging
    const trace = await this.client.submitTrace({
      user_input: userInput,
      model_output: modelOutput,
      model_name: modelName,
      timestamp: new Date().toISOString()
    });
    
    // Evaluate quality
    const evaluation = await this.client.evaluateText(
      userInput,
      modelOutput, 
      modelName,
      ['relevance', 'safety', 'helpfulness']
    );
    
    // Alert if quality is low
    if (evaluation.overall_score < this.scoreThreshold) {
      this.alertLowQuality(evaluation);
    }
    
    return evaluation;
  }
  
  alertLowQuality(evaluation) {
    console.warn(`Low quality response detected!`);
    console.warn(`Score: ${evaluation.overall_score}`);
    console.warn(`Reason: ${evaluation.reasoning}`);
    // Send to monitoring system...
  }
}

// Usage
const monitor = new ModelQualityMonitor(process.env.LLM_EVAL_API_KEY);

// In your application route/handler
app.post('/chat', async (req, res) => {
  const { message } = req.body;
  
  // Get model response
  const modelResponse = await yourLLMService.generateResponse(message);
  
  // Monitor quality (async, don't block response)
  monitor.evaluateProduction(message, modelResponse, 'gpt-4')
    .catch(console.error);
  
  res.json({ response: modelResponse });
});
```

## Error Handling

### Common Error Codes

- `401`: Invalid API key or expired token
- `403`: Insufficient permissions  
- `429`: Rate limit exceeded (includes `Retry-After` header)
- `400`: Bad request (validation errors)
- `500`: Internal server error

### Error Response Format

```json
{
  "detail": "Error description",
  "error_code": "INVALID_API_KEY",
  "timestamp": "2025-01-27T10:00:00.000Z",
  "request_id": "req-123"
}
```

### SDK Error Handling

```python
from backend.api.external_sdk import (
    LLMEvaluationClient, 
    RateLimitError, 
    AuthenticationError,
    LLMEvaluationAPIError
)

client = LLMEvaluationClient("your-api-key")

try:
    result = client.evaluate_text(...)
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after} seconds")
    time.sleep(e.retry_after)
    # Retry the request
except AuthenticationError:
    print("Invalid API key. Check your credentials.")
except LLMEvaluationAPIError as e:
    print(f"API error {e.status_code}: {e}")
    if e.response_data:
        print(f"Details: {e.response_data}")
```

## Best Practices

### 1. API Key Security

- Store API keys securely (environment variables, secret managers)
- Rotate keys regularly
- Use different keys for different environments
- Never commit keys to version control

### 2. Rate Limit Management

- Implement exponential backoff for retries
- Cache results when possible
- Use batch operations for multiple requests
- Monitor usage with `/usage` endpoint

### 3. Error Handling

- Always handle rate limits gracefully
- Implement proper retry logic
- Log errors for debugging
- Validate inputs before API calls

### 4. Performance Optimization

- Use async operations when available
- Batch similar requests together
- Implement request timeouts
- Cache evaluation results when appropriate

### 5. Monitoring and Observability

- Track API usage and costs
- Monitor evaluation quality trends
- Set up alerts for anomalies
- Log important evaluation decisions

### 6. Development Workflow

- Test with different API tiers
- Validate responses against expected schemas
- Use health checks for service discovery
- Implement circuit breakers for resilience

## Support and Documentation

- **API Documentation**: Available at `/docs` when running the server
- **OpenAPI Spec**: Available at `/openapi.json`
- **Status Page**: Check `/api/external/health` for service status
- **SDK Source**: Available in the repository for customization

For additional support or feature requests, please refer to the main project documentation or open an issue in the repository. 