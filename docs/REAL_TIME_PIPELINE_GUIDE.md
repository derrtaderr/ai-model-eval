# Real-Time Data Pipeline Guide

The LLM Evaluation Platform includes a comprehensive real-time data pipeline that enables seamless integration with your LLM applications, providing instant trace ingestion, live updates, and streaming analytics.

## 🔄 **Pipeline Overview**

The real-time pipeline consists of three main components:

1. **Webhook Endpoints** - Receive trace data from external systems
2. **Streaming API** - Provide real-time updates via Server-Sent Events
3. **SDK Clients** - Easy integration libraries for Python and JavaScript

## 📊 **Data Flow Architecture**

```
Your LLM App → SDK/Direct API → Webhook Endpoints → Database + Redis Cache → Streaming API → Frontend Dashboard
```

### Key Features:
- **Real-time ingestion** - Sub-second trace processing
- **Batch processing** - Handle up to 100 traces per batch
- **Live streaming** - Server-Sent Events for real-time updates
- **Scalable caching** - Redis-backed cache for performance
- **Cross-platform SDKs** - Python and JavaScript support

---

## 🔌 **Webhook Endpoints**

### Single Trace Webhook
**POST** `/webhook/trace`

Accepts individual trace data for immediate processing.

**Request Body:**
```json
{
  "trace_id": "unique_trace_id",
  "timestamp": "2025-01-27T12:00:00Z",
  "model_name": "gpt-4",
  "user_query": "What is machine learning?",
  "system_prompt": "You are a helpful AI assistant",
  "ai_response": "Machine learning is a subset of artificial intelligence...",
  "functions_called": [
    {
      "name": "search_knowledge_base",
      "parameters": {"query": "machine learning"},
      "result": "Found 15 relevant articles"
    }
  ],
  "metadata": {
    "temperature": 0.7,
    "max_tokens": 1000,
    "user_id": "user_123"
  },
  "tokens_used": 245,
  "response_time_ms": 850,
  "cost": 0.0012
}
```

**Response:**
```json
{
  "success": true,
  "message": "Trace received and processed successfully",
  "trace_ids": ["unique_trace_id"]
}
```

### Batch Trace Webhook
**POST** `/webhook/batch`

Process multiple traces in a single request for efficiency.

**Request Body:**
```json
{
  "traces": [
    {
      "trace_id": "trace_1",
      "model_name": "gpt-4",
      "user_query": "First query",
      "ai_response": "First response",
      // ... other trace fields
    },
    {
      "trace_id": "trace_2",
      "model_name": "gpt-3.5-turbo",
      "user_query": "Second query",
      "ai_response": "Second response",
      // ... other trace fields
    }
  ],
  "source": "production_app",
  "batch_id": "batch_123"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Processed 2 of 2 traces",
  "processed_count": 2,
  "errors": null,
  "trace_ids": ["trace_1", "trace_2"]
}
```

### Webhook Stats
**GET** `/webhook/stats`

Get real-time processing statistics.

**Response:**
```json
{
  "total_traces": 1500,
  "today_traces": 45,
  "pending_evaluation": 12,
  "timestamp": "2025-01-27T12:00:00Z"
}
```

---

## 📡 **Streaming API**

### Live Events Stream
**GET** `/stream/events`

Stream all real-time events with optional filtering.

**Query Parameters:**
- `client_id` (optional) - Unique client identifier
- `model_name` (optional) - Filter by model name
- `evaluation_status` (optional) - Filter by evaluation status

**Server-Sent Events Format:**
```
data: {"event": "trace_updates", "data": {"action": "new_trace", "trace_id": "abc123"}, "timestamp": "2025-01-27T12:00:00Z"}

data: {"event": "evaluation_updates", "data": {"trace_id": "abc123", "status": "accepted"}, "timestamp": "2025-01-27T12:00:00Z"}
```

### Trace Stream
**GET** `/stream/traces`

Stream recent traces with real-time updates.

**Query Parameters:**
- `client_id` (optional) - Unique client identifier  
- `limit` (default: 10) - Number of initial traces to send

### Metrics Stream
**GET** `/stream/metrics`

Stream live metrics and statistics.

**Query Parameters:**
- `client_id` (optional) - Unique client identifier
- `interval` (default: 5) - Update interval in seconds

**Sample Metrics Event:**
```json
{
  "event": "metrics_update",
  "data": {
    "total_traces": 1500,
    "accepted_traces": 850,
    "rejected_traces": 150,
    "pending_traces": 500,
    "avg_response_time": 1250.5,
    "total_cost": 15.67,
    "today_traces": 45,
    "acceptance_rate": 85.0
  },
  "timestamp": "2025-01-27T12:00:00Z"
}
```

---

## 🐍 **Python SDK**

### Installation
```bash
pip install requests aiohttp
```

### Basic Usage
```python
from llm_eval_sdk import LLMEvalClient

# Initialize client
client = LLMEvalClient("http://localhost:8000")

# Send a trace
response = client.send_trace(
    trace_id="my_trace_123",
    model_name="gpt-4",
    user_query="What is AI?",
    ai_response="AI is artificial intelligence...",
    tokens_used=25,
    response_time_ms=850,
    cost=0.002,
    metadata={"temperature": 0.7}
)

print("Trace sent:", response)
```

### Advanced Usage with Context Manager
```python
from llm_eval_sdk import TraceLogger

# Automatic timing and error handling
with TraceLogger(client, "gpt-4") as logger:
    # Your LLM call here
    result = call_your_llm("What is quantum computing?")
    
    # Log with automatic timing
    logger.log_trace(
        user_query="What is quantum computing?",
        ai_response=result.text,
        tokens_used=result.tokens,
        cost=result.cost
    )
```

### Batch Processing
```python
from llm_eval_sdk import TraceData

traces = [
    TraceData(
        trace_id="trace_1",
        model_name="gpt-4",
        user_query="Query 1",
        ai_response="Response 1"
    ),
    TraceData(
        trace_id="trace_2", 
        model_name="gpt-3.5-turbo",
        user_query="Query 2",
        ai_response="Response 2"
    )
]

response = client.send_traces_batch(traces, source="my_app")
```

### Real-time Streaming
```python
import asyncio
from llm_eval_sdk import AsyncLLMEvalClient

async def stream_updates():
    async_client = AsyncLLMEvalClient("http://localhost:8000")
    
    def handle_event(event):
        print(f"Received: {event}")
    
    # Stream all events
    await async_client.stream_events(handle_event)

# Run streaming
asyncio.run(stream_updates())
```

---

## 🌐 **JavaScript SDK**

### Installation
```bash
npm install axios  # For Node.js
```

### Basic Usage
```javascript
import { LLMEvalClient } from './llm-eval-sdk';

// Initialize client
const client = new LLMEvalClient('http://localhost:8000');

// Send a trace
const response = await client.sendTrace({
  traceId: 'my_trace_123',
  modelName: 'gpt-4',
  userQuery: 'What is AI?',
  aiResponse: 'AI is artificial intelligence...',
  tokensUsed: 25,
  responseTimeMs: 850,
  cost: 0.002,
  metadata: { temperature: 0.7 }
});

console.log('Trace sent:', response);
```

### Using TraceLogger
```javascript
const logger = new TraceLogger(client, 'gpt-4').start();

// Simulate LLM call
await new Promise(resolve => setTimeout(resolve, 500));

// Log with automatic timing
await logger.logTrace(
  'Explain quantum computing',
  'Quantum computing uses quantum mechanics...',
  { tokensUsed: 150, cost: 0.001 }
);
```

### Real-time Streaming (Browser)
```javascript
import { LLMEvalStreamClient } from './llm-eval-sdk';

const streamClient = new LLMEvalStreamClient('http://localhost:8000');

// Stream events
const stopStreaming = streamClient.streamEvents((event) => {
  console.log('Received event:', event);
});

// Stream metrics
const stopMetrics = streamClient.streamMetrics((metrics) => {
  console.log('Metrics update:', metrics);
});

// Stop streaming when done
setTimeout(() => {
  stopStreaming();
  stopMetrics();
}, 30000);
```

### React Integration Example
```jsx
import React, { useState, useEffect } from 'react';
import { LLMEvalStreamClient } from './llm-eval-sdk';

function LiveMetrics() {
  const [metrics, setMetrics] = useState(null);
  
  useEffect(() => {
    const streamClient = new LLMEvalStreamClient('http://localhost:8000');
    
    const stopMetrics = streamClient.streamMetrics((data) => {
      if (data.event === 'metrics_update') {
        setMetrics(data.data);
      }
    });
    
    return () => stopMetrics();
  }, []);
  
  return (
    <div>
      {metrics && (
        <div>
          <h3>Live Metrics</h3>
          <p>Total Traces: {metrics.total_traces}</p>
          <p>Acceptance Rate: {metrics.acceptance_rate}%</p>
          <p>Avg Response Time: {metrics.avg_response_time}ms</p>
        </div>
      )}
    </div>
  );
}
```

---

## 🔧 **Integration Examples**

### OpenAI Integration
```python
import openai
from llm_eval_sdk import TraceLogger

client = LLMEvalClient("http://localhost:8000")

def tracked_openai_call(prompt, model="gpt-4"):
    with TraceLogger(client, model) as logger:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        logger.log_trace(
            user_query=prompt,
            ai_response=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            cost=calculate_cost(response.usage, model),
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "model": response.model
            }
        )
        
        return response

# Use it
result = tracked_openai_call("What is machine learning?")
```

### LangChain Integration
```python
from langchain.callbacks.base import BaseCallbackHandler
from llm_eval_sdk import LLMEvalClient

class EvalCallbackHandler(BaseCallbackHandler):
    def __init__(self, eval_client):
        self.client = eval_client
        self.current_trace = {}
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.current_trace = {
            "trace_id": f"lc_{int(time.time() * 1000)}",
            "model_name": serialized.get("model_name", "unknown"),
            "user_query": prompts[0] if prompts else "",
            "start_time": time.time()
        }
    
    def on_llm_end(self, response, **kwargs):
        if self.current_trace:
            self.current_trace.update({
                "ai_response": response.generations[0][0].text,
                "response_time_ms": int((time.time() - self.current_trace["start_time"]) * 1000)
            })
            
            self.client.send_trace(**self.current_trace)

# Use with LangChain
eval_client = LLMEvalClient("http://localhost:8000")
callback = EvalCallbackHandler(eval_client)

llm = OpenAI(callbacks=[callback])
result = llm("What is the capital of France?")
```

### Express.js Middleware
```javascript
const { LLMEvalClient } = require('./llm-eval-sdk');
const client = new LLMEvalClient('http://localhost:8000');

function trackLLMCall(req, res, next) {
  const originalSend = res.send;
  const startTime = Date.now();
  
  res.send = function(data) {
    // Extract LLM data from response
    if (req.body.track_llm && data.llm_response) {
      client.sendTrace({
        traceId: `express_${Date.now()}_${Math.random()}`,
        modelName: req.body.model || 'unknown',
        userQuery: req.body.query,
        aiResponse: data.llm_response,
        responseTimeMs: Date.now() - startTime,
        metadata: {
          endpoint: req.path,
          method: req.method,
          user_id: req.user?.id
        }
      }).catch(console.error);
    }
    
    originalSend.call(this, data);
  };
  
  next();
}

app.use(trackLLMCall);
```

---

## 🚀 **Production Deployment**

### Environment Variables
```bash
# Required for production
WEBHOOK_SECRET=your_webhook_secret_for_signature_verification

# Redis configuration
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your_redis_password

# Database
DATABASE_URL=postgresql://user:pass@localhost/eval_db
```

### Docker Compose Example
```yaml
version: '3.8'
services:
  api:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@db:5432/eval_db
    depends_on:
      - redis
      - db
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: eval_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
```

### Security Considerations
1. **Webhook Signatures** - Implement HMAC verification for webhook security
2. **Rate Limiting** - Built-in rate limiting prevents abuse
3. **Authentication** - Use API keys for production environments
4. **CORS** - Configure appropriate CORS settings for your domain

### Performance Optimization
1. **Redis Caching** - Automatic caching for frequently accessed data
2. **Connection Pooling** - Database connection pooling for efficiency
3. **Batch Processing** - Process multiple traces in single requests
4. **Async Processing** - Background processing for webhook payloads

---

## 📈 **Monitoring & Analytics**

### Health Checks
```bash
# Basic health check
curl http://localhost:8000/webhook/health

# Detailed stats
curl http://localhost:8000/webhook/stats
```

### Performance Metrics
The platform automatically tracks:
- **Trace ingestion rate** - Traces processed per second
- **Processing latency** - Time from webhook to database storage
- **Error rates** - Failed webhook attempts and their reasons
- **Cache hit rates** - Redis cache performance
- **Stream connection count** - Active real-time connections

### Alerting Integration
Configure monitoring alerts for:
- High error rates (>5%)
- Slow processing (>1000ms)
- Cache misses (>50%)
- Database connection issues

---

## 🔍 **Troubleshooting**

### Common Issues

#### Webhook Timeouts
```python
# Increase timeout in SDK
client = LLMEvalClient("http://localhost:8000", timeout=60)
```

#### Missing Traces
Check webhook response for errors:
```python
try:
    response = client.send_trace(trace_data)
    print("Success:", response)
except Exception as e:
    print("Error:", e)
```

#### Streaming Connection Issues
Verify CORS and network connectivity:
```javascript
// Add error handling
const stopStreaming = streamClient.streamEvents(
  (event) => console.log(event),
  { modelName: 'gpt-4' }
);

// Handle connection errors
streamClient.onError = (error) => {
  console.error('Stream error:', error);
};
```

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

client = LLMEvalClient("http://localhost:8000")
```

---

## 📚 **Additional Resources**

- [API Reference](./API_REFERENCE.md) - Complete API documentation
- [SDK Examples](../examples/) - More integration examples
- [Performance Guide](./PERFORMANCE.md) - Optimization best practices
- [Security Guide](./SECURITY.md) - Security recommendations

---

**Need Help?** Check our [FAQ](./FAQ.md) or create an issue in the repository. 