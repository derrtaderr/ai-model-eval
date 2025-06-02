# Integration Tutorials

## Overview

This guide provides step-by-step instructions for integrating the LLM Evaluation Platform with popular frameworks, tools, and workflows. Choose the integration that best fits your setup.

## Table of Contents

1. [Quick Start Integration](#quick-start-integration)
2. [Framework Integrations](#framework-integrations)
3. [CI/CD Pipeline Integration](#cicd-pipeline-integration)
4. [Data Management Systems](#data-management-systems)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Custom Integrations](#custom-integrations)

## Quick Start Integration

### Basic Python Integration

The simplest way to start logging traces to the evaluation platform:

```python
import requests
import json
from datetime import datetime

class EvalLogger:
    def __init__(self, base_url="http://localhost:8000", api_key="your_api_key"):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def log_trace(self, user_input, model_output, model_name, **kwargs):
        """Log a trace to the evaluation platform"""
        trace_data = {
            "user_input": user_input,
            "model_output": model_output,
            "model_name": model_name,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/api/traces",
            headers=self.headers,
            json=trace_data
        )
        
        if response.status_code == 200:
            return response.json()["trace_id"]
        else:
            print(f"Error logging trace: {response.text}")
            return None

# Usage
logger = EvalLogger(api_key="your_api_key")

# Log a simple interaction
trace_id = logger.log_trace(
    user_input="What is machine learning?",
    model_output="Machine learning is a subset of artificial intelligence...",
    model_name="gpt-4",
    session_id="user_session_123",
    latency_ms=1200,
    cost_usd=0.002
)
```

### Basic JavaScript Integration

For Node.js applications:

```javascript
const axios = require('axios');

class EvalLogger {
    constructor(baseUrl = 'http://localhost:8000', apiKey = 'your_api_key') {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        };
    }
    
    async logTrace(userInput, modelOutput, modelName, options = {}) {
        const traceData = {
            user_input: userInput,
            model_output: modelOutput,
            model_name: modelName,
            timestamp: new Date().toISOString(),
            ...options
        };
        
        try {
            const response = await axios.post(
                `${this.baseUrl}/api/traces`,
                traceData,
                { headers: this.headers }
            );
            return response.data.trace_id;
        } catch (error) {
            console.error('Error logging trace:', error.response?.data || error.message);
            return null;
        }
    }
}

// Usage
const logger = new EvalLogger('http://localhost:8000', 'your_api_key');

await logger.logTrace(
    'What is machine learning?',
    'Machine learning is a subset of artificial intelligence...',
    'gpt-4',
    {
        session_id: 'user_session_123',
        latency_ms: 1200,
        cost_usd: 0.002
    }
);
```

## Framework Integrations

### LangChain Integration

Integrate with LangChain applications using callbacks:

```python
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
import requests
import json
import time
from typing import Any, Dict, List

class EvalPlatformCallback(BaseCallbackHandler):
    """Callback handler for LangChain that logs to the evaluation platform"""
    
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.start_time = None
        self.current_input = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running"""
        self.start_time = time.time()
        self.current_input = prompts[0] if prompts else ""
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM ends running"""
        if not self.start_time or not response.generations:
            return
            
        latency_ms = int((time.time() - self.start_time) * 1000)
        model_output = response.generations[0][0].text
        
        # Extract model name from LLM result
        model_name = getattr(response, 'llm_output', {}).get('model_name', 'unknown')
        
        trace_data = {
            "user_input": self.current_input,
            "model_output": model_output,
            "model_name": model_name,
            "latency_ms": latency_ms,
            "metadata": {
                "framework": "langchain",
                "token_usage": response.llm_output.get('token_usage', {}) if response.llm_output else {}
            }
        }
        
        try:
            requests.post(
                f"{self.base_url}/api/traces",
                headers=self.headers,
                json=trace_data
            )
        except Exception as e:
            print(f"Failed to log trace: {e}")

# Usage with LangChain
from langchain.llms import OpenAI

callback = EvalPlatformCallback(api_key="your_api_key")
llm = OpenAI(callbacks=[callback])

response = llm("What is the capital of France?")
```

### LlamaIndex Integration

For LlamaIndex applications:

```python
from llama_index.callbacks.base import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload
import requests
import json
from typing import Any, Dict, List, Optional

class EvalPlatformLlamaCallback(BaseCallbackHandler):
    """LlamaIndex callback for the evaluation platform"""
    
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[]
        )
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.query_data = {}
    
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        if event_type == CBEventType.QUERY:
            self.query_data[event_id] = {
                "start_time": time.time(),
                "query": payload.get(EventPayload.QUERY_STR, "") if payload else ""
            }
        return event_id
    
    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type == CBEventType.QUERY and event_id in self.query_data:
            query_info = self.query_data[event_id]
            latency_ms = int((time.time() - query_info["start_time"]) * 1000)
            
            response = payload.get(EventPayload.RESPONSE, "") if payload else ""
            
            trace_data = {
                "user_input": query_info["query"],
                "model_output": str(response),
                "model_name": "llamaindex",
                "latency_ms": latency_ms,
                "metadata": {
                    "framework": "llamaindex",
                    "event_id": event_id
                }
            }
            
            try:
                requests.post(
                    f"{self.base_url}/api/traces",
                    headers=self.headers,
                    json=trace_data
                )
            except Exception as e:
                print(f"Failed to log trace: {e}")
            
            del self.query_data[event_id]

# Usage with LlamaIndex
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.callbacks import CallbackManager

callback = EvalPlatformLlamaCallback(api_key="your_api_key")
callback_manager = CallbackManager([callback])

# Set up your index with the callback
index = VectorStoreIndex.from_documents(
    documents,
    callback_manager=callback_manager
)

query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
```

### OpenAI Integration

Direct integration with OpenAI API calls:

```python
import openai
from functools import wraps
import time
import requests

def log_to_eval_platform(api_key: str, base_url: str = "http://localhost:8000"):
    """Decorator to log OpenAI API calls to the evaluation platform"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Call the original function
            response = func(*args, **kwargs)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract information from the response
            if hasattr(response, 'choices') and response.choices:
                user_input = kwargs.get('messages', [{}])[-1].get('content', '') if 'messages' in kwargs else kwargs.get('prompt', '')
                model_output = response.choices[0].message.content if hasattr(response.choices[0], 'message') else response.choices[0].text
                model_name = kwargs.get('model', 'unknown')
                
                trace_data = {
                    "user_input": user_input,
                    "model_output": model_output,
                    "model_name": model_name,
                    "latency_ms": latency_ms,
                    "metadata": {
                        "framework": "openai",
                        "usage": response.usage._asdict() if hasattr(response, 'usage') else {}
                    }
                }
                
                # Calculate cost if usage information is available
                if hasattr(response, 'usage'):
                    # Simplified cost calculation - adjust based on actual pricing
                    cost_per_token = 0.00002  # Example rate
                    total_tokens = response.usage.total_tokens
                    trace_data["cost_usd"] = total_tokens * cost_per_token
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                try:
                    requests.post(
                        f"{base_url}/api/traces",
                        headers=headers,
                        json=trace_data
                    )
                except Exception as e:
                    print(f"Failed to log trace: {e}")
            
            return response
        return wrapper
    return decorator

# Usage
@log_to_eval_platform(api_key="your_eval_platform_api_key")
def call_openai_chat(messages, model="gpt-4"):
    return openai.ChatCompletion.create(
        model=model,
        messages=messages
    )

# Now all calls will be logged
response = call_openai_chat([
    {"role": "user", "content": "What is machine learning?"}
])
```

## CI/CD Pipeline Integration

### GitHub Actions

Add evaluation to your GitHub workflow:

```yaml
# .github/workflows/llm-evaluation.yml
name: LLM Evaluation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install requests pytest
    
    - name: Run LLM Tests
      env:
        EVAL_PLATFORM_API_KEY: ${{ secrets.EVAL_PLATFORM_API_KEY }}
        EVAL_PLATFORM_URL: ${{ secrets.EVAL_PLATFORM_URL }}
      run: |
        python -m pytest tests/test_llm_evaluation.py -v
    
    - name: Upload Test Results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: evaluation-results
        path: test-results/
```

Test file example:

```python
# tests/test_llm_evaluation.py
import pytest
import requests
import os
from your_llm_app import YourLLMApp

class TestLLMEvaluation:
    def setup_method(self):
        self.eval_api_key = os.getenv("EVAL_PLATFORM_API_KEY")
        self.eval_url = os.getenv("EVAL_PLATFORM_URL", "http://localhost:8000")
        self.app = YourLLMApp()
    
    def test_response_quality(self):
        """Test that LLM responses meet quality standards"""
        test_cases = [
            {
                "input": "What is the capital of France?",
                "expected_keywords": ["Paris"],
                "max_length": 100
            },
            {
                "input": "Explain machine learning in simple terms",
                "expected_keywords": ["algorithm", "data", "learn"],
                "max_length": 500
            }
        ]
        
        for case in test_cases:
            response = self.app.generate_response(case["input"])
            
            # Log to evaluation platform
            self.log_trace(case["input"], response, "test-model")
            
            # Run assertions
            assert len(response) <= case["max_length"]
            assert any(keyword.lower() in response.lower() for keyword in case["expected_keywords"])
    
    def log_trace(self, user_input, model_output, model_name):
        """Log trace to evaluation platform"""
        if not self.eval_api_key:
            return
            
        trace_data = {
            "user_input": user_input,
            "model_output": model_output,
            "model_name": model_name,
            "metadata": {
                "test_run": True,
                "ci_environment": "github_actions"
            }
        }
        
        headers = {
            "Authorization": f"Bearer {self.eval_api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            requests.post(
                f"{self.eval_url}/api/traces",
                headers=headers,
                json=trace_data
            )
        except Exception as e:
            print(f"Warning: Failed to log trace: {e}")
```

### Jenkins Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        EVAL_PLATFORM_API_KEY = credentials('eval-platform-api-key')
        EVAL_PLATFORM_URL = 'http://your-eval-platform.com'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Run LLM Tests') {
            steps {
                sh '''
                    python -m pytest tests/test_llm_evaluation.py \
                        --junitxml=test-results.xml \
                        --html=test-report.html
                '''
            }
        }
        
        stage('Upload Results') {
            steps {
                script {
                    // Upload test results to evaluation platform
                    sh '''
                        curl -X POST "${EVAL_PLATFORM_URL}/api/test-runs" \
                             -H "Authorization: Bearer ${EVAL_PLATFORM_API_KEY}" \
                             -H "Content-Type: application/json" \
                             -d @test-results.json
                    '''
                }
            }
        }
    }
    
    post {
        always {
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: '.',
                reportFiles: 'test-report.html',
                reportName: 'LLM Evaluation Report'
            ])
        }
    }
}
```

## Data Management Systems

### PostgreSQL Integration

Direct database integration for high-volume logging:

```python
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime

class PostgreSQLEvalLogger:
    def __init__(self, connection_string):
        self.conn = psycopg2.connect(connection_string)
        self.setup_tables()
    
    def setup_tables(self):
        """Create tables if they don't exist"""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS llm_traces (
                    id SERIAL PRIMARY KEY,
                    trace_id UUID DEFAULT gen_random_uuid(),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_input TEXT NOT NULL,
                    model_output TEXT NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    session_id VARCHAR(255),
                    latency_ms INTEGER,
                    cost_usd DECIMAL(10, 6),
                    metadata JSONB,
                    tags TEXT[]
                );
                
                CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON llm_traces(timestamp);
                CREATE INDEX IF NOT EXISTS idx_traces_model ON llm_traces(model_name);
                CREATE INDEX IF NOT EXISTS idx_traces_session ON llm_traces(session_id);
            """)
        self.conn.commit()
    
    def log_trace(self, user_input, model_output, model_name, **kwargs):
        """Log a trace directly to PostgreSQL"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO llm_traces 
                (user_input, model_output, model_name, session_id, latency_ms, cost_usd, metadata, tags)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING trace_id
            """, (
                user_input,
                model_output,
                model_name,
                kwargs.get('session_id'),
                kwargs.get('latency_ms'),
                kwargs.get('cost_usd'),
                json.dumps(kwargs.get('metadata', {})),
                kwargs.get('tags', [])
            ))
            trace_id = cur.fetchone()[0]
        self.conn.commit()
        return trace_id

# Usage
logger = PostgreSQLEvalLogger("postgresql://user:password@localhost/eval_db")
trace_id = logger.log_trace(
    "What is AI?",
    "AI is artificial intelligence...",
    "gpt-4",
    session_id="session_123",
    latency_ms=1200
)
```

### Redis Integration

For real-time caching and session management:

```python
import redis
import json
from datetime import datetime, timedelta

class RedisEvalCache:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
    
    def cache_trace(self, trace_id, trace_data, ttl_hours=24):
        """Cache trace data in Redis"""
        key = f"trace:{trace_id}"
        self.redis_client.setex(
            key,
            timedelta(hours=ttl_hours),
            json.dumps(trace_data)
        )
    
    def get_cached_trace(self, trace_id):
        """Retrieve cached trace data"""
        key = f"trace:{trace_id}"
        data = self.redis_client.get(key)
        return json.loads(data) if data else None
    
    def cache_session_traces(self, session_id, trace_ids):
        """Cache trace IDs for a session"""
        key = f"session:{session_id}:traces"
        self.redis_client.sadd(key, *trace_ids)
        self.redis_client.expire(key, timedelta(hours=24))
    
    def get_session_traces(self, session_id):
        """Get all trace IDs for a session"""
        key = f"session:{session_id}:traces"
        return [tid.decode() for tid in self.redis_client.smembers(key)]

# Usage
cache = RedisEvalCache()
cache.cache_trace("trace_123", {
    "user_input": "Hello",
    "model_output": "Hi there!",
    "model_name": "gpt-4"
})
```

## Monitoring and Observability

### Prometheus Metrics

Export metrics for monitoring:

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
TRACES_TOTAL = Counter('llm_traces_total', 'Total number of traces logged', ['model_name'])
TRACE_LATENCY = Histogram('llm_trace_latency_seconds', 'Trace logging latency', ['model_name'])
ACTIVE_SESSIONS = Gauge('llm_active_sessions', 'Number of active sessions')
EVALUATION_SCORES = Histogram('llm_evaluation_scores', 'Distribution of evaluation scores', ['model_name'])

class MetricsEvalLogger:
    def __init__(self, eval_platform_logger):
        self.eval_logger = eval_platform_logger
        # Start Prometheus metrics server
        start_http_server(8001)
    
    def log_trace_with_metrics(self, user_input, model_output, model_name, **kwargs):
        """Log trace and update metrics"""
        start_time = time.time()
        
        # Log to evaluation platform
        trace_id = self.eval_logger.log_trace(
            user_input, model_output, model_name, **kwargs
        )
        
        # Update metrics
        TRACES_TOTAL.labels(model_name=model_name).inc()
        TRACE_LATENCY.labels(model_name=model_name).observe(time.time() - start_time)
        
        return trace_id
    
    def record_evaluation(self, model_name, score):
        """Record evaluation score in metrics"""
        EVALUATION_SCORES.labels(model_name=model_name).observe(score)

# Usage
metrics_logger = MetricsEvalLogger(your_eval_logger)
trace_id = metrics_logger.log_trace_with_metrics(
    "What is AI?",
    "AI is...",
    "gpt-4"
)
```

### Grafana Dashboard

Example dashboard configuration:

```json
{
  "dashboard": {
    "title": "LLM Evaluation Metrics",
    "panels": [
      {
        "title": "Traces per Model",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(llm_traces_total[5m])",
            "legendFormat": "{{model_name}}"
          }
        ]
      },
      {
        "title": "Average Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(llm_trace_latency_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Evaluation Score Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(llm_evaluation_scores_bucket[5m])",
            "legendFormat": "{{model_name}}"
          }
        ]
      }
    ]
  }
}
```

## Custom Integrations

### Webhook Integration

Set up webhooks for real-time notifications:

```python
from flask import Flask, request, jsonify
import requests
import hmac
import hashlib

app = Flask(__name__)

@app.route('/webhook/evaluation-complete', methods=['POST'])
def handle_evaluation_complete():
    """Handle evaluation completion webhook"""
    
    # Verify webhook signature (optional but recommended)
    signature = request.headers.get('X-Signature')
    if not verify_signature(request.data, signature):
        return jsonify({'error': 'Invalid signature'}), 401
    
    data = request.json
    
    # Process the evaluation completion
    trace_id = data['trace_id']
    evaluation_score = data['score']
    model_name = data['model_name']
    
    # Trigger actions based on evaluation
    if evaluation_score < 3.0:
        # Low score - trigger alert
        send_alert(f"Low evaluation score ({evaluation_score}) for {model_name}")
    
    # Update your internal systems
    update_model_performance_metrics(model_name, evaluation_score)
    
    return jsonify({'status': 'processed'})

def verify_signature(payload, signature):
    """Verify webhook signature"""
    secret = "your_webhook_secret"
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)

def send_alert(message):
    """Send alert to your notification system"""
    # Implement your alerting logic (Slack, email, etc.)
    pass

def update_model_performance_metrics(model_name, score):
    """Update your internal performance tracking"""
    # Implement your metrics update logic
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Custom SDK

Build a custom SDK for your specific needs:

```python
import requests
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TraceResult:
    trace_id: str
    status: str
    message: str

class CustomEvalSDK:
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def log_trace_async(
        self,
        user_input: str,
        model_output: str,
        model_name: str,
        **kwargs
    ) -> TraceResult:
        """Asynchronously log a trace"""
        trace_data = {
            "user_input": user_input,
            "model_output": model_output,
            "model_name": model_name,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        
        async with self.session.post(
            f"{self.base_url}/api/traces",
            json=trace_data
        ) as response:
            if response.status == 200:
                result = await response.json()
                return TraceResult(
                    trace_id=result["trace_id"],
                    status="success",
                    message="Trace logged successfully"
                )
            else:
                error_text = await response.text()
                return TraceResult(
                    trace_id="",
                    status="error",
                    message=f"Failed to log trace: {error_text}"
                )
    
    async def batch_log_traces(self, traces: List[Dict[str, Any]]) -> List[TraceResult]:
        """Log multiple traces concurrently"""
        tasks = [
            self.log_trace_async(**trace) for trace in traces
        ]
        return await asyncio.gather(*tasks)
    
    def log_trace_sync(self, user_input: str, model_output: str, model_name: str, **kwargs) -> TraceResult:
        """Synchronously log a trace"""
        trace_data = {
            "user_input": user_input,
            "model_output": model_output,
            "model_name": model_name,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/api/traces",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=trace_data
        )
        
        if response.status_code == 200:
            result = response.json()
            return TraceResult(
                trace_id=result["trace_id"],
                status="success",
                message="Trace logged successfully"
            )
        else:
            return TraceResult(
                trace_id="",
                status="error",
                message=f"Failed to log trace: {response.text}"
            )

# Usage
async def main():
    async with CustomEvalSDK("your_api_key") as sdk:
        # Single trace
        result = await sdk.log_trace_async(
            "What is AI?",
            "AI is artificial intelligence...",
            "gpt-4"
        )
        print(f"Logged trace: {result.trace_id}")
        
        # Batch traces
        traces = [
            {"user_input": "Hello", "model_output": "Hi!", "model_name": "gpt-4"},
            {"user_input": "Goodbye", "model_output": "Bye!", "model_name": "gpt-4"}
        ]
        results = await sdk.batch_log_traces(traces)
        print(f"Logged {len(results)} traces")

# Run async example
asyncio.run(main())

# Sync usage
sdk = CustomEvalSDK("your_api_key")
result = sdk.log_trace_sync("Hello", "Hi there!", "gpt-4")
```

## Troubleshooting Integration Issues

### Common Problems

1. **Authentication Errors**
   - Verify API key is correct
   - Check if key has expired
   - Ensure proper header format: `Authorization: Bearer YOUR_KEY`

2. **Network Connectivity**
   - Test connection to evaluation platform
   - Check firewall settings
   - Verify SSL/TLS configuration

3. **Rate Limiting**
   - Implement exponential backoff
   - Batch requests when possible
   - Monitor rate limit headers

4. **Data Format Issues**
   - Validate JSON structure
   - Check required fields
   - Ensure proper encoding (UTF-8)

### Debug Mode

Enable debug logging in your integration:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DebugEvalLogger:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
    
    def log_trace(self, user_input, model_output, model_name, **kwargs):
        trace_data = {
            "user_input": user_input,
            "model_output": model_output,
            "model_name": model_name,
            **kwargs
        }
        
        logger.debug(f"Logging trace: {trace_data}")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/traces",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=trace_data
            )
            
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Response body: {response.text}")
            
            return response.json() if response.status_code == 200 else None
            
        except Exception as e:
            logger.error(f"Error logging trace: {e}")
            return None
```

For additional help with integrations, consult the [API documentation](../api/) or contact support. 