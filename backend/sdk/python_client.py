"""
LLM Evaluation Platform Python SDK

This SDK provides easy integration with the LLM Evaluation Platform,
allowing you to send traces, receive real-time updates, and interact
with the evaluation system.

Installation:
    pip install requests aiohttp asyncio

Usage:
    from llm_eval_sdk import LLMEvalClient
    
    # Initialize client
    client = LLMEvalClient("http://localhost:8000")
    
    # Send a trace
    client.send_trace(
        trace_id="unique_id",
        model_name="gpt-4",
        user_query="What is AI?",
        ai_response="AI is artificial intelligence...",
        metadata={"temperature": 0.7}
    )
"""

import requests
import json
import asyncio
import aiohttp
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
import logging

@dataclass
class TraceData:
    """Data structure for trace information"""
    trace_id: str
    model_name: str
    user_query: str
    ai_response: str
    system_prompt: Optional[str] = None
    functions_called: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None
    response_time_ms: Optional[int] = None
    cost: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.functions_called is None:
            self.functions_called = []
        if self.metadata is None:
            self.metadata = {}

class LLMEvalClient:
    """Main client for interacting with the LLM Evaluation Platform"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize the LLM Evaluation client
        
        Args:
            base_url: Base URL of the evaluation platform (e.g., "http://localhost:8000")
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'LLMEvalSDK/1.0'
        })
        
        if api_key:
            self.session.headers['Authorization'] = f'Bearer {api_key}'
        
        self.logger = logging.getLogger(__name__)
    
    def send_trace(self, **kwargs) -> Dict[str, Any]:
        """
        Send a single trace to the evaluation platform
        
        Args:
            trace_id: Unique identifier for the trace
            model_name: Name of the AI model
            user_query: User's input query
            ai_response: AI model's response
            system_prompt: Optional system prompt
            functions_called: Optional list of function calls
            metadata: Optional metadata dict
            tokens_used: Optional token count
            response_time_ms: Optional response time in milliseconds
            cost: Optional cost in dollars
            
        Returns:
            Response from the webhook endpoint
        """
        try:
            trace = TraceData(**kwargs)
            payload = asdict(trace)
            
            # Convert datetime to ISO string
            if isinstance(payload['timestamp'], datetime):
                payload['timestamp'] = payload['timestamp'].isoformat()
            
            response = self.session.post(
                f"{self.base_url}/webhook/trace",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to send trace: {e}")
            raise
    
    def send_traces_batch(self, traces: List[TraceData], source: str = "python_sdk", batch_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send multiple traces in a single batch request
        
        Args:
            traces: List of TraceData objects
            source: Source system identifier
            batch_id: Optional batch identifier
            
        Returns:
            Response from the batch webhook endpoint
        """
        try:
            # Convert traces to dict format
            trace_dicts = []
            for trace in traces:
                trace_dict = asdict(trace)
                if isinstance(trace_dict['timestamp'], datetime):
                    trace_dict['timestamp'] = trace_dict['timestamp'].isoformat()
                trace_dicts.append(trace_dict)
            
            payload = {
                "traces": trace_dicts,
                "source": source,
                "batch_id": batch_id
            }
            
            response = self.session.post(
                f"{self.base_url}/webhook/batch",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to send batch: {e}")
            raise
    
    def get_traces(self, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get traces from the platform
        
        Args:
            limit: Maximum number of traces to return
            status: Optional status filter (e.g., 'pending', 'accepted', 'rejected')
            
        Returns:
            List of trace data
        """
        try:
            params = {"limit": limit}
            if status:
                params["status"] = status
                
            response = self.session.get(
                f"{self.base_url}/api/traces",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get traces: {e}")
            raise
    
    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """Get a specific trace by ID"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/traces/{trace_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get trace {trace_id}: {e}")
            raise
    
    def evaluate_trace(self, trace_id: str, status: str, reason: Optional[str] = None, notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a trace (accept/reject)
        
        Args:
            trace_id: ID of the trace to evaluate
            status: 'accepted' or 'rejected'
            reason: Optional reason for rejection
            notes: Optional evaluation notes
            
        Returns:
            Evaluation response
        """
        try:
            payload = {
                "status": status,
                "reason": reason,
                "notes": notes
            }
            
            response = self.session.post(
                f"{self.base_url}/api/evaluations",
                json={
                    "trace_id": trace_id,
                    **payload
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate trace {trace_id}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get platform statistics"""
        try:
            response = self.session.get(
                f"{self.base_url}/webhook/stats",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check platform health"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            raise

class AsyncLLMEvalClient:
    """Async version of the LLM Evaluation client for real-time streaming"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {'Content-Type': 'application/json'}
        
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
        
        self.logger = logging.getLogger(__name__)
    
    async def stream_events(self, callback: Callable[[Dict[str, Any]], None], 
                          model_name: Optional[str] = None, 
                          evaluation_status: Optional[str] = None):
        """
        Stream real-time events from the platform
        
        Args:
            callback: Function to call with each event
            model_name: Optional filter by model name
            evaluation_status: Optional filter by evaluation status
        """
        try:
            params = {}
            if model_name:
                params['model_name'] = model_name
            if evaluation_status:
                params['evaluation_status'] = evaluation_status
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/stream/events"
                async with session.get(url, params=params, headers=self.headers) as response:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])  # Remove 'data: ' prefix
                                callback(data)
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            self.logger.error(f"Stream error: {e}")
            raise
    
    async def stream_traces(self, callback: Callable[[Dict[str, Any]], None], limit: int = 10):
        """Stream real-time trace updates"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/stream/traces"
                params = {'limit': limit}
                
                async with session.get(url, params=params, headers=self.headers) as response:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                callback(data)
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            self.logger.error(f"Trace stream error: {e}")
            raise
    
    async def stream_metrics(self, callback: Callable[[Dict[str, Any]], None], interval: int = 5):
        """Stream real-time metrics updates"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/stream/metrics"
                params = {'interval': interval}
                
                async with session.get(url, params=params, headers=self.headers) as response:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data: '):
                            try:
                                data = json.loads(line[6:])
                                callback(data)
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            self.logger.error(f"Metrics stream error: {e}")
            raise

# Context manager for easy trace logging
class TraceLogger:
    """Context manager for automatic trace logging"""
    
    def __init__(self, client: LLMEvalClient, model_name: str, trace_id: Optional[str] = None):
        self.client = client
        self.model_name = model_name
        self.trace_id = trace_id or f"trace_{int(time.time() * 1000)}"
        self.start_time = None
        self.metadata = {}
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Log error traces
            self.metadata['error'] = str(exc_val)
            self.metadata['error_type'] = exc_type.__name__
        
        # Calculate response time
        response_time = int((time.time() - self.start_time) * 1000)
        self.metadata['response_time_ms'] = response_time
    
    def log_trace(self, user_query: str, ai_response: str, **kwargs):
        """Log the trace with timing information"""
        try:
            # Merge metadata
            final_metadata = {**self.metadata, **kwargs.get('metadata', {})}
            
            self.client.send_trace(
                trace_id=self.trace_id,
                model_name=self.model_name,
                user_query=user_query,
                ai_response=ai_response,
                response_time_ms=final_metadata.get('response_time_ms'),
                metadata=final_metadata,
                **{k: v for k, v in kwargs.items() if k != 'metadata'}
            )
        except Exception as e:
            self.client.logger.error(f"Failed to log trace: {e}")

# Example usage and utility functions
def example_usage():
    """Example of how to use the SDK"""
    
    # Initialize client
    client = LLMEvalClient("http://localhost:8000")
    
    # Send a simple trace
    response = client.send_trace(
        trace_id="example_trace_1",
        model_name="gpt-4",
        user_query="What is the capital of France?",
        ai_response="The capital of France is Paris.",
        tokens_used=25,
        response_time_ms=850,
        cost=0.002,
        metadata={"temperature": 0.7, "max_tokens": 100}
    )
    print("Trace sent:", response)
    
    # Using context manager for automatic timing
    with TraceLogger(client, "gpt-3.5-turbo") as logger:
        # Simulate model call
        time.sleep(0.5)  # Simulate processing time
        
        logger.log_trace(
            user_query="Explain quantum computing",
            ai_response="Quantum computing uses quantum mechanics...",
            tokens_used=150,
            cost=0.001
        )
    
    # Get traces
    traces = client.get_traces(limit=10, status="pending")
    print(f"Found {len(traces)} pending traces")
    
    # Check stats
    stats = client.get_stats()
    print("Platform stats:", stats)

async def example_streaming():
    """Example of streaming real-time updates"""
    
    async_client = AsyncLLMEvalClient("http://localhost:8000")
    
    def handle_event(event_data):
        print(f"Received event: {event_data}")
    
    # Stream all events
    await async_client.stream_events(handle_event)

if __name__ == "__main__":
    # Run example
    example_usage()
    
    # Run streaming example
    # asyncio.run(example_streaming()) 