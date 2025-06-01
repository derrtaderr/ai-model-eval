"""
Python SDK for LLM Evaluation Platform External API.
Provides a convenient interface for integrating with the evaluation platform.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
import requests
from urllib.parse import urljoin


@dataclass
class EvaluationCriteria:
    """Evaluation criteria configuration."""
    relevance: float = 1.0
    coherence: float = 1.0
    accuracy: float = 1.0
    safety: float = 1.0
    helpfulness: float = 1.0


@dataclass
class EvaluationRequest:
    """Request data for evaluation."""
    user_input: str
    model_output: str
    model_name: str
    system_prompt: Optional[str] = None
    criteria: List[str] = None
    context: Optional[Dict[str, Any]] = None
    reference_answer: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.criteria is None:
            self.criteria = ["relevance", "coherence"]


@dataclass
class TraceFilter:
    """Filter parameters for listing traces."""
    model_names: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    session_ids: Optional[List[str]] = None
    has_evaluation: Optional[bool] = None
    limit: int = 50
    offset: int = 0


@dataclass
class BatchRequest:
    """Batch operation request."""
    operation: str
    items: List[Dict[str, Any]]
    options: Optional[Dict[str, Any]] = None
    callback_url: Optional[str] = None


class LLMEvaluationAPIError(Exception):
    """Base exception for API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class RateLimitError(LLMEvaluationAPIError):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: int = 3600):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class AuthenticationError(LLMEvaluationAPIError):
    """Exception raised for authentication failures."""
    
    def __init__(self, message: str = "Invalid API key"):
        super().__init__(message, status_code=401)


class LLMEvaluationClient:
    """Synchronous client for the LLM Evaluation Platform API."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the evaluation client.
        
        Args:
            api_key: Your API key for authentication
            base_url: Base URL of the evaluation platform
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "LLMEval-Python-SDK/1.0.0"
        })
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make an HTTP request with retry logic."""
        
        url = urljoin(f"{self.base_url}/", endpoint.lstrip('/'))
        
        for attempt in range(self.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = self.session.get(url, params=params, timeout=self.timeout)
                elif method.upper() == "POST":
                    response = self.session.post(url, json=data, params=params, timeout=self.timeout)
                elif method.upper() == "DELETE":
                    response = self.session.delete(url, timeout=self.timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 3600))
                    raise RateLimitError(
                        f"Rate limit exceeded. Retry after {retry_after} seconds.",
                        retry_after=retry_after
                    )
                
                # Handle authentication errors
                if response.status_code == 401:
                    raise AuthenticationError("Invalid API key or expired token")
                
                # Handle other client errors
                if 400 <= response.status_code < 500:
                    error_data = response.json() if response.content else {}
                    raise LLMEvaluationAPIError(
                        f"Client error: {response.status_code} - {error_data.get('detail', response.text)}",
                        status_code=response.status_code,
                        response_data=error_data
                    )
                
                # Handle server errors with retry
                if response.status_code >= 500:
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    
                    error_data = response.json() if response.content else {}
                    raise LLMEvaluationAPIError(
                        f"Server error: {response.status_code} - {error_data.get('detail', response.text)}",
                        status_code=response.status_code,
                        response_data=error_data
                    )
                
                # Success response
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise LLMEvaluationAPIError(f"Request failed: {str(e)}")
        
        raise LLMEvaluationAPIError("Max retries exceeded")
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        return self._make_request("GET", "/api/external/health")
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available evaluation models and capabilities."""
        return self._make_request("GET", "/api/external/models")
    
    def create_evaluation(self, request: EvaluationRequest) -> Dict[str, Any]:
        """
        Create a new evaluation for an input-output pair.
        
        Args:
            request: Evaluation request data
            
        Returns:
            Evaluation response with scores and reasoning
        """
        data = asdict(request)
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        
        return self._make_request("POST", "/api/external/evaluations", data=data)
    
    def list_evaluations(
        self,
        limit: int = 50,
        offset: int = 0,
        model_name: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        List evaluations with optional filtering.
        
        Args:
            limit: Maximum number of results (1-500)
            offset: Number of results to skip
            model_name: Filter by model name
            min_score: Minimum evaluation score (0-1)
            
        Returns:
            List of evaluation responses
        """
        params = {"limit": limit, "offset": offset}
        if model_name:
            params["model_name"] = model_name
        if min_score is not None:
            params["min_score"] = min_score
        
        return self._make_request("GET", "/api/external/evaluations", params=params)
    
    def submit_trace(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a trace for evaluation.
        
        Args:
            trace_data: Trace data including user_input, model_output, model_name
            
        Returns:
            Trace submission response
        """
        return self._make_request("POST", "/api/external/traces", data=trace_data)
    
    def list_traces(self, filter_params: Optional[TraceFilter] = None) -> List[Dict[str, Any]]:
        """
        List traces with optional filtering.
        
        Args:
            filter_params: Filter parameters for traces
            
        Returns:
            List of trace responses
        """
        if filter_params is None:
            filter_params = TraceFilter()
        
        params = asdict(filter_params)
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        return self._make_request("GET", "/api/external/traces", params=params)
    
    def create_batch_operation(self, batch_request: BatchRequest) -> Dict[str, Any]:
        """
        Create a batch operation for processing multiple items.
        
        Args:
            batch_request: Batch operation request
            
        Returns:
            Batch operation response with job ID and status
        """
        data = asdict(batch_request)
        data = {k: v for k, v in data.items() if v is not None}
        
        return self._make_request("POST", "/api/external/batch", data=data)
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics for the current API key."""
        return self._make_request("GET", "/api/external/usage")
    
    def evaluate_text(
        self,
        user_input: str,
        model_output: str,
        model_name: str,
        criteria: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Convenience method for evaluating a single text response.
        
        Args:
            user_input: The user's input/question
            model_output: The model's response
            model_name: Name of the model that generated the response
            criteria: List of evaluation criteria
            **kwargs: Additional parameters for EvaluationRequest
            
        Returns:
            Evaluation response
        """
        request = EvaluationRequest(
            user_input=user_input,
            model_output=model_output,
            model_name=model_name,
            criteria=criteria or ["relevance", "coherence"],
            **kwargs
        )
        
        return self.create_evaluation(request)
    
    def bulk_evaluate(
        self,
        evaluations: List[EvaluationRequest],
        batch_size: int = 10,
        callback_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple input-output pairs in batch.
        
        Args:
            evaluations: List of evaluation requests
            batch_size: Number of evaluations per batch
            callback_url: Optional webhook URL for completion notification
            
        Returns:
            Batch operation response
        """
        items = [asdict(eval_req) for eval_req in evaluations]
        
        batch_request = BatchRequest(
            operation="evaluate",
            items=items,
            options={"batch_size": batch_size},
            callback_url=callback_url
        )
        
        return self.create_batch_operation(batch_request)
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncLLMEvaluationClient:
    """Asynchronous client for the LLM Evaluation Platform API."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the async evaluation client.
        
        Args:
            api_key: Your API key for authentication
            base_url: Base URL of the evaluation platform
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "User-Agent": "LLMEval-Python-SDK/1.0.0"
        }
        
        self._session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=self.timeout
            )
        return self._session
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make an async HTTP request with retry logic."""
        
        session = await self._get_session()
        url = urljoin(f"{self.base_url}/", endpoint.lstrip('/'))
        
        for attempt in range(self.max_retries + 1):
            try:
                async with session.request(
                    method,
                    url,
                    json=data,
                    params=params
                ) as response:
                    
                    # Handle rate limiting
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 3600))
                        raise RateLimitError(
                            f"Rate limit exceeded. Retry after {retry_after} seconds.",
                            retry_after=retry_after
                        )
                    
                    # Handle authentication errors
                    if response.status == 401:
                        raise AuthenticationError("Invalid API key or expired token")
                    
                    # Handle other client errors
                    if 400 <= response.status < 500:
                        error_data = await response.json() if response.content_length else {}
                        raise LLMEvaluationAPIError(
                            f"Client error: {response.status} - {error_data.get('detail', await response.text())}",
                            status_code=response.status,
                            response_data=error_data
                        )
                    
                    # Handle server errors with retry
                    if response.status >= 500:
                        if attempt < self.max_retries:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))
                            continue
                        
                        error_data = await response.json() if response.content_length else {}
                        raise LLMEvaluationAPIError(
                            f"Server error: {response.status} - {error_data.get('detail', await response.text())}",
                            status_code=response.status,
                            response_data=error_data
                        )
                    
                    # Success response
                    response.raise_for_status()
                    return await response.json()
                    
            except aiohttp.ClientError as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise LLMEvaluationAPIError(f"Request failed: {str(e)}")
        
        raise LLMEvaluationAPIError("Max retries exceeded")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        return await self._make_request("GET", "/api/external/health")
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get list of available evaluation models and capabilities."""
        return await self._make_request("GET", "/api/external/models")
    
    async def create_evaluation(self, request: EvaluationRequest) -> Dict[str, Any]:
        """Create a new evaluation for an input-output pair."""
        data = asdict(request)
        data = {k: v for k, v in data.items() if v is not None}
        
        return await self._make_request("POST", "/api/external/evaluations", data=data)
    
    async def list_evaluations(
        self,
        limit: int = 50,
        offset: int = 0,
        model_name: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """List evaluations with optional filtering."""
        params = {"limit": limit, "offset": offset}
        if model_name:
            params["model_name"] = model_name
        if min_score is not None:
            params["min_score"] = min_score
        
        return await self._make_request("GET", "/api/external/evaluations", params=params)
    
    async def submit_trace(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a trace for evaluation."""
        return await self._make_request("POST", "/api/external/traces", data=trace_data)
    
    async def list_traces(self, filter_params: Optional[TraceFilter] = None) -> List[Dict[str, Any]]:
        """List traces with optional filtering."""
        if filter_params is None:
            filter_params = TraceFilter()
        
        params = asdict(filter_params)
        params = {k: v for k, v in params.items() if v is not None}
        
        return await self._make_request("GET", "/api/external/traces", params=params)
    
    async def create_batch_operation(self, batch_request: BatchRequest) -> Dict[str, Any]:
        """Create a batch operation for processing multiple items."""
        data = asdict(batch_request)
        data = {k: v for k, v in data.items() if v is not None}
        
        return await self._make_request("POST", "/api/external/batch", data=data)
    
    async def get_usage_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics for the current API key."""
        return await self._make_request("GET", "/api/external/usage")
    
    async def evaluate_text(
        self,
        user_input: str,
        model_output: str,
        model_name: str,
        criteria: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Convenience method for evaluating a single text response."""
        request = EvaluationRequest(
            user_input=user_input,
            model_output=model_output,
            model_name=model_name,
            criteria=criteria or ["relevance", "coherence"],
            **kwargs
        )
        
        return await self.create_evaluation(request)
    
    async def bulk_evaluate(
        self,
        evaluations: List[EvaluationRequest],
        batch_size: int = 10,
        callback_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate multiple input-output pairs in batch."""
        items = [asdict(eval_req) for eval_req in evaluations]
        
        batch_request = BatchRequest(
            operation="evaluate",
            items=items,
            options={"batch_size": batch_size},
            callback_url=callback_url
        )
        
        return await self.create_batch_operation(batch_request)
    
    async def stream_evaluations(
        self,
        evaluations: List[EvaluationRequest],
        max_concurrent: int = 5
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream evaluation results as they complete.
        
        Args:
            evaluations: List of evaluation requests
            max_concurrent: Maximum concurrent evaluations
            
        Yields:
            Evaluation results as they complete
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_single(eval_request: EvaluationRequest) -> Dict[str, Any]:
            async with semaphore:
                return await self.create_evaluation(eval_request)
        
        # Create tasks for all evaluations
        tasks = [evaluate_single(eval_req) for eval_req in evaluations]
        
        # Yield results as they complete
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                yield result
            except Exception as e:
                # Yield error information
                yield {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience functions for quick usage
def evaluate_response(
    api_key: str,
    user_input: str,
    model_output: str,
    model_name: str,
    criteria: Optional[List[str]] = None,
    base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Quick evaluation of a single response.
    
    Args:
        api_key: Your API key
        user_input: The user's input/question
        model_output: The model's response
        model_name: Name of the model
        criteria: Evaluation criteria
        base_url: API base URL
        
    Returns:
        Evaluation result
    """
    with LLMEvaluationClient(api_key, base_url) as client:
        return client.evaluate_text(
            user_input=user_input,
            model_output=model_output,
            model_name=model_name,
            criteria=criteria
        )


async def async_evaluate_response(
    api_key: str,
    user_input: str,
    model_output: str,
    model_name: str,
    criteria: Optional[List[str]] = None,
    base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Quick async evaluation of a single response.
    
    Args:
        api_key: Your API key
        user_input: The user's input/question
        model_output: The model's response
        model_name: Name of the model
        criteria: Evaluation criteria
        base_url: API base URL
        
    Returns:
        Evaluation result
    """
    async with AsyncLLMEvaluationClient(api_key, base_url) as client:
        return await client.evaluate_text(
            user_input=user_input,
            model_output=model_output,
            model_name=model_name,
            criteria=criteria
        )


# Example usage
if __name__ == "__main__":
    # Synchronous example
    client = LLMEvaluationClient(api_key="your-api-key-here")
    
    # Check health
    health = client.health_check()
    print(f"API Status: {health['status']}")
    
    # Evaluate a response
    result = client.evaluate_text(
        user_input="What is the capital of France?",
        model_output="The capital of France is Paris.",
        model_name="gpt-4",
        criteria=["accuracy", "completeness"]
    )
    
    print(f"Evaluation Score: {result['overall_score']}")
    print(f"Reasoning: {result['reasoning']}")
    
    client.close() 