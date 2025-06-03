"""
Performance monitoring middleware for the LLM Evaluation Platform.
Tracks request metrics, response times, and system performance.
"""

import time
import logging
import asyncio
import psutil
from typing import Callable, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
from starlette.responses import JSONResponse

from config.performance import MONITORING_SETTINGS, RATE_LIMIT_SETTINGS
from services.cache_service import cache_service

logger = logging.getLogger(__name__)


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor API performance and collect metrics."""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.total_requests = 0
        self.start_time = datetime.utcnow()
    
    async def dispatch(self, request: Request, call_next):
        """Process request and collect performance metrics."""
        start_time = time.time()
        self.total_requests += 1
        
        # Add request ID for tracing
        request_id = f"{int(time.time() * 1000000)}"
        request.state.request_id = request_id
        
        try:
            # Call the endpoint
            response = await call_next(request)
            
            # Calculate response time
            process_time = time.time() - start_time
            response_time_ms = process_time * 1000
            
            # Log slow requests
            if (MONITORING_SETTINGS["enable_request_timing"] and 
                response_time_ms > MONITORING_SETTINGS["api_response_time_target_ms"]):
                logger.warning(
                    f"Slow request: {request.method} {request.url.path} "
                    f"took {response_time_ms:.2f}ms (target: {MONITORING_SETTINGS['api_response_time_target_ms']}ms)"
                )
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            # Store metrics
            endpoint = f"{request.method}:{request.url.path}"
            self.request_times[endpoint].append(response_time_ms)
            
            # Keep only recent metrics (last 1000 requests per endpoint)
            if len(self.request_times[endpoint]) > 1000:
                self.request_times[endpoint] = self.request_times[endpoint][-1000:]
            
            # Cache metrics if Redis is available
            if cache_service.is_available():
                cache_key = f"metrics:response_time:{endpoint}:{int(time.time() // 60)}"  # Per minute
                cache_service.set(cache_key, response_time_ms, 3600)  # 1 hour TTL
            
            return response
            
        except Exception as e:
            # Track errors
            process_time = time.time() - start_time
            endpoint = f"{request.method}:{request.url.path}"
            self.error_counts[endpoint] += 1
            
            logger.error(
                f"Request error: {request.method} {request.url.path} "
                f"after {process_time * 1000:.2f}ms - {str(e)}"
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
                headers={"X-Request-ID": request_id}
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        current_time = datetime.utcnow()
        uptime = (current_time - self.start_time).total_seconds()
        
        # Calculate average response times
        avg_response_times = {}
        for endpoint, times in self.request_times.items():
            if times:
                avg_response_times[endpoint] = {
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "count": len(times)
                }
        
        # Calculate error rates
        error_rates = {}
        for endpoint, error_count in self.error_counts.items():
            total_requests = len(self.request_times.get(endpoint, [])) + error_count
            if total_requests > 0:
                error_rates[endpoint] = error_count / total_requests
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "average_response_times": avg_response_times,
            "error_rates": error_rates,
            "monitored_endpoints": len(self.request_times),
            "timestamp": current_time.isoformat()
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_counts = defaultdict(lambda: defaultdict(int))
        self.last_reset = defaultdict(lambda: defaultdict(lambda: datetime.utcnow()))
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting based on client IP and endpoint."""
        client_ip = request.client.host
        endpoint = request.url.path
        current_time = datetime.utcnow()
        
        # Determine rate limit for this endpoint
        rate_limit = self._get_rate_limit(endpoint)
        if not rate_limit:
            return await call_next(request)
        
        limit, window_minutes = self._parse_rate_limit(rate_limit)
        
        # Check if we need to reset the counter
        last_reset_time = self.last_reset[client_ip][endpoint]
        if (current_time - last_reset_time).total_seconds() >= window_minutes * 60:
            self.request_counts[client_ip][endpoint] = 0
            self.last_reset[client_ip][endpoint] = current_time
        
        # Check rate limit
        current_count = self.request_counts[client_ip][endpoint]
        if current_count >= limit:
            logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "limit": limit,
                    "window_minutes": window_minutes,
                    "retry_after": window_minutes * 60
                },
                headers={
                    "X-Rate-Limit": str(limit),
                    "X-Rate-Limit-Remaining": "0",
                    "X-Rate-Limit-Reset": str(int((last_reset_time + timedelta(minutes=window_minutes)).timestamp()))
                }
            )
        
        # Increment counter and proceed
        self.request_counts[client_ip][endpoint] += 1
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = limit - self.request_counts[client_ip][endpoint]
        response.headers["X-Rate-Limit"] = str(limit)
        response.headers["X-Rate-Limit-Remaining"] = str(remaining)
        response.headers["X-Rate-Limit-Reset"] = str(
            int((last_reset_time + timedelta(minutes=window_minutes)).timestamp())
        )
        
        return response
    
    def _get_rate_limit(self, endpoint: str) -> Optional[str]:
        """Get rate limit for specific endpoint."""
        # Map specific endpoints to rate limits
        endpoint_limits = {
            "/api/auth/": RATE_LIMIT_SETTINGS["auth_rate_limit"],
            "/api/upload": RATE_LIMIT_SETTINGS["upload_rate_limit"],
            "/api/export": RATE_LIMIT_SETTINGS["export_rate_limit"],
            "/api/evaluations": RATE_LIMIT_SETTINGS["evaluation_rate_limit"],
        }
        
        # Check for specific endpoint matches
        for path_prefix, limit in endpoint_limits.items():
            if endpoint.startswith(path_prefix):
                return limit
        
        # Default rate limit
        return RATE_LIMIT_SETTINGS["default_rate_limit"]
    
    def _parse_rate_limit(self, rate_limit: str) -> tuple[int, int]:
        """Parse rate limit string like '1000/minute' into (limit, window_minutes)."""
        try:
            limit_str, period = rate_limit.split("/")
            limit = int(limit_str)
            
            period_minutes = {
                "second": 1/60,
                "minute": 1,
                "hour": 60,
                "day": 1440
            }
            
            window_minutes = int(period_minutes.get(period, 1))
            return limit, window_minutes
            
        except (ValueError, KeyError):
            # Default to 1000 requests per minute
            logger.warning(f"Invalid rate limit format: {rate_limit}")
            return 1000, 1


class CompressionMiddleware(BaseHTTPMiddleware):
    """Simple response compression middleware."""
    
    async def dispatch(self, request: Request, call_next):
        """Apply gzip compression to responses."""
        response = await call_next(request)
        
        # Check if client accepts compression
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response
        
        # Only compress JSON responses above a certain size
        if (hasattr(response, 'body') and 
            response.headers.get("content-type", "").startswith("application/json") and
            len(response.body) > 1024):  # Only compress responses > 1KB
            
            import gzip
            compressed_body = gzip.compress(response.body)
            
            # Only use compression if it actually reduces size
            if len(compressed_body) < len(response.body):
                response.body = compressed_body
                response.headers["content-encoding"] = "gzip"
                response.headers["content-length"] = str(len(compressed_body))
        
        return response


# Global middleware instances (to be added to FastAPI app)
performance_monitor = PerformanceMonitoringMiddleware
rate_limiter = RateLimitMiddleware
compression = CompressionMiddleware


# Utility functions for external access
def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics from monitoring middleware."""
    # This would need to be implemented with a global instance
    # or dependency injection system
    return {
        "message": "Performance metrics middleware not initialized",
        "available_when_middleware_added": True
    }


def reset_rate_limits():
    """Reset all rate limit counters."""
    # This would need access to the rate limit middleware instance
    logger.info("Rate limits reset requested")
    return {"message": "Rate limits reset functionality needs middleware instance"} 