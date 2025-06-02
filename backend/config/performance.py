"""
Performance configuration for the LLM Evaluation Platform.
Includes caching, connection pooling, and monitoring settings.
"""

import os
from typing import Dict, Any

# Database Connection Pool Settings
DATABASE_POOL_SETTINGS = {
    "pool_size": int(os.getenv("DATABASE_POOL_SIZE", "10")),
    "max_overflow": int(os.getenv("DATABASE_MAX_OVERFLOW", "20")),
    "pool_timeout": int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
    "pool_recycle": int(os.getenv("DATABASE_POOL_RECYCLE", "3600")),  # 1 hour
    "pool_pre_ping": True,  # Verify connections before use
}

# Redis Cache Settings
REDIS_CACHE_SETTINGS = {
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", "6379")),
    "db": int(os.getenv("REDIS_CACHE_DB", "0")),
    "password": os.getenv("REDIS_PASSWORD", None),
    "encoding": "utf-8",
    "decode_responses": True,
    "max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
    "socket_timeout": int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
    "connection_pool_kwargs": {
        "retry_on_timeout": True,
        "health_check_interval": 30,
    }
}

# Cache TTL Settings (in seconds)
CACHE_TTL_SETTINGS = {
    "trace_stats": 300,      # 5 minutes
    "user_sessions": 1800,   # 30 minutes
    "experiment_results": 600,  # 10 minutes
    "evaluation_summaries": 900,  # 15 minutes
    "test_run_results": 300,  # 5 minutes
    "model_configs": 3600,   # 1 hour
    "filter_presets": 1800,  # 30 minutes
    "dashboard_analytics": 120,  # 2 minutes
}

# Performance Monitoring Settings
MONITORING_SETTINGS = {
    "enable_request_timing": True,
    "enable_database_monitoring": True,
    "enable_cache_monitoring": True,
    "slow_query_threshold_ms": 1000,  # Log queries taking longer than 1 second
    "api_response_time_target_ms": 200,
    "alert_on_error_rate_threshold": 0.05,  # 5% error rate
    "metrics_collection_interval": 60,  # seconds
}

# Background Job Settings
BACKGROUND_JOB_SETTINGS = {
    "max_workers": int(os.getenv("CELERY_MAX_WORKERS", "4")),
    "task_time_limit": int(os.getenv("CELERY_TASK_TIME_LIMIT", "300")),  # 5 minutes
    "task_soft_time_limit": int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "240")),  # 4 minutes
    "worker_prefetch_multiplier": int(os.getenv("CELERY_PREFETCH_MULTIPLIER", "1")),
    "task_routes": {
        "evaluation.tasks.run_model_evaluation": {"queue": "evaluation"},
        "analytics.tasks.generate_report": {"queue": "analytics"},
        "export.tasks.export_data": {"queue": "export"},
    }
}

# API Rate Limiting Settings
RATE_LIMIT_SETTINGS = {
    "default_rate_limit": "1000/minute",
    "auth_rate_limit": "10/minute",
    "upload_rate_limit": "100/hour",
    "export_rate_limit": "50/hour",
    "evaluation_rate_limit": "500/hour",
}

# Response Size Limits
RESPONSE_SIZE_LIMITS = {
    "max_traces_per_request": 1000,
    "max_evaluations_per_request": 500,
    "max_export_rows": 10000,
    "max_file_upload_size_mb": 100,
    "pagination_default_limit": 50,
    "pagination_max_limit": 500,
}

# Performance Optimization Flags
OPTIMIZATION_FLAGS = {
    "enable_query_cache": True,
    "enable_response_compression": True,
    "enable_etag_caching": True,
    "enable_lazy_loading": True,
    "enable_query_optimization": True,
    "enable_batch_operations": True,
}

def get_performance_config() -> Dict[str, Any]:
    """Get complete performance configuration."""
    return {
        "database_pool": DATABASE_POOL_SETTINGS,
        "redis_cache": REDIS_CACHE_SETTINGS,
        "cache_ttl": CACHE_TTL_SETTINGS,
        "monitoring": MONITORING_SETTINGS,
        "background_jobs": BACKGROUND_JOB_SETTINGS,
        "rate_limits": RATE_LIMIT_SETTINGS,
        "response_limits": RESPONSE_SIZE_LIMITS,
        "optimizations": OPTIMIZATION_FLAGS,
    } 