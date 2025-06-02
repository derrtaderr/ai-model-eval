"""
Redis-based caching service for the LLM Evaluation Platform.
Provides intelligent caching for API responses, database queries, and analytics.
"""

import json
import logging
import pickle
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from functools import wraps
from hashlib import md5

import redis
from fastapi import Request

from config.performance import REDIS_CACHE_SETTINGS, CACHE_TTL_SETTINGS

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-based caching service with intelligent key management and TTL."""
    
    def __init__(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_CACHE_SETTINGS["host"],
                port=REDIS_CACHE_SETTINGS["port"],
                db=REDIS_CACHE_SETTINGS["db"],
                password=REDIS_CACHE_SETTINGS.get("password"),
                encoding=REDIS_CACHE_SETTINGS["encoding"],
                decode_responses=False,  # We'll handle encoding ourselves
                max_connections=REDIS_CACHE_SETTINGS["max_connections"],
                socket_timeout=REDIS_CACHE_SETTINGS["socket_timeout"],
                **REDIS_CACHE_SETTINGS["connection_pool_kwargs"]
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self.redis_client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available."""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False
    
    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a consistent cache key."""
        key_parts = [prefix]
        
        # Add positional arguments
        for arg in args:
            if isinstance(arg, (dict, list)):
                key_parts.append(md5(json.dumps(arg, sort_keys=True).encode()).hexdigest()[:8])
            else:
                key_parts.append(str(arg))
        
        # Add keyword arguments
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            kwargs_str = json.dumps(sorted_kwargs, sort_keys=True)
            key_parts.append(md5(kwargs_str.encode()).hexdigest()[:8])
        
        return ":".join(key_parts)
    
    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage."""
        if isinstance(data, (str, int, float, bool)):
            return json.dumps(data).encode()
        else:
            return pickle.dumps(data)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from storage."""
        try:
            # Try JSON first (for simple types)
            return json.loads(data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle for complex types
            return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if not self.is_available():
            return None
        
        try:
            data = self.redis_client.get(key)
            if data is None:
                return None
            return self._deserialize(data)
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL."""
        if not self.is_available():
            return False
        
        try:
            serialized_data = self._serialize(value)
            if ttl:
                return self.redis_client.setex(key, ttl, serialized_data)
            else:
                return self.redis_client.set(key, serialized_data)
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if not self.is_available():
            return False
        
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        if not self.is_available():
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache delete pattern error for {pattern}: {e}")
            return 0
    
    def invalidate_user_cache(self, user_id: str) -> int:
        """Invalidate all cache entries for a specific user."""
        return self.delete_pattern(f"user:{user_id}:*")
    
    def invalidate_trace_cache(self, trace_id: str) -> int:
        """Invalidate all cache entries for a specific trace."""
        return self.delete_pattern(f"trace:{trace_id}:*")
    
    def invalidate_experiment_cache(self, experiment_id: str) -> int:
        """Invalidate all cache entries for a specific experiment."""
        return self.delete_pattern(f"experiment:{experiment_id}:*")


# Global cache service instance
cache_service = CacheService()


def cache_response(
    cache_key_prefix: str,
    ttl_key: str = None,
    ttl: int = None,
    include_user: bool = False,
    vary_on: List[str] = None
):
    """
    Decorator to cache API responses.
    
    Args:
        cache_key_prefix: Prefix for the cache key
        ttl_key: Key in CACHE_TTL_SETTINGS for TTL value
        ttl: Explicit TTL in seconds (overrides ttl_key)
        include_user: Whether to include user ID in cache key
        vary_on: List of request parameters to include in cache key
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not cache_service.is_available():
                return await func(*args, **kwargs)
            
            # Build cache key
            key_parts = [cache_key_prefix]
            
            # Add user ID if requested
            if include_user:
                # Try to get user from kwargs or function signature
                user_id = kwargs.get('current_user') or kwargs.get('user_id')
                if user_id:
                    key_parts.append(f"user:{user_id}")
            
            # Add specific parameters
            if vary_on:
                for param in vary_on:
                    value = kwargs.get(param)
                    if value is not None:
                        key_parts.append(f"{param}:{value}")
            
            # Add all other relevant kwargs
            cache_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['current_user', 'db'] and v is not None}
            
            cache_key = cache_service._make_key(*key_parts, **cache_kwargs)
            
            # Try to get from cache
            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            # Determine TTL
            cache_ttl = ttl
            if cache_ttl is None and ttl_key:
                cache_ttl = CACHE_TTL_SETTINGS.get(ttl_key, 300)
            
            cache_service.set(cache_key, result, cache_ttl)
            logger.debug(f"Cached result for key: {cache_key} (TTL: {cache_ttl}s)")
            
            return result
        
        return wrapper
    return decorator


def cache_database_query(cache_key_prefix: str, ttl: int = 300):
    """
    Decorator to cache database query results.
    
    Args:
        cache_key_prefix: Prefix for the cache key
        ttl: Cache TTL in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not cache_service.is_available():
                return await func(*args, **kwargs)
            
            # Build cache key from function arguments
            cache_key = cache_service._make_key(cache_key_prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Database cache hit for key: {cache_key}")
                return cached_result
            
            # Execute query and cache result
            result = await func(*args, **kwargs)
            cache_service.set(cache_key, result, ttl)
            logger.debug(f"Cached database result for key: {cache_key}")
            
            return result
        
        return wrapper
    return decorator


class CacheManager:
    """High-level cache management utilities."""
    
    @staticmethod
    def warm_up_cache():
        """Pre-populate cache with frequently accessed data."""
        logger.info("Starting cache warm-up...")
        # This would typically load common filter presets, user sessions, etc.
        # Implementation depends on specific application needs
    
    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """Get cache performance statistics."""
        if not cache_service.is_available():
            return {"status": "unavailable"}
        
        try:
            info = cache_service.redis_client.info()
            return {
                "status": "available",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
                ),
                "total_commands_processed": info.get("total_commands_processed", 0),
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    @staticmethod
    def clear_all_cache():
        """Clear all cache data (use with caution)."""
        if cache_service.is_available():
            try:
                cache_service.redis_client.flushdb()
                logger.info("All cache data cleared")
                return True
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                return False
        return False
    
    @staticmethod
    def invalidate_stale_data():
        """Remove stale cache entries based on patterns."""
        patterns_to_clear = [
            "stats:*",  # Clear old statistics
            "analytics:*",  # Clear old analytics
        ]
        
        total_cleared = 0
        for pattern in patterns_to_clear:
            cleared = cache_service.delete_pattern(pattern)
            total_cleared += cleared
            logger.info(f"Cleared {cleared} keys matching pattern: {pattern}")
        
        return total_cleared


# Initialize cache manager
cache_manager = CacheManager() 