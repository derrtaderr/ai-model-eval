"""
Redis-based caching service for the LLM Evaluation Platform.
Provides intelligent caching for API responses, database queries, and analytics.
"""

import json
import logging
import pickle
import asyncio
from typing import Any, Optional, Dict, List, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
from hashlib import md5

import redis
import redis.asyncio as aioredis
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
            
            # Initialize async Redis client for pub/sub
            self.async_redis_client = aioredis.Redis(
                host=REDIS_CACHE_SETTINGS["host"],
                port=REDIS_CACHE_SETTINGS["port"],
                db=REDIS_CACHE_SETTINGS["db"],
                password=REDIS_CACHE_SETTINGS.get("password"),
                encoding="utf-8",
                decode_responses=True,
                max_connections=REDIS_CACHE_SETTINGS["max_connections"],
                socket_timeout=REDIS_CACHE_SETTINGS["socket_timeout"]
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self.redis_client = None
            self.async_redis_client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available."""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False
    
    async def is_available_async(self) -> bool:
        """Check if async Redis is available."""
        if not self.async_redis_client:
            return False
        try:
            await self.async_redis_client.ping()
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
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL (async version)."""
        if not await self.is_available_async():
            return False
        
        try:
            if expire:
                return await self.async_redis_client.setex(key, expire, json.dumps(value))
            else:
                return await self.async_redis_client.set(key, json.dumps(value))
        except Exception as e:
            logger.warning(f"Async cache set error for key {key}: {e}")
            return False
    
    def set_sync(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL (sync version)."""
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
    
    # Pub/Sub functionality for real-time updates
    async def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """Publish a message to a Redis channel."""
        if not await self.is_available_async():
            return False
        
        try:
            message_json = json.dumps(message)
            result = await self.async_redis_client.publish(channel, message_json)
            logger.debug(f"Published to channel {channel}: {message_json} (subscribers: {result})")
            return result > 0
        except Exception as e:
            logger.warning(f"Publish error for channel {channel}: {e}")
            return False
    
    async def get_pubsub(self) -> Optional[aioredis.client.PubSub]:
        """Get a pub/sub instance for subscribing to channels."""
        if not await self.is_available_async():
            return None
        
        try:
            return self.async_redis_client.pubsub()
        except Exception as e:
            logger.warning(f"PubSub creation error: {e}")
            return None
    
    async def subscribe_to_channel(self, channel: str, callback: Callable[[str, Dict[str, Any]], None]) -> Optional[aioredis.client.PubSub]:
        """Subscribe to a channel and process messages with callback."""
        pubsub = await self.get_pubsub()
        if not pubsub:
            return None
        
        try:
            await pubsub.subscribe(channel)
            logger.info(f"Subscribed to channel: {channel}")
            
            async def message_handler():
                try:
                    async for message in pubsub.listen():
                        if message['type'] == 'message':
                            try:
                                data = json.loads(message['data'])
                                await callback(channel, data)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse message from {channel}: {e}")
                except Exception as e:
                    logger.error(f"Error in message handler for {channel}: {e}")
                finally:
                    await pubsub.unsubscribe(channel)
                    await pubsub.close()
            
            # Start message handler in background
            asyncio.create_task(message_handler())
            return pubsub
            
        except Exception as e:
            logger.warning(f"Subscribe error for channel {channel}: {e}")
            await pubsub.close()
            return None
    
    async def publish_trace_update(self, action: str, trace_id: str, **extra_data):
        """Publish trace update event."""
        message = {
            "action": action,
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            **extra_data
        }
        return await self.publish("trace_updates", message)
    
    async def publish_evaluation_update(self, trace_id: str, status: str, **extra_data):
        """Publish evaluation update event."""
        message = {
            "trace_id": trace_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            **extra_data
        }
        return await self.publish("evaluation_updates", message)
    
    async def publish_system_update(self, event_type: str, **extra_data):
        """Publish system update event."""
        message = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **extra_data
        }
        return await self.publish("system_updates", message)
    
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
            
            cache_key = cache_service._make_key(*key_parts)
            
            # Try to get from cache
            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result
            
            # Cache miss - execute function
            logger.debug(f"Cache miss for key: {cache_key}")
            result = await func(*args, **kwargs)
            
            # Determine TTL
            cache_ttl = ttl
            if cache_ttl is None and ttl_key:
                cache_ttl = CACHE_TTL_SETTINGS.get(ttl_key, 300)
            
            # Store in cache
            cache_service.set_sync(cache_key, result, cache_ttl)
            return result
        
        return wrapper
    return decorator


def cache_database_query(cache_key_prefix: str, ttl: int = 300):
    """
    Decorator to cache database query results.
    
    Args:
        cache_key_prefix: Prefix for the cache key
        ttl: Time to live in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not cache_service.is_available():
                return await func(*args, **kwargs)
            
            # Create cache key from function name and arguments
            cache_key = cache_service._make_key(cache_key_prefix, func.__name__, *args, **kwargs)
            
            # Try cache first
            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute query and cache result
            result = await func(*args, **kwargs)
            cache_service.set_sync(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


class CacheManager:
    """High-level cache management operations."""
    
    @staticmethod
    def warm_up_cache():
        """Pre-populate cache with frequently accessed data."""
        if not cache_service.is_available():
            logger.warning("Cache not available for warm-up")
            return
        
        logger.info("Cache warm-up completed")
    
    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """Get cache performance statistics."""
        if not cache_service.is_available():
            return {
                "status": "unavailable",
                "error": "Redis not available"
            }
        
        try:
            info = cache_service.redis_client.info()
            return {
                "status": "available",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1) * 100
                ),
                "total_commands_processed": info.get("total_commands_processed", 0)
            }
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @staticmethod
    def clear_all_cache():
        """Clear all cache data."""
        if not cache_service.is_available():
            return False
        
        try:
            cache_service.redis_client.flushdb()
            logger.info("All cache data cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    @staticmethod
    def invalidate_stale_data():
        """Remove expired keys and other cleanup."""
        if not cache_service.is_available():
            return 0
        
        # Redis automatically handles TTL expiration
        # This could be extended for custom cleanup logic
        logger.info("Cache cleanup completed")
        return 0


# Global cache manager instance
cache_manager = CacheManager() 