"""
Advanced Redis Service
Provides enterprise-grade Redis caching with connection pooling, serialization, and monitoring.
"""

import json
import pickle
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
import hashlib

import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from redis.asyncio.retry import Retry
from redis.asyncio.client import Pipeline

from config.performance import REDIS_CACHE_SETTINGS, CACHE_TTL_SETTINGS
import os

logger = logging.getLogger(__name__)


class RedisConnectionManager:
    """Manages Redis connections with automatic failover and reconnection."""
    
    def __init__(self):
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[redis.Redis] = None
        self.is_connected = False
        self.connection_attempts = 0
        self.max_retries = 3
        
    async def initialize(self):
        """Initialize Redis connection with retry logic."""
        try:
            # Create connection pool with optimized settings
            self.pool = ConnectionPool(
                host=REDIS_CACHE_SETTINGS["host"],
                port=REDIS_CACHE_SETTINGS["port"],
                db=REDIS_CACHE_SETTINGS["db"],
                password=REDIS_CACHE_SETTINGS["password"],
                encoding=REDIS_CACHE_SETTINGS["encoding"],
                decode_responses=REDIS_CACHE_SETTINGS["decode_responses"],
                max_connections=REDIS_CACHE_SETTINGS["max_connections"],
                socket_timeout=REDIS_CACHE_SETTINGS["socket_timeout"],
                retry_on_timeout=True,
                health_check_interval=30,
                retry=Retry(retries=3)
            )
            
            # Create Redis client
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            self.is_connected = True
            self.connection_attempts = 0
            
            logger.info("✅ Redis connection established successfully")
            
        except Exception as e:
            self.is_connected = False
            self.connection_attempts += 1
            logger.warning(f"⚠️ Redis connection failed (attempt {self.connection_attempts}): {e}")
            
            if self.connection_attempts < self.max_retries:
                await asyncio.sleep(2 ** self.connection_attempts)  # Exponential backoff
                await self.initialize()
            else:
                logger.error("❌ Redis connection failed after max retries. Operating without cache.")
    
    async def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            if not self.client:
                return False
            await self.client.ping()
            if not self.is_connected:
                self.is_connected = True
                logger.info("✅ Redis connection restored")
            return True
        except Exception as e:
            if self.is_connected:
                self.is_connected = False
                logger.warning(f"⚠️ Redis health check failed: {e}")
            return False
    
    async def close(self):
        """Close Redis connections gracefully."""
        try:
            if self.client:
                await self.client.aclose()
            if self.pool:
                await self.pool.aclose()
            self.is_connected = False
            logger.info("Redis connections closed")
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")


class RedisSerializer:
    """Handles serialization and deserialization of cached data."""
    
    @staticmethod
    def serialize(data: Any) -> bytes:
        """Serialize data for Redis storage."""
        try:
            if isinstance(data, (str, int, float, bool)):
                return json.dumps(data).encode('utf-8')
            else:
                # Use pickle for complex objects
                return pickle.dumps(data)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """Deserialize data from Redis."""
        try:
            # Try JSON first (faster and safer)
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fallback to pickle
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise


class RedisCacheMetrics:
    """Tracks Redis cache performance metrics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
        self.total_size = 0
        self.start_time = datetime.utcnow()
    
    def record_hit(self):
        self.hits += 1
    
    def record_miss(self):
        self.misses += 1
    
    def record_set(self, size: int = 0):
        self.sets += 1
        self.total_size += size
    
    def record_delete(self):
        self.deletes += 1
    
    def record_error(self):
        self.errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
            "total_size_bytes": self.total_size,
            "uptime_seconds": uptime,
            "requests_per_second": round(total_requests / uptime, 2) if uptime > 0 else 0
        }


class RedisService:
    """Advanced Redis caching service with comprehensive features."""
    
    def __init__(self):
        self.connection_manager = RedisConnectionManager()
        self.serializer = RedisSerializer()
        self.metrics = RedisCacheMetrics()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the Redis service."""
        if not self._initialized:
            await self.connection_manager.initialize()
            self._initialized = True
    
    def is_available(self) -> bool:
        """Check if Redis is available."""
        return self.connection_manager.is_connected
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        try:
            if not await self.connection_manager.health_check():
                self.metrics.record_error()
                return None
            
            # Get raw data from Redis
            raw_data = await self.connection_manager.client.get(key)
            
            if raw_data is None:
                self.metrics.record_miss()
                logger.debug(f"Cache miss: {key}")
                return None
            
            # Deserialize and return
            data = self.serializer.deserialize(raw_data)
            self.metrics.record_hit()
            logger.debug(f"Cache hit: {key}")
            return data
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache get error for key '{key}': {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """Set a value in cache with optional TTL and conditions."""
        try:
            if not await self.connection_manager.health_check():
                self.metrics.record_error()
                return False
            
            # Serialize data
            serialized_data = self.serializer.serialize(value)
            
            # Set with conditions
            result = await self.connection_manager.client.set(
                key, 
                serialized_data, 
                ex=ttl,
                nx=nx,  # Only set if key doesn't exist
                xx=xx   # Only set if key exists
            )
            
            if result:
                self.metrics.record_set(len(serialized_data))
                logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            
            return bool(result)
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache set error for key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        try:
            if not await self.connection_manager.health_check():
                self.metrics.record_error()
                return False
            
            result = await self.connection_manager.client.delete(key)
            
            if result:
                self.metrics.record_delete()
                logger.debug(f"Cache delete: {key}")
            
            return bool(result)
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache delete error for key '{key}': {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        try:
            if not await self.connection_manager.health_check():
                self.metrics.record_error()
                return 0
            
            # Get all keys matching pattern
            keys = await self.connection_manager.client.keys(pattern)
            
            if not keys:
                return 0
            
            # Delete keys in batches
            deleted_count = 0
            batch_size = 100
            
            for i in range(0, len(keys), batch_size):
                batch = keys[i:i + batch_size]
                result = await self.connection_manager.client.delete(*batch)
                deleted_count += result
                self.metrics.record_delete()
            
            logger.debug(f"Cache pattern delete: {pattern} ({deleted_count} keys)")
            return deleted_count
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache pattern delete error for pattern '{pattern}': {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        try:
            if not await self.connection_manager.health_check():
                return False
            
            result = await self.connection_manager.client.exists(key)
            return bool(result)
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache exists error for key '{key}': {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for a key."""
        try:
            if not await self.connection_manager.health_check():
                self.metrics.record_error()
                return False
            
            result = await self.connection_manager.client.expire(key, ttl)
            return bool(result)
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache expire error for key '{key}': {e}")
            return False
    
    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL for a key."""
        try:
            if not await self.connection_manager.health_check():
                return -1
            
            return await self.connection_manager.client.ttl(key)
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache TTL error for key '{key}': {e}")
            return -1
    
    async def pipeline(self) -> Pipeline:
        """Get Redis pipeline for batch operations."""
        if not await self.connection_manager.health_check():
            raise RuntimeError("Redis not available")
        
        return self.connection_manager.client.pipeline()
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter."""
        try:
            if not await self.connection_manager.health_check():
                self.metrics.record_error()
                return 0
            
            result = await self.connection_manager.client.incrby(key, amount)
            return result
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache increment error for key '{key}': {e}")
            return 0
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get Redis memory usage statistics."""
        try:
            if not await self.connection_manager.health_check():
                return {}
            
            info = await self.connection_manager.client.info("memory")
            return {
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "used_memory_peak": info.get("used_memory_peak", 0),
                "used_memory_peak_human": info.get("used_memory_peak_human", "0B"),
                "memory_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0)
            }
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Redis memory usage error: {e}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        return self.metrics.get_stats()
    
    async def clear_all(self) -> bool:
        """Clear all cache data (use with caution!)."""
        try:
            if not await self.connection_manager.health_check():
                self.metrics.record_error()
                return False
            
            await self.connection_manager.client.flushdb()
            logger.warning("⚠️ All cache data cleared!")
            return True
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache clear all error: {e}")
            return False
    
    async def close(self):
        """Close Redis connections."""
        await self.connection_manager.close()


# Global Redis service instance
redis_service = RedisService()


def cache_result(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    skip_cache: bool = False,
    refresh_cache: bool = False
):
    """Decorator for caching function results."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Skip caching if disabled
            if skip_cache or not redis_service.is_available():
                return await func(*args, **kwargs)
            
            # Generate cache key
            key_data = {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(sorted(kwargs.items()))
            }
            key_str = json.dumps(key_data, sort_keys=True)
            cache_key = f"{key_prefix}:{hashlib.md5(key_str.encode()).hexdigest()}"
            
            # Check cache first (unless refreshing)
            if not refresh_cache:
                cached_result = await redis_service.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache_ttl = ttl or CACHE_TTL_SETTINGS.get(func.__name__, 300)
            await redis_service.set(cache_key, result, cache_ttl)
            
            return result
        
        return wrapper
    return decorator


async def warm_up_cache():
    """Warm up cache with frequently accessed data."""
    logger.info("🔥 Starting cache warm-up...")
    
    try:
        # Initialize Redis service
        await redis_service.initialize()
        
        if not redis_service.is_available():
            logger.warning("⚠️ Redis not available for cache warm-up")
            return
        
        # Warm up common cache keys
        warm_up_data = {
            "system:status": {"status": "operational", "timestamp": datetime.utcnow().isoformat()},
            "config:performance": {"cache_enabled": True, "monitoring_enabled": True},
        }
        
        for key, data in warm_up_data.items():
            await redis_service.set(key, data, ttl=3600)  # 1 hour
        
        logger.info("✅ Cache warm-up completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Cache warm-up failed: {e}")


async def cleanup_expired_cache():
    """Clean up expired cache entries (for maintenance)."""
    logger.info("🧹 Starting cache cleanup...")
    
    try:
        if not redis_service.is_available():
            return
        
        # Redis automatically handles TTL expiration, but we can clean up patterns
        patterns_to_clean = [
            "temp:*",
            "session:expired:*",
            "old_data:*"
        ]
        
        total_deleted = 0
        for pattern in patterns_to_clean:
            deleted = await redis_service.delete_pattern(pattern)
            total_deleted += deleted
        
        logger.info(f"✅ Cache cleanup completed. Deleted {total_deleted} expired entries")
        
    except Exception as e:
        logger.error(f"❌ Cache cleanup failed: {e}")


# Export the service instance
__all__ = ["redis_service", "cache_result", "warm_up_cache", "cleanup_expired_cache"] 