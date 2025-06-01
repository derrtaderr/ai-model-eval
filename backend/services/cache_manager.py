"""
Cache Manager Service

Provides comprehensive caching functionality for performance optimization
of database queries, filter results, and computed data.
"""

import asyncio
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import redis
import pickle
from contextlib import asynccontextmanager

from config.settings import get_settings

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for cache settings."""
    redis_url: Optional[str] = None
    default_ttl: int = 3600  # 1 hour
    max_memory_items: int = 1000
    filter_cache_ttl: int = 1800  # 30 minutes
    taxonomy_cache_ttl: int = 86400  # 24 hours
    query_cache_ttl: int = 900  # 15 minutes
    enabled: bool = True

@dataclass
class CacheItem:
    """Cache item with metadata."""
    key: str
    value: Any
    expires_at: datetime
    size_bytes: int = 0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)

class MemoryCache:
    """In-memory cache with LRU eviction and TTL support."""
    
    def __init__(self, max_items: int = 1000):
        self.max_items = max_items
        self.cache: Dict[str, CacheItem] = {}
        self.access_order: List[str] = []
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        async with self._lock:
            item = self.cache.get(key)
            if not item:
                return None
            
            # Check expiration
            if datetime.utcnow() > item.expires_at:
                await self._remove(key)
                return None
            
            # Update access info
            item.access_count += 1
            item.last_accessed = datetime.utcnow()
            
            # Move to end of access order (most recently used)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return item.value
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set item in cache with TTL."""
        async with self._lock:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            size_bytes = len(pickle.dumps(value)) if value is not None else 0
            
            item = CacheItem(
                key=key,
                value=value,
                expires_at=expires_at,
                size_bytes=size_bytes
            )
            
            # Remove old item if exists
            if key in self.cache:
                await self._remove(key)
            
            # Add new item
            self.cache[key] = item
            self.access_order.append(key)
            
            # Evict if necessary
            await self._evict_if_needed()
    
    async def delete(self, key: str) -> None:
        """Delete item from cache."""
        async with self._lock:
            await self._remove(key)
    
    async def clear(self) -> None:
        """Clear all cache items."""
        async with self._lock:
            self.cache.clear()
            self.access_order.clear()
    
    async def _remove(self, key: str) -> None:
        """Remove item from cache (internal)."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)
    
    async def _evict_if_needed(self) -> None:
        """Evict least recently used items if cache is full."""
        while len(self.cache) > self.max_items:
            if self.access_order:
                lru_key = self.access_order[0]
                await self._remove(lru_key)

class RedisCache:
    """Redis-based cache for distributed caching."""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
            # Test connection
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)
            self._connected = True
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._connected = False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from Redis cache."""
        if not self._connected or not self.redis_client:
            return None
        
        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, key
            )
            return pickle.loads(data) if data else None
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set item in Redis cache with TTL."""
        if not self._connected or not self.redis_client:
            return
        
        try:
            data = pickle.dumps(value)
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.setex, key, ttl, data
            )
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete item from Redis cache."""
        if not self._connected or not self.redis_client:
            return
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.delete, key
            )
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
    
    async def clear(self) -> None:
        """Clear all items from Redis cache."""
        if not self._connected or not self.redis_client:
            return
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.flushdb
            )
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")

class CacheManager:
    """Main cache manager with fallback strategy."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.memory_cache = MemoryCache(self.config.max_memory_items)
        self.redis_cache: Optional[RedisCache] = None
        
        if self.config.redis_url:
            self.redis_cache = RedisCache(self.config.redis_url)
    
    async def initialize(self) -> None:
        """Initialize cache connections."""
        if self.redis_cache:
            await self.redis_cache.connect()
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a consistent cache key from arguments."""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            key_data += f":{':'.join(f'{k}={v}' for k, v in sorted_kwargs)}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache (Redis first, then memory)."""
        if not self.config.enabled:
            return None
        
        # Try Redis first
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                return value
        
        # Fallback to memory cache
        return await self.memory_cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache."""
        if not self.config.enabled:
            return
        
        ttl = ttl or self.config.default_ttl
        
        # Set in both caches
        await self.memory_cache.set(key, value, ttl)
        if self.redis_cache:
            await self.redis_cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> None:
        """Delete item from cache."""
        await self.memory_cache.delete(key)
        if self.redis_cache:
            await self.redis_cache.delete(key)
    
    async def clear(self) -> None:
        """Clear all cache items."""
        await self.memory_cache.clear()
        if self.redis_cache:
            await self.redis_cache.clear()
    
    # High-level caching methods
    
    async def cache_filter_results(
        self, 
        filter_config: Dict[str, Any], 
        results: List[Dict[str, Any]]
    ) -> None:
        """Cache filter query results."""
        key = self._generate_cache_key("filter", **filter_config)
        await self.set(key, results, self.config.filter_cache_ttl)
    
    async def get_cached_filter_results(
        self, 
        filter_config: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached filter results."""
        key = self._generate_cache_key("filter", **filter_config)
        return await self.get(key)
    
    async def cache_taxonomy(self, taxonomy_data: Dict[str, Any]) -> None:
        """Cache taxonomy data."""
        key = "taxonomy:current"
        await self.set(key, taxonomy_data, self.config.taxonomy_cache_ttl)
    
    async def get_cached_taxonomy(self) -> Optional[Dict[str, Any]]:
        """Get cached taxonomy data."""
        return await self.get("taxonomy:current")
    
    async def cache_query_result(
        self, 
        query_hash: str, 
        result: Any, 
        ttl: Optional[int] = None
    ) -> None:
        """Cache database query result."""
        key = f"query:{query_hash}"
        await self.set(key, result, ttl or self.config.query_cache_ttl)
    
    async def get_cached_query_result(self, query_hash: str) -> Optional[Any]:
        """Get cached query result."""
        key = f"query:{query_hash}"
        return await self.get(key)
    
    def cache_decorator(self, ttl: Optional[int] = None, key_prefix: str = "func"):
        """Decorator for caching function results."""
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(
                    f"{key_prefix}:{func.__name__}", 
                    *args, 
                    **kwargs
                )
                
                # Try to get cached result
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator
    
    # Cache invalidation methods
    
    async def invalidate_filter_cache(self) -> None:
        """Invalidate all filter-related cache entries."""
        # This would require a pattern-based deletion in Redis
        # For now, we'll clear all cache
        logger.info("Invalidating filter cache")
        await self.clear()
    
    async def invalidate_taxonomy_cache(self) -> None:
        """Invalidate taxonomy cache."""
        await self.delete("taxonomy:current")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        memory_stats = {
            "items": len(self.memory_cache.cache),
            "max_items": self.memory_cache.max_items,
            "total_size_bytes": sum(item.size_bytes for item in self.memory_cache.cache.values()),
            "avg_access_count": sum(item.access_count for item in self.memory_cache.cache.values()) / len(self.memory_cache.cache) if self.memory_cache.cache else 0
        }
        
        return {
            "enabled": self.config.enabled,
            "redis_connected": self.redis_cache._connected if self.redis_cache else False,
            "memory_cache": memory_stats
        }

# Global cache manager instance
_cache_manager: Optional[CacheManager] = None

async def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        settings = get_settings()
        config = CacheConfig(
            redis_url=getattr(settings, 'redis_url', None),
            enabled=getattr(settings, 'cache_enabled', True)
        )
        _cache_manager = CacheManager(config)
        await _cache_manager.initialize()
    return _cache_manager

# Convenience functions for common cache operations

async def cache_filter_results(filter_config: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    """Cache filter query results."""
    cache_manager = await get_cache_manager()
    await cache_manager.cache_filter_results(filter_config, results)

async def get_cached_filter_results(filter_config: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Get cached filter results."""
    cache_manager = await get_cache_manager()
    return await cache_manager.get_cached_filter_results(filter_config)

async def invalidate_caches() -> None:
    """Invalidate all caches."""
    cache_manager = await get_cache_manager()
    await cache_manager.clear() 